"""World state: who/what is around, robot state, recent events -> the context sent with turn.begin (§9.2)."""
import collections
import time

from beni_common import schemas as S

from .identity import FaceIdentifier

COCO_PERSON = 0


class World:
    def __init__(self, store, identifier: FaceIdentifier, events=None, clock=time.time, places=None):
        self.store, self.ident, self.events, self.clock, self.places = store, identifier, events, clock, places
        self.robot = {}                 # last robot_state message
        self.place = None
        self.persons_in_view = 0
        self.last_person_t = 0.0
        self.last_scene_caption = ""
        self.privacy = False            # §12.3 camera privacy mode (actions.py)
        self.estop = False              # e-stop GPIO held (main.py): no motion requests
        self.recent = collections.deque(maxlen=12)
        self.unknown_tracks = []
        self._names = {}
        self.phones_home = lambda: []   # set by the agent (perception/presence.py)
        self.on_sighting = None         # (pid, name, cam, place) -> None: an arrival (perception/thumbs.py)

    # ------------------------------------------------------------------ inputs
    def on_vision(self, topic, msg):
        if topic != S.T_DET:
            return
        for fr in msg.get("frames", ()):
            persons = sum(1 for o in fr.get("objs", ()) if o.get("cls") == COCO_PERSON and o.get("gie", 1) == 1)
            if fr.get("cam", 0) == 0:
                self.persons_in_view = persons
            if persons:
                self.last_person_t = self.clock()
            for kind, pid, key in self.ident.on_frame(fr):
                self._on_identity(kind, pid, key)

    def _on_identity(self, kind, pid, key):  # key = (cam, track)
        now = self.clock()
        if kind == "unknown_face":
            self.unknown_tracks = [k for k in self.unknown_tracks if k in self.ident.tracks] + [key]
        elif pid:
            name = self.name(pid)
            self.note("%s %s" % (name, "arrived" if kind == "person_seen" else "left"))
            if kind == "person_seen":
                p = self.store.get("person", pid) or {}
                self.store.put("person", {"id": pid, "last_seen": now, "first_seen": p.get("first_seen") or now,
                                          "times_seen": (p.get("times_seen") or 0) + 1})
                if self.on_sighting is not None:
                    self.on_sighting(pid, name, key[0], self.place)
        if self.events is not None:
            self.events.send(b"event", name=kind, person=pid, track=list(key))

    def on_robot_state(self, topic, msg):
        self.robot = msg
        pl = msg.get("place")
        if self.places is not None:
            entered, nav = self.places.on_state(msg)
            pl = entered["name"] if entered else None
            if nav is not None and not nav["success"] and nav["notes"]:
                self.note(nav["notes"])
        if pl and pl != self.place:
            self.place = pl
            self.note("moved to %s" % pl)

    def note(self, text):
        self.recent.append((self.clock(), text))

    # ------------------------------------------------------------------ outputs
    def name(self, pid):
        n = self._names.get(pid)
        if n is None:
            p = self.store.get("person", pid)
            n = self._names[pid] = (p or {}).get("display_name") or pid
        return n

    def invalidate_names(self):
        self._names.clear()

    @property
    def battery(self):
        return self.robot.get("battery")

    def people_present(self):
        return self.ident.present()

    def context(self):
        now = self.clock()
        return {
            "people_present": [{"id": p, "name": self.name(p)} for p in self.people_present()],
            "unknown_faces": len([k for k in self.unknown_tracks if k in self.ident.tracks]),
            "persons_in_view": self.persons_in_view,
            "phones_home": self.phones_home(),
            "place": self.place,
            "time": time.strftime("%A %Y-%m-%d %H:%M", time.localtime(now)),
            "battery": self.battery,
            "robot_state": dict({k: self.robot.get(k) for k in ("docked", "charging", "moving", "nav")
                                 if k in self.robot}, **({"camera_privacy": True} if self.privacy else {})),
            "last_scene_caption": self.last_scene_caption,
            "recent_events": ["%s ago: %s" % (_ago(now - t), s) for t, s in list(self.recent)[-6:]],
        }


def _ago(s):
    return "%ds" % s if s < 90 else "%dmin" % (s // 60) if s < 5400 else "%dh" % (s // 3600)
