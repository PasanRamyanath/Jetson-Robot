"""Spatial memory on the Jetson (§11.10): named places, object sightings -> beliefs, navigation experience.

The agent owns places (its DB is the system of record); the ROS bridge only knows poses. Everything here is cheap
dict/SQL work done at robot_state rate (10 Hz) with the heavy parts (DB writes) rate-limited.
"""
import json
import math
import re
import time

from beni_common import schemas as S

_ART = re.compile(r"^(the|my|our|a|an)\s+|'s\b|\s+(room|area)$")
DOCK_WORDS = {"dock", "charger", "charging station", "base", "home base"}
SIGHT_EVERY_S = 120.0       # one sighting row per label per 2 min unless it moved
SIGHT_MOVE_M = 1.0
SIGHT_MIN_CONF = 0.5
SKIP_CLS = {0}              # people are tracked by identity, not as objects
NAV_OPS = {"goto", "dock"}


def norm(s):
    s = " ".join((s or "").lower().replace("_", " ").split())
    return _ART.sub("", _ART.sub("", s)).strip()


def belief_id(label):
    return "obj-" + label.replace(" ", "_")


class Places:
    def __init__(self, store, clock=time.time):
        self.store, self.clock = store, clock
        self._rows = None
        self._seen = {}             # label -> (t, x, y) of the last sighting row
        self._task = None           # (id, status, from_place, goal) of the last goto/dock seen in robot_state
        self.current = None         # place row the robot is in

    # ------------------------------------------------------------------ places
    def rows(self):
        if self._rows is None:
            self._rows = self.store.q("SELECT * FROM place WHERE deleted=0 AND x IS NOT NULL")
        return self._rows

    def invalidate(self):
        self._rows = None

    def resolve(self, name):
        """Place row for a spoken name: exact name/alias, then dock words, then word containment."""
        n = norm(name)
        if not n:
            return None
        rows = self.rows()
        for r in rows:
            if n == norm(r["name"]) or n in (norm(a) for a in _aliases(r)):
                return r
        if n in DOCK_WORDS:
            return self.dock()
        for r in rows:
            names = [norm(r["name"])] + [norm(a) for a in _aliases(r)]
            if any(x and (re.search(r"\b%s\b" % re.escape(x), n) or re.search(r"\b%s\b" % re.escape(n), x))
                   for x in names):
                return r
        return None

    def dock(self):
        return next((r for r in self.rows() if r.get("kind") == "dock"), None)

    def at(self, x, y):
        """Nearest place whose radius contains (x, y)."""
        best, bd = None, 1e9
        for r in self.rows():
            d = math.hypot(r["x"] - x, r["y"] - y)
            if d <= (r.get("radius") or 0.8) and d < bd:
                best, bd = r, d
        return best

    def save(self, name, pose, kind=None, map_id="home", radius=None):
        """Create or move a named place to `pose` {x, y, yaw}. Returns the row id."""
        name = " ".join((name or "").split())
        if not name or not pose:
            return None
        if kind is None and norm(name) in DOCK_WORDS:
            kind = "dock"
        old = self.dock() if kind == "dock" else None
        old = old or next((r for r in self.rows() if norm(r["name"]) == norm(name)), None)
        row = {"name": name, "map_id": map_id, "x": float(pose["x"]), "y": float(pose["y"]),
               "yaw": float(pose.get("yaw") or 0.0), "kind": kind or (old or {}).get("kind") or "room"}
        if old:
            row["id"] = old["id"]
        if radius:
            row["radius"] = float(radius)
        pid = self.store.put("place", row)
        self.invalidate()
        return pid

    # ------------------------------------------------------------------ robot_state
    def on_state(self, msg):
        """Returns (entered_place_row_or_None, finished_nav_task_row_or_None)."""
        pose, now, entered = msg.get("pose"), self.clock(), None
        if pose:
            p = self.at(pose["x"], pose["y"])
            if p is not None and (self.current is None or p["id"] != self.current["id"]):
                entered = p
            if p is not None or self.current is None or _far(self.current, pose):
                self.current = p
            objs = msg.get("objs")
            if objs:
                self.on_objects(objs, now)
        return entered, self._on_task(msg.get("task"), now)

    def on_objects(self, objs, now=None):
        now = now or self.clock()
        pid = self.current["id"] if self.current else None
        for o in objs:
            c = o.get("cls", -1)
            if c in SKIP_CLS or not 0 <= c < len(S.COCO) or (o.get("conf") or 0) < SIGHT_MIN_CONF or "x" not in o:
                continue
            label, x, y = S.COCO[c], o["x"], o["y"]
            last = self._seen.get(label)
            if last and now - last[0] < SIGHT_EVERY_S and math.hypot(x - last[1], y - last[2]) < SIGHT_MOVE_M:
                continue
            self._seen[label] = (now, x, y)
            self.store.put("object_sighting", {"label": label, "x": x, "y": y, "z": 0.0, "place_id": pid,
                                               "cam": str(o.get("cam", 0)), "conf": o.get("conf"), "t": now})
            b = self.store.get("object_belief", belief_id(label)) or {}
            hist = json.loads(b.get("place_hist") or "{}")
            if pid:
                hist[pid] = hist.get(pid, 0) + 1
            self.store.put("object_belief", {"id": belief_id(label), "label": label, "last_place_id": pid,
                                             "last_xy": json.dumps([round(x, 2), round(y, 2)]), "last_seen": now,
                                             "place_hist": json.dumps(hist)})

    def _on_task(self, task, now):
        if not task or task.get("op") not in NAV_OPS:
            return None
        key, prev = task.get("id"), self._task
        if task.get("status") == "running":
            if prev is None or prev[0] != key:
                goal = "dock" if task["op"] == "dock" else task.get("detail") or None
                self._task = (key, "running", self.current["name"] if self.current else None, goal)
            return None
        if prev is None or prev[0] != key or prev[1] != "running":
            return None
        self._task = (key, task.get("status"), prev[2], prev[3])
        row = {"t": now, "from_place": prev[2], "to_place": prev[3], "success": int(task.get("status") == "succeeded"),
               "duration": task.get("dur"), "recoveries": task.get("recov", 0),
               "stuck_xy": json.dumps(task.get("stuck") or []), "notes": task.get("detail")}
        row["id"] = self.store.put("nav_experience", row)
        return row


def _aliases(r):
    a = r.get("aliases")
    if not a:
        return []
    try:
        v = json.loads(a)
        return v if isinstance(v, list) else [str(v)]
    except ValueError:
        return [s.strip() for s in a.split(",")]


def _far(place, pose):
    return math.hypot(place["x"] - pose["x"], place["y"] - pose["y"]) > (place.get("radius") or 0.8) + 0.5
