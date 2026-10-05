"""Executes tool calls (§10.6) on the robot: `action{name,args,call_id}` from the brain or local reflexes.

Physical tools -> robot_cmd REQ (ROS 2 bridge); expressions -> face_ctrl PUSH; snapshots -> vision_ctrl REQ + the
jpeg arriving on the vision PUB; reminders/enrollment are local (the Jetson owns time and identity).
"""
import asyncio
import itertools
import logging
import time

from beni_common import schemas as S

log = logging.getLogger("actions")
ROBOT_OPS = {"move_to": "goto", "look_at": "look_at", "follow_person": "follow", "stop": "stop",
             "goto_dock": "dock", "come_here": "come_here", "search_room": "search"}
CAMS = {"head": 0, "front": 1}
VISION_UNITS = ("beni-vision-core", "beni-vision")


async def _cmd(*argv):
    """-> (returncode, stdout); 1 when the command can't run (e.g. no systemd on a dev PC)."""
    try:
        p = await asyncio.create_subprocess_exec(*argv, stdout=asyncio.subprocess.PIPE,
                                                 stderr=asyncio.subprocess.DEVNULL)
        out, _ = await asyncio.wait_for(p.communicate(), 30)
        return p.returncode, out.decode(errors="replace").strip()
    except (OSError, asyncio.TimeoutError) as e:
        log.warning("%s: %s", " ".join(argv), e)
        return 1, ""


class Actions:
    def __init__(self, bus, world=None, identifier=None, proactive=None, store=None, sync=None, link=None,
                 places=None):
        self.bus, self.world, self.ident, self.proactive = bus, world, identifier, proactive
        self.store, self.sync, self.link, self.places = store, sync, link, places
        self.face = bus.push(S.EP["face_ctrl"])
        self._snaps = {}
        self._req = itertools.count(1)
        self.enrolling = False
        self.emit = None                # events publisher send(topic, **fields), set by the agent
        self.cmd = _cmd

    # ------------------------------------------------------------------ snapshots
    def on_vision(self, topic, msg):
        if topic == S.T_JPEG:
            fut = self._snaps.pop(msg.get("req"), None)
            if fut is not None and not fut.done():
                fut.set_result(msg)

    async def snapshot(self, cam=0, op="snapshot", timeout=1.5, **kw):
        """A jpeg message from vision: the 1024x576 VLM snapshot, or {op: thumb} crops (perception/thumbs.py)."""
        req = next(self._req)
        fut = asyncio.get_running_loop().create_future()
        self._snaps[req] = fut
        kw.update(op=op, req=req, cam=cam)
        r = await self.bus.request(S.EP["vision_ctrl"], kw)
        if not r.get("ok"):
            self._snaps.pop(req, None)
            return None
        try:
            return await asyncio.wait_for(fut, timeout)
        except asyncio.TimeoutError:
            self._snaps.pop(req, None)
            return None

    async def send_snapshot(self, cam=0, turn=0, req=None):
        m = await self.snapshot(cam)
        if m is None or self.link is None:
            return False
        return await self.link.send("vision.snapshot", turn=turn, req=req, jpeg=m["jpeg"], cam=m.get("cam", cam),
                                    ts=m.get("t"), detections=m.get("objs", []))

    # ------------------------------------------------------------------ dispatch
    async def __call__(self, name, args, turn=0):
        try:
            return await self.run(name, args or {}, turn)
        except Exception as e:
            log.exception("action %s", name)
            return {"ok": False, "error": str(e)}

    async def run(self, name, args, turn=0):
        if name in ROBOT_OPS:
            if name != "stop" and getattr(self.world, "estop", False):
                return {"ok": False, "result": "the emergency stop is on"}
            req = self.robot_request(name, args)
            if "error" in req:
                return {"ok": False, "result": req["error"]}
            r = await self.bus.request(S.EP["robot_cmd"], req, timeout=3.0)
            if name == "search_room" and not r.get("ok") and req.get("object"):
                self.want(req["object"])
            return {"ok": bool(r.get("ok")), "result": r.get("result") or r.get("err")}
        if name == "save_place":
            return await self.save_place(args.get("name") or args.get("place"), args.get("kind"))
        if name == "set_expression":
            self.face.send(S.pack(S.envelope("agent", op="expression", name=args.get("expression", "happy"))))
            return {"ok": True}
        if name == "take_snapshot":
            ok = await self.send_snapshot(CAMS.get(args.get("camera", "head"), 0), turn)
            return {"ok": ok, "result": "image attached" if ok else "camera unavailable"}
        if name == "set_reminder" and self.proactive is not None:
            return self.proactive.add_reminder(args.get("who"), args.get("when"), args.get("what"))
        if name == "enroll_person":
            return await self.enroll(args.get("name", "").strip())
        if name == "forget_person":
            return self.forget(args.get("person_id"))
        if name == "camera_privacy":
            return await self.privacy(str(args.get("on", True)).lower() not in ("false", "0", "off", "no"))
        return {"ok": False, "error": "unknown action %s" % name}

    # ------------------------------------------------------------------ camera privacy (§12.3)
    async def privacy(self, on):
        """Cameras off: the running vision unit is stopped (its pipeline goes to NULL) and the eyes close.
        Off: the unit that was stopped starts again. Kept in kv so a reboot or agent restart re-applies it."""
        state = (self.store.kv_get("privacy") if self.store is not None else None) or {}
        if on:
            units = [u for u in VISION_UNITS if (await self.cmd("systemctl", "is-active", u))[1] == "active"]
            units = units or state.get("units") or []
            ok = all([(await self.cmd("sudo", "-n", "/bin/systemctl", "stop", u))[0] == 0 for u in units])
            state = {"on": True, "units": units or state.get("units") or [VISION_UNITS[0]]}
        else:
            units = state.get("units") or [VISION_UNITS[0]]
            ok = (await self.cmd("sudo", "-n", "/bin/systemctl", "start", units[0]))[0] == 0
            state = {"on": False, "units": units}
        if self.store is not None:
            self.store.kv_set("privacy", state)
        if self.world is not None:
            self.world.privacy = on
        if self.emit is not None:
            self.emit(b"event", name="privacy", on=int(on))
        log.info("camera privacy %s (%s, ok=%s)", "on" if on else "off", state["units"], ok)
        return {"ok": ok, "result": "cameras off, eyes closed" if on else "cameras on"}

    async def restore_privacy(self):
        """Startup: a vision unit that systemd started at boot is stopped again if privacy mode was on."""
        state = (self.store.kv_get("privacy") if self.store is not None else None) or {}
        if state.get("on"):
            await self.privacy(True)

    # ------------------------------------------------------------------ robot ops
    def robot_request(self, name, args):
        """Tool args -> bridge op: the agent owns places and identities, the bridge only poses and track ids."""
        req = dict(args, op=ROBOT_OPS[name])
        if name == "move_to":
            p = self.places.resolve(args.get("place")) if self.places is not None else None
            if p is None:
                return {"error": "I don't know where %s is" % (args.get("place") or "that")}
            req.update(x=p["x"], y=p["y"], yaw=p.get("yaw") or 0.0, place=p["name"])
        elif name == "goto_dock":
            p = self.places.dock() if self.places is not None else None
            if p is not None:
                req.update(x=p["x"], y=p["y"], yaw=p.get("yaw") or 0.0)
        elif name in ("follow_person", "look_at", "come_here"):
            key = self.track_of(args.get("name") or args.get("target"))
            if key is not None:
                req.update(cam=key[0], tid=key[1])
        elif name == "search_room":
            req["object"] = args.get("object") or args.get("target") or ""
        return req

    def track_of(self, who):
        """(cam, tid) of the freshest face track identified as `who` (a display name or person id), else None."""
        who = (who or "").strip().lower()
        if not who or who in ("me", "you", "us", "someone", "anyone") or self.ident is None or self.store is None:
            return None
        pids = {r["id"] for r in self.store.q("SELECT id FROM person WHERE deleted=0 AND (lower(display_name)=? "
                                                "OR id=?)", (who, who))}
        tracks = [t for t in self.ident.tracks.values() if t.person in pids]
        return max(tracks, key=lambda t: t.last).key if tracks else None

    def want(self, label):
        """Asked for, not found: sleep replay looks for it in the day's recordings (perception/replay.py)."""
        if self.store is None:
            return
        label = label.strip().lower()
        w = [x for x in self.store.kv_get("wanted_objects", []) or [] if x != label]
        self.store.kv_set("wanted_objects", (w + [label])[-20:])

    async def save_place(self, name, kind=None):
        if not name or self.places is None:
            return {"ok": False, "error": "need a place name"}
        r = await self.bus.request(S.EP["robot_cmd"], {"op": "save_place"}, timeout=3.0)
        if not r.get("ok"):
            return {"ok": False, "result": r.get("err") or "I don't know where I am"}
        pose = r["result"]
        pid = self.places.save(name, pose, kind, pose.get("map_id") or "home")
        if self.world is not None:
            self.world.place = name
        if self.sync is not None:
            self.sync.nudge()
        return {"ok": True, "place_id": pid, "result": "saved this spot as %s" % name}

    async def on_frame(self, f):
        """Brain -> robot: `action` and `vision.request`."""
        if f["type"] == "action":
            res = await self(f.get("name"), f.get("args"), f.get("turn", 0))
            if self.link is not None:
                await self.link.send("action.result", turn=f.get("turn", 0), call_id=f.get("call_id"),
                                     ok=res.get("ok", False), result=res)
        elif f["type"] == "vision.request":
            await self.send_snapshot(f.get("cam", 0), f.get("turn", 0), f.get("req"))

    # ------------------------------------------------------------------ identity
    async def enroll(self, name, seconds=8.0):
        if not name or self.ident is None or self.store is None:
            return {"ok": False, "error": "need a name"}
        if self.enrolling:
            return {"ok": False, "error": "already enrolling"}
        self.enrolling = True
        try:
            rows = self.store.q("SELECT id FROM person WHERE deleted=0 AND lower(display_name)=lower(?)", (name,))
            pid = rows[0]["id"] if rows else self.store.put(
                "person", {"display_name": name, "relation": "family", "consent_face": 1, "first_seen": time.time()})
            self.ident.start_enroll()
            await asyncio.sleep(seconds)
            n = self.ident.finish_enroll(pid)
            if self.world is not None:
                self.world.invalidate_names()
            if self.sync is not None:
                self.sync.nudge()
            return {"ok": n > 0, "person_id": pid, "exemplars": n,
                    "result": "learned %s's face" % name if n else "I couldn't see a face clearly"}
        finally:
            self.enrolling = False

    def forget(self, pid):
        if not pid or self.store is None:
            return {"ok": False, "error": "no person"}
        n = self.store.forget_person(pid)
        if self.ident is not None:
            self.ident.g.reload()
        if self.world is not None:
            self.world.invalidate_names()
        if self.sync is not None:
            self.sync.nudge()
        return {"ok": True, "removed": n}
