"""End-to-end over a real websocket with stub STT/TTS/LLM: hello, snapshot upload, a voice turn, a tool call that
round-trips through the robot, heartbeat echo, memory deltas both ways, brain.stop -> clean exit."""
import asyncio
import json
import socket

import numpy as np
import pytest
from websockets.asyncio.client import connect

from beni_brain import gateway
from beni_brain.config import Config
from beni_common.memory import MemoryStore
from beni_common.memory.sync import max_hlc
from beni_common.schemas import pack, unpack
from beni_agent.memory.sync import snapshot_bytes

TOKEN = "s3cret"


def free_port():
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


@pytest.fixture
async def brain(tmp_path, monkeypatch):
    made = []

    class B(gateway.Brain):
        def __init__(self, cfg):
            super().__init__(cfg)
            made.append(self)
    monkeypatch.setattr(gateway, "Brain", B)
    cfg = Config(token=TOKEN, host="127.0.0.1", port=free_port(), run_dir=str(tmp_path / "run"),
                 db=str(tmp_path / "mirror.db"), stub=True, tts="stub", stt_model="")
    task = asyncio.create_task(gateway.amain(cfg))
    for _ in range(100):
        if made and made[0].ready:
            break
        await asyncio.sleep(0.05)
    yield made[0], "ws://127.0.0.1:%d" % cfg.port
    made[0].request_stop("test")
    assert await asyncio.wait_for(task, 10) == 0


class Robot:
    def __init__(self, ws):
        self.ws, self.seq, self.inbox, self.seen = ws, 0, [], set()

    async def send(self, type_, **kw):
        self.seq += 1
        kw.update(type=type_, seq=self.seq)
        await self.ws.send(pack(kw))

    async def until(self, pred, timeout=5.0):
        """Next frame matching pred that no earlier until() returned (frames may arrive while waiting for others)."""
        for i, f in enumerate(self.inbox):
            if i not in self.seen and pred(f):
                self.seen.add(i)
                return f

        async def loop():
            while True:
                f = unpack(await self.ws.recv())
                self.inbox.append(f)
                if pred(f):
                    self.seen.add(len(self.inbox) - 1)
                    return f
        return await asyncio.wait_for(loop(), timeout)

    async def say(self, turn, ctx=None):
        await self.send("turn.begin", turn=turn, context=ctx or {})
        pcm = (np.zeros(640, "<i2")).tobytes()
        for _ in range(10):
            await self.send("audio.chunk", turn=turn, pcm16=pcm)
        await self.send("audio.end", turn=turn)


async def upload_snapshot(url, store):
    data = snapshot_bytes(store)
    async with connect(url + "/bulk", max_size=None) as ws:
        await ws.send(pack({"type": "bulk.put", "token": TOKEN, "kind": "memory_snapshot", "name": "memory.db.z",
                            "size": len(data), "meta": {"hlc": max_hlc(store), "codec": "zlib"}}))
        for i in range(0, len(data), 1 << 20):
            await ws.send(data[i:i + (1 << 20)])
        await ws.send(pack({"type": "bulk.end"}))
        return unpack(await ws.recv())


async def test_rejects_bad_token(brain):
    _, url = brain
    async with connect(url + "/ws") as ws:
        await ws.send(pack({"type": "hello", "token": "nope"}))
        assert unpack(await ws.recv())["error"] == "unauthorized"


async def test_full_session(brain, tmp_path):
    b, url = brain
    jet = MemoryStore(str(tmp_path / "jetson.db"), node="jetson")
    pid = jet.put("person", {"display_name": "Nimal", "relation": "owner"})
    jet.add_fact(pid, "likes", "kottu", "Nimal likes kottu")

    async with connect(url + "/ws", max_size=None) as ws:
        r = Robot(ws)
        await r.send("hello", token=TOKEN, robot_id="beni-01", versions={"agent": "0.1.0"})
        ok = unpack(await ws.recv())
        assert ok["type"] == "hello_ok" and ok["memory_since"] is None and "tools" in ok["capabilities"]

        snap_top = max_hlc(jet)
        ack = await upload_snapshot(url, jet)                           # empty brain -> full snapshot
        assert ack["ok"] and ack["rows"] >= 2
        assert b.mirror.store.facts_about(pid)[0]["object"] == "kottu"

        await r.send("heartbeat", rtt_probe=123.5)
        assert (await r.until(lambda f: f["type"] == "heartbeat"))["echo"] == 123.5

        # --- a plain voice turn
        await r.say(1, {"people_present": [{"id": pid, "name": "Nimal"}]})
        final = await r.until(lambda f: f["type"] == "stt.final")
        assert final["text"] == "hello beni"
        end = await r.until(lambda f: f["type"] == "turn.end" and f["turn"] == 1)
        chunks = [f for f in r.inbox if f["type"] == "tts.chunk" and f["turn"] == 1]
        assert chunks[0]["pcm16_24k"] and chunks[0]["emo"] == "happy" and chunks[-1]["final"]
        assert "You said: hello beni." in "".join(f["text"] for f in r.inbox if f["type"] == "llm.delta")
        assert end.get("reason") is None

        # --- a tool call: the brain asks the robot to act, waits for action.result, then answers
        b.stt.text = "spin around"
        await r.say(2)
        act = await r.until(lambda f: f["type"] == "action")
        assert act["name"] == "look_at" and act["args"] == {"target": "left"}
        await r.send("action.result", call_id=act["call_id"], ok=True, result={"ok": True})
        await r.until(lambda f: f["type"] == "turn.end" and f["turn"] == 2)

        # --- reflex: "stop" never reaches the LLM
        b.stt.text = "stop!"
        await r.say(3)
        stop = await r.until(lambda f: f["type"] == "action")
        assert stop["name"] == "stop"
        await r.until(lambda f: f["type"] == "turn.end" and f["turn"] == 3)

        # --- memory: the brain's extracted episodes flow back as deltas; ours flow in and get acked
        delta = await r.until(lambda f: f["type"] == "memory.delta", timeout=10)
        applied, _ = jet.apply_delta(delta["rows"])
        assert applied and all(row["hlc"].endswith("-kaggle") for rows in delta["rows"].values() for row in rows)
        await r.send("memory.ack", hlc=delta["hlc"])
        nid = jet.add_fact(pid, "birthday", "March 3", "Nimal's birthday is March 3")
        rows, cur = jet.changes_since(snap_top, exclude_node="kaggle")
        await r.send("memory.delta", rows=rows, hlc=cur)
        assert (await r.until(lambda f: f["type"] == "memory.ack"))["hlc"] == cur
        assert b.mirror.store.get("fact", nid)["object"] == "March 3"

        await r.send("brain.stop", reason="test")          # consolidate, flush the last deltas, then close
        while True:
            f = await r.until(lambda f: f["type"] in ("memory.delta", "error"), timeout=10)
            if f["type"] == "error":
                break
            jet.apply_delta(f["rows"])
            await r.send("memory.ack", hlc=f["hlc"])
        assert "shutting down" in f["error"]
        assert b.stop_reason == "test"
        assert jet.q("SELECT count(*) AS n FROM episode WHERE hlc LIKE '%-kaggle' AND kind != 'action'")[0]["n"] == 3
        acts = jet.q("SELECT media_ref FROM episode WHERE kind='action'")          # the skill miner's input (§11.9)
        assert len(acts) == 1 and json.loads(acts[0]["media_ref"])["ok"] is True
    jet.close()


async def test_replay_flag(brain):
    b, url = brain
    finds = []

    def find(jpeg, phrase):
        finds.append(phrase)
        return ([[1, 2, 3, 4]] if phrase == "keys" else []), (1280, 720)
    b.vision.find = find
    async with connect(url + "/ws", max_size=None) as ws:
        r = Robot(ws)
        await r.send("hello", token=TOKEN, robot_id="beni-01", versions={"agent": "0.1.0"})
        assert unpack(await ws.recv())["type"] == "hello_ok"
        want = ["keys", "umbrella", "a", "b", "c", "d"]
        await r.send("replay.flag", id="ep1", jpeg=b"\xff\xd8jpg", cam=1, ts=1.0, want=want)
        res = await r.until(lambda f: f["type"] == "replay.result")
        assert res["id"] == "ep1" and res["caption"] == "an indoor room" and res["found"] == ["keys"]
        assert finds == want[:5]
        await r.send("replay.flag", id="ep2", cam=1, ts=1.0)             # no keyframe: empty result
        res = await r.until(lambda f: f["type"] == "replay.result")
        assert res == dict(res, id="ep2", caption="", found=[])
