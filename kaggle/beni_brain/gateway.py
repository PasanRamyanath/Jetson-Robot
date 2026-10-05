"""Brain gateway (§9.2, §10.5): python -m beni_brain.gateway [--port 8765] [--stub]

websockets server: /ws (real-time session, one robot), /bulk (snapshot/exemplar uploads), /health (200 once the LLM,
STT and TTS are loaded). Exits cleanly on SIGTERM or `brain.stop`: consolidate memory if not done recently, flush
memory deltas to the robot, close. The notebook's keep-alive loop then ends, which stops the Kaggle session.
"""
import argparse
import asyncio
import hmac
import logging
import os
import signal
import sys
import time
from http import HTTPStatus

from websockets.asyncio.server import serve
from websockets.exceptions import ConnectionClosed

from beni_common.memory import embed as embed_mod
from beni_common.schemas import pack, unpack

from . import __version__, llm, stt, tts, vision_tools
from . import prompts as P
from .config import Config
from .memory import consolidate
from .memory.extract import Extractor
from .memory.mirror import Mirror
from .memory.rerank import Reranker
from .session import Session

log = logging.getLogger("gateway")
MAX_FRAME = 8 << 20
CONSOLIDATE_EVERY_S = 6 * 3600


class Brain:
    """Process-wide state shared by sessions: models, memory mirror, background work, stop signal."""

    def __init__(self, cfg):
        self.cfg = cfg
        os.makedirs(cfg.run_dir, exist_ok=True)
        embed = (embed_mod.Embedder(cfg.embed_dir, threads=4) if cfg.embed_dir else embed_mod.HashEmbedder())
        rerank = None if cfg.stub or cfg.reranker == "none" else Reranker(cfg.reranker)
        self.mirror = Mirror(cfg.db, embed, rerank)
        self.llm = llm.load(cfg)
        self.stt = stt.load(cfg)
        self.tts = tts.load(cfg)
        self.vision = vision_tools.load(cfg)
        self.extractor = Extractor(self.llm, self.mirror)
        self.session = None
        self.ready = False
        self.stop_event = asyncio.Event()
        self.stop_reason = None
        self.last_activity = time.monotonic()
        self._bg = set()

    def info(self):
        return {"version": __version__, "llm": self.cfg.llm_model, "stt": type(self.stt).__name__,
                "tts": type(self.tts).__name__, "vision": type(self.vision).__name__,
                "embed": type(self.mirror.embed).__name__}

    def touch(self):
        self.last_activity = time.monotonic()

    def spawn(self, coro):
        t = asyncio.create_task(coro)
        self._bg.add(t)
        t.add_done_callback(self._bg.discard)
        return t

    def request_stop(self, reason):
        if not self.stop_event.is_set():
            log.info("stop requested: %s", reason)
            self.stop_reason = reason
            self.stop_event.set()

    async def warmup(self):
        if await self.llm.ready():
            await self.llm.warmup(P.system(self.cfg, {}, ""))
            self.ready = True
            log.info("brain ready: %s", self.info())
        else:
            log.error("LLM never became ready")
            self.request_stop("llm_failed")

    async def consolidate(self, budget_s=600):
        last = self.mirror.store.kv_get("consolidate.last", 0) or 0
        if not self.mirror.loaded or time.time() - last < CONSOLIDATE_EVERY_S:
            return None
        return await consolidate.run(self.mirror, self.extractor, self.llm, budget_s)

    async def maintenance(self):
        """Idle-time consolidation: after cfg.idle_consolidate_s without turns, at most every 6 h."""
        while True:
            await asyncio.sleep(30)
            if self.ready and time.monotonic() - self.last_activity > self.cfg.idle_consolidate_s:
                try:
                    await self.consolidate()
                except Exception:
                    log.exception("consolidation")

    async def flush_memory(self, timeout=30.0):
        s = self.mirror.sync
        if s is None or not s.ready:
            return
        end = time.monotonic() + timeout
        while time.monotonic() < end and s.online():
            if s._inflight is None and not await s.push_once():
                return
            await asyncio.sleep(0.2)

    # ------------------------------------------------------------------ connections
    def health(self, connection, request):
        if request.path.startswith("/health"):
            ok = self.ready and not self.stop_event.is_set()
            return connection.respond(HTTPStatus.OK if ok else HTTPStatus.SERVICE_UNAVAILABLE,
                                      "ok\n" if ok else "starting\n")
        if not request.path.startswith(("/ws", "/bulk")):
            return connection.respond(HTTPStatus.NOT_FOUND, "not found\n")
        return None

    def _auth(self, token):
        return not self.cfg.token or hmac.compare_digest(str(token or ""), self.cfg.token)

    async def handler(self, ws):
        path = ws.request.path
        try:
            if path.startswith("/bulk"):
                await self.bulk(ws)
            else:
                await self.realtime(ws)
        except ConnectionClosed:
            pass

    async def realtime(self, ws):
        hello = unpack(await asyncio.wait_for(ws.recv(), 10))
        if hello.get("type") != "hello" or not self._auth(hello.get("token")):
            await ws.send(pack({"type": "error", "error": "unauthorized", "seq": 0, "t": time.time()}))
            return
        if not self.ready or self.stop_event.is_set():
            await ws.send(pack({"type": "error", "error": "warming up" if not self.ready else "stopping",
                                "seq": 0, "t": time.time()}))
            return
        if self.session is not None and self.session.open:          # the robot reconnected: drop the stale socket
            await self.session.ws.close()
        await ws.send(pack({"type": "hello_ok", "seq": 0, "t": time.time(), "models": self.info(),
                            "capabilities": ["opus", "pcm16", "vision", "tools", "memory"],
                            "memory_since": self.mirror.memory_since()}))
        log.info("robot %s connected (versions %s)", hello.get("robot_id"), hello.get("versions"))
        self.session = s = Session(self, ws, hello)
        self.touch()
        await s.run()
        log.info("robot %s disconnected", hello.get("robot_id"))

    async def bulk(self, ws):
        """bulk.put{token, kind, name, size, meta} -> binary chunks until `size` bytes -> bulk.end -> ack."""
        head = unpack(await asyncio.wait_for(ws.recv(), 10))
        if head.get("type") != "bulk.put" or not self._auth(head.get("token")):
            await ws.send(pack({"ok": False, "error": "unauthorized"}))
            return
        size, parts, got = int(head.get("size") or 0), [], 0
        while got < size:
            b = await asyncio.wait_for(ws.recv(), 60)
            parts.append(b)
            got += len(b)
        end = unpack(await asyncio.wait_for(ws.recv(), 60))
        data = b"".join(parts)
        if end.get("type") != "bulk.end" or got != size:
            await ws.send(pack({"ok": False, "error": "size mismatch"}))
            return
        kind, name = head.get("kind"), os.path.basename(head.get("name") or "blob")
        try:
            if kind == "memory_snapshot":
                n = await asyncio.to_thread(self.mirror.load_snapshot, data, head.get("meta") or {})
                self.mirror.activate()
                ack = {"ok": True, "rows": n}
            else:
                d = os.path.join(self.cfg.run_dir, "bulk", kind or "misc")
                os.makedirs(d, exist_ok=True)
                with open(os.path.join(d, name), "wb") as f:
                    f.write(data)
                ack = {"ok": True, "path": os.path.join(d, name)}
        except Exception as e:
            log.exception("bulk %s", kind)
            ack = {"ok": False, "error": str(e)[:200]}
        log.info("bulk %s/%s %d bytes -> %s", kind, name, size, ack)
        await ws.send(pack(ack))

    async def shutdown(self):
        log.info("shutting down (%s)", self.stop_reason)
        if self.stop_reason not in ("sigterm", "llm_failed"):
            try:
                await self.consolidate(budget_s=300)
            except Exception:
                log.exception("final consolidation")
        await self.flush_memory()
        if self.session is not None and self.session.open:
            await self.session.send("error", error="brain shutting down")
            await self.session.ws.close()
        for t in list(self._bg):
            t.cancel()


async def amain(cfg):
    brain = Brain(cfg)
    loop = asyncio.get_running_loop()
    for sig in (signal.SIGTERM, signal.SIGINT):
        try:
            loop.add_signal_handler(sig, brain.request_stop, "sigterm")
        except (NotImplementedError, RuntimeError):
            pass
    pid_file = os.path.join(cfg.run_dir, "gateway.pid")
    with open(pid_file, "w") as f:
        f.write(str(os.getpid()))
    async with serve(brain.handler, cfg.host, cfg.port, process_request=brain.health, max_size=MAX_FRAME,
                     compression=None, ping_interval=20, ping_timeout=20):
        log.info("listening on %s:%d", cfg.host, cfg.port)
        brain.spawn(brain.warmup())
        brain.spawn(brain.maintenance())
        await brain.stop_event.wait()
        await brain.shutdown()
    try:
        os.unlink(pid_file)
    except OSError:
        pass
    return 0


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--port", type=int)
    ap.add_argument("--host")
    ap.add_argument("--stub", action="store_true", help="stub STT/TTS/LLM (no GPU)")
    a = ap.parse_args(argv)
    if a.stub:
        os.environ["BENI_STUB"] = "1"
    over = {k: v for k, v in (("port", a.port), ("host", a.host)) if v}
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname).1s %(name)s: %(message)s")
    return asyncio.run(amain(Config.from_env(**over)))


if __name__ == "__main__":
    sys.exit(main())
