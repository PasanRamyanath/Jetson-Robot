"""Jetson -> Kaggle brain WebSocket client (§9.2, §9.3). websockets 13.1 legacy client, msgpack binary frames.

Single real-time socket (/ws); bulk transfers (DB snapshots, exemplar batches) use a separate /bulk connection
so they never delay audio. Heartbeat every 5 s carries an RTT probe; 3 missed -> offline.
"""
import asyncio
import logging
import time

import msgpack
import websockets

from ..util import spawn

log = logging.getLogger("cloud")
HB_S, HB_MISS = 5.0, 3
BULK_CHUNK = 1 << 20
VERSIONS = {"proto": 1, "agent": "0.1.0"}


def _pack(d):
    return msgpack.packb(d, use_bin_type=True)


def _unpack(b):
    return msgpack.unpackb(b, raw=False)


def bulk_url(ws_url):
    return ws_url[:-3] + "/bulk" if ws_url.endswith("/ws") else ws_url.rstrip("/") + "/bulk"


class CloudLink:
    """on_frame: async fn(frame dict). on_state: optional fn(online: bool, info: dict)."""

    def __init__(self, urls, token, on_frame, robot_id="beni-01", memory_rev=lambda: "", on_state=None):
        self.urls, self.token, self.on_frame, self.robot_id = list(urls), token, on_frame, robot_id
        self.memory_rev, self.on_state = memory_rev, on_state
        self.ws, self.url, self.online = None, None, asyncio.Event()
        self.seq, self.rtt_ms, self.info = 0, None, {}
        self._last_rx = 0.0
        self._lock = asyncio.Lock()        # one writer at a time keeps seq monotonic on the wire
        self._wanted = asyncio.Event()     # cleared while the lifecycle manager says the brain is down
        self._wanted.set()

    def pause(self):
        """Stop reconnect attempts (brain known to be off: saves radio time and log spam)."""
        self._wanted.clear()
        if self.ws is not None:
            spawn(self.ws.close())

    def resume(self):
        self._wanted.set()

    async def run(self):
        backoff = 1
        while True:
            await self._wanted.wait()
            for url in self.urls:
                try:
                    async with websockets.connect(url, max_size=8 << 20, ping_interval=HB_S, ping_timeout=2 * HB_S,
                                                  compression=None, open_timeout=5, close_timeout=2) as ws:
                        await ws.send(_pack({"type": "hello", "token": self.token, "robot_id": self.robot_id,
                                             "versions": VERSIONS, "memory_rev": self.memory_rev(),
                                             "seq": 0, "t": time.time()}))
                        hello = _unpack(await asyncio.wait_for(ws.recv(), 10))
                        if not isinstance(hello, dict) or hello.get("type") != "hello_ok":
                            raise RuntimeError("handshake rejected: %s" % hello.get("error", hello.get("type")))
                        self.ws, self.url, self.info, backoff = ws, url, hello, 1
                        self._last_rx = time.monotonic()
                        self.online.set()
                        log.info("brain online via %s (%s)", url, hello.get("models"))
                        if self.on_state:
                            self.on_state(True, hello)
                        hb = spawn(self._heartbeat(ws))
                        try:
                            async for raw in ws:
                                self._last_rx = time.monotonic()
                                try:
                                    f = _unpack(raw)
                                except Exception:
                                    f = None
                                if not isinstance(f, dict):
                                    log.warning("dropping a malformed frame (%d bytes)", len(raw))
                                    continue
                                if f.get("type") == "heartbeat":
                                    if f.get("echo"):
                                        self.rtt_ms = (time.time() - f["echo"]) * 1000.0
                                    elif "rtt_probe" in f:
                                        await self.send("heartbeat", echo=f["rtt_probe"])
                                    continue
                                try:
                                    await self.on_frame(f)
                                except Exception:
                                    log.exception("frame %s", f.get("type"))
                        finally:
                            hb.cancel()
                except (OSError, asyncio.TimeoutError, websockets.WebSocketException, RuntimeError) as e:
                    log.info("link %s: %s", url, e)
                except Exception:                  # a bad hello must not end the reconnect loop for good
                    log.exception("link %s", url)
                finally:
                    if self.online.is_set():
                        self.online.clear()
                        if self.on_state:
                            self.on_state(False, {})
                    self.ws = None
                if not self._wanted.is_set():
                    break
            await asyncio.sleep(backoff)
            backoff = min(backoff * 2, 30)

    async def _heartbeat(self, ws):
        while True:
            await asyncio.sleep(HB_S)
            if time.monotonic() - self._last_rx > HB_S * HB_MISS:
                log.warning("brain silent for %.0fs, dropping link", HB_S * HB_MISS)
                await ws.close()
                return
            await self.send("heartbeat", rtt_probe=time.time())

    async def send(self, type_, **kw):
        ws = self.ws
        if ws is None:
            return False
        async with self._lock:
            self.seq += 1
            kw.update(type=type_, seq=self.seq, t=time.time())
            kw.setdefault("turn", 0)
            try:
                await ws.send(_pack(kw))
                return True
            except websockets.ConnectionClosed:
                return False

    # ------------------------------------------------------------------ bulk channel
    async def bulk_put(self, kind, name, data, meta=None, timeout=300):
        """Upload bytes on /bulk in 1 MB frames. Returns the brain's ack dict or None."""
        if not self.url:
            return None
        try:
            async with websockets.connect(bulk_url(self.url), max_size=2 * BULK_CHUNK, compression=None,
                                          open_timeout=5, ping_interval=20) as ws:
                await ws.send(_pack({"type": "bulk.put", "token": self.token, "kind": kind, "name": name,
                                     "size": len(data), "meta": meta or {}}))
                for i in range(0, len(data), BULK_CHUNK):
                    await ws.send(data[i:i + BULK_CHUNK])
                await ws.send(_pack({"type": "bulk.end"}))
                return _unpack(await asyncio.wait_for(ws.recv(), timeout))
        except Exception as e:
            log.warning("bulk_put %s/%s failed: %s", kind, name, e)
            return None
