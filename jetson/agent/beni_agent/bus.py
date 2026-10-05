"""ZeroMQ + msgpack IPC helpers on asyncio (§7.2). One Context per process; all sockets non-blocking."""
import asyncio
import logging
import os

import zmq
import zmq.asyncio

from beni_common import schemas as S

log = logging.getLogger("bus")


class Bus:
    def __init__(self, src="agent"):
        self.src = src
        self.ctx = zmq.asyncio.Context.instance()
        os.makedirs(S.SOCK_DIR, exist_ok=True)
        self._req = {}

    # ------------------------------------------------------------------ pub/sub
    def pub(self, ep, hwm=100):
        s = self.ctx.socket(zmq.PUB)
        s.setsockopt(zmq.SNDHWM, hwm)
        s.setsockopt(zmq.LINGER, 0)
        s.bind(ep)
        return Publisher(s, self.src)

    def sub(self, ep, *topics, conflate=False):
        s = self.ctx.socket(zmq.SUB)
        s.setsockopt(zmq.RCVHWM, 20)
        s.setsockopt(zmq.LINGER, 0)
        if conflate:            # latest-only (state streams); incompatible with multipart topics, so not used for those
            s.setsockopt(zmq.CONFLATE, 1)
        for t in topics or (b"",):
            s.setsockopt(zmq.SUBSCRIBE, t)
        s.connect(ep)
        return s

    def push(self, ep):
        s = self.ctx.socket(zmq.PUSH)
        s.setsockopt(zmq.SNDHWM, 10)
        s.setsockopt(zmq.LINGER, 0)
        s.connect(ep)
        return s

    # ------------------------------------------------------------------ request/reply with timeout
    async def request(self, ep, msg, timeout=2.0):
        """REQ/REP round trip. On timeout the REQ socket is discarded (a REQ can't recover from a lost reply)."""
        s = self._req.get(ep)
        if s is None:
            s = self._req[ep] = self.ctx.socket(zmq.REQ)
            s.setsockopt(zmq.LINGER, 0)
            s.connect(ep)
        try:
            await s.send(S.pack(msg))
            if await s.poll(int(timeout * 1000)):
                return S.unpack(await s.recv())
        except zmq.ZMQError as e:
            log.warning("req %s: %s", ep, e)
        self._req.pop(ep, None).close()
        return {"ok": False, "err": "timeout"}


class Publisher:
    def __init__(self, sock, src):
        self.sock, self.src = sock, src

    def send(self, topic, **fields):
        try:
            self.sock.send_multipart([topic, S.pack(S.envelope(self.src, **fields))], flags=zmq.NOBLOCK)
        except zmq.Again:
            pass


async def recv(sock):
    """(topic, msg) from a SUB socket."""
    parts = await sock.recv_multipart()
    return parts[0], S.unpack(parts[-1])


async def pump(sock, handler, name="sub"):
    """Forward every message on sock to handler(topic, msg); a bad message never kills the loop."""
    while True:
        try:
            topic, msg = await recv(sock)
            r = handler(topic, msg)
            if asyncio.iscoroutine(r):
                await r
        except asyncio.CancelledError:
            raise
        except Exception:
            log.exception("%s handler", name)
