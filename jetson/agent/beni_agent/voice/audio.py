"""Audio I/O against beni_audio (§8.2): RTP L16 in from udp:6000 (16 kHz), paced RTP L16 out to udp:6001 (24 kHz).

Everything inside the agent is little-endian int16 (numpy native, what the brain sends); RTP L16 is big-endian,
so byteswap happens exactly once on each edge.
"""
import asyncio
import glob
import os
import random
import socket
import struct
import time

import numpy as np

RTP_HDR = 12
OUT_RATE = 24000
FRAME_MS = 20
FRAME_BYTES = OUT_RATE * FRAME_MS // 1000 * 2       # 960 B
LEAD_MS = 60                                         # keep this much queued ahead of real time in the jitter buffer
SINK_LATENCY_MS = 100                                # rtpjitterbuffer 40 ms + alsasink 60 ms


class MicReceiver(asyncio.DatagramProtocol):
    """Pushes (pcm_f32, pcm16_le_bytes) per RTP packet into an asyncio.Queue (bounded; drops oldest when late)."""

    def __init__(self, maxsize=200):
        self.q = asyncio.Queue(maxsize)

    def datagram_received(self, pkt, addr):
        if len(pkt) <= RTP_HDR:
            return
        cc = pkt[0] & 0x0F                                # skip CSRCs if any (gst never sets them, cheap to honour)
        le = np.frombuffer(pkt[RTP_HDR + 4 * cc:], dtype=">i2").astype("<i2")
        if self.q.full():
            self.q.get_nowait()
        self.q.put_nowait((le.astype(np.float32) * (1.0 / 32768.0), le.tobytes()))

    @classmethod
    async def open(cls, port=6000, host="127.0.0.1"):
        loop = asyncio.get_running_loop()
        _, proto = await loop.create_datagram_endpoint(cls, local_addr=(host, port))
        return proto


class Player:
    """Paced RTP sender. feed() never blocks; played_ms is what the speaker has actually emitted so far."""

    def __init__(self, port=6001, host="127.0.0.1", fillers_dir=None):
        self.addr = (host, port)
        self.sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        self.sock.setblocking(False)
        self.buf = bytearray()
        self.seq, self.ts, self.ssrc = random.getrandbits(16), random.getrandbits(32), random.getrandbits(32)
        self.started_at, self._t0 = 0.0, time.monotonic()   # wall/mono time the current utterance started
        self.sent_ms = 0
        self._wake = asyncio.Event()
        self._idle = asyncio.Event()
        self._idle.set()
        self._closed = False
        self.fillers = self._load_fillers(fillers_dir)
        self.on_first_audio = None     # callback(t) for latency logging

    @staticmethod
    def _load_fillers(d):
        """Pre-rendered 24 kHz s16le clips ('hmm', 'let me think') in Beni's voice (§10.5 voice identity)."""
        if not d or not os.path.isdir(d):
            return []
        out = []
        for p in sorted(glob.glob(os.path.join(d, "*.raw"))):
            with open(p, "rb") as f:
                out.append(f.read())
        return out

    @property
    def speaking(self):
        return not self._idle.is_set()

    @property
    def played_ms(self):
        if not self.speaking and not self.sent_ms:
            return 0
        wall = (time.monotonic() - self._t0) * 1000.0 if self.speaking else float(self.sent_ms)
        return int(max(0.0, min(float(self.sent_ms), wall) - (SINK_LATENCY_MS if self.speaking else 0)))

    def feed(self, pcm16_le):
        if not pcm16_le:
            return
        if not self.speaking:
            self.started_at, self._t0, self.sent_ms = time.time(), time.monotonic(), 0
            self._idle.clear()
        self.buf += pcm16_le
        self._wake.set()

    def play_filler(self):
        if self.fillers and not self.speaking:
            self.feed(random.choice(self.fillers))
            return True
        return False

    def stop(self):
        """Barge-in: drop everything queued. Returns ms actually heard (for history truncation)."""
        played = self.played_ms
        self.buf.clear()
        self._finish()
        return played

    async def wait_done(self):
        await self._idle.wait()

    def _finish(self):
        self._idle.set()

    def _send(self, frame):
        hdr = struct.pack("!BBHII", 0x80, 96, self.seq, self.ts & 0xFFFFFFFF, self.ssrc)
        self.seq = (self.seq + 1) & 0xFFFF
        self.ts += len(frame) // 2
        be = np.frombuffer(frame, "<i2").astype(">i2").tobytes()
        try:
            self.sock.sendto(hdr + be, self.addr)
        except (BlockingIOError, OSError):
            pass

    async def run(self):
        while not self._closed:
            if not self.buf:
                if self.speaking:
                    # drain: wait until the tail is audible before declaring done
                    remain = self.sent_ms - (time.monotonic() - self._t0) * 1000.0
                    if remain > 0:
                        await asyncio.sleep(remain / 1000.0)
                        continue                    # feed() may have run meanwhile: its wake must not be cleared
                    self._finish()
                self._wake.clear()
                await self._wake.wait()
                continue
            ahead = self.sent_ms - (time.monotonic() - self._t0) * 1000.0
            if ahead > LEAD_MS:
                await asyncio.sleep((ahead - LEAD_MS) / 1000.0)
                continue
            frame = bytes(self.buf[:FRAME_BYTES])
            del self.buf[:FRAME_BYTES]
            if len(frame) < FRAME_BYTES:
                frame += b"\0" * (FRAME_BYTES - len(frame))
            if self.sent_ms == 0 and self.on_first_audio:
                self.on_first_audio(time.time())
            self._send(frame)
            self.sent_ms += FRAME_MS

    def close(self):
        self._closed = True
        self._wake.set()
        self.sock.close()
