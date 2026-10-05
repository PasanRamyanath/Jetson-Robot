"""Face touch input (§3.11, §4 item 57): tap = talk to me, stroke = "pet Beni's face".

Two backends, both stdlib only:
  * XPT2046/ADS7846 on SPI1 (/dev/spidev0.0 after jetson-io enables spi1), read directly over spidev ioctls.
    No ads7846 device-tree overlay is needed. It is sampled at 50 Hz only while PENIRQ (pin 22) is low.
  * A USB-HID touch panel: /dev/input/eventN via evdev (ABS_X/ABS_Y/BTN_TOUCH).
"""
import asyncio
import ctypes
import glob
import logging
import os
import struct
import time

log = logging.getLogger("touch")

TAP_MAX_S, TAP_MAX_PATH = 0.35, 0.05     # normalised screen units
PET_MIN_PATH = 0.15
HOLD_MIN_S = 0.8


class Stroke:
    """Accumulates one touch and classifies it on release: 'tap', 'pet', 'hold' or None."""

    def __init__(self):
        self.t0 = self.last = None
        self.path = 0.0

    def down(self, t, x, y):
        self.t0, self.last, self.path = t, (x, y), 0.0

    def move(self, x, y):
        if self.last is not None:
            self.path += ((x - self.last[0]) ** 2 + (y - self.last[1]) ** 2) ** 0.5
            self.last = (x, y)

    def up(self, t):
        if self.t0 is None:
            return None
        dur, path = t - self.t0, self.path
        self.t0 = self.last = None
        if path >= PET_MIN_PATH:
            return "pet"
        if dur <= TAP_MAX_S and path <= TAP_MAX_PATH:
            return "tap"
        if dur >= HOLD_MIN_S and path <= TAP_MAX_PATH:
            return "hold"
        return None


def parse_cal(s):
    """'x0,x1,y0,y1' raw 12-bit extremes (swap a pair to invert an axis) -> tuple of 4 ints."""
    try:
        v = tuple(int(p) for p in s.split(","))
        return v if len(v) == 4 else None
    except (AttributeError, ValueError):
        return None


def norm(v, lo, hi):
    return min(1.0, max(0.0, float(v - lo) / (hi - lo))) if hi != lo else 0.0


# ---------------------------------------------------------------- XPT2046 over spidev
SPI_IOC_MESSAGE_1 = 0x40206B00          # _IOW('k', 0, struct spi_ioc_transfer[1]), 32 bytes
X_CMD, Y_CMD, Z1_CMD = 0xD0, 0x90, 0xB0  # 12-bit differential, PD=00 keeps PENIRQ armed
Z_PRESSED = 80


class Xpt2046:
    def __init__(self, dev="/dev/spidev0.0", cal=(200, 3900, 200, 3900), hz=1000000):
        import fcntl                            # Linux only; keeps the module importable on dev hosts
        self.ioctl = fcntl.ioctl
        self.fd = os.open(dev, os.O_RDWR)
        self.cal, self.hz = cal, hz
        self.tx, self.rx = ctypes.create_string_buffer(3), ctypes.create_string_buffer(3)

    def _read(self, cmd):
        self.tx.raw = bytes((cmd, 0, 0))
        xfer = struct.pack("QQIIHBBBBH", ctypes.addressof(self.tx), ctypes.addressof(self.rx), 3, self.hz, 0, 8,
                           0, 0, 0, 0)
        self.ioctl(self.fd, SPI_IOC_MESSAGE_1, xfer)
        b = self.rx.raw
        return ((b[1] << 8) | b[2]) >> 3

    def sample(self):
        """-> (x, y) in 0..1, or None when not pressed. Median of 3 per axis rejects the XPT2046's spikes."""
        if self._read(Z1_CMD) < Z_PRESSED:
            return None
        xs = sorted(self._read(X_CMD) for _ in range(3))
        ys = sorted(self._read(Y_CMD) for _ in range(3))
        x0, x1, y0, y1 = self.cal
        return norm(xs[1], x0, x1), norm(ys[1], y0, y1)


# ---------------------------------------------------------------- evdev (USB HID)
EV_KEY, EV_ABS, ABS_X, ABS_Y, BTN_TOUCH = 1, 3, 0, 1, 0x14A
EVENT = struct.Struct("llHHi")           # struct input_event on aarch64 (timeval = 2 longs)


def eviocgabs(axis):
    return 0x80000000 | (24 << 16) | (ord("E") << 8) | (0x40 + axis)


def find_evdev():
    for p in sorted(glob.glob("/dev/input/by-id/*event*")):
        if "touch" in p.lower():
            return p
    return None


class Touch:
    """Runs whichever backend is present; calls on_gesture(name, x, y) on the loop."""

    def __init__(self, on_gesture, spi_dev="/dev/spidev0.0", irq_value=None, cal=None, evdev=None):
        self.on_gesture = on_gesture
        self.spi_dev, self.irq_value, self.cal, self.evdev = spi_dev, irq_value, cal, evdev
        self.stroke = Stroke()
        self.pen = asyncio.Event()

    def pen_irq(self, level):
        """From the GPIO watcher: PENIRQ is active low."""
        if not level:
            self.pen.set()

    async def run_spi(self):
        try:
            ts = Xpt2046(self.spi_dev, self.cal or (200, 3900, 200, 3900))
        except OSError as e:
            log.info("no SPI touch (%s)", e)
            return
        log.info("XPT2046 touch on %s", self.spi_dev)
        while True:
            if self.irq_value is not None:
                await self.pen.wait()
                self.pen.clear()
            p = ts.sample()
            if p is None:
                if self.irq_value is None:
                    await asyncio.sleep(0.1)             # no PENIRQ wired: cheap 10 Hz idle poll
                continue
            self.stroke.down(time.monotonic(), *p)
            last = p
            while p is not None:                         # 50 Hz while the finger is down
                last = p
                self.stroke.move(*p)
                await asyncio.sleep(0.02)
                p = ts.sample()
            self._emit(self.stroke.up(time.monotonic()), last)

    async def run_evdev(self, path):
        import fcntl
        fd = os.open(path, os.O_RDONLY | os.O_NONBLOCK)
        rng = {}
        for axis in (ABS_X, ABS_Y):
            buf = bytearray(24)
            try:
                fcntl.ioctl(fd, eviocgabs(axis), buf)
                _, lo, hi = struct.unpack_from("iii", buf)
            except OSError:
                lo, hi = 0, 4095
            rng[axis] = (lo, hi)
        log.info("evdev touch on %s", path)
        q = asyncio.Queue()
        loop = asyncio.get_running_loop()

        def readable():
            try:
                q.put_nowait(os.read(fd, EVENT.size * 64))
            except BlockingIOError:
                pass
            except OSError as e:                         # unplugged: stop, don't spin on a dead fd
                loop.remove_reader(fd)
                q.put_nowait(e)
        loop.add_reader(fd, readable)
        pos = [0.5, 0.5]
        down = False
        while True:
            data = await q.get()
            if isinstance(data, OSError):
                log.warning("evdev touch lost: %s", data)
                os.close(fd)
                return
            for off in range(0, len(data) - EVENT.size + 1, EVENT.size):
                _, _, typ, code, val = EVENT.unpack_from(data, off)
                now = time.monotonic()
                if typ == EV_ABS and code in rng:
                    pos[code] = norm(val, *rng[code])
                    if down:
                        self.stroke.move(*pos)
                elif typ == EV_KEY and code == BTN_TOUCH:
                    if val and not down:
                        self.stroke.down(now, *pos)
                    elif not val and down:
                        self._emit(self.stroke.up(now), tuple(pos))
                    down = bool(val)

    def _emit(self, gesture, pos):
        if gesture:
            self.on_gesture(gesture, round(pos[0], 3), round(pos[1], 3))

    async def run(self):
        path = self.evdev or find_evdev()
        if path:
            await self.run_evdev(path)
        elif os.path.exists(self.spi_dev):
            await self.run_spi()
