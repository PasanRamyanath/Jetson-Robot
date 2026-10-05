"""Teleop bitrate from the measured link quality (§4 item 19). Wi-Fi signal from /proc/net/wireless; a wired link
keeps the full rate. Hysteresis so a signal hovering at a step edge doesn't flap the encoder."""
import os

FULL_BPS = 2000000                                    # vision_core --teleop-bps default
STEPS = ((-62, FULL_BPS), (-70, 1200000), (-78, 700000), (None, 400000))   # (dBm floor, bps), best first
HYST_DB = 3


def wired(iface="eth0"):
    try:
        with open("/sys/class/net/%s/carrier" % iface) as f:
            return f.read().strip() == "1"
    except OSError:
        return False


def wifi_dbm(path="/proc/net/wireless"):
    """Signal level of the first wireless interface, or None."""
    try:
        with open(path) as f:
            lines = f.read().splitlines()[2:]
    except OSError:
        return None
    for ln in lines:
        parts = ln.split()
        if len(parts) >= 4:
            try:
                v = float(parts[3].rstrip("."))
            except ValueError:
                continue
            return v - 256 if v > 0 else v            # some drivers report unsigned dBm
    return None


def pick_bps(dbm, current=None):
    if dbm is None:
        return current or FULL_BPS
    for floor, bps in STEPS:
        if floor is None or dbm >= floor:
            want = bps
            break
    if current and want > current:                    # step up only once clear of the edge
        floor = next(f for f, b in STEPS if b == want)
        if floor is not None and dbm < floor + HYST_DB:
            return current
    return want


class LinkRate:
    """Call step() every tick; returns the new bitrate when it should change, else None."""

    def __init__(self, eth=os.environ.get("BENI_ETH_IF", "eth0")):
        self.eth, self.bps = eth, None

    def step(self, dbm=None, is_wired=None):
        is_wired = wired(self.eth) if is_wired is None else is_wired
        want = FULL_BPS if is_wired else pick_bps(wifi_dbm() if dbm is None else dbm, self.bps)
        if want == self.bps:
            return None
        self.bps = want
        return want
