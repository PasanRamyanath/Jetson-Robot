"""Bluetooth phone presence (§3.17 M.2 row, §4 item 56): "the owner's phone is home".

It sends a classic-BT name request (`hcitool name MAC`) to each paired phone. That reaches the phone's public
address even when the phone is not discoverable. A BLE scan would not work, because modern phones rotate random
addresses. A phone is "away" only after several misses in a row, because phones doze their radios.
"""
import asyncio
import logging
import time

log = logging.getLogger("presence")

MISSES_AWAY = 4
POLL_HOME_S = 120                   # confirm presence rarely
POLL_AWAY_S = 30                    # but notice arrivals quickly
PROBE_TIMEOUT_S = 10


def parse_phones(s):
    """'Nimal=AA:BB:CC:DD:EE:FF, Kumari=11:22:33:44:55:66' -> {MAC: name}."""
    out = {}
    for part in filter(None, (p.strip() for p in (s or "").split(","))):
        name, _, mac = part.partition("=")
        mac = mac.strip().upper()
        if len(mac) == 17 and mac.count(":") == 5:
            out[mac] = name.strip()
    return out


async def hcitool_name(mac):
    try:
        p = await asyncio.create_subprocess_exec("hcitool", "name", mac, stdout=asyncio.subprocess.PIPE,
                                                 stderr=asyncio.subprocess.DEVNULL)
        out, _ = await asyncio.wait_for(p.communicate(), PROBE_TIMEOUT_S)
        return bool(out.strip())
    except asyncio.TimeoutError:
        p.kill()
        return False
    except OSError:
        return None                 # no BlueZ tools: presence disabled


class Presence:
    def __init__(self, phones, on_change, probe=hcitool_name, clock=time.time):
        self.phones, self.on_change, self.probe, self.clock = phones, on_change, probe, clock
        self.home = {m: None for m in phones}          # None = unknown (no event for the first result)
        self.misses = {m: 0 for m in phones}

    def feed(self, mac, seen):
        was = self.home[mac]
        if seen:
            self.misses[mac] = 0
            now_home = True
        else:
            self.misses[mac] += 1
            now_home = was if self.misses[mac] < MISSES_AWAY and was is not None else False
        self.home[mac] = now_home
        if now_home != was:
            if was is not None or now_home:
                self.on_change(self.phones[mac], now_home)
            return True
        return False

    def names_home(self):
        return sorted(self.phones[m] for m, h in self.home.items() if h)

    async def run(self):
        if not self.phones:
            return
        while True:
            for mac in self.phones:
                seen = await self.probe(mac)
                if seen is None:
                    log.warning("hcitool not found: phone presence disabled")
                    return
                self.feed(mac, seen)
            await asyncio.sleep(POLL_AWAY_S if not all(self.home.values()) else POLL_HOME_S)
