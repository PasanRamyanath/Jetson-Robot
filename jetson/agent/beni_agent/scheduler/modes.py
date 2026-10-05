"""Power/perception modes (§15.1). decide() is pure; Applier performs the transitions with minimal side effects.

nvpmodel ids: 0 = MAXN 10 W, 1 = 5 W, 2 = custom BENI_7W (add the block to /etc/nvpmodel.conf, see §15.1).
Vision `interval` = batches skipped between detector runs at the 60 fps mux: 3 -> 15 Hz, 7 -> 7.5 Hz, 59 -> 1 Hz.
"""
import asyncio
import logging
import time

log = logging.getLogger("modes")

MODES = {
    #            nvpmodel  clocks  interval  background
    "active":   (0,        True,   3,        True),
    "idle":     (2,        False,  59,       True),
    "low":      (1,        False,  7,        False),
    "critical": (1,        False,  None,     False),   # detector off (vision paused by the dock behaviour)
    "replay":   (0,        True,   59,       True),    # §11.9 sleep replay: NVDEC -> TRT at MAXN on the charger
}
CAM0_OFF = ("idle", "replay")
IDLE_AFTER_S = 10 * 60
DWELL_S = 30
LLM_AFTER_OFFLINE_S = 60
LLM_STOP_AFTER_ONLINE_S = 300
LLM_MIN_RAM_MB = 700
HIGH_DRAW_W = 8.0     # INA3221 POM_5V_IN, 30 s mean: near the 10 W MAXN envelope
HIGH_DRAW_PCT = 30    # ...then "low" starts here instead of at 20 % (§15.2 #10: modes from real measurements)
SHUTDOWN_PCT, SHUTDOWN_HOLD_S = 5, 60    # §15.1 critical: power off at 5 % off the charger (held 60 s: no blips)


def decide(batt_pct, charging, docked, last_person_age_s, voice_active, hot=False, night=False, power_w=None):
    if batt_pct is not None and not charging:
        if batt_pct < 10:
            return "critical"
        if batt_pct < 20 or (batt_pct < HIGH_DRAW_PCT and power_w is not None and power_w > HIGH_DRAW_W):
            return "low"
    if voice_active:
        return "active"
    if hot:
        return "idle"  # thermal guard (§15.2): shed the detector and MAXN until the SoC cools
    if night and charging and docked and last_person_age_s > IDLE_AFTER_S:
        return "replay"
    if docked or last_person_age_s > IDLE_AFTER_S:
        return "idle"
    return "active"


def empty_check(since, batt_pct, charging, now):
    """§15.1 critical -> off: (since', power_off). Below SHUTDOWN_PCT off the charger for SHUTDOWN_HOLD_S."""
    if batt_pct is None or batt_pct >= SHUTDOWN_PCT or charging:
        return None, False
    since = since or now
    return since, now - since >= SHUTDOWN_HOLD_S


def want_llm(brain_online, since_change_s, ram_avail_mb, llm_running):
    if brain_online:
        return llm_running and since_change_s < LLM_STOP_AFTER_ONLINE_S
    if llm_running:
        return True
    return since_change_s >= LLM_AFTER_OFFLINE_S and ram_avail_mb >= LLM_MIN_RAM_MB


async def _run(*cmd):
    p = None
    try:
        p = await asyncio.create_subprocess_exec(*cmd, stdout=asyncio.subprocess.DEVNULL,
                                                 stderr=asyncio.subprocess.PIPE)
        _, err = await asyncio.wait_for(p.communicate(), 30)
        if p.returncode:
            log.warning("%s -> %d %s", " ".join(cmd), p.returncode, err.decode(errors="replace").strip()[:200])
        return p.returncode == 0
    except (OSError, asyncio.TimeoutError) as e:
        log.warning("%s: %s", " ".join(cmd), e)
        if p is not None and p.returncode is None:      # timed out: don't leave it running next to the retry
            try:
                p.kill()
                await p.wait()
            except ProcessLookupError:
                pass
        return False


class Applier:
    def __init__(self, request, dry_run=False, clock=time.time):
        self.request = request          # async fn(ep, msg) -> reply (vision_ctrl)
        self.dry, self.clock = dry_run, clock
        self.mode, self.changed_at = None, 0.0
        self.nvp = None
        self.llm_running = False
        self.cam0 = True

    async def set_mode(self, mode, force=False):
        now = self.clock()
        if mode == self.mode or (not force and now - self.changed_at < DWELL_S and mode not in ("critical", "low")):
            return False
        nvp, clocks, interval, _ = MODES[mode]
        log.info("mode %s -> %s", self.mode, mode)
        if not self.dry:
            if nvp != self.nvp:
                if await _run("sudo", "-n", "/usr/sbin/nvpmodel", "-m", str(nvp)):
                    self.nvp = nvp
            if clocks:
                await _run("sudo", "-n", "/usr/bin/jetson_clocks")
            from beni_common import schemas as S
            if interval is not None:
                await self.request(S.EP["vision_ctrl"], {"op": "interval", "n": interval})
            cam0 = mode not in CAM0_OFF
            if cam0 != self.cam0:  # §15.1 idle-watch / replay: CAM0's Argus session stopped (vision_core only)
                r = await self.request(S.EP["vision_ctrl"], {"op": "cam", "cam": 0, "on": int(cam0)})
                self.cam0 = cam0 if (r or {}).get("ok") else None
        self.mode, self.changed_at = mode, now
        return True

    async def poweroff(self):
        """§15.1: clean shutdown before the pack's BMS cuts out (protects the SSD, the memory DB and the vault)."""
        log.critical("battery critical: powering off")
        return self.dry or await _run("sudo", "-n", "/bin/systemctl", "poweroff")

    async def set_llm(self, on):
        if on == self.llm_running:
            return False
        log.info("offline LLM %s", "start" if on else "stop")
        if self.dry or await _run("sudo", "-n", "/bin/systemctl", "start" if on else "stop", "beni-llm"):
            self.llm_running = on
        return True
