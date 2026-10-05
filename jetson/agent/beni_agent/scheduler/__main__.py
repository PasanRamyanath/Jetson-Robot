"""beni-sched: engine/power scheduler (§3.17, §15.1).  python -m beni_agent.scheduler [--dry-run]

Every 2 s: tegrastats window + robot state + voice activity + brain status ->
  * publish {admit, pause, mode, llm} on sched.sock (vision_core / agent pick their background work from it)
  * run/SIGSTOP/SIGCONT the scheduler's own SCHED_IDLE background jobs
  * switch power modes (nvpmodel, jetson_clocks, detector interval) and start/stop the offline LLM
  * drive the pwm-fan curve from the tegrastats temperatures; shed load above 80 C (§15.2)
  * set the teleop bitrate from the measured Wi-Fi signal (§4 item 19)
"""
import argparse
import asyncio
import logging
import os
import signal
import sys
import threading
import time

from beni_common import schemas as S
from beni_common import tegrastats

from .. import bus as B
from ..config import Config, in_windows
from ..util import setup_logging, spawn
from .admit import Stats, admit, job_allowed
from .link import LinkRate
from .modes import MODES, Applier, decide, empty_check, want_llm
from .thermal import Fan, ThermalGuard

log = logging.getLogger("sched")
TICK_S = 2.0
LINK_EVERY_S, LINK_RESEND_S = 10.0, 120.0   # resend: vision may have restarted at its default rate


def _idle_prio():
    os.nice(19)
    try:
        os.sched_setscheduler(0, os.SCHED_IDLE, os.sched_param(0))
    except (AttributeError, OSError):
        pass


class Job:
    """A periodic background-tier subprocess. SIGSTOP pauses it (instant, keeps state); SIGCONT resumes it.
    pausable=False for jobs that hold the memory DB's write lock: stopped mid-transaction, they would block every
    agent write (busy timeout, then "database is locked") for as long as the pause lasts."""

    def __init__(self, name, cmd, needs, period_s, pausable=True):
        self.name, self.cmd, self.needs, self.period_s, self.pausable = name, cmd, needs, period_s, pausable
        self.proc, self.last_start, self.stopped = None, 0.0, False

    def step(self, allowed, now):
        if self.proc is not None and self.proc.poll() is not None:
            log.info("job %s exited %s", self.name, self.proc.returncode)
            self.proc = None
        if self.proc is None:
            if allowed and now - self.last_start >= self.period_s:
                import subprocess
                self.proc = subprocess.Popen(self.cmd, preexec_fn=_idle_prio, stdout=subprocess.DEVNULL,
                                             stderr=subprocess.DEVNULL)
                self.last_start, self.stopped = now, False
                log.info("job %s started", self.name)
        elif allowed and self.stopped:
            self.proc.send_signal(signal.SIGCONT)
            self.stopped = False
        elif not allowed and not self.stopped and self.pausable:
            self.proc.send_signal(signal.SIGSTOP)
            self.stopped = True


def default_jobs(cfg):
    py = sys.executable
    return [
        Job("fts_optimize", [py, "-c", "import sqlite3,sys; c=sqlite3.connect(sys.argv[1], timeout=30); "
                             "c.execute(\"INSERT INTO episode_fts(episode_fts) VALUES('optimize')\"); c.commit()",
                             cfg.db], ("cpu", "ram"), 24 * 3600, pausable=False),
        Job("log_compress", ["find", cfg.logs, "-name", "*.csv", "-mtime", "+1", "-exec", "gzip", "-9", "{}", ";"],
            ("cpu",), 6 * 3600),
    ]


class Scheduler:
    def __init__(self, cfg, dry_run=False):
        self.cfg, self.dry = cfg, dry_run
        self.bus = B.Bus("sched")
        self.pub = self.bus.pub(S.EP["sched"])
        self.stats = Stats()
        self.apply = Applier(self.bus.request, dry_run)
        self.jobs = default_jobs(cfg)
        self.fan, self.guard = Fan(dry_run=dry_run), ThermalGuard()
        self.robot = {}
        self.last_person_t = time.time()
        self.empty_since = None     # battery nearly empty off the charger since (empty_check)
        self.brain_online, self.brain_changed = False, time.time()
        self.link, self.link_at, self.link_sent = LinkRate(), 0.0, 0.0

    def on_event(self, topic, msg):
        n = msg.get("name")
        if n == "voice_state":
            self.stats.voice_active = msg.get("state") not in ("idle", None)
        elif n == "brain":
            if bool(msg.get("online")) != self.brain_online:
                self.brain_online, self.brain_changed = bool(msg.get("online")), time.time()
        elif n in ("person_seen", "speech_start"):
            self.last_person_t = time.time()

    def on_vision(self, topic, msg):
        if any(o.get("cls") == 0 for f in msg.get("frames", ()) for o in f.get("objs", ())):
            self.last_person_t = time.time()

    def on_robot(self, topic, msg):
        self.robot = msg
        self.stats.batt_pct = msg.get("battery")
        self.stats.on_charger = bool(msg.get("charging"))

    def night(self, now):
        lt = time.localtime(now)
        return in_windows(self.cfg.quiet_hours, lt.tm_hour * 60 + lt.tm_min)

    def tegra_thread(self, loop):
        try:
            for line in tegrastats.stream(1000):
                loop.call_soon_threadsafe(self.stats.add, tegrastats.parse(line))
        except OSError as e:
            log.warning("tegrastats unavailable (%s): admission uses defaults", e)

    async def tick(self):
        now = time.time()
        self.stats.ram_avail_mb = tegrastats.mem_available_mb() or self.stats.ram_avail_mb
        d = admit(self.stats)
        temp = max(self.stats.temp_gpu, self.stats._last("temp_cpu")) or None
        self.fan.set(temp)
        mode = decide(self.stats.batt_pct, self.stats.on_charger, bool(self.robot.get("docked")),
                      now - self.last_person_t, self.stats.voice_active, self.guard.feed(temp), self.night(now),
                      self.stats._avg("p_in_mw", 30) / 1000 or None)
        await self.apply.set_mode(mode)
        self.empty_since, off = empty_check(self.empty_since, self.stats.batt_pct, self.stats.on_charger, now)
        if off:
            await self.apply.poweroff()
        await self.apply.set_llm(want_llm(self.brain_online, now - self.brain_changed, self.stats.ram_avail_mb,
                                          self.apply.llm_running))
        if now - self.link_at >= LINK_EVERY_S:
            self.link_at = now
            await self.set_link_rate(now)
        bg_ok = MODES[self.apply.mode][3] if self.apply.mode else True
        for j in self.jobs:
            j.step(bg_ok and job_allowed(d, j.needs), now)
        self.pub.send(b"sched", admit=d["engines"], pause=d["pause"] or not bg_ok, mode=self.apply.mode,
                      llm=self.apply.llm_running, ram_mb=self.stats.ram_avail_mb, hot=self.guard.hot,
                      fan=self.fan.pwm, p_in_mw=self.stats._last("p_in_mw") or None)

    async def set_link_rate(self, now):
        bps = self.link.step()
        if bps is None and now - self.link_sent < LINK_RESEND_S:
            return
        bps = bps or self.link.bps
        if self.dry:
            log.info("teleop bitrate %d", bps)
            return
        r = await self.bus.request(S.EP["vision_ctrl"], {"op": "bitrate", "bps": bps})
        if r.get("ok"):
            self.link_sent = now
        else:
            self.link.bps = None   # vision down or teleop off: retry next time

    async def run(self):
        loop = asyncio.get_running_loop()
        threading.Thread(target=self.tegra_thread, args=(loop,), daemon=True).start()
        ev = self.bus.sub(S.EP["events"], b"event")
        vis = self.bus.sub(S.EP["vision"], S.T_DET)
        rob = self.bus.sub(S.EP["robot_state"], S.T_STATE)
        for s, h, n in ((ev, self.on_event, "events"), (vis, self.on_vision, "vision"), (rob, self.on_robot, "robot")):
            spawn(B.pump(s, h, n))
        await self.apply.set_mode("active", force=True)
        while True:
            try:
                await self.tick()
            except Exception:
                log.exception("tick")
            await asyncio.sleep(TICK_S)


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--dry-run", action="store_true", help="decide and publish, but don't touch nvpmodel/services")
    a = p.parse_args(argv)
    setup_logging()
    try:
        asyncio.run(Scheduler(Config.from_env(), a.dry_run).run())
    except KeyboardInterrupt:
        pass
    return 0


if __name__ == "__main__":
    sys.exit(main())
