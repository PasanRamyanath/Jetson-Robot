"""Kaggle brain lifecycle (§10.4): start the batch kernel during awake hours or on demand, stay under the weekly
GPU budget, ask the brain to stop itself (brain.stop frame) when it's no longer wanted.

Usage accounting is our own (the CLI doesn't expose quota): every poll while the kernel is running/queued adds the
elapsed time to a per-ISO-week counter kept in memory.kv, so it survives restarts and syncs nowhere harmful.
"""
import asyncio
import datetime as dt
import json
import logging
import os
import shutil
import sys
import tempfile
import time

from ..config import in_windows

log = logging.getLogger("lifecycle")
POLL_S = 60
RUNNING = ("running", "queued")
# The CLI lives in the agent venv, which systemd does not put on PATH.
KAGGLE = os.path.join(os.path.dirname(sys.executable), "kaggle")
KAGGLE = KAGGLE if os.path.exists(KAGGLE) else "kaggle"


class Lifecycle:
    def __init__(self, cfg, kv_get, kv_set, link=None, clock=time.time):
        self.cfg, self.kv_get, self.kv_set, self.link, self.clock = cfg, kv_get, kv_set, link, clock
        self.status = "unknown"
        self.on_demand_until = 0.0
        self._last_poll = None
        self._pushed_at = 0.0
        self._kick = asyncio.Event()       # a cold start polls (and pushes) now instead of at the next POLL_S

    # ------------------------------------------------------------------ kaggle CLI
    async def _kaggle(self, *args, timeout=60):
        env = dict(os.environ)
        try:
            p = await asyncio.create_subprocess_exec(KAGGLE, *args, stdout=asyncio.subprocess.PIPE,
                                                     stderr=asyncio.subprocess.STDOUT, env=env)
            out, _ = await asyncio.wait_for(p.communicate(), timeout)
            return p.returncode, out.decode(errors="replace")
        except (OSError, asyncio.TimeoutError) as e:
            return -1, str(e)

    async def kernel_status(self):
        rc, out = await self._kaggle("kernels", "status", self.cfg.kaggle_kernel)
        low = out.lower()
        for s in ("running", "queued", "complete", "error", "cancel"):
            if s in low:
                return s
        return "unknown" if rc else "idle"

    def _render(self):
        """Copy the kernel dir with kernel-metadata.json ids rewritten for KAGGLE_KERNEL's owner."""
        user = self.cfg.kaggle_kernel.split("/")[0]
        d = tempfile.mkdtemp(prefix="beni-kernel-")
        with open(os.path.join(self.cfg.kaggle_dir, "kernel-metadata.json")) as f:
            meta = json.load(f)
        meta["id"] = self.cfg.kaggle_kernel
        meta["dataset_sources"] = [s.replace("YOUR_KAGGLE_USERNAME", user) for s in meta.get("dataset_sources", [])]
        shutil.copy(os.path.join(self.cfg.kaggle_dir, meta["code_file"]), d)
        with open(os.path.join(d, "kernel-metadata.json"), "w") as f:
            json.dump(meta, f)
        return d

    async def push(self):
        d = self._render()
        try:
            rc, out = await self._kaggle("kernels", "push", "-p", d, timeout=180)
        finally:
            shutil.rmtree(d, ignore_errors=True)
        if rc != 0:
            log.error("kaggle push failed: %s", out.strip()[-400:])
            return False
        self._pushed_at = self.clock()
        log.info("brain kernel pushed")
        return True

    # ------------------------------------------------------------------ policy
    def week_key(self, t=None):
        y, w, _ = dt.date.fromtimestamp(t or self.clock()).isocalendar()
        return "lifecycle.used_s.%d-W%02d" % (y, w)

    def used_h(self):
        return (self.kv_get(self.week_key(), 0) or 0) / 3600.0

    def _account(self, running):
        now = self.clock()
        if running and self._last_poll is not None:
            k = self.week_key(now)
            self.kv_set(k, (self.kv_get(k, 0) or 0) + min(now - self._last_poll, 2 * POLL_S))
        self._last_poll = now

    def should_run(self, now=None):
        now = now or self.clock()
        if self.used_h() >= self.cfg.weekly_budget_h:
            return False
        if now < self.on_demand_until:
            return True
        lt = time.localtime(now)
        return in_windows(self.cfg.awake_hours, lt.tm_hour * 60 + lt.tm_min)

    def ensure_running(self, minutes=30):
        """Wake word outside awake hours: keep the brain up for a while. Returns True if a cold start is needed."""
        if self.used_h() >= self.cfg.weekly_budget_h:
            return False
        self.on_demand_until = max(self.on_demand_until, self.clock() + minutes * 60)
        cold = self.status not in RUNNING
        if cold:
            self._kick.set()
            if self.link is not None:
                self.link.resume()
        return cold

    async def tick(self):
        if not self.cfg.kaggle_kernel:
            return
        self.status = await self.kernel_status()
        running = self.status in RUNNING
        self._account(running)
        want = self.should_run()
        if want and not running and self.clock() - self._pushed_at > 10 * 60:
            if self.link is not None:
                self.link.resume()
            await self.push()
        elif not want and running and self.link is not None and self.link.online.is_set():
            log.info("outside awake hours / budget (%.1f h used): asking brain to stop", self.used_h())
            await self.link.send("brain.stop", reason="schedule")
        elif not want and not running and self.link is not None:
            self.link.pause()

    async def run(self):
        while True:
            try:
                await self.tick()
            except Exception:
                log.exception("lifecycle tick")
            self._kick.clear()
            try:
                await asyncio.wait_for(self._kick.wait(), POLL_S)
            except asyncio.TimeoutError:
                pass
