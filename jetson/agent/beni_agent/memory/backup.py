"""Nightly maintenance (§11.5): purge old tombstones, online backup of memory.db, upload to a private HF dataset
with 14-day retention (local copies too). Runs as SCHED_IDLE-ish work inside the agent's executor.

    python -m beni_agent.memory.backup          # one-shot (cron/manual)
"""
import asyncio
import glob
import logging
import os
import re
import time

from . import keepout
from .sync import snapshot_bytes

log = logging.getLogger("backup")
KEEP_DAYS = 14
_NAME = re.compile(r"memory-(\d{8})\.db\.z$")


def backup_once(store, cfg, now=None):
    now = now or time.time()
    purged = store.purge_tombstones(now=now)
    day = time.strftime("%Y%m%d", time.localtime(now))
    os.makedirs(cfg.backup_dir, exist_ok=True)
    path = os.path.join(cfg.backup_dir, "memory-%s.db.z" % day)
    data = snapshot_bytes(store, level=6)
    with open(path + ".tmp", "wb") as f:
        f.write(data)
    os.replace(path + ".tmp", path)
    cutoff = time.strftime("%Y%m%d", time.localtime(now - KEEP_DAYS * 86400))
    for p in glob.glob(os.path.join(cfg.backup_dir, "memory-*.db.z")):
        m = _NAME.search(p)
        if m and m.group(1) < cutoff:
            os.unlink(p)
    uploaded = False
    if cfg.hf_token and cfg.hf_repo:
        try:
            uploaded = _upload(cfg, path, cutoff)
        except Exception as e:
            log.warning("HF upload failed: %s", e)
    log.info("backup %s (%d KB, purged %d tombstones, hf=%s)", path, len(data) // 1024, purged, uploaded)
    return path


def _upload(cfg, path, cutoff):
    from huggingface_hub import HfApi
    api = HfApi(token=cfg.hf_token)
    api.create_repo(cfg.hf_repo, repo_type="dataset", private=True, exist_ok=True)
    api.upload_file(path_or_fileobj=path, path_in_repo="backups/" + os.path.basename(path), repo_id=cfg.hf_repo,
                    repo_type="dataset", commit_message="nightly memory backup")
    for f in api.list_repo_files(cfg.hf_repo, repo_type="dataset"):
        m = _NAME.search(f)
        if f.startswith("backups/") and m and m.group(1) < cutoff:
            api.delete_file(f, cfg.hf_repo, repo_type="dataset", commit_message="retention")
    return True


async def nightly(store, cfg, hour=3, busy=lambda: False):
    """Once a day at `hour` local time, deferred while a conversation is in progress: backup + keepout (§11.10)."""
    loop = asyncio.get_running_loop()
    last = None
    while True:
        await asyncio.sleep(300)
        lt = time.localtime()
        if lt.tm_hour == hour and last != lt.tm_yday and not busy():
            last = lt.tm_yday
            try:
                await loop.run_in_executor(None, backup_once, store, cfg)
            except Exception:
                log.exception("nightly backup")
            try:
                await loop.run_in_executor(None, keepout.rebuild, store, cfg.maps)
            except Exception:
                log.exception("nightly keepout")


if __name__ == "__main__":
    from beni_common.memory import MemoryStore

    from ..config import Config
    logging.basicConfig(level="INFO")
    c = Config.from_env()
    print(backup_once(MemoryStore(c.db), c))
