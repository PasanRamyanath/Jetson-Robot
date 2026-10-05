"""tegrastats line parser (§14.4), shared by tools/tegrastats_logger.py and the scheduler. Python 3.6-safe."""
import re
import subprocess

_RAM = re.compile(r"RAM (\d+)/(\d+)MB")
_CPU = re.compile(r"CPU \[([^\]]+)\]")
_NUM = {
    "emc": re.compile(r"EMC_FREQ (\d+)%"),
    "gpu": re.compile(r"GR3D_FREQ (\d+)%"),
    "nvenc": re.compile(r"NVENC (\d+)"),
    "nvdec": re.compile(r"NVDEC (\d+)"),
    "temp_cpu": re.compile(r"CPU@([\d.]+)C"),
    "temp_gpu": re.compile(r"GPU@([\d.]+)C"),
}
_PWR = {
    "p_in_mw": re.compile(r"POM_5V_IN (\d+)/"),
    "p_gpu_mw": re.compile(r"POM_5V_GPU (\d+)/"),
    "p_cpu_mw": re.compile(r"POM_5V_CPU (\d+)/"),
}
FIELDS = ("t", "ram_mb", "ram_total_mb", "cpu_avg", "cpu_max", "emc", "gpu", "nvenc", "nvdec",
          "temp_cpu", "temp_gpu", "p_in_mw", "p_gpu_mw", "p_cpu_mw")


def parse(line, t=None):
    row = dict.fromkeys(FIELDS)
    row["t"] = t
    m = _RAM.search(line)
    if m:
        row["ram_mb"], row["ram_total_mb"] = int(m.group(1)), int(m.group(2))
    m = _CPU.search(line)
    if m:
        # "12%@1479,off,7%@1479,..." -> offline cores count as 0
        loads = [int(x.split("%")[0]) if "%" in x else 0 for x in m.group(1).split(",")]
        if loads:
            row["cpu_avg"] = float(sum(loads)) / len(loads)
            row["cpu_max"] = max(loads)
    for k, rx in _NUM.items():
        m = rx.search(line)
        if m:
            row[k] = float(m.group(1))
    for k, rx in _PWR.items():
        m = rx.search(line)
        if m:
            row[k] = int(m.group(1))
    return row


def stream(interval_ms=1000, cmd="tegrastats"):
    """Yield raw tegrastats lines forever (caller owns the process lifetime)."""
    p = subprocess.Popen([cmd, "--interval", str(interval_ms)], stdout=subprocess.PIPE,
                         universal_newlines=True, bufsize=1)
    try:
        for line in p.stdout:
            yield line
    finally:
        p.terminate()


def mem_available_mb(path="/proc/meminfo"):
    """MemAvailable is the right free-RAM number (tegrastats RAM 'used' counts reclaimable cache)."""
    with open(path) as f:
        for line in f:
            if line.startswith("MemAvailable:"):
                return int(line.split()[1]) // 1024
    return None
