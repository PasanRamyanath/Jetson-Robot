"""Agent configuration: environment (systemd EnvironmentFile=/etc/beni/beni.env) with sane defaults."""
import os
from dataclasses import dataclass, field
from typing import List, Tuple

ENV_FILE = "/etc/beni/beni.env"


def load_env_file(path=ENV_FILE):
    """Fill os.environ from KEY=VALUE lines (for running outside systemd). Existing vars win."""
    try:
        with open(path) as f:
            for line in f:
                line = line.strip()
                if line and not line.startswith("#") and "=" in line:
                    k, v = line.split("=", 1)
                    os.environ.setdefault(k.strip(), v.strip().strip('"'))
    except OSError:
        pass


def _env(k, d=None):
    return os.environ.get(k, d)


def parse_windows(s) -> List[Tuple[int, int]]:
    """'07:00-12:00,16:00-22:30' -> [(420, 720), (960, 1350)] minutes of day (wrap-around allowed)."""
    out = []
    for part in filter(None, (p.strip() for p in (s or "").split(","))):
        a, b = part.split("-")
        out.append(tuple(int(x.split(":")[0]) * 60 + int(x.split(":")[1]) for x in (a, b)))
    return out


def in_windows(windows, minute_of_day):
    for a, b in windows:
        if (a <= minute_of_day < b) if a <= b else (minute_of_day >= a or minute_of_day < b):
            return True
    return False


@dataclass
class Config:
    token: str = ""
    brain_urls: List[str] = field(default_factory=list)
    uplink: str = "pcm16"
    db: str = "/ssd/beni/memory.db"
    models: str = "/ssd/beni/models"
    logs: str = "/ssd/beni/logs"
    rec_dir: str = "/ssd/beni/rec"          # vision_core --rec-dir (sleep replay reads it)
    backup_dir: str = "/ssd/beni/backup"
    maps: str = "/ssd/maps"                     # mounted as /maps in the ROS container (§7.4)
    kaggle_kernel: str = ""
    kaggle_dir: str = "/opt/beni/kaggle"
    awake_hours: list = field(default_factory=list)
    weekly_budget_h: float = 28.0
    quiet_hours: list = field(default_factory=list)
    hf_token: str = ""
    hf_repo: str = ""
    home_city: str = "Colombo"
    llm_url: str = "http://127.0.0.1:8081"
    mic_port: int = 6000
    tts_port: int = 6001
    robot_id: str = "beni-01"
    gpio_estop: int = 14                        # sysfs number of header pin 13 (ESP32 e-stop out, §13.4)
    phones: str = ""                            # "Name=MAC,..." paired phones for presence (§4 item 56)
    touch: str = "auto"                         # auto | off | /dev/input/eventN (§4 item 57)
    touch_cal: str = ""                         # XPT2046 raw "x0,x1,y0,y1"
    gpio_penirq: int = -1                       # XPT2046 PENIRQ, e.g. 13 = header pin 22; -1 = poll
    gpio_mute: int = -1                         # mic-mute button (active low), e.g. 194 = pin 15; -1 = none

    # voice FSM timings (§8.5)
    filler_after_s: float = 0.7
    echo_ignore_s: float = 0.4
    barge_min_speech_s: float = 0.25
    follow_up_s: float = 6.0
    no_speech_s: float = 5.0
    max_utterance_s: float = 15.0
    endpoint_s: float = 0.6                     # §8.6 #2: silence that ends a turn...
    endpoint_fast_s: float = 0.45               # ...in a quick back-and-forth (a follow-up to a short answer)
    endpoint_long_s: float = 0.8                # ...when the user has started something long ("tell me about")
    offline_after_s: float = 3.0

    @property
    def m(self):
        return lambda *p: os.path.join(self.models, *p)

    @classmethod
    def from_env(cls):
        load_env_file()
        urls = _env("BENI_BRAIN_URLS") or _env("BENI_BRAIN_URL", "ws://beni-brain:8765/ws")
        return cls(
            token=_env("BENI_TOKEN", ""),
            brain_urls=[u.strip() for u in urls.split(",") if u.strip()],
            uplink=_env("BENI_UPLINK", "pcm16"),
            db=_env("BENI_DB", cls.db),
            models=_env("BENI_MODELS", cls.models),
            logs=_env("BENI_LOGS", cls.logs),
            rec_dir=_env("BENI_REC_DIR", cls.rec_dir),
            backup_dir=_env("BENI_BACKUP_DIR", cls.backup_dir),
            maps=_env("BENI_MAPS", cls.maps),
            kaggle_kernel=_env("KAGGLE_KERNEL", ""),
            kaggle_dir=_env("BENI_KAGGLE_DIR", cls.kaggle_dir),
            awake_hours=parse_windows(_env("BENI_AWAKE_HOURS", "07:00-12:00,16:00-22:30")),
            weekly_budget_h=float(_env("BENI_WEEKLY_BUDGET_H", "28")),
            quiet_hours=parse_windows(_env("BENI_QUIET_HOURS", "22:00-07:00")),
            hf_token=_env("HF_TOKEN", ""),
            hf_repo=_env("HF_BACKUP_REPO", ""),
            home_city=_env("BENI_HOME_CITY", "Colombo"),
            llm_url=_env("BENI_LLM_URL", cls.llm_url),
            robot_id=_env("BENI_ROBOT_ID", cls.robot_id),
            gpio_estop=int(_env("BENI_GPIO_ESTOP", cls.gpio_estop)),
            phones=_env("BENI_PHONES", ""),
            touch=_env("BENI_TOUCH", "auto"),
            touch_cal=_env("BENI_TOUCH_CAL", ""),
            gpio_penirq=int(_env("BENI_GPIO_PENIRQ", cls.gpio_penirq)),
            gpio_mute=int(_env("BENI_GPIO_MUTE", cls.gpio_mute)),
        )
