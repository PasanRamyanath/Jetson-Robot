"""Fan curve and thermal guard (§15.2 items 7 and 10, §4 item 48). The curve is pure; `Fan` is a tiny sysfs writer.

L4T 32.x has no nvfancontrol: the kernel pwm-fan driver ramps on its own unless `temp_control` is 0, and
`jetson_clocks` forces the fan to max. The scheduler rewrites `target_pwm` every tick, which undoes both.
"""
import logging
import os

log = logging.getLogger("thermal")

FAN_ROOT = "/sys/devices/pwm-fan"
FAN_MIN, FAN_MAX = 60, 255          # low PWM keeps air moving (hot silicon leaks more), full at FAN_HIGH_C
FAN_LOW_C, FAN_HIGH_C = 55.0, 70.0
FAN_DOWN_STEP = 16                  # per tick: spin up at once, spin down slowly (no hunting, no audible pumping)
HOT_C, COOL_C = 80.0, 72.0          # thermal guard with hysteresis: above HOT_C the scheduler sheds load


def fan_pwm(temp_c):
    if temp_c is None:
        return FAN_MAX
    if temp_c <= FAN_LOW_C:
        return FAN_MIN
    if temp_c >= FAN_HIGH_C:
        return FAN_MAX
    return int(round(FAN_MIN + (FAN_MAX - FAN_MIN) * (temp_c - FAN_LOW_C) / (FAN_HIGH_C - FAN_LOW_C)))


def next_pwm(current, temp_c):
    want = fan_pwm(temp_c)
    if current is None or want >= current:
        return want
    return max(want, current - FAN_DOWN_STEP)


class ThermalGuard:
    def __init__(self):
        self.hot = False

    def feed(self, temp_c):
        if temp_c is not None:
            if temp_c >= HOT_C and not self.hot:
                log.warning("thermal guard on (%.1f C)", temp_c)
                self.hot = True
            elif temp_c <= COOL_C and self.hot:
                log.info("thermal guard off (%.1f C)", temp_c)
                self.hot = False
        return self.hot


class Fan:
    def __init__(self, root=FAN_ROOT, dry_run=False):
        self.root, self.dry = root, dry_run
        self.pwm = None
        self.ok = os.path.exists(os.path.join(root, "target_pwm"))
        if not self.ok:
            log.info("no pwm-fan at %s: fan curve disabled", root)
        elif not dry_run:
            self._write("temp_control", 0)

    def _write(self, name, v):
        try:
            with open(os.path.join(self.root, name), "w") as f:
                f.write(str(v))
            return True
        except OSError as e:
            log.warning("fan %s: %s", name, e)
            self.ok = name != "target_pwm" and self.ok
            return False

    def set(self, temp_c):
        if not self.ok:
            return None
        pwm = next_pwm(self.pwm, temp_c)
        if self.dry or self._write("target_pwm", pwm):
            self.pwm = pwm
        return self.pwm
