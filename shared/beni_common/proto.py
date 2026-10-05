"""ESP32 <-> Jetson binary protocol (§7.5), mirror of shared/proto/base_proto.h. Python 3.6-safe.

Frame on the wire: COBS([type u8][seq u8][len u8][payload][crc16 u16 LE]) 0x00
CRC16-CCITT (poly 0x1021, init 0xFFFF) over type..payload.
"""
import struct

MSG_CMD_VEL = 0x01
MSG_SERVO = 0x02
MSG_EXPRESSION = 0x03
MSG_LED = 0x04
MSG_CONFIG = 0x05
MSG_ESTOP = 0x06
MSG_STATE = 0x81
MSG_EVENT = 0x82
MSG_ACK = 0x83
MSG_LOG = 0x84

# StateMsg (packed, little endian) — 45 bytes.
STATE = struct.Struct("<IiihhfffhhhhHhB4B")
STATE_FIELDS = ("t_ms", "enc_l", "enc_r", "v_mm_s", "w_mrad_s", "x_m", "y_m", "yaw_rad",
                "gyro_z_mdps", "acc_x_mg", "acc_y_mg", "acc_z_mg", "batt_mv", "batt_ma", "flags")
CMD_VEL = struct.Struct("<hh")
CONFIG = struct.Struct("<5f")
ACK = struct.Struct("<BBB")
EV_BOOT, EV_BUMP, EV_CLIFF, EV_ESTOP, EV_LOW_BATT, EV_WATCHDOG, EV_SENSOR_FAULT = range(1, 8)
FLAG_ESTOP, FLAG_BUMP_L, FLAG_BUMP_R, FLAG_CLIFF, FLAG_CHARGING, FLAG_WD = (1 << i for i in range(6))


def _crc_table():
    t = []
    for i in range(256):
        c = i << 8
        for _ in range(8):
            c = ((c << 1) ^ 0x1021) if c & 0x8000 else (c << 1)
        t.append(c & 0xFFFF)
    return t


_CRC = _crc_table()


def crc16(data, crc=0xFFFF):
    for b in bytearray(data):
        crc = ((crc << 8) & 0xFFFF) ^ _CRC[((crc >> 8) ^ b) & 0xFF]
    return crc


def cobs_encode(data):
    out = bytearray()
    for block in bytes(data).split(b"\x00"):
        # Blocks longer than 254 bytes are split into 0xFF-coded runs.
        while len(block) >= 254:
            out.append(0xFF)
            out += block[:254]
            block = block[254:]
        out.append(len(block) + 1)
        out += block
    return bytes(out)


def cobs_decode(data):
    out = bytearray()
    i, n = 0, len(data)
    while i < n:
        code = data[i]
        end = i + code
        if code == 0 or end > n:
            raise ValueError("bad COBS frame")
        out += data[i + 1:end]
        i = end
        if code < 0xFF and i < n:
            out.append(0)
    return bytes(out)


def encode(msg_type, seq, payload=b""):
    if len(payload) > 255:
        raise ValueError("payload too long")
    body = struct.pack("<BBB", msg_type, seq & 0xFF, len(payload)) + bytes(payload)
    return cobs_encode(body + struct.pack("<H", crc16(body))) + b"\x00"


def decode(frame):
    """Decode one frame (without the trailing 0x00). Returns (type, seq, payload)."""
    raw = cobs_decode(frame)
    if len(raw) < 5:
        raise ValueError("short frame")
    body, (crc,) = raw[:-2], struct.unpack("<H", raw[-2:])
    if crc16(body) != crc:
        raise ValueError("crc mismatch")
    msg_type, seq, ln = body[0], body[1], body[2]
    if ln != len(body) - 3:
        raise ValueError("length mismatch")
    return msg_type, seq, bytes(body[3:])


class FrameReader(object):
    """Incremental splitter for a byte stream; yields decoded frames, drops corrupt ones."""

    def __init__(self):
        self._buf = bytearray()
        self.errors = 0

    def feed(self, data):
        self._buf += data
        while True:
            i = self._buf.find(b"\x00")
            if i < 0:
                if len(self._buf) > 1024:     # garbage without delimiter
                    del self._buf[:]
                return
            chunk = bytes(self._buf[:i])
            del self._buf[:i + 1]
            if not chunk:
                continue
            try:
                yield decode(chunk)
            except ValueError:
                self.errors += 1


def cmd_vel(seq, v_mm_s, w_mrad_s):
    clamp = lambda x: max(-32768, min(32767, int(x)))
    return encode(MSG_CMD_VEL, seq, CMD_VEL.pack(clamp(v_mm_s), clamp(w_mrad_s)))


def parse_state(payload):
    vals = STATE.unpack(payload)
    d = dict(zip(STATE_FIELDS, vals[:15]))
    d["range_cm"] = list(vals[15:])
    return d
