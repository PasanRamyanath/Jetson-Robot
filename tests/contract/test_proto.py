"""ESP32 <-> Jetson frame codec (§7.5) and the msgpack envelope."""
import pytest

from beni_common import proto
from beni_common.schemas import CLOUD_K2J, pack, unpack


def test_cobs_roundtrip_all_bytes():
    for data in (b"", b"\x00", b"\x00\x00", bytes(range(256)), b"\x01" * 300):
        enc = proto.cobs_encode(data)
        assert b"\x00" not in enc
        assert proto.cobs_decode(enc) == data


def test_frame_roundtrip_and_crc():
    f = proto.cmd_vel(7, 250, -500)
    assert f.endswith(b"\x00")
    t, seq, payload = proto.decode(f[:-1])
    assert (t, seq) == (proto.MSG_CMD_VEL, 7)
    assert proto.CMD_VEL.unpack(payload) == (250, -500)
    bad = bytearray(proto.cobs_decode(f[:-1]))
    bad[3] ^= 0xFF
    with pytest.raises(ValueError):
        proto.decode(proto.cobs_encode(bytes(bad)))


def test_frame_reader_splits_stream():
    r = proto.FrameReader()
    stream = proto.cmd_vel(1, 10, 20) + proto.cmd_vel(2, 30, 40)
    out = list(r.feed(stream[:5])) + list(r.feed(stream[5:])) + list(r.feed(b"junk\x00"))
    assert [seq for _, seq, _ in out] == [1, 2] and r.errors == 1


def test_state_parse():
    vals = (1000, 5, -5, 120, 30, 1.5, -0.5, 0.25, 10, 1, 2, 1000, 12000, -800, proto.FLAG_CHARGING, 0, 0, 0, 0)
    s = proto.parse_state(proto.STATE.pack(*vals))
    assert s["batt_mv"] == 12000 and s["flags"] == proto.FLAG_CHARGING


def test_msgpack_bytes_survive():
    m = {"type": "tts.chunk", "pcm16_24k": b"\x00\x01" * 10, "final": False}
    assert unpack(pack(m)) == m
    assert "scene" in CLOUD_K2J
