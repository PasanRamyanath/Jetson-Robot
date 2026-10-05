"""§4 item 19: teleop bitrate follows the measured Wi-Fi signal, with hysteresis; wired keeps the full rate."""
from beni_agent.scheduler import link as L


def test_steps_and_hysteresis():
    lr = L.LinkRate()
    assert lr.step(-55, False) == L.FULL_BPS
    assert lr.step(-55, False) is None                  # unchanged: nothing to send
    assert lr.step(-72, False) == 700000
    assert lr.step(-69, False) is None                  # inside the hysteresis band above -70
    assert lr.step(-66, False) == 1200000
    assert lr.step(-90, True) == L.FULL_BPS             # cable plugged in
    assert L.pick_bps(None, 700000) == 700000           # no reading: keep what we have


def test_proc_net_wireless(tmp_path):
    p = tmp_path / "wireless"
    p.write_text("Inter-| sta-|   Quality        |   Discarded packets\n"
                 " face | tus | link level noise |  nwid  crypt   frag  retry   misc | beacon | 22\n"
                 "wlan0: 0000   45.  -65.  -256        0      0      0      0     12        0\n")
    assert L.wifi_dbm(str(p)) == -65.0
    assert L.wifi_dbm(str(tmp_path / "none")) is None
