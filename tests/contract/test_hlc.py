from beni_common.hlc import HLC, parse


def test_monotonic_with_frozen_clock():
    h = HLC("jetson", clock=lambda: 1000.0)
    a, b, c = h.now(), h.now(), h.now()
    assert a < b < c
    assert parse(c)[:2] == (1000000, 2)


def test_update_orders_after_remote():
    local = HLC("jetson", clock=lambda: 1000.0)
    remote = HLC("kaggle", clock=lambda: 2000.0).now()
    assert local.update(remote) > remote
    assert local.now() > remote


def test_lexicographic_equals_causal():
    ts = [HLC("a", clock=lambda: t).now() for t in (9.999, 10.0, 100.0)]
    assert ts == sorted(ts)
