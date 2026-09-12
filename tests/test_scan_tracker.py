"""
Tests for the windowed scan detector.

Synthetic ``(src_ip, dst_ip, dst_port, timestamp)`` tuples only -- no packets,
no sockets, no sleeping. Time is injected into every call, so a 60-second
window is exercised in microseconds and the suite is deterministic.

Needs pandas, sklearn and torch not at all: this file runs in a bare
environment with pytest and the standard library.
"""

from __future__ import annotations

import pathlib
import sys
import threading

import pytest

ROOT = pathlib.Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from backend.live.scan_tracker import ScanAlert, ScanTracker  # noqa: E402


def _tracker(**kw) -> ScanTracker:
    """Small thresholds so tests stay readable."""
    defaults = dict(
        window_seconds=60.0,
        vertical_port_threshold=10,
        horizontal_host_threshold=5,
        max_sources=100,
        alert_cooldown_seconds=60.0,
    )
    defaults.update(kw)
    return ScanTracker(**defaults)


# =====================================================================
# VERTICAL SCAN -- one source, many ports
# =====================================================================


def test_vertical_scan_fires_exactly_at_threshold():
    t = _tracker(vertical_port_threshold=10)
    alerts = []
    for i in range(9):
        alerts += t.observe("10.0.0.9", "10.0.0.1", 1000 + i, now=100.0 + i * 0.01)
    assert alerts == [], "fired before the threshold was reached"

    alerts = t.observe("10.0.0.9", "10.0.0.1", 1009, now=100.1)
    assert len(alerts) == 1
    a = alerts[0]
    assert a.kind == "vertical"
    assert a.src_ip == "10.0.0.9"
    assert a.distinct == 10
    assert a.threshold == 10


def test_repeating_one_port_never_fires():
    """Distinct count, not volume. A chatty client is not a scanner."""
    t = _tracker(vertical_port_threshold=10)
    alerts = []
    for i in range(500):
        alerts += t.observe("10.0.0.9", "10.0.0.1", 443, now=100.0 + i * 0.01)
    assert alerts == []


def test_below_threshold_stays_quiet():
    t = _tracker(vertical_port_threshold=10, horizontal_host_threshold=5)
    out = []
    for p in range(9):
        out += t.observe("10.0.0.9", "10.0.0.1", 2000 + p, now=100.0)
    assert out == []


# =====================================================================
# HORIZONTAL SCAN -- one source, many hosts
# =====================================================================


def test_horizontal_scan_fires_on_distinct_hosts():
    """The shape that a ports-only detector misses entirely."""
    t = _tracker(vertical_port_threshold=999, horizontal_host_threshold=5)
    alerts = []
    for i in range(5):
        alerts += t.observe("10.0.0.9", f"10.0.0.{i}", 445, now=100.0 + i)
    assert len(alerts) == 1
    assert alerts[0].kind == "horizontal"
    assert alerts[0].distinct == 5


def test_both_kinds_can_fire_from_one_observation():
    t = _tracker(vertical_port_threshold=5, horizontal_host_threshold=5)
    alerts = []
    for i in range(4):
        alerts += t.observe("10.0.0.9", f"10.0.0.{i}", 1000 + i, now=100.0)
    assert alerts == []
    alerts = t.observe("10.0.0.9", "10.0.0.4", 1004, now=100.0)
    assert {a.kind for a in alerts} == {"vertical", "horizontal"}


# =====================================================================
# THE SLIDING WINDOW
# =====================================================================


def test_observations_leave_the_window():
    t = _tracker(window_seconds=60.0, vertical_port_threshold=10)
    for p in range(9):
        t.observe("10.0.0.9", "10.0.0.1", 3000 + p, now=100.0)

    # 61 seconds later the first nine have aged out, so one more port is
    # a count of 1 -- not 10.
    alerts = t.observe("10.0.0.9", "10.0.0.1", 3009, now=161.0)
    assert alerts == [], "expired observations still counted toward the threshold"


def test_a_slow_scan_spread_past_the_window_never_fires():
    """9 ports/minute for ten minutes is not a scan, and must not look like one."""
    t = _tracker(window_seconds=60.0, vertical_port_threshold=10)
    alerts = []
    for i in range(100):
        alerts += t.observe("10.0.0.9", "10.0.0.1", 4000 + i, now=100.0 + i * 7.0)
    assert alerts == []


def test_expiry_decrements_rather_than_deletes_a_repeated_port():
    """A set cannot do this: it has forgotten how many times a port was seen."""
    t = _tracker(window_seconds=60.0, vertical_port_threshold=999)
    t.observe("10.0.0.9", "10.0.0.1", 80, now=100.0)
    t.observe("10.0.0.9", "10.0.0.1", 80, now=140.0)   # same port, later
    # At t=161 the first is out of the window but the second is not,
    # so port 80 must still be present.
    t.observe("10.0.0.9", "10.0.0.1", 81, now=161.0)
    top = t.top_sources(now=161.0)[0]
    assert top["distinct_ports"] == 2, "port dropped while a later observation was live"


def test_sweep_removes_silent_sources():
    t = _tracker(window_seconds=60.0)
    t.observe("10.0.0.9", "10.0.0.1", 80, now=100.0)
    assert t.snapshot(now=100.0)["tracked_sources"] == 1
    assert t.sweep(now=200.0) == 1
    assert t.snapshot(now=200.0)["tracked_sources"] == 0


# =====================================================================
# THE DETECTOR MUST NOT BE AN ATTACK SURFACE
# =====================================================================


def test_source_table_is_bounded_under_a_spoofed_source_flood():
    """nmap -D, or any flood with randomised sources.

    Without a cap this allocates one window per fake address and the detector
    becomes the memory-exhaustion vector. This is the security test.
    """
    t = _tracker(max_sources=100)
    for i in range(10_000):
        t.observe(f"10.{i // 65536}.{(i // 256) % 256}.{i % 256}",
                  "10.0.0.1", 80, now=100.0 + i * 0.001)
    snap = t.snapshot(now=110.0)
    assert snap["tracked_sources"] <= 100
    assert snap["sources_evicted"] >= 9_900


def test_eviction_is_least_recently_observed_not_arbitrary():
    t = _tracker(max_sources=3)
    t.observe("A", "10.0.0.1", 80, now=100.0)
    t.observe("B", "10.0.0.1", 80, now=101.0)
    t.observe("C", "10.0.0.1", 80, now=102.0)
    t.observe("A", "10.0.0.1", 81, now=103.0)   # A is now the freshest
    t.observe("D", "10.0.0.1", 80, now=104.0)   # forces one eviction

    tracked = {r["src_ip"] for r in t.top_sources(limit=10, now=104.0)}
    assert "B" not in tracked, "evicted the wrong source"
    assert tracked == {"A", "C", "D"}


def test_an_active_scanner_is_not_evicted_by_decoy_traffic():
    """The eviction policy has to survive the obvious evasion.

    A scanner that keeps transmitting stays at the fresh end of the LRU, so
    decoys cannot push it out from under the detector.
    """
    t = _tracker(max_sources=50, vertical_port_threshold=10)
    fired = []
    for i in range(200):
        # one real scan observation ...
        fired += t.observe("10.6.6.6", "10.0.0.1", 5000 + i, now=100.0 + i * 0.01)
        # ... buried in decoy noise from fresh addresses every time
        for d in range(5):
            t.observe(f"172.16.{i % 256}.{d}", "10.0.0.1", 80, now=100.0 + i * 0.01)
    assert any(a.src_ip == "10.6.6.6" for a in fired), "scanner evicted by decoys"


# =====================================================================
# ALERT SUPPRESSION
# =====================================================================


def test_one_alert_per_source_per_kind_within_the_cooldown():
    """Without this, every packet past the threshold emits another alert."""
    t = _tracker(vertical_port_threshold=10, alert_cooldown_seconds=60.0)
    alerts = []
    for i in range(200):
        alerts += t.observe("10.0.0.9", "10.0.0.1", 6000 + i, now=100.0 + i * 0.01)
    assert len(alerts) == 1
    assert t.snapshot(now=102.0)["alerts_suppressed"] > 100


def test_alerting_resumes_after_the_cooldown_expires():
    t = _tracker(window_seconds=600.0, vertical_port_threshold=10,
                 alert_cooldown_seconds=60.0)
    first = []
    for i in range(20):
        first += t.observe("10.0.0.9", "10.0.0.1", 7000 + i, now=100.0)
    assert len(first) == 1

    later = t.observe("10.0.0.9", "10.0.0.1", 7999, now=100.0 + 61.0)
    assert len(later) == 1, "stayed silent after the cooldown had expired"


# =====================================================================
# ALLOWLIST
# =====================================================================


def test_ignored_sources_are_never_tracked_or_alerted():
    """NAT gateways, proxies and your own vulnerability scanner."""
    t = _tracker(vertical_port_threshold=5, ignore_sources=["10.0.0.254"])
    alerts = []
    for p in range(50):
        alerts += t.observe("10.0.0.254", "10.0.0.1", 8000 + p, now=100.0)
    assert alerts == []
    assert t.snapshot(now=100.0)["tracked_sources"] == 0, "allowlisted source consumed a slot"


# =====================================================================
# CONCURRENCY
# =====================================================================


def test_concurrent_observers_do_not_corrupt_the_counts():
    """The capture thread writes while the API thread reads.

    Eight threads, 500 observations each, all distinct ports from one source.
    Every observation must be accounted for and no exception may escape.
    """
    t = _tracker(window_seconds=3600.0, vertical_port_threshold=10**9,
                 horizontal_host_threshold=10**9)
    errors: list = []

    def worker(base: int) -> None:
        try:
            for i in range(500):
                t.observe("10.0.0.9", "10.0.0.1", base + i, now=100.0)
        except Exception as exc:      # noqa: BLE001
            errors.append(exc)

    threads = [threading.Thread(target=worker, args=(b * 1000,)) for b in range(8)]
    for th in threads:
        th.start()
    for th in threads:
        th.join()

    assert not errors, f"exceptions escaped: {errors}"
    snap = t.snapshot(now=100.0)
    assert snap["observations"] == 8 * 500
    assert t.top_sources(now=100.0)[0]["distinct_ports"] == 8 * 500


def test_reader_does_not_block_or_crash_while_writers_run():
    t = _tracker(window_seconds=3600.0, vertical_port_threshold=10**9)
    stop = threading.Event()
    errors: list = []

    def writer() -> None:
        i = 0
        while not stop.is_set():
            t.observe("10.0.0.9", "10.0.0.1", i % 60000, now=100.0)
            i += 1

    def reader() -> None:
        try:
            for _ in range(300):
                t.snapshot(now=100.0)
                t.top_sources(limit=5, now=100.0)
        except Exception as exc:      # noqa: BLE001
            errors.append(exc)

    w = threading.Thread(target=writer, daemon=True)
    r = threading.Thread(target=reader)
    w.start(); r.start(); r.join(); stop.set(); w.join(timeout=2)
    assert not errors, f"reader failed while writers ran: {errors}"


# =====================================================================
# CONTRACT
# =====================================================================


@pytest.mark.parametrize("bad", [
    dict(window_seconds=0),
    dict(window_seconds=-1),
    dict(max_sources=0),
])
def test_nonsense_configuration_is_rejected_at_construction(bad):
    with pytest.raises(ValueError):
        _tracker(**bad)


def test_alert_is_immutable_and_self_describing():
    t = _tracker(vertical_port_threshold=3)
    alerts = []
    for p in (80, 443, 8080):
        alerts += t.observe("10.0.0.9", "10.0.0.1", p, now=100.0)
    a = alerts[0]
    assert isinstance(a, ScanAlert)
    with pytest.raises(Exception):
        a.distinct = 99                      # frozen dataclass
    text = a.describe()
    assert "10.0.0.9" in text and "vertical" in text


def test_alert_carries_a_sample_for_the_analyst():
    t = _tracker(vertical_port_threshold=5)
    alerts = []
    for p in range(10):
        alerts += t.observe("10.0.0.9", "10.0.0.1", 9000 + p, now=100.0)
    assert alerts[0].sample, "alert carried no example ports"
    assert len(alerts[0].sample) <= 10
