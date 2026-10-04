"""
Tests for the offline behaviour layer -- TORCH-FREE.

Hand-built ``(src, dst, port, time)`` tables, so every expected count can be
checked by eye.  The live tracker has its own suite (test_scan_tracker.py);
this one covers the dataset-side code and the agreement between the two.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "src"))

from backend.live.scan_tracker import ScanTracker  # noqa: E402
from behaviour import (  # noqa: E402
    behaviour_layer_scores, behaviour_score, canonical_endpoints, flow_keys,
    replay_scan_tracker, top_sources, week_behaviour, window_counts,
)
from config import (  # noqa: E402
    META_CAPTURE, META_DST_IP, META_DST_PORT, META_PROTO, META_ROW, META_SRC_IP,
    META_SRC_PORT, META_TS,
)

T0 = 1_499_400_000.0          # an arbitrary epoch second on a minute boundary


def _meta(rows, capture=0):
    """rows: (src, sport, dst, dport, seconds after T0)"""
    src, sport, dst, dport, t = zip(*rows)
    n = len(rows)
    return pd.DataFrame({
        META_SRC_IP: list(src), META_DST_IP: list(dst),
        META_SRC_PORT: np.asarray(sport, dtype=np.int32),
        META_DST_PORT: np.asarray(dport, dtype=np.int32),
        META_PROTO: np.full(n, 6, dtype=np.int16),
        META_TS: T0 + np.asarray(t, dtype=np.float64),
        META_ROW: np.arange(n, dtype=np.int32),
        META_CAPTURE: np.full(n, capture, dtype=np.int16),
    })


def _scan(src="172.16.0.1", dst="192.168.10.50", n=150, start=0.0, spread=50.0, first_port=1):
    return [(src, 40000 + i, dst, first_port + i, start + spread * i / n) for i in range(n)]


# =====================================================================
# DIRECTION
# =====================================================================


def test_server_as_source_flow_is_turned_round():
    m = _meta([("104.16.1.1", 443, "192.168.10.5", 54865, 0.0),     # reply seen first
               ("192.168.10.5", 54866, "104.16.1.1", 443, 1.0)])    # recorded correctly
    ep = canonical_endpoints(m, canonical=True)
    assert list(ep.swapped) == [True, False]
    assert ep.address(ep.src[0]) == "192.168.10.5" and ep.address(ep.dst[0]) == "104.16.1.1"
    assert list(ep.dport) == [443, 443]
    raw = canonical_endpoints(m, canonical=False)
    assert not raw.swapped.any() and list(raw.dport) == [54865, 443]


def test_ambiguous_port_pairs_are_left_as_recorded():
    m = _meta([("a", 53, "b", 53, 0.0),          # both service ports
               ("a", 50000, "b", 51000, 0.0)])   # neither
    assert not canonical_endpoints(m).swapped.any()


def test_busy_server_looks_like_a_scan_only_without_the_repair():
    """THE artefact: one web server answering 150 clients' ephemeral ports in a
    minute is 'one source, 150 distinct destination ports' as recorded."""
    rows = [("104.16.1.1", 443, f"192.168.10.{5 + i % 20}", 50000 + i, i * 0.3)
            for i in range(150)]
    m = _meta(rows)
    raw = window_counts(canonical_endpoints(m, canonical=False), 60.0)
    fixed = window_counts(canonical_endpoints(m, canonical=True), 60.0)
    assert raw["ports"].max() == 150
    assert fixed["ports"].max() == 1          # twenty clients, each touching port 443


# =====================================================================
# WINDOW COUNTS
# =====================================================================


def test_vertical_scan_counts_distinct_ports_not_flows():
    rows = _scan(n=120) + [("172.16.0.1", 41000, "192.168.10.50", 1, 10.0)] * 30   # repeats
    c = window_counts(canonical_endpoints(_meta(rows)), 60.0)
    assert set(c["ports"]) == {120} and set(c["hosts"]) == {1} and set(c["flows"]) == {150}


def test_horizontal_sweep_counts_distinct_hosts():
    rows = [("10.9.9.9", 40000 + i, f"192.168.10.{i}", 445, i * 0.5) for i in range(80)]
    c = window_counts(canonical_endpoints(_meta(rows)), 60.0)
    assert set(c["hosts"]) == {80} and set(c["ports"]) == {1}
    assert (behaviour_score(c, port_threshold=100, host_threshold=50) >= 1.0).all()


def test_windows_and_sources_do_not_mix():
    rows = (_scan(src="1.1.1.1", n=60, start=0.0, spread=50.0)
            + _scan(src="1.1.1.1", n=60, start=60.0, spread=50.0, first_port=1000)
            + _scan(src="2.2.2.2", n=60, start=0.0, spread=50.0))
    c = window_counts(canonical_endpoints(_meta(rows)), 60.0)
    assert set(c["ports"]) == {60}                     # never 120, never 180
    assert c["group"].nunique() == 3
    assert set(window_counts(canonical_endpoints(_meta(rows)), 120.0)["ports"]) == {120, 60}


def test_benign_clients_score_far_below_a_scanner():
    benign = [(f"192.168.10.{5 + i % 10}", 50000 + i, "104.16.1.1", 443, i * 0.4)
              for i in range(120)]
    rows = benign + _scan(n=150)
    s, _c = behaviour_layer_scores(_meta(rows))
    assert s[:120].max() < 0.1 and s[120:].min() >= 1.0


def test_behaviour_layer_refuses_to_run_without_identifiers():
    m = _meta(_scan(n=5))
    m[META_SRC_IP] = None
    with pytest.raises(RuntimeError, match="source IP"):
        canonical_endpoints(m)


# =====================================================================
# REPLAY THROUGH THE DEPLOYED TRACKER
# =====================================================================


def test_replay_and_window_score_agree_on_who_scans():
    rows = ([(f"192.168.10.{5 + i % 10}", 50000 + i, "104.16.1.1", 443, i * 0.4)
             for i in range(100)] + _scan(n=150))
    m = _meta(rows)
    labels = np.array(["BENIGN"] * 100 + ["PortScan"] * 150, dtype=object)
    rep = replay_scan_tracker(m, labels=labels)
    s, _c = behaviour_layer_scores(m)
    assert set(rep.alerts["src"]) == {"172.16.0.1"}
    assert set(np.asarray(m[META_SRC_IP])[s >= 1.0]) == {"172.16.0.1"}
    assert not rep.in_alert[:100].any()
    assert (rep.alerts["attack_share"] == 1.0).all()


def test_online_coverage_does_not_count_flows_before_the_threshold():
    """The first 99 ports are observed BEFORE the hundredth trips the tracker.
    Counting them as detected would describe a different, retrospective system."""
    m = _meta(_scan(n=150))
    rep = replay_scan_tracker(m, port_threshold=100)
    assert int(rep.in_alert.sum()) == 51 and not rep.in_alert[:99].any()
    s, _c = behaviour_layer_scores(m, port_threshold=100)
    assert (s >= 1.0).all()                    # the window score is retrospective


def _minute_stamped(rows):
    """The same flows with their times floored to the minute, as four of the
    five days of the dataset are stamped."""
    return _meta([(s, sp, d, dp, 60.0 * np.floor(t / 60.0)) for s, sp, d, dp, t in rows])


def test_replay_window_is_one_stamped_minute_not_two():
    """60 ports in one minute and 60 different ports in the next is 60 a
    minute, not 120.  A closed 60-second window on minute stamps would hold
    both minutes and raise an alert the deployed thresholds do not justify."""
    rows = (_scan(n=60, start=0.0, spread=50.0)
            + _scan(n=60, start=60.0, spread=50.0, first_port=5000))
    m = _minute_stamped(rows)
    assert set(m[META_TS] - T0) == {0.0, 60.0}
    rep = replay_scan_tracker(m, port_threshold=100)
    assert len(rep.alerts) == 0 and not rep.in_alert.any()
    assert window_counts(canonical_endpoints(m), 60.0)["ports"].max() == 60


def test_replay_and_window_counts_flag_the_same_source_minutes_on_minute_stamps():
    """Eight sources at different scan rates over five stamped minutes.  The
    set of (source, minute) pairs past the threshold must be the same whether
    it is counted in one pass or met online by the deployed tracker."""
    rng = np.random.default_rng(0)
    rows = []
    for s in range(8):
        for minute in range(5):
            k = int(rng.integers(20, 220))
            ports = rng.integers(1, 60000, k)
            rows += [(f"10.0.0.{s}", 40000 + i, "192.168.10.50", int(p), 60.0 * minute + 59.0 * i / k)
                     for i, p in enumerate(ports)]
    m = _minute_stamped(rows)
    src, minute = m[META_SRC_IP].to_numpy(), ((m[META_TS] - T0) // 60).to_numpy()

    score, _c = behaviour_layer_scores(m, port_threshold=100)
    by_count = {(a, b) for a, b in zip(src[score >= 1.0], minute[score >= 1.0])}
    rep = replay_scan_tracker(m, port_threshold=100)
    by_replay = {(a, b) for a, b in zip(src[rep.in_alert], minute[rep.in_alert])}
    assert by_count == by_replay and 5 < len(by_count) < 40
    assert len(rep.alerts) == len(by_count)            # one alert per flagged source-minute


def test_alert_on_benign_labelled_scan_is_reported_as_such():
    """A scan the dataset labels BENIGN must surface as an alert with
    attack_share 0 -- that row is the evidence of a labelling error."""
    m = _meta(_scan(src="192.168.10.8", n=150))
    rep = replay_scan_tracker(m, labels=np.array(["BENIGN"] * 150, dtype=object))
    assert len(rep.alerts) >= 1 and (rep.alerts["attack_share"] == 0.0).all()
    top = top_sources(m, window_counts(canonical_endpoints(m), 60.0),
                      np.array(["BENIGN"] * 150, dtype=object), k=1)
    assert top.loc[0, "src"] == "192.168.10.8" and top.loc[0, "attack_share"] == 0.0


def test_source_counts_reads_the_window_without_changing_it():
    t = ScanTracker(vertical_port_threshold=10, horizontal_host_threshold=5)
    for p in range(7):
        t.observe("1.1.1.1", "2.2.2.2", p, now=float(p))
    assert t.source_counts("1.1.1.1") == (7, 1)
    assert t.source_counts("1.1.1.1") == (7, 1)        # idempotent
    assert t.source_counts("9.9.9.9") == (0, 0)
    assert t.stats["observations"] == 7


# =====================================================================
# WHOLE-WEEK COUNTS
# =====================================================================


def test_the_study_uses_the_numbers_the_live_tracker_ships_with():
    """The paper says the behaviour layer is measured "at its deployed
    thresholds".  The study keeps its own copy of those three numbers (it must
    not import the backend to read a config value), so this pins the copy to
    the original.  Change one without the other and this fails."""
    from backend.live import scan_tracker
    from config import (BEHAVIOUR_HOST_THRESHOLD, BEHAVIOUR_PORT_THRESHOLD,
                        BEHAVIOUR_WINDOW_SECONDS)

    assert BEHAVIOUR_PORT_THRESHOLD == scan_tracker.VERTICAL_PORT_THRESHOLD == 100
    assert BEHAVIOUR_HOST_THRESHOLD == scan_tracker.HORIZONTAL_HOST_THRESHOLD == 50
    assert BEHAVIOUR_WINDOW_SECONDS == scan_tracker.WINDOW_SECONDS == 60.0
    t = ScanTracker()
    assert (t.window_seconds, t.vertical_port_threshold, t.horizontal_host_threshold) == (
        BEHAVIOUR_WINDOW_SECONDS, BEHAVIOUR_PORT_THRESHOLD, BEHAVIOUR_HOST_THRESHOLD)


def test_week_counts_are_not_halved_by_a_split():
    """A split that holds half of a window must still see the whole window's
    count -- otherwise validation thresholds come out about half as high as
    they should."""
    m = _meta(_scan(n=160))
    week = week_behaviour(m, port_threshold=100)
    half = m.iloc[::2].reset_index(drop=True)
    s_week, c_week = week.lookup(half)
    s_split, c_split = behaviour_layer_scores(half, port_threshold=100)
    assert set(c_week["ports"]) == {160} and (s_week >= 1.0).all()
    assert set(c_split["ports"]) == {80} and (s_split < 1.0).all()


def test_week_lookup_follows_row_order_and_rejects_foreign_rows():
    rows = _scan(n=120) + [("192.168.10.5", 50001, "104.16.1.1", 443, 5.0)]
    m = _meta(rows)
    week = week_behaviour(m)
    shuffled = m.sample(frac=1.0, random_state=0).reset_index(drop=True)
    s, _c = week.lookup(shuffled)
    is_scanner = (shuffled[META_SRC_IP] == "172.16.0.1").to_numpy()
    assert (s[is_scanner] >= 1.0).all() and (s[~is_scanner] < 1.0).all()
    assert np.array_equal(np.sort(week.scan_like()), np.sort(flow_keys(m)[:120]))
    with pytest.raises(RuntimeError, match="not in the week-level"):
        week.lookup(_meta(_scan(n=3), capture=7))


def test_flow_keys_are_unique_across_captures():
    a, b = _meta(_scan(n=5), capture=0), _meta(_scan(n=5), capture=1)
    keys = flow_keys(pd.concat([a, b], ignore_index=True))
    assert np.unique(keys).size == 10
