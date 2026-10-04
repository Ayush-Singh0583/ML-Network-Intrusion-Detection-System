"""
Tests for the evaluation arithmetic -- TORCH-FREE.

Thresholds, block-bootstrap intervals, the two fusion rules and the two
helpers that edit a split (``sanitise_bundle``) or re-fit on it
(``lodo_benign_scores``).  Everything the paper's tables are made of passes
through this code, so each test pins one property a reviewer would ask about.
"""

from __future__ import annotations

import dataclasses
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from sklearn.preprocessing import LabelEncoder, StandardScaler

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "src"))

from behaviour import flow_keys  # noqa: E402
from config import META_CAPTURE, META_ROW, META_TS  # noqa: E402
from fusion import (  # noqa: E402
    bonferroni_fired, empirical_pvalues, layer_attribution, layer_subsets, minp_scores,
)
from preprocessing import FrameCleaner, SplitBundle  # noqa: E402
from stats import (  # noqa: E402
    ANY_ATTACK, MIN_BLOCKS_FOR_INTERVAL, auc_table, block_ids, bootstrap_rates,
    budget_threshold, detection_table, fired_at_budget, paired_difference,
    projected_precision, summarise_seeds,
)

T0 = 1_499_400_000.0


# =====================================================================
# BLOCKS
# =====================================================================


def test_blocks_are_capture_minutes_when_there_are_timestamps():
    meta = pd.DataFrame({
        META_CAPTURE: np.array([0, 0, 0, 0, 1, 1], dtype=np.int16),
        META_TS: T0 + np.array([0.0, 59.0, 60.0, 61.0, 0.0, 30.0]),
        META_ROW: np.arange(6, dtype=np.int32),
    })
    b = block_ids(meta, 6, block_seconds=60.0)
    assert b[0] == b[1] and b[2] == b[3] and b[4] == b[5]
    assert len({b[0], b[2], b[4]}) == 3          # minute and capture both separate blocks


def test_blocks_fall_back_to_runs_of_rows_without_timestamps():
    meta = pd.DataFrame({
        META_CAPTURE: np.array([0] * 6 + [1] * 4, dtype=np.int16),
        META_TS: np.full(10, np.nan),
        META_ROW: np.array([0, 1, 2, 3, 4, 5, 0, 1, 2, 3], dtype=np.int32),
    })
    b = block_ids(meta, 10, fallback_rows=3)
    assert list(b) == [0, 0, 0, 1, 1, 1, 2, 2, 2, 3]
    assert list(block_ids(None, 7, fallback_rows=3)) == [0, 0, 0, 1, 1, 1, 2]


# =====================================================================
# BOOTSTRAP
# =====================================================================


def test_point_estimate_is_the_plain_rate_whatever_the_blocks():
    rng = np.random.default_rng(0)
    fired = rng.random(5000) < 0.3
    mask = rng.random(5000) < 0.5
    for blocks in (None, np.arange(5000), np.arange(5000) // 100):
        r = bootstrap_rates(fired, {"c": mask}, blocks, n_boot=200, seed=1)["c"]
        assert r["rate"] == pytest.approx(fired[mask].mean()) and r["n"] == int(mask.sum())


def test_block_interval_is_wide_when_detection_comes_in_bursts():
    """THE reason for the block bootstrap.  20,000 flows in 20 one-minute
    blocks; a block is either wholly detected or wholly missed.  Treating the
    flows as independent gives an interval under 2 points wide.  Resampling
    blocks gives one tens of points wide, which is the honest answer when the
    evidence is really twenty observations."""
    n_blocks, per = 20, 1000
    blocks = np.repeat(np.arange(n_blocks), per)
    fired = np.repeat(np.arange(n_blocks) % 2 == 0, per)          # 10 of 20 blocks fire
    mask = np.ones(n_blocks * per, dtype=bool)
    by_block = bootstrap_rates(fired, {"c": mask}, blocks, n_boot=2000, seed=0)["c"]
    by_flow = bootstrap_rates(fired, {"c": mask}, np.arange(mask.size), n_boot=300, seed=0)["c"]
    assert by_block["rate"] == by_flow["rate"] == 0.5
    assert by_flow["hi"] - by_flow["lo"] < 0.02
    assert by_block["hi"] - by_block["lo"] > 0.30
    assert by_block["lo"] < 0.5 < by_block["hi"]


def test_bootstrap_is_reproducible_and_handles_degenerate_input():
    fired = np.array([True, False, True, True])
    blocks = np.array([0, 0, 1, 1])
    a = bootstrap_rates(fired, {"c": np.ones(4, bool)}, blocks, n_boot=100, seed=3)
    b = bootstrap_rates(fired, {"c": np.ones(4, bool)}, blocks, n_boot=100, seed=3)
    assert a == b
    empty = bootstrap_rates(fired, {"none": np.zeros(4, bool)}, blocks, n_boot=50)["none"]
    assert empty["n"] == 0 and np.isnan(empty["rate"])
    plain = bootstrap_rates(fired, {"c": np.ones(4, bool)}, None)["c"]
    assert plain["rate"] == 0.75 and np.isnan(plain["lo"])
    with pytest.raises(ValueError, match="blocks has"):
        bootstrap_rates(fired, {"c": np.ones(4, bool)}, np.array([0, 1]), n_boot=10)


# =====================================================================
# THRESHOLDS
# =====================================================================


@pytest.mark.parametrize("n", [7, 100, 999, 1000, 40_000, 143_917])
@pytest.mark.parametrize("budget", [0.001, 0.005, 0.01, 0.001 / 3, 0.05])
def test_budget_threshold_never_lets_more_than_the_budget_through(n, budget):
    """The guarantee, checked on awkward sizes: on the sample the threshold
    was fitted on, the share strictly above it is at most the budget -- and it
    is the tightest such threshold, so no more than one flow is left unused."""
    ref = np.random.default_rng(n).normal(size=n)
    tau = budget_threshold(ref, budget)
    above = int((ref > tau).sum())
    assert above <= n * budget + 1e-9
    assert above == int(np.floor(n * budget + 1e-9))
    assert tau in ref                                   # an observed score, not an interpolation


def test_interpolated_quantile_breaks_the_guarantee_this_function_keeps():
    """Why ``np.quantile`` is not used: 40,000 benign flows, a 0.1% budget
    shared by three layers.  40 false alarms are allowed."""
    ref = np.random.default_rng(3).normal(size=40_000)
    share = 0.001 / 3
    assert 3 * int((ref > np.quantile(ref, 1.0 - share)).sum()) == 42
    assert 3 * int((ref > budget_threshold(ref, share)).sum()) == 39


def test_budget_threshold_edge_cases():
    assert budget_threshold(np.array([1.0, 2.0, 3.0]), 0.01) == 3.0     # too small: nothing fires
    assert budget_threshold(np.r_[np.zeros(990), np.ones(10)], 0.01) == 0.0   # exactly 1% at 1
    assert budget_threshold(np.r_[np.zeros(980), np.ones(20)], 0.01) == 1.0   # 2% tie: none fire
    assert budget_threshold(np.array([np.nan, 1.0, np.inf, 2.0]), 0.5) == 1.0  # non-finite ignored
    with pytest.raises(ValueError, match="no validation benign"):
        budget_threshold(np.array([np.nan]), 0.01)
    for bad in (0.0, 1.0, -0.1, 1.5):
        with pytest.raises(ValueError, match="between 0 and 1"):
            budget_threshold(np.arange(10.0), bad)


# =====================================================================
# DETECTION TABLE
# =====================================================================


def test_threshold_comes_from_validation_benign_only():
    """Changing the test scores must not move the threshold; changing the
    validation benign scores must."""
    rng = np.random.default_rng(0)
    ref = rng.random(10_000)
    y = np.array(["BENIGN"] * 500 + ["DDoS"] * 500, dtype=object)
    s1 = np.r_[rng.random(500), rng.random(500) + 0.5]
    t1 = detection_table(y, s1, ref, budgets=[0.01], n_boot=0)
    t2 = detection_table(y, s1 * 3.0, ref, budgets=[0.01], n_boot=0)
    t3 = detection_table(y, s1, ref * 0.5, budgets=[0.01], n_boot=0)
    assert int((ref > t1["threshold"].iloc[0]).sum()) == 100          # 1% of 10,000, exactly
    assert t2["threshold"].iloc[0] == t1["threshold"].iloc[0]
    assert t3["threshold"].iloc[0] == pytest.approx(0.5 * t1["threshold"].iloc[0])


def test_table_has_one_row_per_budget_and_class_plus_any_attack():
    y = np.array(["BENIGN"] * 100 + ["DDoS"] * 50 + ["Bot"] * 10, dtype=object)
    s = np.r_[np.zeros(100), np.ones(50), np.zeros(10)]
    ref = np.linspace(0.0, 0.5, 1000)
    t = detection_table(y, s, ref, budgets=[0.001, 0.01], n_boot=0, detector="x")
    assert len(t) == 2 * 3 and set(t["class"]) == {"DDoS", "Bot", ANY_ATTACK}
    row = t[(t["budget"] == 0.01)].set_index("class")
    assert row.loc["DDoS", "detection_rate"] == 1.0 and row.loc["Bot", "detection_rate"] == 0.0
    assert row.loc[ANY_ATTACK, "detection_rate"] == pytest.approx(50 / 60)
    assert row.loc[ANY_ATTACK, "n"] == 60 and row.loc["DDoS", "n_benign"] == 100
    assert row["observed_fpr"].nunique() == 1 and row["observed_fpr"].iloc[0] == 0.0
    assert (t["detector"] == "x").all()


def test_ties_at_the_quantile_never_push_validation_false_alarms_over_budget():
    """The behaviour layer's score is a count, so thousands of benign flows tie.
    2% of validation benign flows sit at score 1.  With ``>=`` a 1% budget
    would fire on all of them (2%); with strict ``>`` it fires on none."""
    ref = np.r_[np.zeros(980), np.ones(20)]
    y = np.array(["BENIGN"] * 1000, dtype=object)
    t = detection_table(np.r_[y, ["PortScan"]], np.r_[ref, 5.0], ref, budgets=[0.01], n_boot=0)
    assert t["threshold"].iloc[0] == 1.0
    assert t["observed_fpr"].iloc[0] == 0.0 <= 0.01
    assert t.set_index("class").loc["PortScan", "detection_rate"] == 1.0


def test_detection_table_refuses_unusable_input():
    y = np.array(["BENIGN", "DDoS"], dtype=object)
    with pytest.raises(ValueError, match="no validation benign"):
        detection_table(y, np.zeros(2), np.array([np.nan]), budgets=[0.01], n_boot=0)
    with pytest.raises(ValueError, match="scores"):
        detection_table(y, np.zeros(3), np.zeros(5), budgets=[0.01], n_boot=0)


def test_observed_false_alarm_rate_reports_drift_instead_of_hiding_it():
    """Validation benign scores sit lower than test benign scores (a new day).
    The table must show the budget AND the larger rate it turned into."""
    rng = np.random.default_rng(1)
    ref = rng.normal(0.0, 1.0, 20_000)
    y = np.array(["BENIGN"] * 20_000 + ["DDoS"] * 100, dtype=object)
    s = np.r_[rng.normal(0.5, 1.0, 20_000), np.full(100, 10.0)]
    t = detection_table(y, s, ref, budgets=[0.01], n_boot=0)
    assert t["budget"].iloc[0] == 0.01 and t["observed_fpr"].iloc[0] > 0.02


def test_n_blocks_exposes_an_interval_that_has_nothing_to_resample():
    """A class inside ONE block gets a zero-width interval from the block
    bootstrap.  That is not certainty; ``n_blocks`` is how the reader (and the
    decision rules) can tell."""
    blocks = np.r_[np.zeros(100, dtype=int), np.repeat(np.arange(1, 41), 25)]
    y = np.array(["Heartbleed"] * 100 + ["BENIGN"] * 1000, dtype=object)
    rng = np.random.default_rng(0)
    s = np.r_[(np.arange(100) < 60).astype(float) * 9.0, rng.random(1000)]
    t = detection_table(y, s, np.linspace(0, 1, 5000), budgets=[0.01], blocks=blocks,
                        n_boot=300).set_index("class")
    assert t.loc["Heartbleed", "detection_rate"] == 0.6
    assert t.loc["Heartbleed", "det_lo"] == t.loc["Heartbleed", "det_hi"] == 0.6
    assert t.loc["Heartbleed", "n_blocks"] == 1 < MIN_BLOCKS_FOR_INTERVAL
    # ... and the table says so itself, in a column, on every row
    assert not t.loc["Heartbleed", "interval_counts"]
    r = bootstrap_rates(s > 0.5, {"benign": y == "BENIGN"}, blocks, n_boot=100)["benign"]
    assert r["n_blocks"] == 40 >= MIN_BLOCKS_FOR_INTERVAL and r["lo"] < r["hi"]
    d = paired_difference(s > 0.5, s > 0.9, {"hb": y == "Heartbleed"}, blocks, n_boot=50)["hb"]
    assert d["n_blocks"] == 1


# =====================================================================
# COMPARING TWO DETECTORS
# =====================================================================


def _bursty_pair(extra=0.02, n_blocks=20, per=2000, seed=0):
    """Detector B's hit rate swings from 10% to 90% between blocks; detector A
    catches everything B does plus ``extra`` of each block."""
    rng = np.random.default_rng(seed)
    blocks = np.repeat(np.arange(n_blocks), per)
    p_block = np.linspace(0.1, 0.9, n_blocks)[blocks]
    u = rng.random(blocks.size)
    b = u < p_block
    a = u < p_block + extra
    return a, b, blocks


def test_paired_difference_sees_a_small_gain_that_separate_intervals_hide():
    """The two detection rates differ by 2 points while each one's own
    interval is tens of points wide, because the test traffic is bursty.
    Reading the two intervals side by side says "no difference".  The paired
    interval, which resamples the same minutes for both, says otherwise."""
    a, b, blocks = _bursty_pair()
    m = {"c": np.ones(a.size, dtype=bool)}
    ia = bootstrap_rates(a, m, blocks, n_boot=1000, seed=0)["c"]
    ib = bootstrap_rates(b, m, blocks, n_boot=1000, seed=0)["c"]
    assert ia["lo"] < ib["hi"] and ib["lo"] < ia["hi"]                 # they overlap widely
    assert ia["hi"] - ia["lo"] > 0.15
    d = paired_difference(a, b, m, blocks, n_boot=1000, seed=0)["c"]
    assert d["diff"] == pytest.approx(a.mean() - b.mean()) == pytest.approx(0.02, abs=0.004)
    assert 0 < d["lo"] < d["diff"] < d["hi"] and d["hi"] - d["lo"] < 0.01
    assert d["rate_a"] == pytest.approx(a.mean()) and d["rate_b"] == pytest.approx(b.mean())


def test_paired_difference_of_a_detector_with_itself_is_exactly_zero():
    a, _b, blocks = _bursty_pair()
    d = paired_difference(a, a, {"c": np.ones(a.size, dtype=bool)}, blocks, n_boot=200)["c"]
    assert d["diff"] == d["lo"] == d["hi"] == 0.0


def test_paired_difference_can_be_negative_and_handles_degenerate_input():
    a, b, blocks = _bursty_pair()
    half = np.arange(a.size) % 2 == 0
    r = paired_difference(b, a, {"all": np.ones(a.size, bool), "half": half,
                                 "none": np.zeros(a.size, bool)}, blocks, n_boot=300)
    assert r["all"]["hi"] < 0 and r["half"]["n"] == int(half.sum())
    assert r["none"]["n"] == 0 and np.isnan(r["none"]["diff"])
    plain = paired_difference(a, b, {"all": np.ones(a.size, bool)}, None)["all"]
    assert plain["diff"] > 0 and np.isnan(plain["lo"])
    with pytest.raises(ValueError, match="different rows"):
        paired_difference(a, b[:-1], {"all": np.ones(a.size, bool)}, blocks)
    with pytest.raises(ValueError, match="blocks has"):
        paired_difference(a, b, {"all": np.ones(a.size, bool)}, blocks[:-1], n_boot=10)


def test_fired_at_budget_is_the_rule_the_detection_table_applies():
    rng = np.random.default_rng(0)
    ref = rng.random(5000)
    y = np.array(["BENIGN"] * 3000 + ["DDoS"] * 1000, dtype=object)
    s = np.r_[rng.random(3000), rng.random(1000) + 0.6]
    fired = fired_at_budget(s, ref, 0.01)
    t = detection_table(y, s, ref, budgets=[0.01], n_boot=0).set_index("class")
    assert fired[y == "DDoS"].mean() == t.loc["DDoS", "detection_rate"]
    assert fired[y == "BENIGN"].mean() == t.loc["DDoS", "observed_fpr"]


def test_projected_precision_is_the_base_rate_arithmetic():
    # half of the attacks caught, 1% false alarms, attacks 0.1% of traffic:
    # 5 true alerts for every 99.9 false ones
    assert projected_precision(0.5, 0.01, 0.001) == pytest.approx(0.0005 / (0.0005 + 0.00999))
    assert projected_precision(0.5, 0.01, 0.001) < 0.05 < 0.95 < projected_precision(0.5, 0.01, 0.5)
    assert projected_precision(1.0, 0.0, 0.001) == 1.0
    assert np.isnan(projected_precision(0.0, 0.0, 0.01))


# =====================================================================
# AUC
# =====================================================================


def test_auc_table_separates_ranking_from_usefulness_at_low_false_alarms():
    """A class scored 0.6 against benign scores uniform on [0, 1] ranks above
    60% of benign flows (AUROC 0.6) and above none of the top 1% -- partial
    AUC at or below chance.  The second number is the one that matters."""
    benign = np.linspace(0.0, 1.0, 10_001)
    y = np.array(["BENIGN"] * benign.size + ["Slow"] * 200 + ["Loud"] * 200, dtype=object)
    s = np.r_[benign, np.full(200, 0.6), np.full(200, 2.0)]
    t = auc_table(y, s, max_fpr=0.01, detector="d").set_index("class")
    assert t.loc["Loud", "auroc"] == 1.0 and t.loc["Loud", "pauc@0.01"] == 1.0
    assert t.loc["Slow", "auroc"] == pytest.approx(0.6, abs=0.01)
    assert t.loc["Slow", "pauc@0.01"] <= 0.5
    assert ANY_ATTACK in t.index and t.loc[ANY_ATTACK, "n"] == 400


def test_auc_table_is_nan_without_both_populations():
    t = auc_table(np.array(["DDoS"] * 5, dtype=object), np.arange(5.0))
    assert t["auroc"].isna().all()


def test_summarise_seeds_reports_mean_spread_and_count():
    frames = [pd.DataFrame({"class": ["a", "b"], "rate": [v, 1.0]}) for v in (0.1, 0.2, 0.3)]
    out = summarise_seeds(frames, ["class"], ["rate"]).set_index("class")
    assert out.loc["a", "rate_mean"] == pytest.approx(0.2)
    assert out.loc["a", "rate_sd"] == pytest.approx(0.1)
    assert out.loc["b", "rate_sd"] == 0.0 and out.loc["a", "n_seeds"] == 3
    assert summarise_seeds([], ["class"], ["rate"]).empty


# =====================================================================
# FUSION
# =====================================================================


def test_empirical_pvalues_follow_the_definition():
    ref = np.array([4.0, 1.0, 3.0, 2.0])
    p = empirical_pvalues(ref, np.array([2.5, 100.0, 0.0, 3.0]))
    assert list(p) == pytest.approx([3 / 5, 1 / 5, 5 / 5, 3 / 5])      # ties count against
    assert (p > 0).all()
    with pytest.raises(ValueError):
        empirical_pvalues(np.array([np.nan]), np.zeros(1))


def test_minp_fires_on_a_flow_that_is_extreme_in_any_one_layer():
    rng = np.random.default_rng(0)
    vb = {"a": rng.random(5000), "b": rng.random(5000)}
    s = {"a": np.array([0.5, 9.0, 0.5, 0.5]), "b": np.array([0.5, 0.5, 9.0, 0.5])}
    fused = minp_scores(vb, s, ["a", "b"])
    assert fused[1] == fused[2] > fused[0] == fused[3]
    assert fused[1] == pytest.approx(-np.log10(1 / 5001))
    assert np.array_equal(minp_scores(vb, s, ["a"]), -np.log10(empirical_pvalues(vb["a"], s["a"])))
    with pytest.raises(ValueError):
        minp_scores(vb, s, [])


@pytest.mark.parametrize("k", [1, 2, 3])
def test_both_fusion_rules_keep_validation_false_alarms_within_the_budget(k):
    """The comparison the paper rests on: k layers fused must cost no more
    benign false alarms on validation than one layer does."""
    rng = np.random.default_rng(k)
    layers = [f"l{i}" for i in range(k)]
    vb = {name: rng.normal(size=40_000) for name in layers}
    for budget in (0.001, 0.01):
        fired, per = bonferroni_fired(vb, vb, layers, budget)
        assert fired.mean() <= budget + 1e-12
        assert np.array_equal(fired, np.logical_or.reduce([per[name] for name in layers]))
        fused = minp_scores(vb, vb, layers)
        assert (fused > budget_threshold(fused, budget)).mean() <= budget + 1e-12


def test_minp_spends_the_budget_that_bonferroni_leaves_unused():
    """Two layers that fire on the SAME benign flows.  Bonferroni halves each
    layer's budget and so uses about half of it; min-p uses nearly all."""
    rng = np.random.default_rng(0)
    base = rng.normal(size=50_000)
    vb = {"a": base, "b": base.copy()}
    fired, _ = bonferroni_fired(vb, vb, ["a", "b"], 0.01)
    fused = minp_scores(vb, vb, ["a", "b"])
    used_minp = (fused > budget_threshold(fused, 0.01)).mean()
    assert fired.mean() == pytest.approx(0.005) and used_minp == pytest.approx(0.01)


def test_layer_subsets_is_the_full_ablation_grid():
    subs = layer_subsets(["c", "n", "b"])
    assert len(subs) == 7 and subs[:3] == [("c",), ("n",), ("b",)] and subs[-1] == ("c", "n", "b")


def test_attribution_shares_sum_to_one_per_class_and_name_the_layers():
    y = np.array(["BENIGN"] * 4 + ["PortScan"] * 6, dtype=object)
    per = {
        "classifier": np.array([0, 0, 0, 1, 1, 1, 0, 0, 0, 0], dtype=bool),
        "behaviour":  np.array([0, 0, 0, 0, 1, 0, 1, 1, 1, 0], dtype=bool),
    }
    a = layer_attribution(per, y)
    assert a.groupby("class")["share"].sum().round(12).eq(1.0).all()
    ps = a[a["class"] == "PortScan"].set_index("layers")
    assert ps.loc["classifier+behaviour", "flows"] == 1
    assert ps.loc["classifier", "flows"] == 1 and ps.loc["behaviour", "flows"] == 3
    assert ps.loc["none", "flows"] == 1 and ps.loc["none", "share"] == pytest.approx(1 / 6)
    assert (ps["class_total"] == 6).all()


# =====================================================================
# HELPERS THAT EDIT OR RE-FIT A SPLIT
# =====================================================================


def _bundle(n_train=60, n_val=40, n_test=30, seed=0):
    """A small, valid SplitBundle.  X[:, 0] carries the row's own number, so a
    misaligned side-table shows up as a wrong number, not a silent pass."""
    rng = np.random.default_rng(seed)

    def part(n, capture, labels):
        X = rng.normal(size=(n, 4)).astype(np.float32)
        X[:, 0] = np.arange(n, dtype=np.float32) / 100.0
        meta = pd.DataFrame({
            META_CAPTURE: np.full(n, capture, dtype=np.int16),
            META_ROW: np.arange(n, dtype=np.int32),
            META_TS: T0 + np.arange(n, dtype=np.float64),
        })
        return X, np.asarray(labels, dtype=object), meta

    Xtr, ytr, mtr = part(n_train, 0, ["BENIGN"] * (n_train - 10) + ["DoS"] * 10)
    Xva, yva, mva = part(n_val, 3, ["BENIGN"] * (n_val - 5) + ["DoS"] * 5)
    Xte, yte, mte = part(n_test, 5, ["BENIGN"] * (n_test - 5) + ["PortScan"] * 5)
    enc = LabelEncoder().fit(ytr)
    names = ["f0", "f1", "f2", "f3"]
    lookup = {c: i for i, c in enumerate(enc.classes_)}
    code = lambda y: np.array([lookup.get(v, -1) for v in y], dtype=np.int64)   # noqa: E731
    b = SplitBundle(
        X_train=Xtr, X_val=Xva, X_test=Xte,
        y_train=code(ytr), y_val=code(yva), y_test=code(yte),
        y_train_str=ytr, y_val_str=yva, y_test_str=yte,
        encoder=enc, scaler=StandardScaler().fit(Xtr),
        cleaner=FrameCleaner().fit(pd.DataFrame(rng.normal(size=(8, 4)), columns=names)),
        feature_names=names, protocol="crossday", known_classes=list(enc.classes_),
        meta_train=mtr, meta_val=mva, meta_test=mte, info={"protocol": "crossday"},
    )
    b.validate()
    return b


def test_sanitise_removes_only_benign_labelled_flagged_flows_from_train_and_val():
    from study import sanitise_bundle

    b = _bundle()
    k_tr, k_va, k_te = flow_keys(b.meta_train), flow_keys(b.meta_val), flow_keys(b.meta_test)
    flagged = np.r_[k_tr[[3, 4]],        # BENIGN in train     -> removed
                    k_tr[[55]],          # DoS in train        -> kept: it is an attack
                    k_va[[7]],           # BENIGN in val       -> removed
                    k_va[[38]],          # DoS in val          -> kept
                    k_te[[0, 1, 29]]]    # test                -> never touched
    out = sanitise_bundle(b, flagged, log=lambda *_: None)

    assert len(out.X_train) == 58 and len(out.X_val) == 39 and len(out.X_test) == 30
    assert out.X_test is b.X_test and out.meta_test is b.meta_test
    assert not np.isin(k_tr[[3, 4]], flow_keys(out.meta_train)).any()
    assert np.isin(k_tr[[55]], flow_keys(out.meta_train)).all()
    assert not np.isin(k_va[[7]], flow_keys(out.meta_val)).any()
    assert (out.y_train_str == "DoS").sum() == 10 and (out.y_val_str == "DoS").sum() == 5
    # features, labels and side-table are still the same rows
    assert np.allclose(out.X_train[:, 0] * 100.0, out.meta_train[META_ROW].to_numpy(), atol=1e-3)
    assert np.allclose(out.X_val[:, 0] * 100.0, out.meta_val[META_ROW].to_numpy(), atol=1e-3)
    assert out.info["sanitise_removed_train"] == 2 and out.info["sanitise_removed_val"] == 1
    assert "sanitised" not in b.info                       # the input is not edited in place
    out.validate()


def test_sanitise_with_nothing_flagged_changes_nothing():
    from study import sanitise_bundle

    b = _bundle()
    out = sanitise_bundle(b, np.empty(0, dtype=np.int64), log=lambda *_: None)
    assert np.array_equal(out.X_train, b.X_train) and np.array_equal(out.y_val, b.y_val)


def _three_day_bundle(seed=0):
    """Train = captures 0, 1, 2 (Monday, Tuesday, Wednesday); benign is one
    blob, the attack another, and each day's benign blob is shifted a little."""
    rng = np.random.default_rng(seed)
    X, y, cap = [], [], []
    for capture, shift, n_attack in ((0, 0.0, 0), (1, 0.3, 60), (2, -0.3, 60)):
        X.append(rng.normal(shift, 1.0, size=(300, 4)))
        y += ["BENIGN"] * 300
        cap += [capture] * 300
        if n_attack:
            X.append(rng.normal(4.0, 1.0, size=(n_attack, 4)))
            y += ["DoS"] * n_attack
            cap += [capture] * n_attack
    X = np.vstack(X).astype(np.float32)
    y = np.asarray(y, dtype=object)
    b = _bundle()
    enc = LabelEncoder().fit(y)
    meta = pd.DataFrame({META_CAPTURE: np.asarray(cap, dtype=np.int16),
                         META_ROW: np.arange(len(y), dtype=np.int32),
                         META_TS: T0 + np.arange(len(y), dtype=np.float64)})
    return dataclasses.replace(
        b, X_train=X, y_train=enc.transform(y).astype(np.int64), y_train_str=y,
        meta_train=meta, encoder=enc, scaler=StandardScaler().fit(X),
        X_val=X[::7].copy(), y_val=enc.transform(y[::7]).astype(np.int64), y_val_str=y[::7],
        meta_val=meta.iloc[::7].reset_index(drop=True),
    )


def test_leave_one_day_out_scores_every_benign_flow_once_with_a_model_that_never_saw_its_day():
    from study import SCORE_RULES, lodo_benign_scores

    b = _three_day_bundle()
    lines = []
    out = lodo_benign_scores(b, "dt", seed=0, log=lines.append)
    assert set(out) == set(SCORE_RULES)                   # both score rules, same flows
    for s in out.values():
        assert s.shape == (900,)                          # 3 days x 300 benign flows
        assert np.isfinite(s).all() and (s >= 0).all() and (s <= 1).all()
        assert np.median(s) < 0.5                         # benign flows mostly look benign
    assert sum("held out" in ln for ln in lines) == 3


def test_leave_one_day_out_fits_on_unique_rows_and_scores_the_day_as_recorded():
    """The fold model is fitted on de-duplicated flows, like a deployed model.
    The held-out day is scored as captured: a benign flow recorded 50 times is
    50 flows the threshold will meet, not one."""
    from study import SCORE_RULES, lodo_benign_scores

    b = _three_day_bundle()
    mon = np.flatnonzero(b.meta_train[META_CAPTURE].to_numpy() == 0)[:1]
    tue = np.flatnonzero((b.meta_train[META_CAPTURE].to_numpy() == 1)
                         & (b.y_train_str == "BENIGN"))[:1]
    extra = np.r_[np.repeat(mon, 50), np.repeat(tue, 50)]             # 50 copies each
    b = dataclasses.replace(
        b, X_train=np.vstack([b.X_train, b.X_train[extra]]),
        y_train=np.r_[b.y_train, b.y_train[extra]],
        y_train_str=np.r_[b.y_train_str, b.y_train_str[extra]],
        meta_train=pd.concat([b.meta_train, b.meta_train.iloc[extra]], ignore_index=True))
    lines = []
    out = lodo_benign_scores(b, "dt", seed=0, log=lines.append)
    assert out[SCORE_RULES[0]].shape == (1000,)           # 900 + the 100 recorded copies
    # holding out Wednesday: Monday's and Tuesday's copies collapse in the fit set
    wed = next(ln for ln in lines if "Wednesday" in ln)
    assert "fitted on 660 de-duplicated flows" in wed     # 300 + 360, not 760


def test_leave_one_day_out_skips_a_day_whose_complement_has_no_attack():
    """Hold out the only days with attacks and nothing is left to learn from:
    that fold is skipped and said so, not fitted on one class."""
    from study import lodo_benign_scores

    b = _three_day_bundle()
    keep = ~((b.y_train_str == "DoS") & (b.meta_train[META_CAPTURE].to_numpy() == 2))
    b = dataclasses.replace(b, X_train=b.X_train[keep], y_train=b.y_train[keep],
                            y_train_str=b.y_train_str[keep],
                            meta_train=b.meta_train[keep].reset_index(drop=True))
    lines = []
    s = lodo_benign_scores(b, "dt", seed=0, log=lines.append)["max attack probability"]
    assert s.shape == (600,)                              # Tuesday's fold was skipped
    assert any("skip Tuesday" in ln for ln in lines)


def test_leave_one_day_out_needs_capture_ids():
    from study import lodo_benign_scores

    b = dataclasses.replace(_bundle(), meta_train=None)
    with pytest.raises(RuntimeError, match="capture ids"):
        lodo_benign_scores(b, "dt", seed=0, log=lambda *_: None)



# =====================================================================
# RECALIBRATION, AND WHAT IT CAN AND CANNOT CHANGE
# =====================================================================


def test_a_strictly_increasing_map_of_the_score_fires_on_exactly_the_same_flows():
    """The part of the argument that is a theorem: re-map the scalar score with
    a strictly increasing function, refit the threshold, and nothing moves."""
    rng = np.random.default_rng(0)
    val, test = rng.random(20_000), rng.random(50_000) ** 0.7
    for g in (np.log1p, lambda v: v ** 3, lambda v: 1 / (1 + np.exp(-12 * (v - 0.5)))):
        for b in (0.001, 0.01):
            assert np.array_equal(fired_at_budget(test, val, b), fired_at_budget(g(test), g(val), b))


def test_a_map_with_plateaus_can_only_stop_flows_firing_never_start_them():
    """Isotonic regression is increasing but not strictly: it has flat steps.
    Flows that land on the threshold's step stop firing.  So the fired set can
    shrink -- false alarms AND detections together -- and can never grow."""
    rng = np.random.default_rng(1)
    val, test = rng.random(20_000), rng.random(50_000)
    step = lambda v: np.floor(v * 40) / 40                              # noqa: E731
    before = fired_at_budget(test, val, 0.01)
    after = fired_at_budget(step(test), step(val), 0.01)
    assert not (after & ~before).any()                 # nothing new fires
    assert (before & ~after).any()                     # some flows stopped firing
    assert after.mean() < before.mean()


def test_per_class_recalibration_is_not_a_monotone_map_of_the_score():
    """What ``CalibratedClassifierCV`` does -- one map per class, then
    renormalise -- CAN reorder flows by their largest attack probability.  So
    the theorem above says nothing about it, and E6 measures it instead."""
    from layers import isotonic_recalibration, max_attack_probability

    rng = np.random.default_rng(2)
    n, k = 30_000, 4                                   # class 0 = benign, three attack classes
    y = rng.choice(k, size=n, p=[0.7, 0.1, 0.1, 0.1])
    logits = rng.normal(size=(n, k)) + 2.5 * np.eye(k)[y] * np.array([1.0, 0.4, 1.5, 0.8])
    P = np.exp(logits) / np.exp(logits).sum(axis=1, keepdims=True)
    apply = isotonic_recalibration(P[:15_000], y[:15_000])
    Q = apply(P[15_000:])
    assert np.allclose(Q.sum(axis=1), 1.0) and (Q >= 0).all()
    raw, cal = max_attack_probability(P[15_000:], 0), max_attack_probability(Q, 0)
    # a monotone map would preserve every pairwise order; this one does not
    i, j = np.argsort(raw)[:-1], np.argsort(raw)[1:]
    assert (cal[i] > cal[j] + 1e-12).any()
    # and so the set of flows above a budget threshold is not the same set
    ref_raw, ref_cal = raw[y[15_000:] == 0], cal[y[15_000:] == 0]
    assert not np.array_equal(fired_at_budget(raw, ref_raw, 0.01),
                              fired_at_budget(cal, ref_cal, 0.01))


def test_recalibration_handles_a_class_missing_from_the_fitting_sample():
    from layers import isotonic_recalibration

    P = np.array([[0.7, 0.2, 0.1], [0.2, 0.7, 0.1], [0.6, 0.3, 0.1], [0.1, 0.8, 0.1]])
    y = np.array([0, 1, 0, 1])                         # class 2 never occurs
    Q = isotonic_recalibration(P, y)(P)
    assert np.allclose(Q.sum(axis=1), 1.0) and np.allclose(Q[:, 2], 0.0)


def test_report_takes_the_newest_run_by_time_not_by_commit_hash(tmp_path, monkeypatch):
    """Run folders are named <sha>-<date>-<time>-<tag>.  Sorted by name, a run
    from commit f00d123 beats a later one from commit 0a1b2c3."""
    import json

    import study

    for name in ("f00d123-20261005-090000-study-e1", "0a1b2c3-20261009-090000-study-e1"):
        d = tmp_path / name
        d.mkdir()
        (d / "results.json").write_text(json.dumps({"experiment": "e1", "quick": False}))
    monkeypatch.setattr(study, "RUNS_DIR", tmp_path)
    assert study.latest_runs()["e1"].name.startswith("0a1b2c3-20261009")


def _fake_run(root, name, experiment, quick, files, **args):
    import json

    d = root / name
    d.mkdir(parents=True)
    (d / "results.json").write_text(json.dumps(
        {"experiment": experiment, "quick": quick, "seconds": 1.0,
         "environment": {"git_sha": name.split("-")[0]}}))
    # the command line as `study.py` records it: the defaults, plus what was changed
    import study
    saved = {k: v for k, v in vars(study.build_parser().parse_args([experiment])).items()
             if k != "func"}
    saved.update(quick=quick, **args)
    (d / "config.json").write_text(json.dumps({"experiment": experiment, "args": saved}))
    for fname, text in files.items():
        (d / fname).write_text(text)
    return d


def test_report_rebuilds_the_paper_folders_and_keeps_quick_runs_out_of_them(tmp_path, monkeypatch):
    """Two ways a table that is not a result could end up among the results,
    both closed here.  (1) ``report`` used to copy on top of what was there,
    so a table of an older run stayed in ``paper/tables`` with no entry in
    SOURCES.md.  (2) ``all --quick`` used to write its smoke-test tables into
    the same folder as the real ones."""
    import argparse

    import study

    runs, paper = tmp_path / "runs", tmp_path / "paper"
    _fake_run(runs, "aaaaaaa-20261005-090000-study-e1", "e1", False,
              {"e1_detection.csv": "class,budget\nDDoS,0.01\n", "e1_protocol_effect.png": "png"})
    _fake_run(runs, "aaaaaaa-20261006-090000-study-e2-quick", "e2", True,
              {"e2_seen_unseen.csv": "class,budget\nDDoS,0.01\n"})
    monkeypatch.setattr(study, "RUNS_DIR", runs)
    monkeypatch.setattr(study, "PAPER_DIR", paper)

    study.run_report(argparse.Namespace(allow_quick=False))
    assert {f.name for f in (paper / "tables").iterdir()} == {"e1_detection.csv", "e1_detection.md"}
    assert {f.name for f in (paper / "figures").iterdir()} == {"e1_protocol_effect.png"}
    src = (paper / "SOURCES.md").read_text()
    assert "| `tables/e1_detection.csv` | `runs/aaaaaaa-20261005-090000-study-e1` |" in src
    assert "study-e2-quick" not in src and "| e2 | *not run* |" in src
    assert not (paper / "quick").exists()

    # A person adds files of their own -- one of them named like a study table.
    (paper / "tables" / "my_notes.txt").write_text("written by a person\n")
    (paper / "tables" / "e1_detection_for_slides.xlsx").write_text("hand-made\n")
    # A newer run of the same experiment writes a DIFFERENT set of tables.
    _fake_run(runs, "bbbbbbb-20261007-090000-study-e1", "e1", False,
              {"e1_summary.csv": "protocol,model\ncrossday,xgb\n"})
    study.run_report(argparse.Namespace(allow_quick=False))
    tables = {f.name for f in (paper / "tables").iterdir()}
    # the older report's tables are gone, the person's files are untouched
    assert tables == {"e1_summary.csv", "e1_summary.md", "my_notes.txt",
                      "e1_detection_for_slides.xlsx"}
    assert list((paper / "figures").iterdir()) == []
    src = (paper / "SOURCES.md").read_text()
    assert "e1_detection.csv" not in src and "| `tables/e1_summary.csv` |" in src
    # ... and the report says they are there and are not its own, so the folder
    # never holds a file that SOURCES.md does not explain
    assert "## Other files in `tables/` and `figures/`" in src
    assert "- `tables/my_notes.txt`" in src and "- `tables/e1_detection_for_slides.xlsx`" in src

    # the quick report goes to its own folder and leaves the results alone
    study.run_report(argparse.Namespace(allow_quick=True))
    assert {f.name for f in (paper / "tables").iterdir()} == tables
    quick = paper / "quick"
    assert (quick / "tables" / "e2_seen_unseen.csv").exists()
    assert (quick / "tables" / "e1_summary.csv").exists()         # newest of each, quick or not
    qsrc = (quick / "SOURCES.md").read_text()
    assert qsrc.startswith("# QUICK RUNS") and "study-e2-quick" in qsrc and "| YES |" in qsrc
    assert (paper / "SOURCES.md").read_text() == src              # untouched


def test_report_says_which_runs_did_not_follow_the_plan(tmp_path, monkeypatch, capsys):
    """Only ``--quick`` marks a run as "not a result".  A run started with
    ``--seeds 1 --models dt`` is collected like any other, so the report must
    say what was changed: the command line's defaults are the plan."""
    import argparse

    import study

    runs, paper = tmp_path / "runs", tmp_path / "paper"
    _fake_run(runs, "aaaaaaa-20261005-090000-study-e1", "e1", False, {"e1_summary.csv": "a\n1\n"})
    _fake_run(runs, "aaaaaaa-20261005-100000-study-e5", "e5", False, {"e5_summary.csv": "a\n1\n"},
              seeds=1, models=["dt"], quiet=True, device="cpu")
    monkeypatch.setattr(study, "RUNS_DIR", runs)
    monkeypatch.setattr(study, "PAPER_DIR", paper)

    study.run_report(argparse.Namespace(allow_quick=False))
    src = (paper / "SOURCES.md").read_text()
    e1 = next(line for line in src.splitlines() if line.startswith("| e1 |"))
    e5 = next(line for line in src.splitlines() if line.startswith("| e5 |"))
    assert e1.rstrip().endswith("| none |")
    assert "`seeds=1`" in e5 and "`classifier=dt`" in e5
    assert "quiet" not in e5 and "device" not in e5               # these change no number in the plan
    assert "**Not run as planned: e5.**" in src and "Deviations" in src
    assert "NOT RUN AS PLANNED: e5" in capsys.readouterr().out

    # a run whose command line was not recorded cannot be shown to follow the plan
    d = _fake_run(runs, "aaaaaaa-20261005-110000-study-e4", "e4", False, {"e4_alerts.csv": "a\n1\n"})
    (d / "config.json").unlink()
    study.run_report(argparse.Namespace(allow_quick=False))
    src = (paper / "SOURCES.md").read_text()
    e4 = next(line for line in src.splitlines() if line.startswith("| e4 |"))
    assert "`settings=unknown`" in e4 and "**Not run as planned: e4, e5.**" in src


def _defaults(study, experiment, **changed):
    saved = {k: v for k, v in vars(study.build_parser().parse_args([experiment])).items()
             if k != "func"}
    saved.update(changed)
    return saved


def test_a_run_is_off_plan_only_for_a_setting_its_experiment_reads(tmp_path):
    """The check compares what a run actually did with what the plan fixes --
    not the command line as text.  Giving a default explicitly changes
    nothing; an argument the experiment never reads changes nothing; and the
    arguments that do matter are caught under the name of what they set."""
    import study

    same = study.settings_changed
    for exp in ("doctor", "e1", "e2", "e3", "e4", "e5", "e6"):
        assert same(exp, _defaults(study, exp)) == {}, exp
        assert same(exp, _defaults(study, exp, quiet=True, device="cpu")) == {}, exp
        assert same(exp, {}) == {"settings": "unknown"} and same(exp, None) == {"settings": "unknown"}

    # a default, spelled out
    assert same("e3", _defaults(study, "e3", epochs=30, n_boot=1000,
                                budgets=[0.001, 0.005, 0.01])) == {}
    assert same("e1", _defaults(study, "e1", models=["xgb", "rf"],
                                protocols=["random", "blocked", "crossday"])) == {}
    assert same("e5", _defaults(study, "e5", novelty="autoencoder", models=["xgb"])) == {}
    assert same("e4", _defaults(study, "e4", window=60.0, port_threshold=100, host_threshold=50)) == {}

    # an argument the experiment does not read
    assert same("e4", _defaults(study, "e4", seeds=1, models=["dt"], epochs=3)) == {}
    assert same("e2", _defaults(study, "e2", seeds=1)) == {}      # E2 uses one seed whatever it says
    assert same("e6", _defaults(study, "e6", seeds=1, epochs=3)) == {}
    assert same("doctor", _defaults(study, "doctor", seeds=1, models=["xgb"])) == {}
    assert same("e1", _defaults(study, "e1", quick_rows=9)) == {}  # counts only with --quick

    # what does change a result
    assert same("e1", _defaults(study, "e1", seeds=1)) == {"seeds": 1}
    assert same("e1", _defaults(study, "e1", models=["xgb"])) == {"models": ["xgb"]}
    assert same("e2", _defaults(study, "e2", models=["rf"])) == {"model": "rf"}
    assert same("e2", _defaults(study, "e2", classes=["DDoS"])) == {"classes": ["DDoS"]}
    assert same("e3", _defaults(study, "e3", epochs=5)) == {"epochs": 5}
    assert same("e3", _defaults(study, "e3", no_reference=True)) == {"reference": None}
    assert same("e3", _defaults(study, "e3", detectors=["pca"])) == {"detectors": ["pca"]}
    assert same("e4", _defaults(study, "e4", port_threshold=20)) == {"port_threshold": 20}
    assert same("e5", _defaults(study, "e5", seeds=1, models=["dt"])) == {"seeds": 1,
                                                                          "classifier": "dt"}
    assert same("e5", _defaults(study, "e5", no_sanitise=True)) == {"ablation": False}
    assert same("e5", _defaults(study, "e5", val_mode="tail")) == {"val_mode": "tail"}
    assert same("e6", _defaults(study, "e6", models=["xgb"])) == {"models": ["xgb"]}
    assert same("e6", _defaults(study, "e6", budgets=[0.05])) == {"budgets": [0.05]}
    assert same("e6", _defaults(study, "e6", n_boot=50)) == {"n_boot": 50}
    # `doctor --data elsewhere` describes another folder than the experiments read
    other = same("doctor", _defaults(study, "doctor", data=str(tmp_path)))
    assert list(other) == ["data"] and other["data"] == str(tmp_path.resolve())
    # a quick run differs in everything quick mode touches
    q = same("e3", _defaults(study, "e3", quick=True))
    assert q["quick"] is True and q["n_boot"] == 200 and q["epochs"] == 4 and "quick_rows" in q


def test_report_keeps_the_tables_of_an_experiment_whose_run_folder_is_gone(tmp_path, monkeypatch,
                                                                         capsys):
    """Run folders are large; people delete them.  The tables in paper/ may
    then be the only copy.  A report with nothing to copy from must leave
    them where they are -- and say that they were kept, not refreshed."""
    import argparse
    import json
    import shutil

    import study

    runs, paper = tmp_path / "runs", tmp_path / "paper"
    e1 = _fake_run(runs, "aaaaaaa-20261005-090000-study-e1", "e1", False,
                   {"e1_summary.csv": "a\n1\n"})
    e5 = _fake_run(runs, "aaaaaaa-20261005-100000-study-e5", "e5", False,
                   {"e5_summary.csv": "a\n1\n", "e5_layer_attribution.png": "png"})
    monkeypatch.setattr(study, "RUNS_DIR", runs)
    monkeypatch.setattr(study, "PAPER_DIR", paper)
    study.run_report(argparse.Namespace(allow_quick=False))
    e5_files = {"tables/e5_summary.csv", "tables/e5_summary.md", "figures/e5_layer_attribution.png"}
    assert all((paper / f).exists() for f in e5_files)

    shutil.rmtree(e5)                                   # freed some disk space
    study.run_report(argparse.Namespace(allow_quick=False))
    assert all((paper / f).exists() for f in e5_files)
    src = (paper / "SOURCES.md").read_text()
    assert f"| e5 | *no run folder found: 3 file(s) kept from `runs/{e5.name}`* |" in src
    assert "**Kept from an earlier report: e5.**" in src
    assert f"| `tables/e5_summary.csv` | `runs/{e5.name}` (kept) |" in src
    assert "KEPT, NOT REFRESHED: e5" in capsys.readouterr().out
    listed = json.loads((paper / study.REPORT_MANIFEST).read_text())
    assert e5_files <= set(listed) and listed["tables/e5_summary.csv"]["kept"] is True

    # every run folder gone: nothing at all is removed
    shutil.rmtree(e1)
    study.run_report(argparse.Namespace(allow_quick=False))
    assert all((paper / f).exists() for f in e5_files | {"tables/e1_summary.csv"})

    # the experiment is run again: now its old files are replaced
    _fake_run(runs, "bbbbbbb-20261009-090000-study-e5", "e5", False, {"e5_fusion.csv": "a\n1\n"})
    study.run_report(argparse.Namespace(allow_quick=False))
    assert not any((paper / f).exists() for f in e5_files)
    assert (paper / "tables" / "e5_fusion.csv").exists()
    assert (paper / "tables" / "e1_summary.csv").exists()          # e1 is still only kept


def test_report_prefers_the_run_that_follows_the_plan_over_a_newer_one_that_does_not(
        tmp_path, monkeypatch):
    """A later look at something with other settings must not replace the
    planned run's tables just by being newer."""
    import argparse

    import study

    runs, paper = tmp_path / "runs", tmp_path / "paper"
    planned = _fake_run(runs, "aaaaaaa-20261005-090000-study-e5", "e5", False,
                        {"e5_summary.csv": "which\nplanned\n"})
    later = _fake_run(runs, "aaaaaaa-20261008-090000-study-e5", "e5", False,
                      {"e5_summary.csv": "which\nseeds1\n"}, seeds=1)
    only = _fake_run(runs, "aaaaaaa-20261008-100000-study-e1", "e1", False,
                     {"e1_summary.csv": "a\n1\n"}, models=["xgb"])
    monkeypatch.setattr(study, "RUNS_DIR", runs)
    monkeypatch.setattr(study, "PAPER_DIR", paper)

    picks = study.runs_for_report()
    assert picks["e5"][0] == planned and picks["e5"][1] == {} and picks["e5"][2] == [later.name]
    # no run of e1 follows the plan: the newest is taken, and named as off-plan
    assert picks["e1"][0] == only and picks["e1"][1] == {"models": ["xgb"]}
    # the smoke-test report simply takes the newest
    assert study.runs_for_report(allow_quick=True)["e5"][0] == later

    study.run_report(argparse.Namespace(allow_quick=False))
    assert "planned" in (paper / "tables" / "e5_summary.csv").read_text()
    src = (paper / "SOURCES.md").read_text()
    assert "**Newer runs that were not used**" in src and later.name in src
    assert "**Not run as planned: e1.**" in src


def test_report_never_deletes_a_file_it_did_not_write(tmp_path):
    import json

    import study

    dest = tmp_path / "paper"
    (dest / "tables").mkdir(parents=True)
    (dest / "figures").mkdir()
    (dest / "tables" / "e1_old.csv").write_text("x")
    (dest / "tables" / "e1_mine.csv").write_text("x")
    (tmp_path / "outside.txt").write_text("x")
    assert study._load_report_list(dest) == {}                     # no list: nothing is its own
    hostile = ["tables/e1_old.csv", "tables/gone_already.csv", "../outside.txt", "SOURCES.md",
               str(tmp_path / "outside.txt"), "tables/../../outside.txt"]
    assert study._remove_report_files(dest, hostile) == (1, [])
    assert not (dest / "tables" / "e1_old.csv").exists()
    assert (dest / "tables" / "e1_mine.csv").exists() and (tmp_path / "outside.txt").exists()

    # the list of the first version (paths only) is still understood
    (dest / study.REPORT_MANIFEST).write_text(json.dumps(["tables/e3_auc.csv", "tables/doctor.json", 7]))
    old = study._load_report_list(dest)
    assert old == {"tables/e3_auc.csv": {"experiment": "e3", "run": "?"},
                   "tables/doctor.json": {"experiment": "doctor", "run": "?"}}
    (dest / study.REPORT_MANIFEST).write_text("not json")
    assert study._load_report_list(dest) == {}
