"""
Tests for the decision rules of ``paper/PROTOCOL.md`` -- TORCH-FREE.

``src/rules.py`` turns the protocol's three yes/no rules (RQ3, RQ5, RQ6) into
code.  Each test here builds the experiment's table by hand, with numbers
chosen so that the answer is obvious, and checks that the rule gives it --
including the cases the rule exists to refuse: an interval that excludes zero
only because the class sits in one time block, a gain that is paid for with
more false alarms, a deep model that beats two baselines and loses to a third,
and a table with a row missing.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "src"))

from config import BENIGN_LABEL  # noqa: E402
from rules import (  # noqa: E402
    NOT_EVALUATED, false_alarm_check, reading_rule, rq3_supported, rq3_verdict, rq5_supported,
    rq5_verdict, rq6_supported, rq6_verdict,
)
from stats import (  # noqa: E402
    ANY_ATTACK, MIN_BLOCKS_FOR_INTERVAL, difference_supported, interval_counts,
)

B = 0.01                       # the primary budget
RULES = ("max attack probability", "1 - P(benign)")
CLASSICAL = ("iforest", "pca", "mahalanobis")
LAYERS = ("classifier", "novelty", "behaviour")


# =====================================================================
# THE READING RULE FOR ONE INTERVAL
# =====================================================================


def test_an_interval_counts_only_from_twenty_blocks_up():
    assert MIN_BLOCKS_FOR_INTERVAL == 20
    assert not interval_counts(0) and not interval_counts(1) and not interval_counts(19)
    assert interval_counts(20) and interval_counts(500)
    assert not interval_counts(None) and not interval_counts("many")


def test_a_difference_needs_an_interval_off_zero_and_enough_blocks():
    assert difference_supported(0.01, 0.05, 40)          # above zero
    assert difference_supported(-0.05, -0.01, 40)        # below zero
    assert not difference_supported(-0.01, 0.05, 40)     # straddles zero
    assert not difference_supported(0.0, 0.05, 40)       # touches zero
    # the case the rule exists for: one block, zero-width interval, "excludes zero"
    assert not difference_supported(0.03, 0.03, 1)
    assert not difference_supported(0.01, 0.05, 19)
    assert not difference_supported(np.nan, np.nan, 40)  # no bootstrap was run


def test_the_reading_rule_is_printed_with_its_number():
    assert str(MIN_BLOCKS_FOR_INTERVAL) in reading_rule() and "differs" in reading_rule()


# =====================================================================
# "NOT BOUGHT WITH FALSE ALARMS"
# =====================================================================


def _benign(rate, diff, lo, hi):
    return pd.Series({"diff": diff, "diff_lo": lo, "diff_hi": hi, "rate_system": rate})


def test_false_alarms_are_acceptable_within_the_budget_or_when_an_excess_is_not_shown():
    # within the 1% budget, although it fires more than a quiet comparator: acceptable
    c = false_alarm_check(_benign(0.008, +0.007, +0.006, +0.008), "rate_system", B)
    assert c["within_budget"] and not c["excess_not_shown"] and c["false_alarms_ok"]
    assert not c["no_excess"]
    # over the budget (the day drifted), an excess is not shown: acceptable
    c = false_alarm_check(_benign(0.044, +0.001, -0.002, +0.004), "rate_system", B)
    assert not c["within_budget"] and c["excess_not_shown"] and c["false_alarms_ok"]
    # over the budget AND clearly worse than the comparator: refused
    c = false_alarm_check(_benign(0.044, +0.030, +0.025, +0.035), "rate_system", B)
    assert not c["within_budget"] and not c["excess_not_shown"] and not c["false_alarms_ok"]
    # exactly on the budget counts as within it
    assert false_alarm_check(_benign(0.01, 0.005, 0.004, 0.006), "rate_system", B)["within_budget"]
    # no interval and over budget: nothing shows the gain was not bought, so no pass
    c = false_alarm_check(_benign(0.044, 0.0, np.nan, np.nan), "rate_system", B)
    assert not c["false_alarms_ok"]


def test_the_rule_is_lenient_about_a_noisy_excess_and_the_strict_column_is_not():
    """The asymmetry the rule's own text admits: detection must be shown to be
    higher, false alarms only not shown to be higher.  4.4% on a 1% budget and
    three points more than the comparator -- but the interval straddles zero.
    That passes the rule.  It does not pass the strict reading, which asks
    that the observed rate is not above the comparator's at all."""
    c = false_alarm_check(_benign(0.044, +0.030, -0.001, +0.065), "rate_system", B)
    assert c["excess_not_shown"] and c["false_alarms_ok"] and not c["no_excess"]
    even = false_alarm_check(_benign(0.044, 0.0, -0.002, 0.002), "rate_system", B)
    lower = false_alarm_check(_benign(0.044, -0.004, -0.006, -0.002), "rate_system", B)
    assert even["no_excess"] and lower["no_excess"]


# =====================================================================
# RQ3  deep versus classical benign-only detectors
# =====================================================================


def _pair(cls, base, diff, lo, hi, n_blocks, deep="autoencoder", protocol="crossday", budget=B,
          rate_deep=0.5):
    return {"protocol": protocol, "budget": budget, "class": cls, "deep": deep, "baseline": base,
            "diff": diff, "diff_lo": lo, "diff_hi": hi, "rate_deep": rate_deep,
            "rate_baseline": rate_deep - diff, "n": 1000, "n_blocks": n_blocks, "seed": 42}


def _quiet(bases=CLASSICAL, **kw):
    """BENIGN rows: the deep detector fires on 0.8% of benign test flows -- inside
    the 1% budget -- and no differently from each baseline."""
    return [_pair(BENIGN_LABEL, b, 0.0, -0.001, 0.001, 400, rate_deep=0.008, **kw) for b in bases]


def test_rq3_passes_only_when_the_deep_model_beats_every_baseline():
    pairs = pd.DataFrame([
        # DDoS: beats all three, on 60 blocks                       -> passes
        _pair("DDoS", "iforest", 0.10, 0.06, 0.14, 60),
        _pair("DDoS", "pca", 0.04, 0.01, 0.07, 60),
        _pair("DDoS", "mahalanobis", 0.08, 0.02, 0.15, 60),
        # PortScan: beats two, ties with the third                  -> fails
        _pair("PortScan", "iforest", 0.10, 0.06, 0.14, 80),
        _pair("PortScan", "pca", 0.00, -0.02, 0.02, 80),
        _pair("PortScan", "mahalanobis", 0.08, 0.02, 0.15, 80),
        # Bot: "beats" all three, but lives in 3 blocks             -> fails
        _pair("Bot", "iforest", 0.30, 0.30, 0.30, 3),
        _pair("Bot", "pca", 0.20, 0.20, 0.20, 3),
        _pair("Bot", "mahalanobis", 0.25, 0.25, 0.25, 3),
    ] + _quiet())
    v = rq3_verdict(pairs, B, required_baselines=CLASSICAL).set_index("class")
    assert set(v.index) == {"DDoS", "PortScan", "Bot"}          # one row per attack class
    assert v["baselines_complete"].all() and v["false_alarms_ok"].all()
    assert (v["deep_false_alarm_rate"] == 0.008).all() and v["within_budget"].all()
    assert v.loc["DDoS", "passes"] and v.loc["DDoS", "baselines"] == 3
    assert v.loc["DDoS", "passes_strict"]                       # no excess at all here
    assert v.loc["DDoS", "smallest_gain"] == 0.04 and v.loc["DDoS", "smallest_gain_lo"] == 0.01
    assert not v.loc["PortScan", "beats_every_baseline"] and not v.loc["PortScan", "passes"]
    assert v.loc["Bot", "beats_every_baseline"] and not v.loc["Bot", "interval_counts"]
    assert not v.loc["Bot", "passes"]
    assert rq3_supported(rq3_verdict(pairs, B, required_baselines=CLASSICAL)) is True
    # without DDoS no class passes
    rest = pairs[pairs["class"] != "DDoS"]
    assert rq3_supported(rq3_verdict(rest, B, required_baselines=CLASSICAL)) is False


def test_rq3_every_baseline_means_every_baseline_the_protocol_names():
    """Beating PCA alone is not beating "every classical baseline".  If a
    detector the protocol names was not run, the question was not asked."""
    pairs = pd.DataFrame([_pair("DDoS", "pca", 0.10, 0.06, 0.14, 60)] + _quiet(("pca",)))
    v = rq3_verdict(pairs, B, required_baselines=CLASSICAL)
    assert not v["baselines_complete"].any() and not v["passes"].any()
    assert rq3_supported(v) == NOT_EVALUATED
    # with no list of required baselines the caller takes responsibility for the set
    assert rq3_supported(rq3_verdict(pairs, B)) is True


def test_rq3_a_gain_bought_with_false_alarms_does_not_count():
    """The deep detector "beats" every baseline on DDoS -- while firing on 6% of
    benign test flows at a 1% budget, three points more than PCA does.  That is
    a looser threshold, not a better detector."""
    win = [_pair("DDoS", b, 0.10, 0.06, 0.14, 60) for b in ("iforest", "pca")]
    loud = [_pair(BENIGN_LABEL, "iforest", 0.00, -0.002, 0.002, 400, rate_deep=0.06),
            _pair(BENIGN_LABEL, "pca", 0.03, 0.025, 0.035, 400, rate_deep=0.06)]
    v = rq3_verdict(pd.DataFrame(win + loud), B).set_index("class")
    assert v.loc["DDoS", "beats_every_baseline"] and v.loc["DDoS", "interval_counts"]
    assert not v.loc["DDoS", "false_alarms_ok"] and not v.loc["DDoS", "passes"]

    # the same day drifted for everyone: 6% for the deep model, and no less for the baselines
    even = [_pair(BENIGN_LABEL, b, 0.00, -0.002, 0.002, 400, rate_deep=0.06)
            for b in ("iforest", "pca")]
    v = rq3_verdict(pd.DataFrame(win + even), B).set_index("class")
    assert v.loc["DDoS", "false_alarms_ok"] and v.loc["DDoS", "passes"]
    assert v.loc["DDoS", "passes_strict"]

    # a noisy excess passes the rule and fails the strict reading
    noisy = [_pair(BENIGN_LABEL, "iforest", 0.00, -0.002, 0.002, 400, rate_deep=0.06),
             _pair(BENIGN_LABEL, "pca", 0.03, -0.001, 0.065, 400, rate_deep=0.06)]
    v = rq3_verdict(pd.DataFrame(win + noisy), B).set_index("class")
    assert v.loc["DDoS", "passes"] and not v.loc["DDoS", "passes_strict"]
    assert rq3_supported(v.reset_index()) is True
    assert rq3_supported(v.reset_index(), strict=True) is False


def test_rq3_an_incomplete_table_never_passes():
    win = [_pair("DDoS", b, 0.10, 0.06, 0.14, 60) for b in ("iforest", "pca")]
    even = [_pair(BENIGN_LABEL, b, 0.00, -0.002, 0.002, 400, rate_deep=0.06)
            for b in ("iforest", "pca")]
    # no BENIGN rows at all: nothing shows where the gain came from
    v = rq3_verdict(pd.DataFrame(win), B).set_index("class")
    assert not v.loc["DDoS", "false_alarms_ok"] and not v.loc["DDoS", "passes"]
    # a BENIGN row for one baseline only
    v = rq3_verdict(pd.DataFrame(win + even[:1]), B).set_index("class")
    assert not v.loc["DDoS", "passes"]
    # two BENIGN rows for ONE baseline and none for the other: counting rows is not enough
    v = rq3_verdict(pd.DataFrame(win + [even[0], even[0]]), B).set_index("class")
    assert not v.loc["DDoS", "passes"]
    # a class compared with only one of the baselines the others were compared with
    lop = win + [_pair("Bot", "iforest", 0.10, 0.06, 0.14, 60)] + even
    v = rq3_verdict(pd.DataFrame(lop), B).set_index("class")
    assert v.loc["DDoS", "passes"] and not v.loc["Bot", "beats_every_baseline"]
    assert not v.loc["Bot", "passes"]


def test_rq3_reads_only_the_named_detector_protocol_and_budget():
    pairs = pd.DataFrame([
        _pair("DDoS", "pca", 0.10, 0.06, 0.14, 60, deep="deep_svdd"),        # not the autoencoder
        _pair("DDoS", "pca", 0.10, 0.06, 0.14, 60, protocol="blocked"),      # not cross-day
        _pair("DDoS", "pca", 0.10, 0.06, 0.14, 60, budget=0.001),            # not the primary budget
        _pair(ANY_ATTACK, "pca", 0.10, 0.06, 0.14, 60),                      # not a class
    ] + _quiet(("pca",)) + _quiet(("pca",), deep="deep_svdd"))
    assert len(rq3_verdict(pairs, B)) == 0
    assert rq3_supported(rq3_verdict(pairs, B)) == NOT_EVALUATED
    assert rq3_supported(rq3_verdict(pd.DataFrame(), B)) == NOT_EVALUATED
    assert rq3_supported(None) == NOT_EVALUATED
    assert rq3_supported(rq3_verdict(pairs, B, deep="deep_svdd")) is True


# =====================================================================
# RQ5  the full system against each single layer
# =====================================================================

OVER = 0.044        # the full system's false-alarm rate on a day that drifted: 4.4% at a 1% budget


def _gain(layer, cls, diff, lo, hi, n_blocks=200, cond="as labelled", seed=42, rule=RULES[0],
          n_layers=1, budget=B, rate_system=None):
    if rate_system is None:
        rate_system = OVER if cls == BENIGN_LABEL else 0.5
    return {"condition": cond, "seed": seed, "score_rule": rule, "budget": budget, "class": cls,
            "system": "classifier+novelty+behaviour", "compared_with": layer,
            "n_layers_compared": n_layers, "diff": diff, "diff_lo": lo, "diff_hi": hi,
            "rate_system": rate_system, "rate_compared": rate_system - diff, "n": 1000,
            "n_blocks": n_blocks}


def _good(layer, **kw):
    """More detection than ``layer``, and -- on a day where the system is over
    its budget -- no more false alarms than it."""
    return [_gain(layer, ANY_ATTACK, 0.10, 0.05, 0.15, **kw),
            _gain(layer, BENIGN_LABEL, 0.000, -0.001, 0.001, **kw)]


def test_rq5_passes_when_every_single_layer_is_beaten_at_no_extra_false_alarms():
    gain = pd.DataFrame(_good("classifier") + _good("novelty") + _good("behaviour"))
    v = rq5_verdict(gain, B)
    assert len(v) == 3 and v["complete"].all() and v["passes"].all() and v["fusion_helps"].all()
    assert v["passes_strict"].all()
    assert rq5_supported(v, "as labelled", 42, RULES[0], required_layers=LAYERS) is True


def test_rq5_fails_when_one_layer_is_not_beaten():
    rows = _good("classifier") + _good("novelty") + [
        _gain("behaviour", ANY_ATTACK, 0.01, -0.02, 0.04),          # interval holds zero
        _gain("behaviour", BENIGN_LABEL, 0.0, -0.001, 0.001)]
    v = rq5_verdict(pd.DataFrame(rows), B).set_index("single_layer")
    assert v.loc["classifier", "passes"] and not v.loc["behaviour", "more_detection"]
    assert not v["fusion_helps"].any()                    # one failure fails the system
    assert rq5_supported(v.reset_index(), "as labelled", 42, RULES[0]) is False


def test_rq5_fails_when_the_gain_is_bought_with_false_alarms():
    """Over the budget AND clearly more false alarms than the layer it is compared with."""
    rows = _good("novelty") + _good("behaviour") + [
        _gain("classifier", ANY_ATTACK, 0.10, 0.05, 0.15),
        _gain("classifier", BENIGN_LABEL, 0.004, 0.002, 0.006)]    # more false alarms, for sure
    v = rq5_verdict(pd.DataFrame(rows), B).set_index("single_layer")
    assert v.loc["classifier", "more_detection"] and not v.loc["classifier", "within_budget"]
    assert not v.loc["classifier", "excess_not_shown"] and not v.loc["classifier", "passes"]
    assert not v["fusion_helps"].any()


def test_rq5_does_not_punish_the_system_for_using_a_budget_a_quiet_layer_leaves_unused():
    """The behaviour layer's score is a count and ties heavily: at a 1% budget
    it may fire on 0.01% of benign flows.  The full system, at 0.8%, is inside
    the same budget.  It raises more false alarms than that layer and still
    spends less than it was given, so the gain counts -- and the strict column
    shows the harsher reading beside it."""
    rows = _good("classifier", rate_system=0.008) + _good("novelty", rate_system=0.008) + [
        _gain("behaviour", ANY_ATTACK, 0.40, 0.22, 0.63, rate_system=0.5),
        _gain("behaviour", BENIGN_LABEL, 0.0079, 0.006, 0.010, rate_system=0.008)]
    v = rq5_verdict(pd.DataFrame(rows), B)
    b = v.set_index("single_layer").loc["behaviour"]
    assert b["within_budget"] and not b["excess_not_shown"] and b["false_alarms_ok"]
    assert b["passes"] and not b["passes_strict"] and b["system_false_alarm_rate"] == 0.008
    assert v["fusion_helps"].all()
    assert rq5_supported(v, "as labelled", 42, RULES[0]) is True
    assert rq5_supported(v, "as labelled", 42, RULES[0], strict=True) is False


def test_rq5_a_noisy_excess_over_budget_passes_the_rule_but_not_the_strict_reading():
    rows = _good("novelty") + _good("behaviour") + [
        _gain("classifier", ANY_ATTACK, 0.10, 0.05, 0.15),
        _gain("classifier", BENIGN_LABEL, 0.030, -0.001, 0.065)]    # +3 points, interval holds 0
    v = rq5_verdict(pd.DataFrame(rows), B)
    c = v.set_index("single_layer").loc["classifier"]
    assert c["excess_not_shown"] and not c["no_excess"] and c["passes"] and not c["passes_strict"]
    assert rq5_supported(v, "as labelled", 42, RULES[0]) is True
    assert rq5_supported(v, "as labelled", 42, RULES[0], strict=True) is False


def test_rq5_does_not_count_a_gain_measured_on_a_handful_of_blocks():
    rows = _good("novelty") + _good("behaviour") + [
        _gain("classifier", ANY_ATTACK, 0.10, 0.10, 0.10, n_blocks=2),
        _gain("classifier", BENIGN_LABEL, 0.0, -0.001, 0.001)]
    v = rq5_verdict(pd.DataFrame(rows), B).set_index("single_layer")
    assert not v.loc["classifier", "more_detection"] and v.loc["classifier", "attack_n_blocks"] == 2


def test_rq5_is_answered_separately_per_condition_seed_and_score_rule():
    rows = (_good("classifier") + _good("novelty") + _good("behaviour")             # passes
            + _good("classifier", rule=RULES[1]) + _good("novelty", rule=RULES[1])
            + [_gain("behaviour", ANY_ATTACK, 0.0, -0.03, 0.03, rule=RULES[1]),     # fails
               _gain("behaviour", BENIGN_LABEL, 0.0, -0.001, 0.001, rule=RULES[1])]
            + _good("classifier", seed=43) + _good("novelty", seed=43)
            + _good("behaviour", seed=43)
            # two-layer systems and other budgets are not part of the rule
            + [_gain("classifier+novelty", ANY_ATTACK, -0.5, -0.6, -0.4, n_layers=2),
               _gain("classifier+novelty", BENIGN_LABEL, 0.5, 0.4, 0.6, n_layers=2),
               _gain("classifier", ANY_ATTACK, -0.5, -0.6, -0.4, budget=0.001),
               _gain("classifier", BENIGN_LABEL, 0.5, 0.4, 0.6, budget=0.001)])
    v = rq5_verdict(pd.DataFrame(rows), B)
    assert len(v) == 9 and set(v["single_layer"]) == set(LAYERS)
    assert rq5_supported(v, "as labelled", 42, RULES[0]) is True
    assert rq5_supported(v, "as labelled", 42, RULES[1]) is False
    assert rq5_supported(v, "as labelled", 43, RULES[0]) is True
    assert rq5_supported(v, "as labelled", 44, RULES[0]) == NOT_EVALUATED
    assert rq5_supported(v, "scan-like benign flows removed", 42, RULES[0]) == NOT_EVALUATED
    assert rq5_supported(rq5_verdict(pd.DataFrame(), B), "as labelled", 42) == NOT_EVALUATED


def test_rq5_an_incomplete_table_never_passes():
    # no false-alarm interval, and over budget
    rows = [_gain("classifier", ANY_ATTACK, 0.10, 0.05, 0.15),
            _gain("classifier", BENIGN_LABEL, 0.0, np.nan, np.nan)]
    v = rq5_verdict(pd.DataFrame(rows), B)
    assert not v["excess_not_shown"].iloc[0] and not v["passes"].iloc[0]

    # a layer whose BENIGN row is missing is a failed row, not a skipped one
    rows = _good("classifier") + _good("novelty") + [
        _gain("behaviour", ANY_ATTACK, 0.10, 0.05, 0.15)]
    v = rq5_verdict(pd.DataFrame(rows), B).set_index("single_layer")
    assert len(v) == 3 and not v.loc["behaviour", "complete"] and not v.loc["behaviour", "passes"]
    assert not v["fusion_helps"].any()
    assert rq5_supported(v.reset_index(), "as labelled", 42, RULES[0]) is False
    # ... and so is one whose any-attack row is missing
    rows = _good("classifier") + _good("novelty") + [
        _gain("behaviour", BENIGN_LABEL, 0.0, -0.001, 0.001)]
    v = rq5_verdict(pd.DataFrame(rows), B).set_index("single_layer")
    assert not v.loc["behaviour", "complete"] and not v.loc["behaviour", "passes"]


def test_rq5_the_three_layer_question_is_not_answered_by_a_two_layer_system():
    """Without source address and timestamp the behaviour layer cannot run and
    E5 fuses two layers.  That table can be read, but it does not answer RQ5."""
    two = pd.DataFrame(_good("classifier") + _good("novelty"))
    v = rq5_verdict(two, B)                           # no system named: the caller's business
    assert v["fusion_helps"].all()
    assert rq5_supported(v, "as labelled", 42, RULES[0]) is True
    assert rq5_supported(v, "as labelled", 42, RULES[0], required_layers=LAYERS) == NOT_EVALUATED
    # told which system the question is about, the TABLE says so too: a reader of
    # the CSV must not find "fusion_helps = True" for a question that was not asked
    v = rq5_verdict(two, B, required_layers=LAYERS)
    assert v["passes"].all() and not v["layers_complete"].any() and not v["fusion_helps"].any()
    assert rq5_supported(v, "as labelled", 42, RULES[0]) == NOT_EVALUATED
    # and with all three layers the column is True
    v = rq5_verdict(pd.DataFrame(_good("classifier") + _good("novelty") + _good("behaviour")),
                    B, required_layers=LAYERS)
    assert v["layers_complete"].all() and v["fusion_helps"].all()
    # complete for one seed, incomplete for another: judged per (rule, condition, seed)
    mixed = pd.DataFrame(_good("classifier") + _good("novelty") + _good("behaviour")
                         + _good("classifier", seed=43) + _good("novelty", seed=43))
    v = rq5_verdict(mixed, B, required_layers=LAYERS)
    assert v[v["seed"] == 42]["fusion_helps"].all() and not v[v["seed"] == 43]["fusion_helps"].any()


# =====================================================================
# RQ6  does recalibration repair the threshold
# =====================================================================


def _cal(cls, diff, lo, hi, unc, model="xgb", rule=RULES[0], n_blocks=300, budget=B):
    return {"model": model, "score_rule": rule, "budget": budget, "class": cls, "diff": diff,
            "diff_lo": lo, "diff_hi": hi, "rate_recalibrated": unc + diff,
            "rate_uncalibrated": unc, "n": 1000, "n_blocks": n_blocks}


def test_rq6_a_repair_needs_something_broken_a_lower_rate_and_a_rate_within_budget():
    cal = pd.DataFrame([
        # 4.4% -> 0.8% at a 1% budget: over before, lower, within after     -> repairs
        _cal(BENIGN_LABEL, -0.036, -0.040, -0.032, 0.044),
        _cal(ANY_ATTACK, -0.020, -0.030, -0.010, 0.285),
        # 4.4% -> 1.4%: lower, still over                                   -> improves
        _cal(BENIGN_LABEL, -0.030, -0.035, -0.025, 0.044, model="rf"),
        _cal(ANY_ATTACK, 0.0, -0.01, 0.01, 0.285, model="rf"),
        # 4.4% -> 4.3%, interval holds zero                                 -> neither
        _cal(BENIGN_LABEL, -0.001, -0.004, 0.002, 0.044, model="lgbm"),
        # 0.95% -> 0.56%: lower, but nothing was over budget                -> neither
        _cal(BENIGN_LABEL, -0.0039, -0.005, -0.003, 0.0095, model="dt"),
        # worse
        _cal(BENIGN_LABEL, 0.010, 0.005, 0.015, 0.044, rule=RULES[1]),
    ])
    v = rq6_verdict(cal, B).set_index(["model", "score_rule"])
    row = v.loc[("xgb", RULES[0])]
    assert row["over_budget_before"] and row["lowers_false_alarms"] and row["within_budget_after"]
    assert row["repairs"] and not row["improves"]
    assert abs(row["detection_diff"] + 0.020) < 1e-12      # and it cost two points of detection
    rf = v.loc[("rf", RULES[0])]
    assert rf["lowers_false_alarms"] and not rf["within_budget_after"]
    assert rf["improves"] and not rf["repairs"] and abs(rf["rate_recalibrated"] - 0.014) < 1e-12
    lg = v.loc[("lgbm", RULES[0])]
    assert not lg["lowers_false_alarms"] and not lg["repairs"] and not lg["improves"]
    dt = v.loc[("dt", RULES[0])]
    # a stricter threshold under another name: lower, but there was nothing to repair
    assert dt["lowers_false_alarms"] and not dt["over_budget_before"]
    assert not dt["repairs"] and not dt["improves"]
    assert not v.loc[("xgb", RULES[1]), "lowers_false_alarms"]

    flat = v.reset_index()
    assert rq6_supported(flat, "xgb", RULES[0]) is True
    assert rq6_supported(flat, "rf", RULES[0]) is False
    assert rq6_supported(flat, "rf", RULES[0], column="improves") is True
    assert rq6_supported(flat, "dt", RULES[0]) is False
    assert rq6_supported(flat, "dt", RULES[0], column="lowers_false_alarms") is True
    assert rq6_supported(flat, "mlp", RULES[0]) == NOT_EVALUATED
    assert rq6_supported(rq6_verdict(pd.DataFrame(), B), "xgb", RULES[0]) == NOT_EVALUATED


def test_rq6_reads_the_primary_budget_only_and_needs_enough_blocks():
    cal = pd.DataFrame([_cal(BENIGN_LABEL, -0.03, -0.035, -0.025, 0.044, budget=0.001)])
    assert len(rq6_verdict(cal, B)) == 0
    few = pd.DataFrame([_cal(BENIGN_LABEL, -0.036, -0.036, -0.036, 0.044, n_blocks=3)])
    v = rq6_verdict(few, B)
    assert not v["lowers_false_alarms"].iloc[0] and not v["repairs"].iloc[0]
