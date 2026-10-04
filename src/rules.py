"""
The decision rules of ``paper/PROTOCOL.md`` section 5, as code.

WHY THIS FILE EXISTS
--------------------
A rule that is written down before the runs and then applied by hand after
them still leaves room to read a table generously.  Three of the six research
questions have a yes/no rule (RQ3, RQ5, RQ6).  Each is applied here by a
function that takes the experiment's own table and returns one row per case
with every condition of the rule as its own column, so a reader can see which
condition failed.

Nothing here looks at a score, a model or a label.  The functions read the
paired-difference tables the experiments write (``e3_deep_vs_classical``,
``e5_gain``, ``e6_calibration_effect``) and nothing else.

RQ1, RQ2 and RQ4 have no yes/no rule: they are answered by the tables
themselves, and ``PROTOCOL.md`` says how to read them.

AN INCOMPLETE TABLE NEVER PASSES.  A comparison whose rows are missing (a
baseline that was not run, a layer with no false-alarm row) is reported as
``not evaluated`` or as a failed row -- never skipped, because skipping a
condition is the same as granting it.

THE TWO CONDITIONS EVERY "A DETECTS MORE THAN B" HAS TO MEET
-----------------------------------------------------------
1. *More detection.*  The paired interval of "A minus B" lies above zero, on
   a class that spans at least ``MIN_BLOCKS_FOR_INTERVAL`` blocks.

2. *Not bought with false alarms.*  Both detectors are fitted to the same
   budget on validation, but on the test day their false-alarm rates differ,
   and a detector that fires more often detects more for free.  A's gain
   counts only if A's false alarms on the test day are **acceptable**:

       A is at or under the budget on the test day            (``within_budget``)
       OR  A is not shown to raise more false alarms than B:
           the paired interval of the BENIGN difference does
           not lie above zero                                 (``excess_not_shown``)

   Why two ways to pass.  The second alone would punish A for using a budget
   that B leaves unused -- the behaviour layer's score is a count, ties
   heavily, and often fires on far fewer benign flows than it is allowed.
   The first alone could never be met on a day when every threshold drifts
   above its budget, which is what the baseline run showed.

   WHAT THIS RULE DOES NOT DO.  It is asymmetric: detection must be SHOWN to
   be higher, false alarms only NOT SHOWN to be higher.  An excess that is
   large but noisy (interval straddling zero) passes it.  So every verdict
   also carries the harsher reading, ``passes_strict``: more detection, and
   A's observed false-alarm rate is not above B's at all (``no_excess``),
   whatever the budget.  The two are reported side by side.
"""

from __future__ import annotations

from typing import Optional, Sequence

import numpy as np
import pandas as pd

from config import BENIGN_LABEL
from stats import ANY_ATTACK, MIN_BLOCKS_FOR_INTERVAL, interval_counts

NOT_EVALUATED = "not evaluated"


def _at_budget(df: pd.DataFrame, budget: float) -> pd.DataFrame:
    """Rows at ``budget``, compared with a tolerance (budgets are floats)."""
    return df[np.isclose(df["budget"].astype(float), float(budget), rtol=0, atol=1e-12)]


def false_alarm_check(benign_row, rate_column: str, budget: float) -> dict:
    """Condition 2 of the module docstring, for one comparison "A minus B".

    ``benign_row`` is the BENIGN row of a paired-difference table and
    ``rate_column`` names A's own false-alarm rate in it.  A missing interval
    (no bootstrap was run) or a missing rate is never a pass.
    """
    lo = float(benign_row["diff_lo"])
    diff = float(benign_row["diff"])
    rate = float(benign_row[rate_column])
    within = bool(np.isfinite(rate) and rate <= float(budget) + 1e-12)
    not_shown = bool(np.isfinite(lo) and lo <= 0.0)
    no_excess = bool(np.isfinite(diff) and diff <= 0.0)
    return {"false_alarm_rate": rate, "false_alarm_diff": diff, "false_alarm_lo": lo,
            "false_alarm_hi": float(benign_row["diff_hi"]),
            "within_budget": within, "excess_not_shown": not_shown, "no_excess": no_excess,
            "false_alarms_ok": bool(within or not_shown)}


_FAILED_CHECK = {"false_alarm_rate": float("nan"), "false_alarm_diff": float("nan"),
                 "false_alarm_lo": float("nan"), "false_alarm_hi": float("nan"),
                 "within_budget": False, "excess_not_shown": False, "no_excess": False,
                 "false_alarms_ok": False}


# =====================================================================
# RQ3
# =====================================================================


def rq3_verdict(
    pairs: pd.DataFrame,
    primary_budget: float,
    deep: str = "autoencoder",
    protocol: str = "crossday",
    required_baselines: Optional[Sequence[str]] = None,
) -> pd.DataFrame:
    """RQ3: does the deep benign-only detector beat EVERY classical one?

    The rule (``PROTOCOL.md``): under the cross-day protocol, at the primary
    budget, for at least one test-day attack class that spans at least 20
    blocks, the autoencoder detects more than every classical baseline with a
    paired interval of the difference that lies above zero -- and its false
    alarms are acceptable against every one of them (module docstring).

    ``required_baselines`` names the classical detectors the protocol fixes.
    If one of them was not run, "every baseline" was not tested and no row
    passes (``baselines_complete`` is False).

    One row per attack class:

    ``baselines``            how many classical detectors it was compared with
    ``baselines_complete``   every required baseline is among them
    ``smallest_gain``        the smallest "deep minus classical" difference
    ``smallest_gain_lo``     the lowest lower end among those intervals
    ``beats_every_baseline`` every interval lies above zero
    ``interval_counts``      the class spans enough blocks for an interval
    ``deep_false_alarm_rate``the deep detector's false-alarm rate on the test day
    ``within_budget``        that rate is at or under the budget
    ``excess_not_shown``     against every baseline, the BENIGN interval is
                             not above zero
    ``no_excess``            against every baseline, the deep detector's
                             observed false-alarm rate is not higher
    ``false_alarms_ok``      within budget, or excess not shown
    ``passes``               complete, beats every baseline, interval counts,
                             false alarms acceptable
    ``passes_strict``        the same with ``no_excess`` in place of
                             ``false_alarms_ok``

    The false-alarm columns do not depend on the class and repeat on every
    row.  RQ3 is answered "yes" when at least one row passes
    (``rq3_supported``).  The any-attack row is left out on purpose: the rule
    is about classes.
    """
    cols = ["class", "deep", "n_blocks", "baselines", "baselines_complete", "smallest_gain",
            "smallest_gain_lo", "beats_every_baseline", "interval_counts",
            "deep_false_alarm_rate", "within_budget", "excess_not_shown", "no_excess",
            "false_alarms_ok", "passes", "passes_strict"]
    if pairs is None or not len(pairs):
        return pd.DataFrame(columns=cols)
    df = _at_budget(pairs, primary_budget)
    df = df[(df["protocol"] == protocol) & (df["deep"] == deep)]
    classes = df[~df["class"].isin([BENIGN_LABEL, ANY_ATTACK])]
    if not len(classes):
        return pd.DataFrame(columns=cols)

    compared = sorted(set(classes["baseline"]))
    complete = (set(required_baselines) <= set(compared)) if required_baselines else True

    # One BENIGN row PER BASELINE is required.  Without it the gain over that
    # baseline cannot be shown not to come from extra false alarms.
    benign = df[df["class"] == BENIGN_LABEL].drop_duplicates("baseline").set_index("baseline")
    checks = [false_alarm_check(benign.loc[b], "rate_deep", primary_budget)
              if b in benign.index else dict(_FAILED_CHECK) for b in compared]
    within = all(c["within_budget"] for c in checks)
    not_shown = all(c["excess_not_shown"] for c in checks)
    no_excess = all(c["no_excess"] for c in checks)
    fa_ok = all(c["false_alarms_ok"] for c in checks)
    rates = [c["false_alarm_rate"] for c in checks if np.isfinite(c["false_alarm_rate"])]
    fa_rate = float(rates[0]) if rates else float("nan")

    rows = []
    for c, g in classes.groupby("class", sort=True):
        n_blocks = int(g["n_blocks"].min())
        # every compared baseline must have a row for this class
        has_all = set(g["baseline"]) == set(compared)
        beats = bool(has_all and (g["diff_lo"].astype(float) > 0.0).all())
        counts = interval_counts(n_blocks)
        base = bool(complete and beats and counts)
        rows.append({
            "class": c, "deep": deep, "n_blocks": n_blocks,
            "baselines": int(g["baseline"].nunique()), "baselines_complete": bool(complete),
            "smallest_gain": float(g["diff"].min()),
            "smallest_gain_lo": float(g["diff_lo"].min()),
            "beats_every_baseline": beats, "interval_counts": counts,
            "deep_false_alarm_rate": fa_rate, "within_budget": bool(within),
            "excess_not_shown": bool(not_shown), "no_excess": bool(no_excess),
            "false_alarms_ok": bool(fa_ok),
            "passes": bool(base and fa_ok), "passes_strict": bool(base and no_excess),
        })
    return pd.DataFrame(rows, columns=cols)


def rq3_supported(verdict: Optional[pd.DataFrame], strict: bool = False):
    """``True`` / ``False``, or ``"not evaluated"`` when there was nothing to
    compare or a baseline the protocol names was not run."""
    if verdict is None or not len(verdict):
        return NOT_EVALUATED
    if not bool(verdict["baselines_complete"].all()):
        return NOT_EVALUATED
    return bool(verdict["passes_strict" if strict else "passes"].any())


# =====================================================================
# RQ5
# =====================================================================


def rq5_verdict(gain: pd.DataFrame, primary_budget: float,
                required_layers: Optional[Sequence[str]] = None) -> pd.DataFrame:
    """RQ5: does the full system beat EACH single layer at the same budget?

    The rule (``PROTOCOL.md``): at the primary budget under min-p fusion, the
    full system detects more attack flows (the any-attack row) than each single
    layer with a paired interval of the difference that lies above zero, AND
    its false alarms are acceptable against that layer (module docstring).

    ``gain`` is ``e5_gain``, alone or stacked with the same table computed
    under the second score rule.  One row per (score rule, condition, seed,
    single layer):

    ``complete``              both rows the rule needs (any-attack, BENIGN)
                              are in the table; if not, the row fails
    ``more_detection``        the any-attack interval lies above zero, on at
                              least 20 blocks
    ``within_budget``         the full system's false-alarm rate on the test
                              day is at or under the budget
    ``excess_not_shown``      the BENIGN interval does not lie above zero
    ``no_excess``             the system's observed false-alarm rate is not
                              above the layer's
    ``false_alarms_ok``       within budget, or excess not shown
    ``passes``                more detection, at acceptable false alarms
    ``passes_strict``         more detection, and no excess at all
    ``layers_complete``       the single layers of that (rule, condition,
                              seed) are exactly ``required_layers``.  RQ5 is a
                              question about the three-layer system; when the
                              behaviour layer could not run, the table can be
                              read but the question was not asked
    ``fusion_helps``          the layers are complete AND every one of them
                              passes -- the same value on each of its rows
    """
    keys = [k for k in ("score_rule", "condition", "seed")
            if gain is not None and k in gain.columns]
    cols = [*keys, "single_layer", "complete", "attack_gain", "attack_lo", "attack_hi",
            "attack_n_blocks", "system_false_alarm_rate", "false_alarm_diff", "false_alarm_lo",
            "false_alarm_hi", "more_detection", "within_budget", "excess_not_shown",
            "no_excess", "false_alarms_ok", "passes", "passes_strict", "layers_complete",
            "fusion_helps"]
    if gain is None or not len(gain):
        return pd.DataFrame(columns=cols)
    df = _at_budget(gain, primary_budget)
    df = df[df["n_layers_compared"] == 1]
    rows = []
    nan = float("nan")
    for key, g in df.groupby([*keys, "compared_with"], sort=False):
        key = key if isinstance(key, tuple) else (key,)
        atk = g[g["class"] == ANY_ATTACK]
        ben = g[g["class"] == BENIGN_LABEL]
        complete = bool(len(atk) and len(ben))
        a = atk.iloc[0] if len(atk) else None
        fa = (false_alarm_check(ben.iloc[0], "rate_system", primary_budget)
              if len(ben) else dict(_FAILED_CHECK))
        more = bool(a is not None and float(a["diff_lo"]) > 0.0
                    and interval_counts(a["n_blocks"]))
        rows.append({
            **dict(zip(keys, key[:-1])), "single_layer": key[-1], "complete": complete,
            "attack_gain": float(a["diff"]) if a is not None else nan,
            "attack_lo": float(a["diff_lo"]) if a is not None else nan,
            "attack_hi": float(a["diff_hi"]) if a is not None else nan,
            "attack_n_blocks": int(a["n_blocks"]) if a is not None else 0,
            "system_false_alarm_rate": fa["false_alarm_rate"],
            "false_alarm_diff": fa["false_alarm_diff"], "false_alarm_lo": fa["false_alarm_lo"],
            "false_alarm_hi": fa["false_alarm_hi"],
            "more_detection": more, "within_budget": fa["within_budget"],
            "excess_not_shown": fa["excess_not_shown"], "no_excess": fa["no_excess"],
            "false_alarms_ok": fa["false_alarms_ok"],
            "passes": bool(complete and more and fa["false_alarms_ok"]),
            "passes_strict": bool(complete and more and fa["no_excess"]),
        })
    out = pd.DataFrame(rows)
    if not len(out):
        return pd.DataFrame(columns=cols)
    need = set(required_layers) if required_layers is not None else None
    if keys:
        grouped = out.groupby(keys, sort=False)
        all_pass = grouped["passes"].transform("all")
        if need is None:
            out["layers_complete"] = True
        else:
            out["layers_complete"] = grouped["single_layer"].transform(
                lambda layers: set(layers) == need)
    else:
        all_pass = pd.Series(bool(out["passes"].all()), index=out.index)
        out["layers_complete"] = True if need is None else (set(out["single_layer"]) == need)
    out["layers_complete"] = out["layers_complete"].astype(bool)
    out["fusion_helps"] = (all_pass.astype(bool) & out["layers_complete"]).astype(bool)
    return out[cols]


def rq5_supported(verdict: Optional[pd.DataFrame], condition: str, seed: int,
                  score_rule: Optional[str] = None, strict: bool = False,
                  required_layers: Optional[Sequence[str]] = None):
    """The answer for one (condition, seed[, score rule]) cell of ``rq5_verdict``.

    ``required_layers`` names the single layers the protocol's system has.
    If the table does not hold exactly those (the behaviour layer could not
    run, say), the three-layer question was not asked: ``"not evaluated"``.
    """
    if verdict is None or not len(verdict):
        return NOT_EVALUATED
    v = verdict[(verdict["condition"] == condition) & (verdict["seed"] == seed)]
    if score_rule is not None and "score_rule" in v.columns:
        v = v[v["score_rule"] == score_rule]
    if not len(v):
        return NOT_EVALUATED
    if required_layers is not None and set(v["single_layer"]) != set(required_layers):
        return NOT_EVALUATED
    if "layers_complete" in v.columns and not bool(v["layers_complete"].all()):
        return NOT_EVALUATED
    return bool(v["passes_strict" if strict else "passes"].all())


# =====================================================================
# RQ6
# =====================================================================


def rq6_verdict(cal: pd.DataFrame, primary_budget: float) -> pd.DataFrame:
    """RQ6: does per-class recalibration REPAIR the threshold on the test day?

    The rule (``PROTOCOL.md``): "recalibration repairs the threshold" is
    supported only if, on the same model and the same reference flows,
    (a) the uncalibrated threshold was over the budget on the test day,
    (b) the recalibrated one's false-alarm rate is lower, with a paired
    interval that lies below zero, and (c) the recalibrated rate is at or
    under the budget.

    One row per (model, score rule) at the primary budget:

    ``lowers_false_alarms``  (b)
    ``over_budget_before``   (a)
    ``within_budget_after``  (c)
    ``repairs``              (a) and (b) and (c)
    ``improves``             (a) and (b) but not (c): lower, still over budget.
                             An improvement, not a repair.
    ``detection_diff``       what the same change does to any-attack detection

    When (a) is false there was nothing to repair: a lower rate is then a
    stricter threshold under another name, and neither column is set.
    """
    cols = ["model", "score_rule", "budget", "rate_uncalibrated", "rate_recalibrated",
            "false_alarm_diff", "false_alarm_lo", "false_alarm_hi", "n_blocks",
            "lowers_false_alarms", "over_budget_before", "within_budget_after",
            "repairs", "improves", "detection_diff", "detection_lo", "detection_hi"]
    if cal is None or not len(cal):
        return pd.DataFrame(columns=cols)
    df = _at_budget(cal, primary_budget)
    rows = []
    tol = float(primary_budget) + 1e-12
    for (model, rule), g in df.groupby(["model", "score_rule"], sort=False):
        ben = g[g["class"] == BENIGN_LABEL]
        if not len(ben):
            continue
        b = ben.iloc[0]
        atk = g[g["class"] == ANY_ATTACK]
        a = atk.iloc[0] if len(atk) else None
        hi = float(b["diff_hi"])
        before, after = float(b["rate_uncalibrated"]), float(b["rate_recalibrated"])
        lowers = bool(np.isfinite(hi) and hi < 0.0 and interval_counts(b["n_blocks"]))
        over_before = bool(np.isfinite(before) and before > tol)
        within_after = bool(np.isfinite(after) and after <= tol)
        rows.append({
            "model": model, "score_rule": rule, "budget": float(primary_budget),
            "rate_uncalibrated": before, "rate_recalibrated": after,
            "false_alarm_diff": float(b["diff"]), "false_alarm_lo": float(b["diff_lo"]),
            "false_alarm_hi": hi, "n_blocks": int(b["n_blocks"]),
            "lowers_false_alarms": lowers, "over_budget_before": over_before,
            "within_budget_after": within_after,
            "repairs": bool(over_before and lowers and within_after),
            "improves": bool(over_before and lowers and not within_after),
            "detection_diff": float(a["diff"]) if a is not None else np.nan,
            "detection_lo": float(a["diff_lo"]) if a is not None else np.nan,
            "detection_hi": float(a["diff_hi"]) if a is not None else np.nan,
        })
    return pd.DataFrame(rows, columns=cols)


def rq6_supported(verdict: Optional[pd.DataFrame], model: str, score_rule: str,
                  column: str = "repairs"):
    """One (model, score rule) cell of ``rq6_verdict``.  ``column`` is
    ``"repairs"`` (the protocol's claim), ``"improves"`` or
    ``"lowers_false_alarms"``."""
    if verdict is None or not len(verdict):
        return NOT_EVALUATED
    v = verdict[(verdict["model"] == model) & (verdict["score_rule"] == score_rule)]
    if not len(v):
        return NOT_EVALUATED
    return bool(v[column].iloc[0])


def reading_rule() -> str:
    """One line for the run logs, so the rule is printed next to its verdict."""
    return (f"An interval counts only for a class that spans at least "
            f"{MIN_BLOCKS_FOR_INTERVAL} blocks (column `interval_counts`); `differs` is "
            f"True only when it also excludes zero.")
