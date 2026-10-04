"""
Combining layers under ONE false-alarm budget.

THE COMPARISON THAT MUST NOT BE RIGGED
--------------------------------------
"The hybrid detects more than the classifier alone" is true of any OR of two
detectors and proves nothing: the OR also raises more false alarms.  Three
detectors each run at 1% false alarms cost up to 3% together.  The only fair
question is what the layers catch *at the same total false-alarm rate* as the
single detector they are compared with.

Two rules are implemented, both calibrated on validation benign flows only.

``bonferroni``  Each of k layers gets budget B/k and its own threshold; a flow
                fires if any layer fires.  The union bound guarantees the
                validation false-alarm rate is at most B.  Simple, and
                conservative when layers fire on the same benign flows.

``min-p``       Each layer's score is turned into an empirical p-value against
                validation benign scores ("what share of benign flows score at
                least this high").  A flow's fused score is its smallest
                p-value, and ONE threshold on that fused score is set at
                budget B on validation benign.  This is the same OR, with the
                budget shared by the data instead of split by hand, so it does
                not waste budget when layers overlap.

Neither rule sees an attack label.  A learned fusion (a classifier stacked on
the layer scores) would have to be trained on attacks, and would then be
tuned to the attack classes that happen to be in validation -- the opposite of
what a study of UNSEEN attacks needs.

``layer_attribution`` answers the question in the paper's title: of the attack
flows the fused system caught, which layer(s) caught them.
"""

from __future__ import annotations

from itertools import combinations
from typing import Dict, Iterable, List, Sequence, Tuple

import numpy as np
import pandas as pd

from stats import budget_threshold


def empirical_pvalues(reference: np.ndarray, scores: np.ndarray) -> np.ndarray:
    """P(reference score >= s), with the +1 correction that keeps it above zero.

    ``reference`` is the validation benign score sample for one layer.  Ties
    count against the flow (``>=``), which matters for the behaviour layer's
    integer counts: a benign-typical count gets a large p-value, as it should.
    """
    ref = np.sort(np.asarray(reference, dtype=np.float64))
    ref = ref[np.isfinite(ref)]
    if ref.size == 0:
        raise ValueError("empirical_pvalues needs a non-empty reference sample")
    s = np.asarray(scores, dtype=np.float64)
    # number of reference values >= s  ==  n - (number strictly below s)
    ge = ref.size - np.searchsorted(ref, s, side="left")
    return (ge + 1.0) / (ref.size + 1.0)


def minp_scores(
    val_benign: Dict[str, np.ndarray],
    scores: Dict[str, np.ndarray],
    layers: Sequence[str],
) -> np.ndarray:
    """Fused score = -log10(min over layers of the empirical p-value).

    Higher = more suspicious, like every other score in the study, so it can
    go straight into ``stats.detection_table``.
    """
    if not layers:
        raise ValueError("minp_scores needs at least one layer")
    p = None
    for name in layers:
        pv = empirical_pvalues(val_benign[name], scores[name])
        p = pv if p is None else np.minimum(p, pv)
    return -np.log10(p)


def bonferroni_fired(
    val_benign: Dict[str, np.ndarray],
    scores: Dict[str, np.ndarray],
    layers: Sequence[str],
    budget: float,
) -> Tuple[np.ndarray, Dict[str, np.ndarray]]:
    """OR of the layers, each at ``budget / len(layers)``.

    Returns ``(fired, per_layer_fired)``.
    """
    if not layers:
        raise ValueError("bonferroni_fired needs at least one layer")
    share = float(budget) / len(layers)
    per: Dict[str, np.ndarray] = {}
    fired = None
    for name in layers:
        tau = budget_threshold(val_benign[name], share)
        f = np.asarray(scores[name], dtype=np.float64) > tau
        per[name] = f
        fired = f if fired is None else (fired | f)
    return fired, per


def layer_subsets(layers: Sequence[str]) -> List[Tuple[str, ...]]:
    """Every non-empty subset, singles first -- the ablation grid."""
    out: List[Tuple[str, ...]] = []
    for k in range(1, len(layers) + 1):
        out.extend(combinations(layers, k))
    return out


def layer_attribution(
    per_layer_fired: Dict[str, np.ndarray],
    y_true_str: Iterable,
    benign_label: str = "BENIGN",
) -> pd.DataFrame:
    """For each class: how its flows split by WHICH layers fired on them.

    One row per (class, combination of layers), e.g. ``classifier+behaviour``,
    plus ``none``.  Shares within a class sum to 1.  The ``none`` share is what
    the whole system missed.
    """
    names = list(per_layer_fired)
    y = np.asarray(list(y_true_str), dtype=object)
    n = y.size
    code = np.zeros(n, dtype=np.int64)
    for i, name in enumerate(names):
        code |= (np.asarray(per_layer_fired[name], dtype=bool).astype(np.int64) << i)

    def label(c: int) -> str:
        on = [names[i] for i in range(len(names)) if (c >> i) & 1]
        return "+".join(on) if on else "none"

    rows = []
    for cls in pd.unique(y):
        m = y == cls
        tot = int(m.sum())
        vals, cnts = np.unique(code[m], return_counts=True)
        for c, k in zip(vals.tolist(), cnts.tolist()):
            rows.append({"class": cls, "layers": label(c), "n_layers": bin(c).count("1"),
                         "flows": int(k), "share": k / tot, "class_total": tot})
    return pd.DataFrame(rows).sort_values(["class", "n_layers", "layers"]).reset_index(drop=True)
