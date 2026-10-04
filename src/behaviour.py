"""
The behaviour layer, measured offline on dataset flows.

WHAT THIS LAYER IS
------------------
A per-flow model answers "does this flow look like an attack".  Some attacks
are not a property of any one flow.  A port scan is one source touching many
ports; a sweep is one source touching many hosts.  The evidence is in the
*set*, so this layer counts over a (source, time-window) group instead of
classifying a row:

    ports(flow)  = distinct destination ports its source touched in its window
    hosts(flow)  = distinct destination hosts its source touched in its window

That idea is not new and is not claimed as new.  KDD'99's "same host" features
count connections over two seconds; TRW (Jung et al., 2004) tests connection
outcomes per source; Ring et al. (2018) aggregate flows per source IP per 60
seconds.  What this module adds is a measurement: the layer is scored in the
same currency as the per-flow detectors -- per-class detection at a benign
false-alarm budget fitted on a validation split -- so that its contribution
can be set beside theirs instead of asserted.

TWO WAYS OF RUNNING IT, ON PURPOSE
----------------------------------
``window_counts`` computes the counts for every flow at once (tumbling
windows).  It is fast, it gives a *score*, and a score can be thresholded at
any false-alarm budget.

``replay_scan_tracker`` feeds the flows, in time order, through the very
``ScanTracker`` class the live backend runs.  It gives *alerts* -- how many an
analyst would have received, when, and about whom.  If the two disagree about
which sources scan, one of them is wrong, and a test asserts they do not.

THE DIRECTION ARTEFACT
----------------------
CICFlowMeter names as "source" whichever endpoint sent the first packet it
saw.  When that packet is a reply -- a flow already under way when the capture
or a flow timeout began -- the *server* is recorded as source and the client's
ephemeral port as "destination port".  A busy web server then appears to be
touching hundreds of distinct destination ports a minute, which is exactly the
signature of a vertical scan.

``canonical_endpoints`` repairs this with the standard heuristic: when the
source port is a service port and the destination port is not, the endpoints
are swapped.  The effect of the repair is measured (``canonical=False``), not
assumed: the uncorrected false-alarm count is part of the result.
"""

from __future__ import annotations

import sys
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from config import (
    BEHAVIOUR_HOST_THRESHOLD,
    BEHAVIOUR_PORT_THRESHOLD,
    BEHAVIOUR_WINDOW_SECONDS,
    META_CAPTURE,
    META_DST_IP,
    META_DST_PORT,
    META_ROW,
    META_SRC_IP,
    META_SRC_PORT,
    META_TS,
    PROJECT_ROOT,
    SERVICE_PORT_MAX,
)
from meta import require_identifiers


# =====================================================================
# ENDPOINTS
# =====================================================================


@dataclass
class Endpoints:
    """Per-flow endpoints as integer codes, after the optional direction repair."""

    src: np.ndarray            # int64 code of the (canonical) source address
    dst: np.ndarray            # int64 code of the (canonical) destination address
    dport: np.ndarray          # int64 (canonical) destination port
    ts: np.ndarray             # float64 seconds
    swapped: np.ndarray        # bool: this flow's endpoints were swapped
    addresses: np.ndarray      # code -> address string

    def address(self, code: int) -> str:
        return str(self.addresses[int(code)])


def canonical_endpoints(
    meta: pd.DataFrame,
    canonical: bool = True,
    service_port_max: int = SERVICE_PORT_MAX,
) -> Endpoints:
    """Endpoints of every flow, with server-as-source flows turned round.

    A flow is swapped when its source port is below ``service_port_max`` and
    its destination port is not.  When both ports are service ports, or
    neither is, there is no basis for a decision and the flow is left as
    recorded.
    """
    require_identifiers(meta, "The behaviour layer")

    src_s = meta[META_SRC_IP].astype("object").to_numpy()
    dst_s = meta[META_DST_IP].astype("object").to_numpy()
    codes, addresses = pd.factorize(np.concatenate([src_s, dst_s]), sort=False)
    n = len(meta)
    src, dst = codes[:n].astype(np.int64), codes[n:].astype(np.int64)

    sport = meta[META_SRC_PORT].to_numpy(dtype=np.int64)
    dport = meta[META_DST_PORT].to_numpy(dtype=np.int64)
    ts = pd.Series(meta[META_TS].to_numpy(dtype=np.float64)).ffill().bfill().to_numpy()

    swapped = np.zeros(n, dtype=bool)
    if canonical:
        swapped = (sport >= 0) & (sport < service_port_max) & (dport >= service_port_max)
        src, dst = np.where(swapped, dst, src), np.where(swapped, src, dst)
        dport = np.where(swapped, sport, dport)

    return Endpoints(src=src, dst=dst, dport=dport, ts=ts, swapped=swapped,
                     addresses=np.asarray(addresses, dtype=object))


# =====================================================================
# WINDOW COUNTS  (the score)
# =====================================================================


def window_counts(
    ep: Endpoints,
    window_seconds: float = BEHAVIOUR_WINDOW_SECONDS,
) -> pd.DataFrame:
    """Per flow: how busy its source was in the window the flow falls in.

    Tumbling windows of ``window_seconds``.  Returns a frame with, per row,
    ``ports`` (distinct destination ports), ``hosts`` (distinct destination
    hosts) and ``flows`` (flow count) of that flow's (source, window) group,
    plus the integer ``group`` id.

    WHY TUMBLING, NOT SLIDING.  Most captures in this dataset are stamped to
    the minute.  A window that slides by less than a minute cannot be computed
    from them, and pretending otherwise would manufacture precision.  With a
    60-second window on minute stamps, a group is simply "the flows of one
    source that carry the same timestamp".  The price is that a scan
    straddling a window boundary is split in two; the window-length ablation
    shows how much that costs.
    """
    if window_seconds <= 0:
        raise ValueError("window_seconds must be positive")
    bucket = np.floor(ep.ts / float(window_seconds)).astype(np.int64)
    group, _ = pd.factorize(pd.MultiIndex.from_arrays([ep.src, bucket]))

    g = pd.DataFrame({"g": group, "p": ep.dport, "h": ep.dst})
    by = g.groupby("g", sort=False)
    out = pd.DataFrame({
        "ports": by["p"].transform("nunique").to_numpy(dtype=np.int64),
        "hosts": by["h"].transform("nunique").to_numpy(dtype=np.int64),
        "flows": by["p"].transform("size").to_numpy(dtype=np.int64),
        "group": group.astype(np.int64),
    })
    return out


def behaviour_score(
    counts: pd.DataFrame,
    port_threshold: int = BEHAVIOUR_PORT_THRESHOLD,
    host_threshold: int = BEHAVIOUR_HOST_THRESHOLD,
) -> np.ndarray:
    """One number per flow: how far its source is past either scan threshold.

    ``max(ports / port_threshold, hosts / host_threshold)``.  A value of 1.0 or
    more is what the deployed tracker alerts on; as a continuous score it can
    also be thresholded at a validation-fitted false-alarm budget like every
    other detector in the study.
    """
    return np.maximum(
        counts["ports"].to_numpy(dtype=np.float64) / float(port_threshold),
        counts["hosts"].to_numpy(dtype=np.float64) / float(host_threshold),
    )


def behaviour_layer_scores(
    meta: pd.DataFrame,
    window_seconds: float = BEHAVIOUR_WINDOW_SECONDS,
    canonical: bool = True,
    port_threshold: int = BEHAVIOUR_PORT_THRESHOLD,
    host_threshold: int = BEHAVIOUR_HOST_THRESHOLD,
) -> Tuple[np.ndarray, pd.DataFrame]:
    """Convenience: meta side-table -> (score per flow, counts frame)."""
    ep = canonical_endpoints(meta, canonical=canonical)
    counts = window_counts(ep, window_seconds)
    return behaviour_score(counts, port_threshold, host_threshold), counts


# =====================================================================
# WHOLE-WEEK COUNTS, LOOKED UP PER SPLIT
# =====================================================================


def flow_keys(meta: pd.DataFrame) -> np.ndarray:
    """A unique integer per flow: its capture and its row in that capture."""
    if META_CAPTURE not in meta.columns or META_ROW not in meta.columns:
        raise RuntimeError("flow_keys needs the cache's meta__capture and meta__row columns")
    return ((meta[META_CAPTURE].to_numpy(dtype=np.int64) << 32)
            | meta[META_ROW].to_numpy(dtype=np.int64))


@dataclass
class WeekBehaviour:
    """Window counts for EVERY flow in the cache, addressable by flow key.

    WHY THE WHOLE WEEK.  A window count is a property of all the flows a
    source sent in that window.  A split holds only some of them: the
    validation half of a randomly split Thursday sees half of each window,
    training has lost its exact duplicates to de-duplication (which for a port
    scan is almost all of it).  Counting inside a split would therefore
    understate every validation count by about half and set the threshold too
    low.  So the counts are computed once over all flows and each split looks
    its own rows up.
    """

    keys: np.ndarray          # sorted flow keys
    ports: np.ndarray
    hosts: np.ndarray
    score: np.ndarray
    params: Dict[str, object] = field(default_factory=dict)

    def lookup(self, meta: pd.DataFrame) -> Tuple[np.ndarray, pd.DataFrame]:
        """(score, counts) for the rows of ``meta``, in ``meta``'s order."""
        k = flow_keys(meta)
        pos = np.searchsorted(self.keys, k)
        pos = np.clip(pos, 0, len(self.keys) - 1)
        if not np.array_equal(self.keys[pos], k):
            missing = int((self.keys[pos] != k).sum())
            raise RuntimeError(f"{missing} flows of this split are not in the week-level "
                               "behaviour table; was it built from a different cache?")
        return self.score[pos], pd.DataFrame({"ports": self.ports[pos], "hosts": self.hosts[pos]})

    def scan_like(self) -> np.ndarray:
        """Keys of flows whose source is at or past the deployed thresholds."""
        return self.keys[self.score >= 1.0]


def week_behaviour(
    meta: pd.DataFrame,
    window_seconds: float = BEHAVIOUR_WINDOW_SECONDS,
    canonical: bool = True,
    port_threshold: int = BEHAVIOUR_PORT_THRESHOLD,
    host_threshold: int = BEHAVIOUR_HOST_THRESHOLD,
) -> WeekBehaviour:
    """Window counts for every flow in ``meta`` (normally: the whole cache)."""
    ep = canonical_endpoints(meta, canonical=canonical)
    counts = window_counts(ep, window_seconds)
    score = behaviour_score(counts, port_threshold, host_threshold)
    keys = flow_keys(meta)
    order = np.argsort(keys, kind="stable")
    if np.unique(keys).size != keys.size:
        raise RuntimeError("flow keys are not unique: (capture, row) should identify a flow")
    return WeekBehaviour(
        keys=keys[order], ports=counts["ports"].to_numpy()[order],
        hosts=counts["hosts"].to_numpy()[order], score=score[order],
        params={"window_seconds": window_seconds, "canonical": canonical,
                "port_threshold": port_threshold, "host_threshold": host_threshold},
    )


# =====================================================================
# REPLAY THROUGH THE DEPLOYED TRACKER  (the alerts)
# =====================================================================


# One millisecond: a thousand times finer than the finest timestamp in the
# dataset (seconds), several thousand times coarser than float64 rounding at
# epoch scale (about 2e-7 s), so the comparison it decides is never a tie.
HALF_OPEN_EPSILON = 1e-3


@dataclass
class ReplayResult:
    in_alert: np.ndarray                       # bool per flow, in the INPUT row order
    alerts: pd.DataFrame                       # one row per alert the tracker emitted
    stats: Dict[str, int] = field(default_factory=dict)


def _scan_tracker_class():
    """Import the live component without importing the live package's
    dependencies (scapy, fastapi): scan_tracker.py needs the stdlib only."""
    root = str(PROJECT_ROOT)
    if root not in sys.path:
        sys.path.insert(0, root)
    from backend.live.scan_tracker import ScanTracker
    return ScanTracker


def replay_scan_tracker(
    meta: pd.DataFrame,
    labels: Optional[np.ndarray] = None,
    canonical: bool = True,
    window_seconds: float = BEHAVIOUR_WINDOW_SECONDS,
    port_threshold: int = BEHAVIOUR_PORT_THRESHOLD,
    host_threshold: int = BEHAVIOUR_HOST_THRESHOLD,
    benign_label: str = "BENIGN",
) -> ReplayResult:
    """Feed dataset flows, in time order, to the tracker the backend runs.

    For each flow the tracker is asked, after it has observed the flow, whether
    that flow's source is at or above either threshold.  ``in_alert`` is that
    answer.  It is deliberately the ONLINE answer: the first ninety-nine ports
    of a scan are observed before the hundredth trips the threshold, and they
    are not counted as detected.  A retrospective count would be higher and
    would describe a different system.

    ``alerts`` carries, per alert, what share of the source's flows in the
    preceding window were labelled as an attack -- the evidence for calling an
    alert true or false.

    TIMESTAMP RESOLUTION.  The tracker keeps every observation whose time is
    at least ``now - window_seconds``: a closed interval.  On a live clock that
    detail is invisible.  On a dataset stamped to the minute it is not: every
    flow of a minute carries the same reading, the previous minute's flows are
    exactly 60 s old, and a closed 60-second window therefore holds TWO stamped
    minutes of traffic -- twice what the name says.  The replay hands the
    tracker a window one millisecond shorter, which makes the interval
    half-open, ``(now - window, now]``.  For minute stamps that is exactly one
    minute; for second stamps exactly sixty seconds; for a continuous clock no
    change at all.  A consequence worth having: on minute-stamped captures the
    replay and ``window_counts`` now agree on which (source, minute) groups
    cross a threshold, so the two can check each other.
    """
    ScanTracker = _scan_tracker_class()
    ep = canonical_endpoints(meta, canonical=canonical)
    n = len(meta)

    cap = (meta[META_CAPTURE].to_numpy(dtype=np.int64) if META_CAPTURE in meta.columns
           else np.zeros(n, dtype=np.int64))
    row = (meta[META_ROW].to_numpy(dtype=np.int64) if META_ROW in meta.columns
           else np.arange(n, dtype=np.int64))
    order = np.lexsort((row, cap, ep.ts))          # time, then capture, then file row

    tracker = ScanTracker(
        window_seconds=max(float(window_seconds) - HALF_OPEN_EPSILON, HALF_OPEN_EPSILON),
        vertical_port_threshold=port_threshold,
        horizontal_host_threshold=host_threshold,
        max_sources=max(50_000, len(ep.addresses) + 1),
        alert_cooldown_seconds=window_seconds,
    )

    in_alert = np.zeros(n, dtype=bool)
    rows: List[dict] = []
    src_l, dst_l, dp_l, ts_l = ep.src.tolist(), ep.dst.tolist(), ep.dport.tolist(), ep.ts.tolist()

    for i in order.tolist():
        s, t = src_l[i], ts_l[i]
        fired = tracker.observe(str(s), str(dst_l[i]), dp_l[i], now=t)
        ports, hosts = tracker.source_counts(str(s))
        in_alert[i] = ports >= port_threshold or hosts >= host_threshold
        for a in fired:
            rows.append({
                "ts": t, "kind": a.kind, "src_code": s, "src": ep.address(s),
                "distinct": a.distinct, "threshold": a.threshold,
                "observations": a.observations, "capture": int(cap[i]),
            })

    # What were the alerting source's flows labelled, in the window that led
    # to the alert?  Looked up after the replay from a (source, time) index:
    # keeping a per-source history inside the loop would hold a Python tuple
    # for every flow of the week.
    if rows:
        by_src = np.lexsort((ep.ts, ep.src))
        src_sorted, ts_sorted = ep.src[by_src], ep.ts[by_src]
        is_attack = (None if labels is None
                     else (np.asarray(labels, dtype=object) != benign_label)[by_src])
        for r in rows:
            lo = int(np.searchsorted(src_sorted, r["src_code"], side="left"))
            hi = int(np.searchsorted(src_sorted, r["src_code"], side="right"))
            seg = ts_sorted[lo:hi]
            # the same half-open window the tracker was given: (ts - window, ts]
            a = lo + int(np.searchsorted(seg, r["ts"] - window_seconds, side="right"))
            b = lo + int(np.searchsorted(seg, r["ts"], side="right"))
            r["flows_in_window"] = b - a
            r["attack_share"] = (float(is_attack[a:b].mean())
                                 if is_attack is not None and b > a else float("nan"))
            del r["src_code"]

    alerts = pd.DataFrame(rows, columns=["ts", "kind", "src", "distinct", "threshold",
                                         "observations", "capture", "flows_in_window",
                                         "attack_share"])
    return ReplayResult(in_alert=in_alert, alerts=alerts, stats=dict(tracker.stats))


# =====================================================================
# SUMMARIES
# =====================================================================


def top_sources(
    meta: pd.DataFrame,
    counts: pd.DataFrame,
    labels: np.ndarray,
    ep: Optional[Endpoints] = None,
    k: int = 10,
    benign_label: str = "BENIGN",
) -> pd.DataFrame:
    """The busiest (source, window) groups, with what their flows were labelled.

    This is the table to read before believing any false-alarm number: it shows
    WHO tripped the counter.  A row whose flows are all labelled BENIGN is
    either a false alarm or a labelling error, and the address usually says
    which.
    """
    ep = ep or canonical_endpoints(meta)
    lab = np.asarray(labels, dtype=object)
    d = pd.DataFrame({
        "group": counts["group"].to_numpy(),
        "src": ep.src, "ts": ep.ts,
        "ports": counts["ports"].to_numpy(), "hosts": counts["hosts"].to_numpy(),
        "attack": (lab != benign_label).astype(np.float64), "label": lab,
    })
    g = d.groupby("group", sort=False)
    agg = g.agg(src=("src", "first"), start=("ts", "min"), ports=("ports", "first"),
                hosts=("hosts", "first"), flows=("attack", "size"),
                attack_share=("attack", "mean"),
                top_label=("label", lambda s: s.value_counts().index[0]))
    agg = agg.sort_values(["ports", "hosts"], ascending=False).head(k).reset_index(drop=True)
    agg["src"] = [ep.address(c) for c in agg["src"]]
    agg["start"] = pd.to_datetime(agg["start"], unit="s").astype(str)
    return agg
