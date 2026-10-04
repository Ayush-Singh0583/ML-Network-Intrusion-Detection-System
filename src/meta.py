"""
Identifier side-table: who talked to whom, on which port, and when.

WHY THIS EXISTS
---------------
``Destination Port``, the IP addresses and the timestamp are removed from the
feature set on purpose (see ``config.IDENTIFIER_COLUMNS``): a tree that may
look at them learns "port 21 -> FTP-Patator" and nothing about traffic.  That
decision stands.

Removing them from the *features* is not the same as throwing them away.  Two
parts of the study need them as side information:

* a time-blocked split has to know which flow came first;
* the behaviour layer counts distinct ports per source per minute, which is a
  statement about exactly the three things a per-flow model must not see.

So the identifiers travel beside the features in ``meta__*`` columns.  One
function, ``feature_columns``, decides what a model may see, and every matrix
in the pipeline is built through it.

THE TIMESTAMP FORMAT, WHICH IS WORSE THAN IT LOOKS
--------------------------------------------------
The GeneratedLabelledFlows CSVs of CIC-IDS2017 print timestamps as
``7/7/2017 3:30`` -- day first, minute resolution in most files, and a
**12-hour clock with no AM/PM marker**.  ``3:30`` on Friday is 15:30.  Read
naively, the afternoon captures sort *before* the morning ones and every
time-ordered split is scrambled without any error being raised.

The capture ran 09:00-17:00, so an hour below 8 can only be afternoon.  That
rule is applied here, it is stated in the paper, and ``study.py doctor`` prints
the hours every capture parses onto -- and lists one that parses outside
08:00-18:00 as a problem -- so that a wrong assumption is visible rather than
silent.
"""

from __future__ import annotations

from typing import Iterable, List, Optional, Sequence

import numpy as np
import pandas as pd

from config import (
    DAY_COL,
    DAY_ORDER,
    IDENTIFIER_COVERAGE,
    LABEL_COL,
    META_CAPTURE,
    META_COLUMNS,
    META_DST_IP,
    META_DST_PORT,
    META_PREFIX,
    META_PROTO,
    META_ROW,
    META_SRC_IP,
    META_SRC_PORT,
    META_TS,
    RAW_META_ALIASES,
    TIMESTAMP_PM_BELOW_HOUR,
)

# The capture week.  Used only to choose between day-first and month-first
# when a file leaves it ambiguous, and by ``doctor`` to flag a capture whose
# parsed date is not the day its file name claims.
EXPECTED_DAY_DATES = {
    "Monday": "2017-07-03",
    "Tuesday": "2017-07-04",
    "Wednesday": "2017-07-05",
    "Thursday": "2017-07-06",
    "Friday": "2017-07-07",
}

_SLASH = (
    r"^\s*(?P<a>\d{1,2})/(?P<b>\d{1,2})/(?P<y>\d{4})\s+"
    r"(?P<h>\d{1,2}):(?P<mi>\d{2})(?::(?P<s>\d{2})(?:\.\d+)?)?\s*(?P<ap>[AaPp][Mm])?\s*$"
)
_ISO = (
    r"^\s*(?P<y>\d{4})-(?P<mo>\d{1,2})-(?P<d>\d{1,2})[ T]"
    r"(?P<h>\d{1,2}):(?P<mi>\d{2})(?::(?P<s>\d{2})(?:\.\d+)?)?\s*$"
)


# =====================================================================
# WHAT A MODEL MAY SEE
# =====================================================================


def is_meta(col: str) -> bool:
    return str(col).startswith(META_PREFIX)


def feature_columns(columns: Iterable[str]) -> List[str]:
    """Columns a model may be trained on: everything except the label, the
    day tag and the ``meta__`` side-table.

    Accepts a DataFrame or any iterable of column names.
    """
    cols = columns.columns if hasattr(columns, "columns") else columns
    return [c for c in cols if c not in (LABEL_COL, DAY_COL) and not is_meta(c)]


def meta_columns_present(columns: Iterable[str]) -> List[str]:
    cols = columns.columns if hasattr(columns, "columns") else columns
    return [c for c in cols if is_meta(c)]


# =====================================================================
# TIMESTAMPS
# =====================================================================


def _to_epoch(y, mo, d, h, mi, s) -> np.ndarray:
    """Component arrays -> float64 seconds since the epoch; NaN where invalid."""
    parts = pd.DataFrame({
        "year": y, "month": mo, "day": d, "hour": h, "minute": mi, "second": s,
    })
    ok = parts.notna().all(axis=1).to_numpy()
    out = np.full(len(parts), np.nan, dtype=np.float64)
    if ok.any():
        dt = pd.to_datetime(parts[ok].astype("int64"), errors="coerce")
        good = dt.notna().to_numpy()
        vals = (dt.to_numpy().astype("datetime64[s]").astype("int64")).astype(np.float64)
        vals[~good] = np.nan
        out[np.flatnonzero(ok)] = vals
    return out


def parse_cic_timestamps(
    raw: Sequence,
    dayfirst: Optional[bool] = None,
    expected_date: Optional[str] = None,
    pm_below_hour: int = TIMESTAMP_PM_BELOW_HOUR,
) -> np.ndarray:
    """CIC-IDS2017 timestamp strings -> seconds since the epoch (float64).

    Handles the three shapes seen in distributions of this dataset::

        7/7/2017 3:30            day/month/year, 12-hour clock, no marker
        03/07/2017 08:55:58      the same with seconds
        2017-07-07 15:30:00.123  ISO, 24-hour (corrected re-releases)

    ``dayfirst=None`` tries both orders and keeps the one that puts more rows
    on ``expected_date`` (falling back to day-first, which is what the
    original release uses).  Unparseable rows come back as NaN -- never as a
    guessed value.

    An hour below ``pm_below_hour`` with no AM/PM marker is read as afternoon;
    see the module docstring for why that is the least-wrong reading.
    """
    s = pd.Series(raw, dtype="object").astype("string")
    n = len(s)
    out = np.full(n, np.nan, dtype=np.float64)
    if n == 0:
        return out

    num = lambda col: pd.to_numeric(col, errors="coerce")  # noqa: E731

    # ---- ISO: unambiguous, already 24-hour --------------------------------
    iso = s.str.extract(_ISO)
    iso_ok = iso["y"].notna().to_numpy()
    if iso_ok.any():
        sub = iso[iso_ok]
        out[iso_ok] = _to_epoch(
            num(sub["y"]), num(sub["mo"]), num(sub["d"]),
            num(sub["h"]), num(sub["mi"]), num(sub["s"]).fillna(0),
        )

    # ---- slash form: day/month order and AM/PM both need deciding ----------
    sl = s.str.extract(_SLASH)
    sl_ok = sl["y"].notna().to_numpy() & ~iso_ok
    if sl_ok.any():
        sub = sl[sl_ok]
        a, b = num(sub["a"]), num(sub["b"])
        h = num(sub["h"]).astype("float64")
        marker = sub["ap"].str.lower()
        has_marker = marker.notna()

        h24 = h.copy()
        pm = has_marker & (marker == "pm") & (h < 12)
        am12 = has_marker & (marker == "am") & (h == 12)
        h24[pm] = h[pm] + 12
        h24[am12] = 0
        # no marker: the 12-hour clock of the original release
        bare_pm = (~has_marker) & (h < pm_below_hour)
        h24[bare_pm] = h[bare_pm] + 12

        common = (num(sub["y"]), h24, num(sub["mi"]), num(sub["s"]).fillna(0))
        day_first = _to_epoch(common[0], b, a, common[1], common[2], common[3])
        if dayfirst is True:
            chosen = day_first
        else:
            month_first = _to_epoch(common[0], a, b, common[1], common[2], common[3])
            if dayfirst is False:
                chosen = month_first
            elif expected_date is not None:
                lo = pd.Timestamp(expected_date).timestamp()
                hits = lambda v: int(np.sum((v >= lo) & (v < lo + 86400)))  # noqa: E731
                chosen = month_first if hits(month_first) > hits(day_first) else day_first
            else:
                # the original release is day-first; prefer it unless it fails
                # to parse rows the other order can
                chosen = (month_first
                          if np.isnan(day_first).sum() > np.isnan(month_first).sum()
                          else day_first)
        out[sl_ok] = chosen

    return out


# =====================================================================
# BUILDING THE SIDE-TABLE FROM A RAW CAPTURE
# =====================================================================


def extract_meta(
    df: pd.DataFrame,
    capture_index: int,
    day: Optional[str] = None,
    row_index: Optional[np.ndarray] = None,
) -> pd.DataFrame:
    """Meta columns for one raw capture, aligned with ``df``'s index.

    Call this BEFORE the identifier columns are dropped.  A column the capture
    does not carry (the MachineLearningCSV files have no source IP and no
    timestamp) comes back null/-1, so the schema is the same for every
    capture and absence is explicit.
    """
    n = len(df)

    def find(meta_col: str) -> Optional[str]:
        return next((a for a in RAW_META_ALIASES[meta_col] if a in df.columns), None)

    out = pd.DataFrame(index=df.index)

    for col in (META_SRC_IP, META_DST_IP):
        src = find(col)
        if src is None:
            out[col] = pd.Series([None] * n, index=df.index, dtype="object")
        else:
            out[col] = df[src].astype("string").str.strip().astype("object")

    for col, dtype in ((META_SRC_PORT, np.int32), (META_DST_PORT, np.int32),
                       (META_PROTO, np.int16)):
        src = find(col)
        if src is None:
            out[col] = np.full(n, -1, dtype=dtype)
        else:
            v = pd.to_numeric(df[src], errors="coerce").fillna(-1)
            out[col] = v.to_numpy().astype(dtype)

    src = find(META_TS)
    if src is None:
        out[META_TS] = np.full(n, np.nan, dtype=np.float64)
    else:
        out[META_TS] = parse_cic_timestamps(
            df[src].to_numpy(), expected_date=EXPECTED_DAY_DATES.get(day or "")
        )

    out[META_ROW] = (np.arange(n, dtype=np.int32) if row_index is None
                     else np.asarray(row_index, dtype=np.int32))
    out[META_CAPTURE] = np.full(n, int(capture_index), dtype=np.int16)
    return out[META_COLUMNS]


def has_identifiers(meta: Optional[pd.DataFrame],
                    min_coverage: float = IDENTIFIER_COVERAGE) -> bool:
    """True when (almost) every row has a source IP and a timestamp."""
    if meta is None or len(meta) == 0:
        return False
    if META_SRC_IP not in meta.columns or META_TS not in meta.columns:
        return False
    ip_ok = float(meta[META_SRC_IP].notna().mean())
    ts_ok = float(np.isfinite(meta[META_TS].to_numpy(dtype=np.float64)).mean())
    return ip_ok >= min_coverage and ts_ok >= min_coverage


def require_identifiers(meta: Optional[pd.DataFrame], what: str) -> None:
    if not has_identifiers(meta):
        raise RuntimeError(
            f"{what} needs a source IP and a timestamp for every flow, and this "
            "cache has none.\n"
            "The MachineLearningCSV files of CIC-IDS2017 do not carry them. "
            "Download GeneratedLabelledFlows.zip from the same CIC page, put its "
            "eight CSVs (same file names) in data/, then run:\n"
            "    python src/run.py cache --rebuild\n"
            "    python src/study.py doctor"
        )


# =====================================================================
# TIME ORDER
# =====================================================================


def capture_ids(df: pd.DataFrame) -> np.ndarray:
    """An integer per row naming the capture it came from.

    The cache records it (``meta__capture``).  A frame built by hand, as the
    tests do, has only a day tag, so the day's position in the week stands in.
    """
    n = len(df)
    if META_CAPTURE in df.columns:
        return df[META_CAPTURE].to_numpy(dtype=np.int64)
    if DAY_COL in df.columns:
        rank = {d: i for i, d in enumerate(DAY_ORDER)}
        return (df[DAY_COL].astype(str).map(lambda d: rank.get(d, len(rank)))
                .to_numpy(dtype=np.int64))
    return np.zeros(n, dtype=np.int64)


def time_order(df: pd.DataFrame, min_coverage: float = IDENTIFIER_COVERAGE) -> np.ndarray:
    """Positions that sort ``df`` into capture order, then time order.

    Within one capture the key is the timestamp when the capture has one for
    (almost) every row, and the original row index otherwise -- never a mix,
    because a row index and an epoch second are not comparable.  Ties (a
    minute-resolution timestamp makes many) are broken by row index, which
    keeps the sort stable and reproducible.
    """
    n = len(df)
    if n == 0:
        return np.empty(0, dtype=np.int64)

    cap = capture_ids(df)

    row = (df[META_ROW].to_numpy(dtype=np.int64) if META_ROW in df.columns
           else np.arange(n, dtype=np.int64))

    key = row.astype(np.float64)
    if META_TS in df.columns:
        ts = df[META_TS].to_numpy(dtype=np.float64)
        for c in np.unique(cap):
            m = cap == c
            good = np.isfinite(ts[m])
            if good.mean() >= min_coverage:
                t = pd.Series(ts[m]).ffill().bfill().to_numpy()
                key[m] = t
    return np.lexsort((row, key, cap))


def time_position(df: pd.DataFrame) -> np.ndarray:
    """Rank of every row in ``time_order``: 0 = earliest."""
    order = time_order(df)
    pos = np.empty(len(df), dtype=np.int64)
    pos[order] = np.arange(len(df), dtype=np.int64)
    return pos
