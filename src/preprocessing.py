"""
Leak-free preprocessing for CIC-IDS2017.

What changed relative to the previous version, and why:

1.  ``load_all_datasets`` globbed ``data/*.csv``.  That silently ingested
    ``data/custom_sample.csv`` -- 115 rows derived from the Friday DDoS capture,
    tagged ``Day="Custom"``.  Replaced with an explicit manifest that raises on
    a missing file.

2.  ``clean_dataset`` computed the imputation median, the >50%-missing column
    threshold and the constant-column list over the WHOLE dataset, before any
    split existed.  All three are now fitted on the training split only, inside
    a ``FrameCleaner`` with ``fit`` / ``transform``.

3.  ``drop_duplicates()`` keyed on every column *including* ``Day``, so a flow
    that appeared identically on Tuesday and Wednesday survived twice and could
    land on both sides of a random split.  Deduplication is now on feature
    columns only, and the count removed is reported.

4.  Seven exactly-collinear duplicate features (``Fwd Header Length.1``, the
    four ``Subflow *`` twins, both ``Avg * Segment Size``) were retained.  They
    split tree feature-importance arbitrarily between twins.  Dropped.

5.  Everything was float64.  2.8M x 69 x 8 B = 1.54 GB per array, with two more
    copies created by ``fit_transform`` + ``transform``.  Now float32
    end-to-end.

6.  ``prepare_data(split_by_day=True)`` returned ``y_train`` as encoded int64
    and ``y_test`` as raw str.  Every consumer had to know this.  Splits now
    return a typed ``SplitBundle`` where the encoding contract is explicit.

7.  Six near-identical ``prepare_*`` functions collapsed into one
    ``build_splits(protocol=...)``.
"""

from __future__ import annotations

import os
import warnings
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler

from config import (
    BENIGN_LABEL,
    CLEAN_PARQUET,
    DATA_DIR,
    DAY_COL,
    DAY_FILES,
    DUPLICATE_FEATURES,
    IDENTIFIER_COLUMNS,
    LABEL_ALIASES,
    LABEL_COL,
    SEED,
    TEST_DAYS,
    THURSDAY_VAL_FRAC,
    TRAIN_DAYS,
    UNKNOWN_LABEL,
    VAL_DAYS,
)

# ---------------------------------------------------------------------
# Loading
# ---------------------------------------------------------------------


def _normalise_labels(s: pd.Series) -> pd.Series:
    s = (
        s.astype(str)
        .str.strip()
        .str.replace("�", "-", regex=False)
        .str.replace("–", "-", regex=False)
        .str.replace("", "-", regex=False)
    )
    s = s.replace(LABEL_ALIASES)
    # after dash normalisation the aliases may need a second pass
    s = s.replace(
        {
            "Web Attack - Brute Force": "WebAttack_BruteForce",
            "Web Attack - XSS": "WebAttack_XSS",
            "Web Attack - Sql Injection": "WebAttack_SQLInjection",
        }
    )
    return s


def load_all_datasets(
    folder_path: os.PathLike | str = DATA_DIR,
    day_files: Optional[Dict[str, List[str]]] = None,
    verbose: bool = True,
) -> pd.DataFrame:
    """Load the CIC-IDS2017 captures named in the manifest. Fails loudly."""
    folder = Path(folder_path)
    manifest = day_files or DAY_FILES

    frames: List[pd.DataFrame] = []
    if verbose:
        print("\n========== LOADING DATASETS ==========\n")

    for day, files in manifest.items():
        for fname in files:
            path = folder / fname
            if not path.exists():
                raise FileNotFoundError(
                    f"Manifest file missing: {path}\n"
                    f"Expected files for {day}: {files}\n"
                    "Fix config.DAY_FILES or place the capture in data/."
                )
            if verbose:
                print(f"  {day:<10s} {fname}")
            df = pd.read_csv(path, low_memory=False)
            df.columns = df.columns.str.strip()
            if LABEL_COL not in df.columns:
                raise ValueError(f"{path} has no '{LABEL_COL}' column")
            df[LABEL_COL] = _normalise_labels(df[LABEL_COL])
            df[DAY_COL] = day
            frames.append(df)

    merged = pd.concat(frames, ignore_index=True, copy=False)
    del frames

    if verbose:
        print("\n----------------------------------------")
        print(f"Rows    : {merged.shape[0]:,}")
        print(f"Columns : {merged.shape[1]}")
        print("----------------------------------------")

    return merged


# ---------------------------------------------------------------------
# Structural cleaning (split-independent, no statistics learned)
# ---------------------------------------------------------------------



def _clean_frame(df: pd.DataFrame) -> pd.DataFrame:
    """Per-capture structural cleaning.

    Learns NOTHING from the data distribution, so it is safe to apply one
    capture at a time -- which is exactly what lets ``build_cache`` stream
    instead of concatenating the whole week into memory first.
    """
    df = df.copy()

    if "Flow Duration" in df.columns:
        df = df[df["Flow Duration"] >= 0].copy()

    # drop identifiers (incl. Destination Port -- deliberate anti-shortcut)
    df = df.drop(columns=IDENTIFIER_COLUMNS, errors="ignore")

    # drop exactly-collinear CICFlowMeter twins
    df = df.drop(columns=[c for c in DUPLICATE_FEATURES if c in df.columns])

    feature_cols = [c for c in df.columns if c not in (LABEL_COL, DAY_COL)]
    for c in feature_cols:
        # NOT `df[c].dtype == object`.  pandas 3.0 made the default string
        # dtype Arrow-backed `str` rather than `object`, so that test silently
        # stops firing and the "Infinity" strings CIC-IDS2017 ships never get
        # coerced -- the astype(float32) below then raises.  Ask what the dtype
        # IS, not what it used to be called.
        if not pd.api.types.is_numeric_dtype(df[c]):
            df[c] = pd.to_numeric(df[c], errors="coerce")
    df[feature_cols] = df[feature_cols].replace([np.inf, -np.inf], np.nan)
    df[feature_cols] = df[feature_cols].astype(np.float32)

    return df.dropna(subset=[LABEL_COL])


def structural_clean(df: pd.DataFrame, verbose: bool = True) -> pd.DataFrame:
    """In-memory variant: clean, then de-duplicate globally.

    Kept for callers that already hold the whole frame, and for the tests.
    ``build_cache`` no longer uses it: its peak memory is ~4x the dataset
    because it needs the entire week materialised at once.
    """
    if verbose:
        print("\n========== STRUCTURAL CLEAN ==========")
        print(f"Initial : {df.shape}")

    df = _clean_frame(df)
    feature_cols = [c for c in df.columns if c not in (LABEL_COL, DAY_COL)]

    # BUG FIX (retained): dedup on FEATURES only.  The original call keyed on
    # Day + Label too, so cross-day duplicates survived into both halves of a
    # random split.
    before = len(df)
    df = df.drop_duplicates(subset=feature_cols, keep="first")
    removed = before - len(df)
    if verbose:
        print(f"Removed {removed:,} duplicate flows ({removed / max(before,1):.2%})")
        print(f"Final   : {df.shape}")
        print("\n---------- LABEL DISTRIBUTION ----------")
        print(df[LABEL_COL].value_counts())

    return df.reset_index(drop=True)


# ---------------------------------------------------------------------
# Distribution-dependent cleaning: FIT ON TRAIN ONLY
# ---------------------------------------------------------------------


class FrameCleaner:
    """
    Learns three things from the training split and applies them everywhere:

      * which columns to drop for exceeding a missing-value rate
      * which columns to drop for being constant
      * the per-column median used for imputation

    All three were previously computed over train+test together.
    """

    def __init__(self, missing_threshold: float = 0.5):
        self.missing_threshold = missing_threshold
        self.feature_names_: List[str] = []
        self.medians_: Optional[pd.Series] = None
        self.dropped_missing_: List[str] = []
        self.dropped_constant_: List[str] = []
        self._fitted = False

    def fit(self, X: pd.DataFrame) -> "FrameCleaner":
        cols = list(X.columns)

        miss_rate = X.isna().mean()
        self.dropped_missing_ = [c for c in cols if miss_rate[c] > self.missing_threshold]
        keep = [c for c in cols if c not in self.dropped_missing_]

        sub = X[keep]
        nun = sub.nunique(dropna=True)
        self.dropped_constant_ = [c for c in keep if nun[c] <= 1]
        keep = [c for c in keep if c not in self.dropped_constant_]

        self.feature_names_ = keep
        self.medians_ = X[keep].median(numeric_only=True).astype(np.float32)
        self.medians_ = self.medians_.fillna(np.float32(0.0))
        self._fitted = True
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        if not self._fitted:
            raise RuntimeError("FrameCleaner.transform called before fit")

        missing = [c for c in self.feature_names_ if c not in X.columns]
        if missing:
            raise ValueError(
                f"{len(missing)} training features absent at transform time: "
                f"{missing[:8]}{'...' if len(missing) > 8 else ''}"
            )

        out = X.loc[:, self.feature_names_].copy()
        out = out.fillna(self.medians_)
        return out.astype(np.float32)

    def fit_transform(self, X: pd.DataFrame) -> pd.DataFrame:
        return self.fit(X).transform(X)


# ---------------------------------------------------------------------
# Split bundle
# ---------------------------------------------------------------------


@dataclass
class SplitBundle:
    """
    Explicit, typed container.  The previous code returned a 7-tuple whose
    element types changed depending on a boolean argument.

    Contract, enforced by ``validate()``:
      * X_* are float32 ndarrays, scaled, same column order
      * y_*_str are string label arrays (the single source of truth)
      * y_* are int64 arrays encoded with ``encoder``, or -1 for labels the
        encoder has never seen (unknown/novel classes in the test split)
    """

    X_train: np.ndarray
    X_val: np.ndarray
    X_test: np.ndarray

    y_train: np.ndarray
    y_val: np.ndarray
    y_test: np.ndarray

    y_train_str: np.ndarray
    y_val_str: np.ndarray
    y_test_str: np.ndarray

    encoder: LabelEncoder
    scaler: StandardScaler
    cleaner: FrameCleaner
    feature_names: List[str]
    protocol: str
    known_classes: List[str] = field(default_factory=list)

    @property
    def n_features(self) -> int:
        return int(self.X_train.shape[1])

    @property
    def n_classes(self) -> int:
        return int(len(self.encoder.classes_))

    def class_counts(self) -> pd.Series:
        return pd.Series(self.y_train_str).value_counts()

    def validate(self) -> None:
        for name in ("X_train", "X_val", "X_test"):
            arr = getattr(self, name)
            assert arr.dtype == np.float32, f"{name} dtype {arr.dtype} != float32"
            assert np.isfinite(arr).all(), f"{name} contains non-finite values"
            assert arr.shape[1] == len(self.feature_names), f"{name} width mismatch"

        assert len(self.X_train) == len(self.y_train) == len(self.y_train_str)
        assert len(self.X_val) == len(self.y_val) == len(self.y_val_str)
        assert len(self.X_test) == len(self.y_test) == len(self.y_test_str)

        # train and val must contain only known classes
        assert (self.y_train >= 0).all(), "unencoded label in y_train"
        assert (self.y_val >= 0).all(), "unencoded label in y_val"

        # scaler must have been fitted on the training width
        assert self.scaler.n_features_in_ == len(self.feature_names)

    def summary(self) -> str:
        n_unknown = int((self.y_test < 0).sum())
        lines = [
            "",
            "========== SPLIT SUMMARY ==========",
            f"Protocol      : {self.protocol}",
            f"Features      : {self.n_features}",
            f"Known classes : {self.n_classes}  {list(self.encoder.classes_)}",
            f"Train         : {self.X_train.shape[0]:,}",
            f"Val           : {self.X_val.shape[0]:,}",
            f"Test          : {self.X_test.shape[0]:,}",
            f"Test unknown  : {n_unknown:,} ({n_unknown / max(len(self.y_test),1):.2%})",
        ]
        if n_unknown:
            novel = sorted(set(self.y_test_str[self.y_test < 0]))
            lines.append(f"Novel classes : {novel}")
        lines.append("===================================")
        return "\n".join(lines)


# ---------------------------------------------------------------------
# Cache
# ---------------------------------------------------------------------



def build_cache(
    folder_path: os.PathLike | str = DATA_DIR,
    out_path: os.PathLike | str = CLEAN_PARQUET,
    day_files: Optional[Dict[str, List[str]]] = None,
    verbose: bool = True,
    resume: bool = True,
    max_captures: Optional[int] = None,
) -> Optional[Path]:
    """Parse the manifest ONE CAPTURE AT A TIME into a float32 Parquet cache.

    WHY THIS IS STREAMED
    --------------------
    The previous implementation was::

        df = load_all_datasets(...)   # all 8 frames held at once, float64
        df = structural_clean(df)     # pd.concat -> copy 2;  .copy() -> copy 3
        df.to_parquet(...)

    pandas reads CIC-IDS2017's numerics as float64, so 2.8M x 78 x 8 B is
    ~1.75 GB before anything else happens; the concat and the defensive
    ``.copy()`` each add a full duplicate.  Peak was roughly 4x the dataset
    and the job was OOM-killed on a 4 GB box holding 863 MB of CSV.

    Three candidate fixes, NOT equally important:

      * per-capture processing -- changes the ORDER of the requirement, from
        O(whole dataset) to O(largest capture).  This is the fix.
      * appending instead of concatenating -- saves nothing on its own.  It is
        the ENABLER: without it you would still hold every frame at the end.
      * float32 at read time -- a 2x constant factor.  Cannot rescue an
        O(whole dataset) design.  Applied at the cleaning step instead,
        because CIC-IDS2017 ships literal "Infinity" strings that a ``dtype=``
        on read_csv cannot parse.

    WHY IT IS RESUMABLE
    -------------------
    A 15-minute job that dies at capture 7 of 8 and restarts from zero is a
    job nobody runs twice.  Each capture is written as its own part under
    ``<cache>_parts/`` and recorded in ``_state.json``; re-running skips what
    is already done.  ``max_captures`` processes a bounded number and returns
    ``None`` to say "not finished yet" -- so the work can be driven in short
    slices by a caller with a timeout.

    THE CACHE DOES NOT DE-DUPLICATE.  DELIBERATELY.
    ------------------------------------------------
    It used to.  That was an architectural error: the cache is shared by both
    protocols, but the justification for de-duplication is protocol-DEPENDENT.

      * ``closedset`` is a stratified RANDOM split, so a flow appearing
        identically on Tuesday and Wednesday really can land on both sides.
        That is contamination and de-duplication is the right fix.
      * ``crossday`` is a TEMPORAL split.  Train is Mon-Wed, test is Friday;
        they cannot overlap, because the day defines the separation, not row
        identity.  A Friday flow identical to a Monday flow is not a leaked
        row -- it is the same traffic pattern recurring, which is exactly what
        happens in deployment.

    Applying it to both cost 22.3% of all rows and, specifically:

        PortScan   158,930  ->  1,851     (98.8% of the class deleted)

    A port scan is the same flow repeated against different ports: identical
    duration, packet counts, lengths and flags.  Once ``Destination Port`` is
    dropped as a shortcut feature, the flows are genuinely identical (measured:
    1,958 unique without the port, 90,819 with it), so de-duplication removes
    the class.  Two individually correct decisions, catastrophic in
    composition.

    Worse, ``keep="first"`` runs in manifest order, so Monday kept everything
    and Friday lost 72,232 benign flows -- making the TEST set's composition a
    function of the TRAINING set.  You may de-duplicate a training set.  You
    may never de-duplicate a test set.

    De-duplication now lives in ``build_splits``, applied per protocol and
    never to validation or test.
    """
    import json

    import numpy as _np
    import pyarrow as pa
    import pyarrow.parquet as pq

    folder = Path(folder_path)
    manifest = day_files or DAY_FILES
    out = Path(out_path)
    out.parent.mkdir(parents=True, exist_ok=True)

    parts_dir = out.parent / f"{out.stem}_parts"
    parts_dir.mkdir(parents=True, exist_ok=True)
    state_path = parts_dir / "_state.json"

    state = json.loads(state_path.read_text()) if (resume and state_path.exists()) else {}
    done: List[str] = list(state.get("done", []))
    canonical: Optional[List[str]] = state.get("canonical")
    label_counts: Dict[str, int] = {k: int(v) for k, v in state.get("label_counts", {}).items()}
    n_read = int(state.get("n_read", 0))


    todo = [(d, f) for d, files in manifest.items() for f in files if f not in done]
    if verbose:
        print("\n===== BUILDING CACHE (streaming, resumable) =====")
        print(f"  already done : {len(done)}    remaining : {len(todo)}")

    processed = 0
    for day, fname in todo:
        if max_captures is not None and processed >= max_captures:
            break
        path = folder / fname
        if not path.exists():
            raise FileNotFoundError(
                f"Manifest file missing: {path}\n"
                f"Fix config.DAY_FILES or place the capture in data/."
            )

        df = pd.read_csv(path, low_memory=False)
        df.columns = df.columns.str.strip()
        if LABEL_COL not in df.columns:
            raise ValueError(f"{path} has no '{LABEL_COL}' column")
        df[LABEL_COL] = _normalise_labels(df[LABEL_COL])
        df[DAY_COL] = day
        n_read += len(df)

        df = _clean_frame(df)

        # SCHEMA GUARD.  The captures are not all the same shape:
        # Friday-...-DDos.csv ships 85 columns (GeneratedLabelledFlows
        # distribution) while the other seven ship 79.  Harmless *today* only
        # because all six extras are identifiers that _clean_frame drops.  Add
        # one entry to IDENTIFIER_COLUMNS and it silently stops being
        # harmless, so assert the invariant rather than rely on luck.
        cols = [c for c in df.columns if c != DAY_COL]
        if canonical is None:
            canonical = cols
        elif set(cols) != set(canonical):
            raise ValueError(
                f"{fname} has a different schema after cleaning.\n"
                f"  only in this file : {sorted(set(cols) - set(canonical))}\n"
                f"  only in the others: {sorted(set(canonical) - set(cols))}"
            )
        df = df.reindex(columns=canonical + [DAY_COL])

        for lbl, cnt in df[LABEL_COL].value_counts().items():
            label_counts[str(lbl)] = label_counts.get(str(lbl), 0) + int(cnt)

        part = parts_dir / f"{len(done):02d}_{Path(fname).stem}.parquet"
        pq.write_table(pa.Table.from_pandas(df, preserve_index=False), part, compression="zstd")

        done.append(fname)
        processed += 1
        if verbose:
            print(f"  {day:<10s} {fname[:48]:<50s} kept {len(df):>9,}")

        # checkpoint AFTER the part is on disk, so a crash never records
        # work that was not persisted
        state_path.write_text(json.dumps({
            "done": done, "canonical": canonical, "label_counts": label_counts,
            "n_read": n_read,
        }))
        del df

    remaining = [f for d, files in manifest.items() for f in files if f not in done]
    if remaining:
        if verbose:
            print(f"\n  PAUSED -- {len(remaining)} capture(s) still to do. Re-run to continue.")
        return None

    # ---- all captures done: stream the parts into one file, in order ----
    writer = None
    n_kept = 0
    try:
        for part in sorted(parts_dir.glob("*.parquet")):
            table = pq.read_table(part)
            if writer is None:
                writer = pq.ParquetWriter(out, table.schema, compression="zstd")
            writer.write_table(table)
            n_kept += table.num_rows
            del table
    finally:
        if writer is not None:
            writer.close()

    if verbose:
        mb = out.stat().st_size / 1e6
        print("\n----------------------------------------")
        print(f"Rows read     : {n_read:,}")
        print(f"Rows written  : {n_kept:,}")
        print(f"Features      : {len(canonical) - 1}")
        print(f"Cache         : {out}  ({mb:.1f} MB)")
        print("\n---------- LABEL DISTRIBUTION ----------")
        for lbl, cnt in sorted(label_counts.items(), key=lambda kv: -kv[1]):
            print(f"  {lbl:<26s} {cnt:>9,}")
        print("----------------------------------------")

    return out


def load_clean(
    cache_path: os.PathLike | str = CLEAN_PARQUET,
    folder_path: os.PathLike | str = DATA_DIR,
    rebuild: bool = False,
    verbose: bool = True,
    days: Optional[Sequence[str]] = None,
    columns: Optional[Sequence[str]] = None,
) -> pd.DataFrame:
    """Load the cached clean frame, optionally only certain days.

    MEMORY.  The previous body was one line -- ``pd.read_parquet(cache)`` --
    and it was OOM-killed in 3.5 seconds at 3.8 GB.  Three reasons, all of
    which this version fixes:

      1. It materialised the whole week (2.83M x 72) when every caller
         immediately partitions it by day.  ``days=`` pushes that filter down
         into the Parquet reader, so only the requested captures are read.
      2. Arrow's table and the pandas frame exist simultaneously during
         ``to_pandas``, so peak is 2x the final frame.  ``self_destruct=True``
         releases each Arrow column as it is converted.
      3. ``Label`` and ``Day`` came back as object dtype -- 2.83M individual
         Python strings, which cost more than all 70 float32 features
         combined.  They are dictionary-encoded (pandas ``category``) instead.

    This is the same defect ``build_cache`` had, one level up: fixing a hot
    spot is not the same as fixing a design.  The rule both now follow is
    "never materialise more than the caller asked for".
    """
    import pyarrow.dataset as pads

    cache = Path(cache_path)
    parts_dir = cache.parent / f"{cache.stem}_parts"

    if rebuild or not (cache.exists() or any(parts_dir.glob("*.parquet"))):
        build_cache(folder_path, cache, verbose=verbose)

    # Prefer the partitioned parts: one file per capture means the reader can
    # skip whole files rather than filtering rows it already decoded.
    source = parts_dir if any(parts_dir.glob("*.parquet")) else cache
    dataset = pads.dataset(source, format="parquet")

    flt = pads.field(DAY_COL).isin(list(days)) if days else None
    table = dataset.to_table(columns=list(columns) if columns else None, filter=flt)

    df = table.to_pandas(self_destruct=True, split_blocks=True, types_mapper=None)
    del table

    for c in (LABEL_COL, DAY_COL):
        # Same pandas 3.0 concern: under the new string dtype this test would
        # not fire, the dictionary encoding would be skipped, and the frame
        # would silently cost hundreds of MB more than it should.
        if c in df.columns and not pd.api.types.is_numeric_dtype(df[c]):
            if str(df[c].dtype) != "category":
                df[c] = df[c].astype("category")

    if verbose:
        mb = df.memory_usage(deep=True).sum() / 1e6
        scope = f" days={list(days)}" if days else ""
        print(f"Loaded clean frame: {df.shape}{scope}  ({mb:.0f} MB in memory)")
    return df


def _encode(encoder: LabelEncoder, y_str: np.ndarray) -> np.ndarray:
    """Encode, mapping unseen labels to -1 instead of raising."""
    lookup = {c: i for i, c in enumerate(encoder.classes_)}
    return np.fromiter((lookup.get(v, -1) for v in y_str), dtype=np.int64, count=len(y_str))


def _frame_to_matrix(df: pd.DataFrame, cleaner: "FrameCleaner") -> np.ndarray:
    """Materialise ONE split as a float32 matrix, one column at a time.

    Replaces ``cleaner.transform(df[feat]).to_numpy(dtype=np.float32)``, which
    allocated a whole second DataFrame and then a whole third array.  Here the
    destination is allocated once and each column is copied and imputed
    straight into it, so peak is the source frame plus the output -- not four
    copies of the data.
    """
    names = list(cleaner.feature_names_)
    out = np.empty((len(df), len(names)), dtype=np.float32)
    for j, c in enumerate(names):
        col = df[c].to_numpy(dtype=np.float32, copy=False)
        np.copyto(out[:, j], col)
        bad = ~np.isfinite(out[:, j])
        if bad.any():
            out[bad, j] = np.float32(cleaner.medians_[c])   # TRAINING median
    return out


def _labels_of(df: pd.DataFrame) -> np.ndarray:
    """Label column as an object array of SHARED string objects.

    ``.to_numpy(dtype=object).astype(str)`` -- the previous form -- produces a
    numpy ``<U22`` array: 88 bytes per row, ~250 MB across the week.  Going via
    ``astype(str)`` on the categorical keeps one Python string per class and an
    array of pointers: ~8 bytes per row.
    """
    return df[LABEL_COL].astype(str).to_numpy(dtype=object)


def _finalise(
    frames: "List[Optional[pd.DataFrame]]",
    protocol: str,
    verbose: bool = True,
) -> SplitBundle:
    """Shared tail: clean -> encode -> scale.  Everything fitted on train.

    MEMORY.  ``frames`` is a MUTABLE list ``[train, val, test]`` and each slot
    is set to ``None`` the moment that split has been converted to a matrix.
    That is not a style choice: the caller must not keep its own names bound to
    these frames, or the ``del`` here frees nothing and the process is
    OOM-killed at ~3.8 GB -- which is exactly what happened.

    The old body held, simultaneously: three source frames, three cleaned
    DataFrames, three ``to_numpy`` copies and three scaled copies.  On 793 MB
    of features that is over 3 GB before overhead.
    """
    import gc

    train_df = frames[0]
    assert train_df is not None
    feat = [c for c in train_df.columns if c not in (LABEL_COL, DAY_COL)]

    cleaner = FrameCleaner().fit(train_df[feat])
    feature_names = list(cleaner.feature_names_)

    # Labels first: small, and needed after the frames are released.
    ytr_s = _labels_of(frames[0])
    yva_s = _labels_of(frames[1])
    yte_s = _labels_of(frames[2])

    # Encoder is fitted on TRAINING labels only.
    encoder = LabelEncoder().fit(ytr_s)
    known = list(encoder.classes_)

    # Scale in place.  copy=False makes StandardScaler write back into the
    # array we already allocated instead of returning a fresh one.
    scaler = StandardScaler(copy=False)

    Xtr_a = _frame_to_matrix(frames[0], cleaner)
    frames[0] = None
    gc.collect()
    scaler.fit(Xtr_a)
    Xtr_a = scaler.transform(Xtr_a)

    Xva_a = _frame_to_matrix(frames[1], cleaner)
    frames[1] = None
    gc.collect()
    Xva_a = scaler.transform(Xva_a)

    Xte_a = _frame_to_matrix(frames[2], cleaner)
    frames[2] = None
    gc.collect()
    Xte_a = scaler.transform(Xte_a)

    unseen_val = sorted(set(yva_s) - set(known))
    if unseen_val:
        if verbose:
            print(
                f"\n[warn] {len(unseen_val)} validation classes absent from training: "
                f"{unseen_val}\n       Their rows are dropped from validation so that "
                "model selection stays closed-set."
            )
        keep = np.isin(yva_s, known)
        Xva_a = Xva_a[keep]
        yva_s = yva_s[keep]

    ytr = _encode(encoder, ytr_s)
    yva = _encode(encoder, yva_s)
    yte = _encode(encoder, yte_s)   # -1 marks a novel class

    # StandardScaler divides by a std that can be ~0 for near-constant columns,
    # producing huge magnitudes.  Clip to a sane z-range; keeps AMP stable.
    for a in (Xtr_a, Xva_a, Xte_a):
        np.nan_to_num(a, copy=False, nan=0.0, posinf=0.0, neginf=0.0)
        np.clip(a, -10.0, 10.0, out=a)

    bundle = SplitBundle(
        X_train=Xtr_a, X_val=Xva_a, X_test=Xte_a,
        y_train=ytr, y_val=yva, y_test=yte,
        y_train_str=ytr_s, y_val_str=yva_s, y_test_str=yte_s,
        encoder=encoder, scaler=scaler, cleaner=cleaner,
        feature_names=feature_names, protocol=protocol,
        known_classes=known,
    )
    bundle.validate()
    if verbose:
        print(bundle.summary())
    return bundle


def _dedup_split(df: pd.DataFrame, what: str, verbose: bool = True) -> pd.DataFrame:
    """De-duplicate ONE split on feature columns, keep="first".

    Never call this on validation or test.  De-duplicating a held-out set
    changes its composition as a function of what happened to be in training,
    and removes traffic the model will certainly meet in deployment.
    """
    feat = [c for c in df.columns if c not in (LABEL_COL, DAY_COL)]
    before = len(df)
    out = df.drop_duplicates(subset=feat, keep="first")
    removed = before - len(out)
    if verbose and removed:
        print(f"[dedup] {what}: removed {removed:,} duplicate flows "
              f"({removed / max(before, 1):.2%}), {len(out):,} remain")
    return out


def build_splits(
    protocol: str = "crossday",
    df: Optional[pd.DataFrame] = None,
    seed: int = SEED,
    verbose: bool = True,
    closedset_sizes: Tuple[float, float] = (0.15, 0.15),
    thursday_val_frac: Optional[float] = None,
) -> SplitBundle:
    """
    protocol="crossday"  -- Mon/Tue/Wed (+ part of Thu) train, Thu val, Fri test.
                            Friday carries DDoS / PortScan / Bot, which are
                            absent from training, so y_test contains -1 for
                            those rows.  This is the open-set protocol.

    protocol="closedset" -- known classes only (those present Mon-Thu),
                            de-duplicated, stratified train/val/test.
                            Optimistic relative to the temporal protocol;
                            used only for the architecture comparison table.
    """
    if protocol == "crossday":
        # Load each day-group separately.  The union of the three is never
        # held at once, which is what keeps this inside 4 GB.
        if df is None:
            base = load_clean(days=TRAIN_DAYS, verbose=verbose)
            thu = load_clean(days=VAL_DAYS, verbose=verbose)
            te = load_clean(days=TEST_DAYS, verbose=verbose)
        else:
            base = df[df[DAY_COL].isin(TRAIN_DAYS)]
            thu = df[df[DAY_COL].isin(VAL_DAYS)]
            te = df[df[DAY_COL].isin(TEST_DAYS)]
        te_ = te

        for name, part, days in (("train", base, TRAIN_DAYS),
                                 ("val", thu, VAL_DAYS),
                                 ("test", te, TEST_DAYS)):
            if len(part) == 0:
                raise ValueError(f"{name} split is empty for days {days}")

        frac = float(thursday_val_frac if thursday_val_frac is not None else THURSDAY_VAL_FRAC)
        if not 0.0 < frac <= 1.0:
            raise ValueError("thursday_val_frac must be in (0, 1]")

        if frac >= 1.0:
            tr, va = base, thu
            tr = _dedup_split(tr, "train", verbose=verbose)
            if verbose:
                print(
                    "\n[warn] THURSDAY_VAL_FRAC = 1.0: the whole of Thursday is held out.\n"
                    "       Thursday is the only day with Infiltration and the Web attacks,\n"
                    "       so validation will contain classes absent from training; those\n"
                    "       rows are dropped and validation may collapse to BENIGN only.\n"
                    "       Model selection then selects on noise. Prefer frac < 1."
                )
        else:
            # stratified: half of Thursday joins training so that validation
            # covers every known class, half is held out for model selection
            y_thu = thu[LABEL_COL].to_numpy()
            counts = pd.Series(y_thu).value_counts()
            stratify = y_thu if (counts >= 2).all() else None
            if stratify is None and verbose:
                print("[warn] a Thursday class has <2 samples; falling back to a "
                      "non-stratified Thursday split")
            thu_tr, thu_va = train_test_split(
                np.arange(len(thu)), test_size=frac,
                random_state=seed, stratify=stratify,
            )
            for _f in (base, thu):
                if str(_f[DAY_COL].dtype) == "category":
                    _f[DAY_COL] = _f[DAY_COL].astype(str)
            tr = pd.concat([base, thu.iloc[thu_tr]], ignore_index=True)
            va = thu.iloc[thu_va]

            # TRAIN ONLY.  Repeated flows bias the training distribution, so
            # collapsing them there is defensible.  Validation and test are
            # left exactly as the capture recorded them.
            tr = _dedup_split(tr, "train", verbose=verbose)
            if verbose:
                print(
                    f"\n[protocol] train = {TRAIN_DAYS} + {1 - frac:.0%} of Thursday "
                    f"({len(thu_tr):,} rows)\n"
                    f"           val   = {frac:.0%} of Thursday ({len(thu_va):,} rows)\n"
                    f"           test  = {TEST_DAYS} ({len(te):,} rows, never seen)"
                )

        frames = [tr, va, te]
        del tr, va, te, base, thu, te_  # release every other reference
        return _finalise(frames, "crossday", verbose=verbose)

    if protocol == "closedset":
        known_days = TRAIN_DAYS + VAL_DAYS
        sub = (load_clean(days=known_days, verbose=verbose) if df is None
               else df[df[DAY_COL].isin(known_days)].copy())

        # RANDOM split: an identical flow really can land on both sides, so
        # here de-duplication before splitting is the correct fix.
        sub = _dedup_split(sub, "closedset pool", verbose=verbose)

        # classes with too few samples cannot be stratified three ways
        counts = sub[LABEL_COL].value_counts()
        too_rare = counts[counts < 10].index.tolist()
        if too_rare:
            if verbose:
                print(f"[warn] dropping classes with <10 samples: {dict(counts[too_rare])}")
            sub = sub[~sub[LABEL_COL].isin(too_rare)]

        val_frac, test_frac = closedset_sizes
        y = sub[LABEL_COL].to_numpy()
        idx = np.arange(len(sub))

        idx_tr, idx_hold = train_test_split(
            idx, test_size=val_frac + test_frac, random_state=seed, stratify=y
        )
        rel = test_frac / (val_frac + test_frac)
        idx_va, idx_te = train_test_split(
            idx_hold, test_size=rel, random_state=seed, stratify=y[idx_hold]
        )

        if verbose:
            print(
                "\n[note] closedset protocol is a stratified split of de-duplicated\n"
                "       Mon-Thu traffic.  It is OPTIMISTIC relative to the temporal\n"
                "       protocol because bursty flows remain temporally adjacent.\n"
                "       Report it as an architecture comparison, not as a detection result."
            )

        frames = [sub.iloc[idx_tr], sub.iloc[idx_va], sub.iloc[idx_te]]
        del sub
        return _finalise(frames, "closedset", verbose=verbose)

    raise ValueError(f"unknown protocol {protocol!r}; expected 'crossday' or 'closedset'")


# ---------------------------------------------------------------------
# Helpers used by the open-set / one-class paths
# ---------------------------------------------------------------------


def benign_index(encoder: LabelEncoder) -> int:
    """
    Index of BENIGN.  The old code assumed ``y_train == 0`` because
    LabelEncoder happens to sort "BENIGN" first.  Unchecked invariant.
    """
    classes = list(encoder.classes_)
    if BENIGN_LABEL not in classes:
        raise ValueError(f"{BENIGN_LABEL!r} not among encoder classes {classes}")
    return classes.index(BENIGN_LABEL)


def open_set_truth(y_str: np.ndarray, known_classes: Sequence[str]) -> np.ndarray:
    """Map any label outside `known_classes` to UNKNOWN_LABEL. dtype=object."""
    known = set(known_classes)
    return np.array(
        [lbl if lbl in known else UNKNOWN_LABEL for lbl in y_str], dtype=object
    )


def to_binary(y_str: np.ndarray) -> np.ndarray:
    return np.where(y_str == BENIGN_LABEL, BENIGN_LABEL, "ATTACK").astype(object)


# ---------------------------------------------------------------------
# Backwards-compatible shims (deprecated)
# ---------------------------------------------------------------------


def prepare_data(*args, **kwargs):  # pragma: no cover - compatibility only
    warnings.warn(
        "prepare_data() is deprecated; use build_splits(protocol=...). "
        "The old function fitted imputation on train+test and returned a "
        "7-tuple whose types depended on split_by_day.",
        DeprecationWarning,
        stacklevel=2,
    )
    protocol = "crossday" if kwargs.get("split_by_day") else "closedset"
    return build_splits(protocol=protocol)
