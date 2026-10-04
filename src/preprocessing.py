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
    BLOCKED_FRACS,
    CACHE_VERSION,
    CLEAN_PARQUET,
    DATA_DIR,
    DAY_COL,
    DAY_FILES,
    DAY_ORDER,
    DUPLICATE_FEATURES,
    IDENTIFIER_COLUMNS,
    IDENTIFIER_COVERAGE,
    LABEL_ALIASES,
    LABEL_COL,
    META_DST_IP,
    META_SRC_IP,
    META_TS,
    SEED,
    TEST_DAYS,
    THURSDAY_VAL_FRAC,
    TRAIN_DAYS,
    UNKNOWN_LABEL,
    VAL_DAYS,
)
from meta import (
    extract_meta,
    feature_columns,
    meta_columns_present,
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


def read_capture(path: os.PathLike | str) -> pd.DataFrame:
    """Read one capture CSV exactly as shipped, header whitespace stripped.

    Two properties of the GeneratedLabelledFlows distribution break a plain
    ``pd.read_csv(path)``, and neither raises a helpful error:

    * ``Thursday-...-WebAttacks`` writes its label dash as the single byte
      0x96 (Windows-1252).  That is not valid UTF-8, so the read dies with a
      UnicodeDecodeError several hundred thousand rows in.  The
      MachineLearningCSV copy of the same file carries U+FFFD instead and
      reads fine, which is why this never showed up before.
    * the same file ends in ~288,000 rows that are empty in every column.
      ``astype(str)`` turns their missing label into the string "nan", which
      then survives ``dropna`` and becomes a thirteenth class.

    So: fall back to cp1252 when UTF-8 fails, and drop rows with no label
    here, before anything can stringify them.
    """
    path = Path(path)
    encoding = "utf-8"
    try:
        df = pd.read_csv(path, low_memory=False)
    except UnicodeDecodeError:
        encoding = "cp1252"
        df = pd.read_csv(path, low_memory=False, encoding=encoding)
    df.columns = df.columns.str.strip()
    df.attrs["encoding"] = encoding
    df.attrs["raw_columns"] = int(df.shape[1])
    df.attrs["raw_rows"] = int(len(df))
    if LABEL_COL not in df.columns:
        raise ValueError(f"{path} has no '{LABEL_COL}' column")

    # position in the file as shipped -- recorded before any row is dropped,
    # so it stays a faithful proxy for arrival order
    df["__file_row"] = np.arange(len(df), dtype=np.int64)

    lab = df[LABEL_COL]
    blank = lab.isna() | lab.astype(str).str.strip().str.lower().isin(["", "nan", "none"])
    if blank.any():
        attrs = dict(df.attrs)
        df = df[~blank].copy()
        df.attrs.update(attrs)
    return df


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
            df = read_capture(path).drop(columns="__file_row")
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



def _clean_frame(
    df: pd.DataFrame,
    capture_index: Optional[int] = None,
    day: Optional[str] = None,
) -> pd.DataFrame:
    """Per-capture structural cleaning.

    Learns NOTHING from the data distribution, so it is safe to apply one
    capture at a time -- which is exactly what lets ``build_cache`` stream
    instead of concatenating the whole week into memory first.

    With ``capture_index`` given, the identifiers are copied into ``meta__*``
    columns before they are dropped from the features (see ``meta.py``).
    Without it the frame comes back exactly as before: features, label, day.
    """
    df = df.copy()

    file_row = df.pop("__file_row").to_numpy() if "__file_row" in df.columns else None

    if "Flow Duration" in df.columns:
        keep = (pd.to_numeric(df["Flow Duration"], errors="coerce") >= 0).to_numpy()
        df = df[keep].copy()
        if file_row is not None:
            file_row = file_row[keep]

    side = None
    if capture_index is not None:
        side = extract_meta(df, capture_index, day=day, row_index=file_row)

    # drop identifiers (incl. Destination Port -- deliberate anti-shortcut)
    df = df.drop(columns=IDENTIFIER_COLUMNS, errors="ignore")

    # drop exactly-collinear CICFlowMeter twins
    df = df.drop(columns=[c for c in DUPLICATE_FEATURES if c in df.columns])

    feature_cols = feature_columns(df)
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

    if side is not None:
        # same index as df, so the join cannot misalign a flow with another
        # flow's address or timestamp
        df = pd.concat([df, side.loc[df.index]], axis=1)

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
    feature_cols = feature_columns(df)

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

    # Identifier side-table, row-aligned with X_* / y_*.  None when the frame
    # the splits were built from carried no ``meta__`` columns.  These are
    # NEVER features: nothing in X_* is derived from them.
    meta_train: Optional[pd.DataFrame] = None
    meta_val: Optional[pd.DataFrame] = None
    meta_test: Optional[pd.DataFrame] = None

    # How the split was made: validation mode, held-out class, rows removed by
    # de-duplication.  Written into every run's results so a number can be
    # traced to the split that produced it.
    info: Dict[str, object] = field(default_factory=dict)

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

        for name, X in (("meta_train", self.X_train), ("meta_val", self.X_val),
                        ("meta_test", self.X_test)):
            m = getattr(self, name)
            assert m is None or len(m) == len(X), f"{name} is not row-aligned"
        # the side-table must not have leaked into the feature matrix
        assert not meta_columns_present(self.feature_names), "meta column used as a feature"

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

    import pyarrow.parquet as pq

    folder = Path(folder_path)
    manifest = day_files or DAY_FILES
    out = Path(out_path)
    out.parent.mkdir(parents=True, exist_ok=True)

    parts_dir = out.parent / f"{out.stem}_parts"
    parts_dir.mkdir(parents=True, exist_ok=True)
    state_path = parts_dir / "_state.json"

    on_disk = json.loads(state_path.read_text()) if state_path.exists() else {}
    state = on_disk if resume else {}

    # WHEN THE PARTS ON DISK MUST GO.  Three cases, each stated when it happens:
    #
    #   * the caller asked for a rebuild (``resume=False``);
    #   * CACHE VERSION: parts written before the meta side-table existed have
    #     no meta__ columns.  Resuming on top of them would produce a cache
    #     whose first captures lack the columns the last ones carry -- or,
    #     worse, a complete old cache that the study reads as "this dataset has
    #     no timestamps";
    #   * the CSVs changed since the parts were written (see ``sources_changed``).
    stale = [p for p in parts_dir.glob("*.parquet")]
    changed = sources_changed(on_disk, folder)
    if not resume:
        reason = "rebuild requested"
    elif int(on_disk.get("version", 1)) != CACHE_VERSION:
        reason = f"layout version {on_disk.get('version', 1)} != {CACHE_VERSION}"
    elif changed:
        reason = f"the CSV changed since it was cached: {', '.join(changed)}"
    else:
        reason = None
    if reason and (on_disk or stale or out.exists()):
        # LOOK BEFORE DELETING.  The old cache is about to be thrown away so
        # that it can be rebuilt from the CSVs.  If the CSVs are not all there
        # the rebuild cannot finish, and deleting first would leave a machine
        # that had a working (if outdated) cache with nothing at all.  So:
        # check first, and if a capture is missing, stop without touching
        # anything.
        missing = [f for _d, files in manifest.items() for f in files
                   if not (folder / f).exists()]
        if missing:
            raise FileNotFoundError(
                f"The cache has to be rebuilt ({reason}), but {len(missing)} of the "
                f"{sum(len(v) for v in manifest.values())} CSVs it would be rebuilt from are "
                f"not in {folder}:\n  " + "\n  ".join(missing)
                + "\nNothing was deleted: the existing cache is still on disk.\n"
                  "Put the captures in data/ (see data/Readme.md) and run the command again."
            )
        if verbose:
            print(f"\n[cache] {reason}; discarding {len(stale)} old part(s) and "
                  "rebuilding from the CSVs.")
        for p in stale:
            p.unlink()
        if out.exists():
            out.unlink()
        if state_path.exists():
            state_path.unlink()
        state = {}

    done: List[str] = list(state.get("done", []))
    canonical: Optional[List[str]] = state.get("canonical")
    label_counts: Dict[str, int] = {k: int(v) for k, v in state.get("label_counts", {}).items()}
    n_read = int(state.get("n_read", 0))
    identifiers: Dict[str, bool] = dict(state.get("identifiers", {}))
    sources: Dict[str, int] = {k: int(v) for k, v in state.get("sources", {}).items()}

    capture_order = [f for _d, files in manifest.items() for f in files]
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

        sources[fname] = int(path.stat().st_size)
        df = read_capture(path)
        df[LABEL_COL] = _normalise_labels(df[LABEL_COL])
        df[DAY_COL] = day
        n_read += len(df)

        df = _clean_frame(df, capture_index=capture_order.index(fname), day=day)
        identifiers[fname] = bool(
            len(df)
            and df[META_SRC_IP].notna().mean() >= IDENTIFIER_COVERAGE
            and np.isfinite(df[META_TS].to_numpy(dtype=np.float64)).mean() >= IDENTIFIER_COVERAGE
        )

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

        part = parts_dir / f"{capture_order.index(fname):02d}_{Path(fname).stem}.parquet"
        pq.write_table(_to_arrow(df), part, compression="zstd")

        done.append(fname)
        processed += 1
        if verbose:
            ids = "src IP + time" if identifiers[fname] else "NO src IP/time"
            print(f"  {day:<10s} {fname[:48]:<50s} kept {len(df):>9,}   [{ids}]")

        # checkpoint AFTER the part is on disk, so a crash never records
        # work that was not persisted
        state_path.write_text(json.dumps({
            "version": CACHE_VERSION,
            "done": done, "canonical": canonical, "label_counts": label_counts,
            "n_read": n_read, "identifiers": identifiers, "sources": sources,
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
        print(f"Features      : {len(feature_columns(canonical))}")
        print(f"Cache         : {out}  ({mb:.1f} MB)")
        n_ids = sum(bool(v) for v in identifiers.values())
        print(f"Identifiers   : {n_ids} of {len(identifiers)} captures carry "
              f"source IP + timestamp")
        if n_ids < len(identifiers):
            print("                (behaviour layer and timestamp ordering need all of "
                  "them -- see data/Readme.md)")
        print("\n---------- LABEL DISTRIBUTION ----------")
        for lbl, cnt in sorted(label_counts.items(), key=lambda kv: -kv[1]):
            print(f"  {lbl:<26s} {cnt:>9,}")
        print("----------------------------------------")

    return out


def _to_arrow(df: pd.DataFrame):
    """DataFrame -> Arrow table with EXPLICIT string types.

    Left to inference, a capture with no source IP yields an all-null column
    that Arrow types as ``null``, while its neighbour's is ``string``.  The
    parts then refuse to merge, and which part is read first decides the
    dataset schema.  Naming the type makes every part identical.
    """
    import pyarrow as pa

    arrays, names = [], []
    for c in df.columns:
        col = df[c]
        if c in (META_SRC_IP, META_DST_IP, LABEL_COL, DAY_COL):
            vals = col.astype("object").where(col.notna(), None).to_numpy(dtype=object)
            arrays.append(pa.array(vals, type=pa.string(), from_pandas=True))
        else:
            arrays.append(pa.array(col.to_numpy(), from_pandas=True))
        names.append(c)
    return pa.Table.from_arrays(arrays, names=names)


def cache_state(cache_path: os.PathLike | str = CLEAN_PARQUET) -> Dict:
    """The cache's own record of what it holds (empty dict if there is none)."""
    import json

    cache = Path(cache_path)
    state_path = cache.parent / f"{cache.stem}_parts" / "_state.json"
    return json.loads(state_path.read_text()) if state_path.exists() else {}


def sources_changed(state: Dict, folder_path: os.PathLike | str = DATA_DIR) -> List[str]:
    """Captures whose CSV is no longer the file the cache was built from.

    THE FAILURE THIS PREVENTS.  The cache outlives the CSVs it was parsed
    from.  Replace the eight files in ``data/`` with the other distribution of
    the dataset -- the one that carries source addresses and timestamps -- and
    the old cache still loads, still has every row, and still says "no
    identifiers".  Every later command then reports, correctly and uselessly,
    that the behaviour layer cannot run.  Nothing fails; the new files are
    simply never read.

    The check is the file SIZE recorded when each capture was parsed.  Size,
    not modification time: copying a project between machines changes every
    mtime and no content, and a rebuild that fires on that is a rebuild people
    learn to distrust.  The two distributions differ by tens of megabytes per
    file, so size is enough to tell them apart.

    A capture that is absent from ``folder_path`` is not "changed": a cache
    may legitimately be used on a machine that does not hold the CSVs.
    """
    folder = Path(folder_path)
    out = []
    for fname, size in (state.get("sources") or {}).items():
        path = folder / fname
        if path.exists() and int(path.stat().st_size) != int(size):
            out.append(fname)
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

    state = cache_state(cache)
    outdated = bool(state) and int(state.get("version", 1)) != CACHE_VERSION
    legacy = not state and (cache.exists() or any(parts_dir.glob("*.parquet")))
    changed = bool(sources_changed(state, folder_path))
    if (rebuild or outdated or legacy or changed
            or not (cache.exists() or any(parts_dir.glob("*.parquet")))):
        # build_cache prints which of these it was, even when verbose is off:
        # a cache silently rebuilt is a five-minute pause with no explanation
        build_cache(folder_path, cache, verbose=verbose or changed or outdated or legacy,
                    resume=not rebuild)

    # Prefer the partitioned parts: one file per capture means the reader can
    # skip whole files rather than filtering rows it already decoded.
    source = parts_dir if any(parts_dir.glob("*.parquet")) else cache
    dataset = pads.dataset(source, format="parquet")

    flt = pads.field(DAY_COL).isin(list(days)) if days else None
    table = dataset.to_table(columns=list(columns) if columns else None, filter=flt)

    df = table.to_pandas(self_destruct=True, split_blocks=True, types_mapper=None)
    del table

    for c in (LABEL_COL, DAY_COL, META_SRC_IP, META_DST_IP):
        # Same pandas 3.0 concern: under the new string dtype this test would
        # not fire, the dictionary encoding would be skipped, and the frame
        # would silently cost hundreds of MB more than it should.  The two
        # address columns are the same case: ~20k distinct strings repeated
        # 2.8M times.
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


def _meta_of(df: pd.DataFrame) -> Optional[pd.DataFrame]:
    """The ``meta__`` side-table of one split, re-indexed 0..n-1, or None."""
    cols = meta_columns_present(df)
    if not cols:
        return None
    return df[cols].reset_index(drop=True)


def _finalise(
    frames: "List[Optional[pd.DataFrame]]",
    protocol: str,
    verbose: bool = True,
    info: Optional[Dict[str, object]] = None,
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
    # feature_columns() excludes the label, the day and every meta__ column.
    # This one call is what keeps addresses, ports and timestamps out of X.
    feat = feature_columns(train_df)

    cleaner = FrameCleaner().fit(train_df[feat])
    feature_names = list(cleaner.feature_names_)

    # Labels first: small, and needed after the frames are released.
    ytr_s = _labels_of(frames[0])
    yva_s = _labels_of(frames[1])
    yte_s = _labels_of(frames[2])

    # Side-table: small, copied out before the frames go.
    metas = [_meta_of(frames[0]), _meta_of(frames[1]), _meta_of(frames[2])]

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
        if metas[1] is not None:
            metas[1] = metas[1][keep].reset_index(drop=True)

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
        meta_train=metas[0], meta_val=metas[1], meta_test=metas[2],
        info=dict(info or {}),
    )
    bundle.validate()
    if verbose:
        print(bundle.summary())
    return bundle


def _dedup_split(
    df: pd.DataFrame,
    what: str,
    verbose: bool = True,
    stats: Optional[Dict[str, object]] = None,
) -> pd.DataFrame:
    """De-duplicate ONE split on feature columns, keep="first".

    Never call this on validation or test.  De-duplicating a held-out set
    changes its composition as a function of what happened to be in training,
    and removes traffic the model will certainly meet in deployment.

    ``stats``, when given, receives the counts, so the number removed can be
    reported with the result it affected.
    """
    feat = feature_columns(df)
    before = len(df)
    out = df.drop_duplicates(subset=feat, keep="first")
    removed = before - len(out)
    if stats is not None:
        stats["dedup_split"] = what
        stats["rows_before_dedup"] = int(before)
        stats["rows_removed_by_dedup"] = int(removed)
    if verbose and removed:
        print(f"[dedup] {what}: removed {removed:,} duplicate flows "
              f"({removed / max(before, 1):.2%}), {len(out):,} remain")
    return out


def _group_rank(df: pd.DataFrame, by_capture: bool):
    """Rank of every row inside its (capture, class) -- or class -- group, in
    time order, and the size of that group.  Returns ``(rank, size)``."""
    from meta import capture_ids, time_position

    g = pd.DataFrame({
        "lab": df[LABEL_COL].astype(str).to_numpy(),
        "pos": time_position(df),
    })
    keys = ["lab"]
    if by_capture:
        g["cap"] = capture_ids(df)
        keys = ["cap", "lab"]
    grp = g.groupby(keys, sort=False)["pos"]
    rank = grp.rank(method="first").to_numpy().astype(np.int64) - 1
    size = grp.transform("size").to_numpy().astype(np.int64)
    return rank, size


def blocked_assignment(
    df: pd.DataFrame,
    fracs: Tuple[float, float, float] = BLOCKED_FRACS,
) -> np.ndarray:
    """0 = train, 1 = validation, 2 = test for every row of ``df``.

    Per (capture, class), in time order: the earliest ``fracs[0]`` of the
    flows train, the next ``fracs[1]`` validate, the rest test.

    WHY PER CLASS.  An attack in this dataset runs for twenty minutes to an
    hour inside an eight-hour capture.  Cutting each capture at 60% and 80% of
    its *duration* would put most attacks wholly inside one split, and the
    "seen" condition this protocol exists to measure would not be seen at all.
    Cutting each class's own flows keeps every class in all three splits while
    still never letting a later flow train a model that is tested on an
    earlier one of the same run.

    WHAT IT DOES NOT CLAIM.  Blocks are contiguous within a capture, not
    across the week: Friday morning's test block precedes Friday afternoon's
    training block.  This is an in-distribution split with the interleaving
    removed, not a forecast.  The cross-day protocol is the forecast.

    A group with fewer than three rows goes to training whole: it cannot be
    split three ways, and one row of a class in test is not a measurement.
    """
    f_tr, f_va, _f_te = fracs
    if not (0 < f_tr < 1 and 0 < f_va < 1 and f_tr + f_va < 1):
        raise ValueError(f"fracs must leave room for all three splits, got {fracs}")

    rank, size = _group_rank(df, by_capture=True)
    cut1 = np.floor(size * f_tr).astype(np.int64)
    cut2 = np.floor(size * (f_tr + f_va)).astype(np.int64)
    # every split non-empty whenever the group can afford it
    cut1 = np.clip(cut1, 1, np.maximum(size - 2, 1))
    cut2 = np.clip(cut2, cut1 + 1, np.maximum(size - 1, cut1 + 1))

    assign = np.where(rank < cut1, 0, np.where(rank < cut2, 1, 2)).astype(np.int8)
    assign[size < 3] = 0
    return assign


def build_splits(
    protocol: str = "crossday",
    df: Optional[pd.DataFrame] = None,
    seed: int = SEED,
    verbose: bool = True,
    closedset_sizes: Tuple[float, float] = (0.15, 0.15),
    thursday_val_frac: Optional[float] = None,
    val_mode: str = "random",
    holdout_class: Optional[str] = None,
    blocked_fracs: Tuple[float, float, float] = BLOCKED_FRACS,
    dedup_train: bool = True,
) -> SplitBundle:
    """
    ``dedup_train=False`` leaves the training split exactly as recorded.  It is
    for callers that need the RAW flows of the training days -- the
    leave-one-day-out calibration scores each day's benign traffic as it was
    captured, not what survives de-duplication -- and it has no effect on the
    ``closedset`` protocol, which de-duplicates its pool before splitting.

    protocol="crossday"  -- Mon/Tue/Wed (+ part of Thu) train, Thu val, Fri test.
                            Friday carries DDoS / PortScan / Bot, which are
                            absent from training, so y_test contains -1 for
                            those rows.  This is the open-set protocol.
                            ``val_mode="random"`` splits Thursday at random
                            (stratified); ``val_mode="tail"`` trains on the
                            EARLIER part of each Thursday class and validates
                            on the LATER part, so that validation benign flows
                            are not interleaved with training ones.

    protocol="closedset" -- known classes only (those present Mon-Thu),
                            de-duplicated, stratified train/val/test.
                            Optimistic relative to the temporal protocol;
                            used only for the architecture comparison table.

    protocol="random"    -- ALL five days pooled and split at random,
                            stratified 70/15/15; then the TRAINING split is
                            de-duplicated, as in every other study protocol.
                            This is the split most published CIC-IDS2017
                            numbers come from.  It leaks (flows of one burst,
                            and exact copies, land on both sides) and exists
                            here only to measure how much that leak is worth.
                            Validation and test are raw flows, so its rates
                            are per flow, like those of the other protocols.

    protocol="blocked"   -- all five days, time-blocked per (capture, class):
                            earliest 60% train, next 20% val, last 20% test.
                            Every class is seen; nothing is interleaved.

    protocol="loco"      -- "blocked" with ``holdout_class`` removed from train
                            and validation and ALL of its flows moved to test,
                            where they are an unseen class (y_test == -1).
                            Leave-one-class-out: the unseen half of the
                            seen-versus-unseen comparison.
    """
    if val_mode not in ("random", "tail"):
        raise ValueError(f"val_mode must be 'random' or 'tail', got {val_mode!r}")
    if protocol == "loco" and not holdout_class:
        raise ValueError("protocol='loco' needs holdout_class=<attack class name>")
    if protocol != "loco" and holdout_class:
        raise ValueError("holdout_class is only meaningful with protocol='loco'")

    info: Dict[str, object] = {"protocol": protocol, "seed": int(seed)}

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
        info.update(val_mode=val_mode if frac < 1.0 else "whole-day",
                    thursday_val_frac=frac)

        if frac >= 1.0:
            tr, va = base, thu
            if dedup_train:
                tr = _dedup_split(tr, "train", verbose=verbose, stats=info)
            if verbose:
                print(
                    "\n[warn] THURSDAY_VAL_FRAC = 1.0: the whole of Thursday is held out.\n"
                    "       Thursday is the only day with Infiltration and the Web attacks,\n"
                    "       so validation will contain classes absent from training; those\n"
                    "       rows are dropped and validation may collapse to BENIGN only.\n"
                    "       Model selection then selects on noise. Prefer frac < 1."
                )
        else:
            if val_mode == "tail":
                # TIME-ORDERED Thursday split, per class: the earlier flows of
                # each class train, the later ones validate.
                #
                # WHY.  A random split of one day puts flows from the same
                # session, milliseconds apart, on both sides.  The validation
                # benign scores are then scores on near-copies of training
                # rows -- lower and tighter than a new day's will be -- and a
                # threshold set at their 99th percentile is too tight for
                # Friday.  That is a candidate cause of a 1% false-alarm budget
                # arriving at 4.4%, and this mode exists to test it.
                rank, size = _group_rank(thu, by_capture=False)
                cut = np.floor(size * (1.0 - frac)).astype(np.int64)
                cut = np.clip(cut, 1, np.maximum(size - 1, 1))
                is_val = (rank >= cut) & (size >= 2)
                thu_tr = np.flatnonzero(~is_val)
                thu_va = np.flatnonzero(is_val)
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
            if dedup_train:
                tr = _dedup_split(tr, "train", verbose=verbose, stats=info)
            if verbose:
                how = ("later part of each class" if val_mode == "tail"
                       else "random, stratified")
                print(
                    f"\n[protocol] train = {TRAIN_DAYS} + {1 - frac:.0%} of Thursday "
                    f"({len(thu_tr):,} rows)\n"
                    f"           val   = {frac:.0%} of Thursday ({len(thu_va):,} rows; {how})\n"
                    f"           test  = {TEST_DAYS} ({len(te):,} rows, never seen)"
                )

        frames = [tr, va, te]
        del tr, va, te, base, thu, te_  # release every other reference
        return _finalise(frames, "crossday", verbose=verbose, info=info)

    if protocol in ("closedset", "random"):
        if protocol == "closedset":
            pool_days = TRAIN_DAYS + VAL_DAYS
        else:
            pool_days = list(DAY_ORDER)
        sub = (load_clean(days=pool_days, verbose=verbose) if df is None
               else df[df[DAY_COL].isin(pool_days)].copy())

        if protocol == "closedset":
            # An identical flow really can land on both sides of a random
            # split, so for the architecture comparison the POOL is
            # de-duplicated before splitting.
            sub = _dedup_split(sub, "closedset pool", verbose=verbose, stats=info)
        # protocol == "random" does NOT de-duplicate the pool.  It is the
        # leaky baseline: de-duplicating first would remove exactly the leak
        # it exists to measure, and would turn its test set into a set of
        # unique vectors while every other protocol tests on raw flows -- two
        # changes at once, and per-class rates that mean different things.
        # Only its training split is de-duplicated, below, like the others.

        # classes with too few samples cannot be stratified three ways
        counts = sub[LABEL_COL].value_counts()
        too_rare = counts[(counts < 10) & (counts > 0)].index.tolist()
        if too_rare:
            if verbose:
                print(f"[warn] dropping classes with <10 samples: {dict(counts[too_rare])}")
            sub = sub[~sub[LABEL_COL].isin(too_rare)]
            info["dropped_rare_classes"] = [str(c) for c in too_rare]

        val_frac, test_frac = closedset_sizes
        y = sub[LABEL_COL].astype(str).to_numpy()
        idx = np.arange(len(sub))

        idx_tr, idx_hold = train_test_split(
            idx, test_size=val_frac + test_frac, random_state=seed, stratify=y
        )
        rel = test_frac / (val_frac + test_frac)
        idx_va, idx_te = train_test_split(
            idx_hold, test_size=rel, random_state=seed, stratify=y[idx_hold]
        )

        if verbose and protocol == "closedset":
            print(
                "\n[note] closedset protocol is a stratified split of de-duplicated\n"
                "       Mon-Thu traffic.  It is OPTIMISTIC relative to the temporal\n"
                "       protocol because bursty flows remain temporally adjacent.\n"
                "       Report it as an architecture comparison, not as a detection result."
            )
        if verbose and protocol == "random":
            print(
                "\n[note] random protocol: all five days pooled and split at random;\n"
                "       only the training split is de-duplicated.  Flows from one burst,\n"
                "       and exact copies, sit on both sides.  This is the LEAKY baseline\n"
                "       -- it is here to be compared against, never quoted as a\n"
                "       detection result."
            )

        tr = sub.iloc[idx_tr]
        if protocol == "random" and dedup_train:
            tr = _dedup_split(tr, "train", verbose=verbose, stats=info)
        frames = [tr, sub.iloc[idx_va], sub.iloc[idx_te]]
        del sub, tr
        return _finalise(frames, protocol, verbose=verbose, info=info)

    if protocol in ("blocked", "loco"):
        whole = load_clean(verbose=verbose) if df is None else df
        labels = whole[LABEL_COL].astype(str).to_numpy()

        if protocol == "loco":
            present = set(pd.unique(labels))
            if holdout_class not in present:
                raise ValueError(
                    f"holdout_class {holdout_class!r} is not in the data. "
                    f"Classes present: {sorted(present)}"
                )
            if holdout_class == BENIGN_LABEL:
                raise ValueError("BENIGN cannot be the held-out class")

        assign = blocked_assignment(whole, blocked_fracs)
        if protocol == "loco":
            # every flow of the held-out class is unseen, so every one is test
            assign[labels == holdout_class] = 2
            info["holdout_class"] = holdout_class
            info["holdout_rows"] = int((labels == holdout_class).sum())
        info["blocked_fracs"] = [float(f) for f in blocked_fracs]

        tr = whole[assign == 0]
        va = whole[assign == 1]
        te = whole[assign == 2]
        del whole, labels
        for name, part in (("train", tr), ("val", va), ("test", te)):
            if len(part) == 0:
                raise ValueError(f"{name} split is empty under protocol={protocol!r}")

        # TRAIN ONLY, as in crossday: the held-out splits stay as recorded.
        if dedup_train:
            tr = _dedup_split(tr, "train", verbose=verbose, stats=info)
        if verbose:
            extra = (f"; {holdout_class} held out ({info['holdout_rows']:,} flows, "
                     "all in test)" if protocol == "loco" else "")
            print(
                f"\n[protocol] {protocol}: per (capture, class), in time order -- "
                f"earliest {blocked_fracs[0]:.0%} train, next {blocked_fracs[1]:.0%} val, "
                f"last {blocked_fracs[2]:.0%} test{extra}"
            )

        frames = [tr, va, te]
        del tr, va, te
        return _finalise(frames, protocol, verbose=verbose, info=info)

    raise ValueError(
        f"unknown protocol {protocol!r}; expected one of "
        "'crossday', 'closedset', 'random', 'blocked', 'loco'"
    )


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
