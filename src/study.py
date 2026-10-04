#!/usr/bin/env python
"""
The experiments behind the paper.

One command per research question.  Each writes its tables, figures and a
``results.json`` into its own ``runs/<sha>-<timestamp>-study-<name>/`` and
overwrites nothing.  ``report`` then copies, for each experiment, the newest
full run that follows the plan into ``paper/`` (see ``runs_for_report``).

    python src/study.py doctor     # check the CSVs before trusting anything
    python src/study.py e1         # RQ1  what the split protocol is worth
    python src/study.py e2         # RQ2  seen versus unseen, per attack class
    python src/study.py e3         # RQ3  benign-only novelty detectors
    python src/study.py e4         # RQ4  the behaviour layer
    python src/study.py e5         # RQ5  fusion under one false-alarm budget
    python src/study.py e6         # RQ6  do thresholds survive a new day
    python src/study.py all        # everything above, in order; stops after
                                   # doctor if it lists a problem with the CSVs
    python src/study.py report     # collect the newest runs into paper/

Add ``--quick`` to any experiment for a smoke run on a training subsample (not
"a few minutes" on the real files: only the training rows are subsampled, and
E1 alone took about ten minutes on a laptop; ``--seeds 1`` shortens it).
A quick run is stamped as such and plain ``report`` never uses it: it exists
to show that the pipeline runs on your machine, not to produce a number.
``all --quick`` and ``report --allow-quick`` collect quick runs into
``paper/quick/`` -- a separate folder -- and never into ``paper/tables`` or
``paper/figures``, which hold results only.

THE RULE EVERY EXPERIMENT FOLLOWS
---------------------------------
A detector is a function from a flow to a score.  A false-alarm budget becomes
a threshold on VALIDATION benign scores.  The table reports, per attack class,
the share of TEST flows above that threshold, with a block-bootstrap interval,
and beside it the false-alarm rate the threshold actually produced on test
benign flows.  No threshold, model choice or hyper-parameter is selected using
the test split, and where a number would require that, it is labelled
"oracle" and never used for a claim.
"""

from __future__ import annotations

import argparse
import dataclasses
import gc
import json
import os
import platform
import re
import shutil
import sys
import time
from pathlib import Path
from typing import Dict, List, Optional, Sequence

SRC_DIR = Path(__file__).resolve().parent
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

import numpy as np
import pandas as pd

from config import (
    BEHAVIOUR_HOST_THRESHOLD,
    BEHAVIOUR_PORT_THRESHOLD,
    BEHAVIOUR_WINDOW_SECONDS,
    BENIGN_LABEL,
    BOOTSTRAP_REPLICATES,
    DATA_DIR,
    DAY_COL,
    DAY_FILES,
    DAY_ORDER,
    IDENTIFIER_COVERAGE,
    LABEL_COL,
    META_CAPTURE,
    META_COLUMNS,
    PROJECT_ROOT,
    RUNS_DIR,
    SEED,
    STUDY_BUDGETS,
    TrainConfig,
)
from evaluation import evaluate_predictions, new_run_dir, save_json
from fusion import (
    bonferroni_fired,
    layer_attribution,
    layer_subsets,
    minp_scores,
)
from layers import (
    DEEP_NOVELTY,
    NOVELTY_DETECTORS,
    LayerScores,
    behaviour_scores,
    classifier_scores,
    fit_sklearn,
    max_attack_probability,
    novelty_scores,
)
from meta import (
    EXPECTED_DAY_DATES,
    feature_columns,
    has_identifiers,
    parse_cic_timestamps,
    require_identifiers,
)
from preprocessing import (
    SplitBundle,
    benign_index,
    blocked_assignment,
    build_splits,
    cache_state,
    load_clean,
    read_capture,
)
from rules import (
    reading_rule,
    rq3_supported,
    rq3_verdict,
    rq5_supported,
    rq5_verdict,
    rq6_supported,
    rq6_verdict,
)
from seeding import git_sha
from stats import (
    ANY_ATTACK,
    MIN_BLOCKS_FOR_INTERVAL,
    auc_table,
    block_ids,
    bootstrap_rates,
    detection_table,
    difference_supported,
    fired_at_budget,
    interval_counts,
    paired_difference,
    projected_precision,
    summarise_seeds,
)

PAPER_DIR = Path(os.environ.get("NIDS_PAPER_DIR", PROJECT_ROOT / "paper"))
PRIMARY_BUDGET = 0.01

# Named in paper/PROTOCOL.md before the runs: the novelty layer of the fused
# system, and the deep detector RQ3's rule is about.
PRIMARY_NOVELTY = "autoencoder"
# "Every classical baseline" in RQ3's rule, and the three layers of RQ5's.
CLASSICAL_BASELINES = tuple(d for d in NOVELTY_DETECTORS if d not in DEEP_NOVELTY)
E5_LAYERS = ("classifier", "novelty", "behaviour")

# The capture is documented as 09:00-17:00.  ``doctor`` raises a problem when a
# file's timestamps parse outside this window, which allows an hour of slack on
# each side (a judgement call).  WHAT IT CAN SEE: a file with hours 6 or 7,
# which the 12-hour rule turns into 18:xx and 19:xx.  WHAT IT CANNOT: traffic
# that really ran at 8-11 in the evening is printed "8"-"11" and is
# indistinguishable from the morning.  Nothing in the files could tell.
DOCTOR_FIRST_HOUR, DOCTOR_LAST_HOUR = 8, 18

# "07-07-2017 03:30": day-month-year with dashes.  None of the three shapes
# meta.parse_cic_timestamps reads.  `doctor` uses this only to word its
# message: it is what a spreadsheet program writes when it re-saves a CSV.
DASH_DATE = re.compile(r"^\s*\d{1,2}-\d{1,2}-\d{4}\s")


# =====================================================================
# RUN BOOK-KEEPING
# =====================================================================


class Tee:
    """Console log that is also written into the run directory."""

    def __init__(self, path: Path):
        self.f = open(path, "w", encoding="utf-8")

    def __call__(self, msg: str = "") -> None:
        print(msg)
        self.f.write(str(msg) + "\n")
        self.f.flush()

    def close(self) -> None:
        self.f.close()


def environment() -> Dict[str, str]:
    env = {"python": platform.python_version(), "platform": platform.platform(),
           "git_sha": git_sha()}
    for mod in ("numpy", "pandas", "sklearn", "xgboost", "lightgbm", "torch", "pyarrow"):
        try:
            env[mod] = getattr(__import__(mod), "__version__", "?")
        except Exception:
            env[mod] = "not installed"
    return env


class Run:
    def __init__(self, name: str, args):
        self.name = name
        self.args = args
        tag = f"study-{name}" + ("-quick" if getattr(args, "quick", False) else "")
        self.dir = new_run_dir(tag)
        self.log = Tee(self.dir / "log.txt")
        self.t0 = time.time()
        self.tables: Dict[str, str] = {}
        self.figures: Dict[str, str] = {}
        state = cache_state()
        save_json({
            "experiment": name,
            "args": {k: v for k, v in vars(args).items() if k != "func"},
            # the command line as given.  An experiment that uses one seed by
            # design (E2, E6) still shows the default `seeds` here; what was
            # actually used is `seeds_used` in results.json.
            "args_note": "command line as given; see results.json for seeds_used",
            "environment": environment(),
            "cache_identifiers": state.get("identifiers", {}),
            "quick": bool(getattr(args, "quick", False)),
        }, self.dir / "config.json")
        self.log(f"========== STUDY {name.upper()}  ->  {self.dir.name} ==========")
        if getattr(args, "quick", False):
            self.log("[quick] SUBSAMPLED SMOKE RUN.  Not a result.  Plain `report` ignores it; "
                     "`all --quick` collects quick runs into paper/quick/ only.")

    def table(self, df: pd.DataFrame, name: str) -> pd.DataFrame:
        path = self.dir / f"{name}.csv"
        df.to_csv(path, index=False)
        self.tables[name] = path.name
        return df

    def figure(self, path: Optional[Path], name: str) -> None:
        if path is not None:
            self.figures[name] = Path(path).name

    def finish(self, results: Optional[dict] = None) -> Path:
        out = dict(results or {})
        out.update({
            "experiment": self.name,
            "quick": bool(getattr(self.args, "quick", False)),
            "seconds": time.time() - self.t0,
            "tables": self.tables, "figures": self.figures,
            "environment": environment(),
            # read again here: the cache may have been (re)built during the run
            "cache_identifiers": cache_state().get("identifiers", {}),
        })
        save_json(out, self.dir / "results.json")
        self.log(f"\nDone in {out['seconds']:.0f}s.  Artifacts: {self.dir}")
        self.log.close()
        return self.dir


# =====================================================================
# SHARED HELPERS
# =====================================================================


def seeds_of(args) -> List[int]:
    return [SEED + i for i in range(max(int(args.seeds), 1))]


def budgets_of(args) -> List[float]:
    return [float(b) for b in (args.budgets or STUDY_BUDGETS)]


def n_boot_of(args) -> int:
    if args.n_boot is not None:
        return int(args.n_boot)
    return 200 if args.quick else BOOTSTRAP_REPLICATES


def train_config(args) -> TrainConfig:
    cfg = TrainConfig()
    cfg.epochs = int(args.epochs) if args.epochs else (4 if args.quick else 30)
    cfg.patience = min(cfg.patience, max(cfg.epochs // 3, 2))
    cfg.warmup_epochs = min(cfg.warmup_epochs, max(cfg.epochs // 5, 1))
    if args.device:
        cfg.device = args.device
    return cfg


def quick_overrides(model: str, args) -> Optional[dict]:
    if not args.quick:
        return None
    return {"xgb": {"n_estimators": 60}, "xgboost": {"n_estimators": 60},
            "rf": {"n_estimators": 40, "max_depth": 16},
            "random_forest": {"n_estimators": 40, "max_depth": 16},
            "lgbm": {"n_estimators": 60}, "lightgbm": {"n_estimators": 60}}.get(model.lower())


def subsample_train(bundle: SplitBundle, max_rows: int, seed: int) -> SplitBundle:
    """Stratified cut of the TRAINING split only, for ``--quick``.

    Validation and test are left whole: a quick run should fail for the same
    reasons a full run would, just sooner.
    """
    n = len(bundle.y_train)
    if n <= max_rows:
        return bundle
    rng = np.random.default_rng(seed)
    keep = []
    for c in np.unique(bundle.y_train):
        idx = np.flatnonzero(bundle.y_train == c)
        quota = max(min(len(idx), 300), int(round(max_rows * len(idx) / n)))
        keep.append(idx if len(idx) <= quota else rng.choice(idx, size=quota, replace=False))
    keep = np.sort(np.concatenate(keep))
    return dataclasses.replace(
        bundle,
        X_train=bundle.X_train[keep], y_train=bundle.y_train[keep],
        y_train_str=bundle.y_train_str[keep],
        meta_train=(None if bundle.meta_train is None
                    else bundle.meta_train.iloc[keep].reset_index(drop=True)),
        info={**bundle.info, "quick_train_rows": int(len(keep))},
    )


def make_bundle(args, log, protocol: str, seed: int, **kw) -> SplitBundle:
    bundle = build_splits(protocol=protocol, seed=seed, verbose=not args.quiet, **kw)
    if args.quick:
        bundle = subsample_train(bundle, args.quick_rows, seed)
    log(f"[split] {protocol} {kw or ''} seed={seed}: train {len(bundle.X_train):,} / "
        f"val {len(bundle.X_val):,} / test {len(bundle.X_test):,}; "
        f"known classes {bundle.n_classes}")
    return bundle


SCORE_RULES = ("max attack probability", "1 - P(benign)")


def classifier_rule_scores(L: LayerScores) -> Dict[str, tuple]:
    """Both ways of turning class probabilities into one detection score."""
    return {
        SCORE_RULES[0]: (L.val, L.test),
        SCORE_RULES[1]: (L.extra["one_minus_benign_val"], L.extra["one_minus_benign_test"]),
    }


def load_week_meta():
    """(meta side-table, labels, day) for every flow in the cache."""
    df = load_clean(columns=[LABEL_COL, DAY_COL, *META_COLUMNS], verbose=False)
    y = df[LABEL_COL].astype(str).to_numpy(dtype=object)
    day = df[DAY_COL].astype(str).to_numpy(dtype=object)
    meta = df[META_COLUMNS].reset_index(drop=True)
    return meta, y, day


def sanitise_bundle(bundle: SplitBundle, scan_like_keys: np.ndarray, log) -> SplitBundle:
    """Drop from TRAIN and VALIDATION the BENIGN-labelled flows whose source was
    past the deployed scan thresholds when they were sent.

    WHY THIS EXISTS.  The original CIC-IDS2017 labels mark some scanning
    traffic BENIGN (documented by Engelen et al. and Liu et al.; the behaviour
    experiment shows which sources).  Those flows then do three things at
    once: they teach a benign-only model that scan flows are normal, they
    teach a classifier that scan flows are benign, and they sit in the top
    percentile of the "benign" validation scores, where they set the
    threshold.

    WHAT IT IS NOT.  It is not a relabelling and it never touches the test
    split.  It uses no label except BENIGN and no information from the test
    day: the flag comes from counting ports per source on the training and
    validation days.  It is reported as a separate condition beside the
    as-labelled result, never instead of it.
    """
    from behaviour import flow_keys

    def keep(meta, y_str) -> np.ndarray:
        flagged = np.isin(flow_keys(meta), scan_like_keys)
        return ~(flagged & (np.asarray(y_str, dtype=object) == BENIGN_LABEL))

    ktr = keep(bundle.meta_train, bundle.y_train_str)
    kva = keep(bundle.meta_val, bundle.y_val_str)
    log(f"[sanitise] removed {int((~ktr).sum()):,} of {len(ktr):,} training flows and "
        f"{int((~kva).sum()):,} of {len(kva):,} validation flows "
        "(BENIGN-labelled, source past the scan thresholds)")
    return dataclasses.replace(
        bundle,
        X_train=bundle.X_train[ktr], y_train=bundle.y_train[ktr],
        y_train_str=bundle.y_train_str[ktr],
        meta_train=bundle.meta_train[ktr].reset_index(drop=True),
        X_val=bundle.X_val[kva], y_val=bundle.y_val[kva], y_val_str=bundle.y_val_str[kva],
        meta_val=bundle.meta_val[kva].reset_index(drop=True),
        info={**bundle.info, "sanitised": True,
              "sanitise_removed_train": int((~ktr).sum()),
              "sanitise_removed_val": int((~kva).sum())},
    )


def attack_classes(labels: Sequence) -> List[str]:
    return sorted(c for c in pd.unique(np.asarray(labels, dtype=object)) if c != BENIGN_LABEL)


def md_table(df: pd.DataFrame, floatfmt: str = "{:.4f}") -> str:
    """A plain Markdown table.  (``DataFrame.to_markdown`` needs ``tabulate``,
    which is not a dependency of this project.)"""
    def cell(v) -> str:
        if isinstance(v, (float, np.floating)):
            return "" if v != v else floatfmt.format(v)
        return str(v)

    head = "| " + " | ".join(map(str, df.columns)) + " |"
    rule = "| " + " | ".join("---" for _ in df.columns) + " |"
    body = ["| " + " | ".join(cell(v) for v in row) + " |" for row in df.itertuples(index=False)]
    return "\n".join([head, rule, *body]) + "\n"


def row_hashes(frame: pd.DataFrame) -> np.ndarray:
    """One 64-bit hash per row.  Two rows hash alike iff every feature is equal."""
    return pd.util.hash_pandas_object(frame, index=False).to_numpy()


# =====================================================================
# DOCTOR
# =====================================================================


def run_doctor(args) -> Path:
    """Report what the CSVs in data/ actually contain.

    Run this first, and read it.  Every assumption the study makes about the
    files -- which distribution they are, whether timestamps parse, the hours
    they parse onto, whether file order is time order -- is printed here as a
    measured fact, and written to ``doctor.json``.

    THE ONE LINE TO LOOK AT TWICE is the parsed time range of each capture.
    The identifier CSVs print a 12-hour clock with no AM/PM marker, and
    ``meta.parse_cic_timestamps`` reads an hour below 8 as afternoon because
    the capture is documented as running 09:00-17:00.  If a capture parses
    onto hours outside the working day, that reading is wrong for it, every
    time-ordered split built from it is scrambled, and nothing else will say
    so.  ``doctor`` prints the range and raises it as a problem.  (It catches
    hours that land after 18:00.  It cannot catch evening traffic printed as
    8-11 o'clock, which reads as morning: see ``DOCTOR_LAST_HOUR``.)
    """
    run = Run("doctor", args)
    log = run.log
    folder = Path(args.data or DATA_DIR)
    rows, problems = [], []
    order = [(d, f) for d, files in DAY_FILES.items() for f in files]

    for day, fname in order:
        path = folder / fname
        rec: Dict[str, object] = {"day": day, "file": fname, "exists": path.exists()}
        if not path.exists():
            problems.append(f"missing: {fname}")
            rows.append(rec)
            continue
        df = read_capture(path)
        cols = set(df.columns)
        enc = df.attrs.get("encoding", "utf-8")
        rec["encoding"] = enc if enc == "utf-8" else f"{enc} (the file is not valid UTF-8)"
        rec.update({
            "mb": round(path.stat().st_size / 1e6, 1),
            "columns": int(df.attrs.get("raw_columns", len(df.columns) - 1)),
            "rows_with_label": int(len(df)),
            "blank_rows_dropped": int(df.attrs.get("raw_rows", len(df)) - len(df)),
            "has_source_ip": bool({"Source IP", "Src IP"} & cols),
            "has_timestamp": "Timestamp" in cols,
            "has_dst_port": bool({"Destination Port", "Dst Port"} & cols),
            "labels": {str(k): int(v) for k, v in df[LABEL_COL].astype(str).str.strip()
                       .value_counts().items()},
        })
        if rec["has_timestamp"]:
            ts_raw = df["Timestamp"]
            ts = parse_cic_timestamps(ts_raw.to_numpy(), expected_date=EXPECTED_DAY_DATES.get(day))
            ok = np.isfinite(ts)
            lo = pd.Timestamp(EXPECTED_DAY_DATES[day]).timestamp()
            on_day = ok & (ts >= lo) & (ts < lo + 86400)
            rec.update({
                "timestamp_examples": [str(v) for v in pd.unique(ts_raw.astype(str))[:3]],
                "timestamp_parse_rate": float(ok.mean()),
                "timestamp_on_expected_date": float(on_day.mean()),
                # 60 = stamped to the minute, 1 = to the second.  The behaviour
                # layer's windows cannot be finer than this.
                "timestamp_resolution_seconds": (
                    (1 if np.any(np.mod(ts[ok], 60.0) != 0) else 60) if ok.any() else None),
                "time_first": str(pd.to_datetime(np.nanmin(ts), unit="s")) if ok.any() else None,
                "time_last": str(pd.to_datetime(np.nanmax(ts), unit="s")) if ok.any() else None,
                # is file order time order?  1.0 = never goes backwards
                "file_order_non_decreasing": float(np.mean(np.diff(ts[ok]) >= 0)) if ok.sum() > 1 else None,
            })
            # A Timestamp COLUMN that cannot be read is not a timestamp.  The
            # 99% floor is the one the cache uses (preprocessing.build_cache,
            # meta.has_identifiers).  Here it is counted over the labelled rows
            # as read; the cache counts over the rows it keeps.
            rec["time_usable"] = bool(rec["timestamp_parse_rate"] >= IDENTIFIER_COVERAGE)
            if not rec["time_usable"]:
                bad = [str(v) for v in pd.unique(ts_raw[~ok].astype(str))[:2]]
                rec["timestamp_unparsed_examples"] = bad
                msg = f"{fname}: only {rec['timestamp_parse_rate']:.1%} of timestamps can be read"
                if bad:
                    msg += f" (for example {bad[0]!r})"
                msg += "."
                if any(DASH_DATE.match(v) for v in bad):
                    # Seen on Ayush's copy of the Friday DDoS capture, 2026-10-04:
                    # "07-07-2017 03:30".  The three shapes this dataset is known
                    # to use (meta.parse_cic_timestamps) have slashes or start
                    # with the year.  The parser is NOT taught this one: a file
                    # that went through a spreadsheet may differ from the
                    # original in other columns too, and nothing here could tell.
                    msg += ("  This dataset writes day/month/year with slashes (7/7/2017 3:30). "
                            "Day-month-year with dashes is what a spreadsheet program writes when "
                            "it opens a CSV and saves it, so this file is probably not the "
                            "original.  Take it from the zip again and do not open it in a "
                            "spreadsheet on the way.")
                problems.append(msg)
            # among the timestamps that DO read; a file where none reads has
            # already been reported above, once
            off_date = float(1.0 - on_day[ok].mean()) if ok.any() else 0.0
            rec["timestamp_off_date_among_parsed"] = off_date
            if off_date > 1.0 - IDENTIFIER_COVERAGE:
                problems.append(f"{fname}: {off_date:.1%} of the timestamps that can be read "
                                f"are not on {EXPECTED_DAY_DATES[day]}")
            if ok.any():
                first = pd.to_datetime(np.nanmin(ts), unit="s")
                last = pd.to_datetime(np.nanmax(ts), unit="s")
                rec["time_range"] = f"{first:%H:%M}-{last:%H:%M}"
                too_early = first.hour < DOCTOR_FIRST_HOUR
                too_late = (last.hour, last.minute, last.second) > (DOCTOR_LAST_HOUR, 0, 0)
                if too_early or too_late:
                    problems.append(
                        f"{fname}: timestamps parse onto {rec['time_range']}.  The capture is "
                        "documented as 09:00-17:00, and anything outside "
                        f"{DOCTOR_FIRST_HOUR:02d}:00-{DOCTOR_LAST_HOUR:02d}:00 is flagged.  Check "
                        "how the 12-hour clock was read (meta.parse_cic_timestamps) before "
                        "trusting any time-ordered split.")
        if rec["has_source_ip"]:
            src_col = "Source IP" if "Source IP" in cols else "Src IP"
            rec["distinct_sources"] = int(df[src_col].nunique())
            rec["source_ip_coverage"] = float(df[src_col].notna().mean()) if len(df) else 0.0
            if rec["source_ip_coverage"] < IDENTIFIER_COVERAGE:
                problems.append(f"{fname}: only {rec['source_ip_coverage']:.1%} of rows have a "
                                "source IP")
        # can the behaviour layer use this capture?  (the cache asks the same
        # two questions with the same floor)
        rec["identifiers_usable"] = bool(
            rec.get("source_ip_coverage", 0.0) >= IDENTIFIER_COVERAGE and rec.get("time_usable"))
        rows.append(rec)
        res = rec.get("timestamp_resolution_seconds")
        if rec.get("time_usable"):
            when = "yes, to the " + ("minute" if res == 60 else "second")
        elif rec["has_timestamp"]:
            when = "UNREADABLE (the column is there; see [problem] below)"
        else:
            when = "NO "
        log(f"  {day:<10s} {fname[:52]:<54s} {rec['rows_with_label']:>9,} rows  "
            f"{rec['columns']} cols  src-ip={'yes' if rec['has_source_ip'] else 'NO '}  "
            f"time={when}")
        if rec.get("time_range"):
            fwd = rec.get("file_order_non_decreasing")
            log(f"  {'':<10s} parsed times {rec['time_range']}; "
                f"{rec['timestamp_parse_rate']:.1%} of timestamps parse; "
                + ("file order is time order" if fwd == 1.0 else
                   f"{fwd:.1%} of consecutive rows go forward in time" if fwd is not None
                   else "one row"))
        del df
        gc.collect()

    present = [r for r in rows if r.get("exists")]
    # Two different questions.  Does a capture HAVE the identifier columns
    # (which download is it)?  And can they be USED (do the timestamps read)?
    has_cols = [bool(r.get("has_source_ip") and r.get("has_timestamp")) for r in present]
    usable = [bool(r.get("identifiers_usable")) for r in present]
    full_ids = bool(present) and all(usable)
    if any(has_cols) and not all(has_cols):
        n_with = sum(has_cols)
        problems.append(
            f"MIXED DISTRIBUTIONS: the source IP and timestamp columns are in {n_with} of the "
            f"{len(present)} captures and missing from the other {len(present) - n_with}.  Use "
            "the eight files of ONE download: all eight from GeneratedLabelledFlows.zip (folder "
            "TrafficLabelling).")
    verdict = {
        "all_files_present": len(present) == len(order),
        "identifiers_in_every_capture": full_ids,
        "can_run": ["e1", "e2", "e3", "e6", "e5 (classifier + novelty layers)"] +
                   (["e4", "e5 (all three layers)"] if full_ids else []),
        "cannot_run": [] if full_ids else ["e4", "behaviour layer of e5"],
        "problems": problems,
    }
    log("")
    log(f"all eight captures present      : {verdict['all_files_present']}")
    log(f"source IP + timestamp everywhere: {full_ids}")
    if not full_ids:
        if present and all(has_cols):
            log("  -> every capture has the two columns, but in at least one they cannot be used\n"
                "     (see [problem] below).  E4 and the behaviour layer need them in every capture.")
        else:
            log("  -> E4 and the behaviour layer need all eight files from GeneratedLabelledFlows.zip\n"
                "     (same CIC page; the folder inside the zip is TrafficLabelling; the eight files\n"
                "     have the same names, so they REPLACE the ones in data/).  Without them only\n"
                "     E1, E2, E3, E6 and a two-layer E5 run, and the paper's question about the\n"
                "     third layer cannot be answered.")
    for p in problems:
        log(f"  [problem] {p}")
    # which folder this describes: `--data` can point doctor somewhere else than
    # the folder the experiments read, and the record must say so
    described = str(folder.resolve())
    log(f"folder described: {described}")
    save_json({"folder": described, "captures": rows, "verdict": verdict},
              run.dir / "doctor.json")
    return run.finish({"verdict": verdict, "folder": described})


# =====================================================================
# E1 -- PROTOCOL EFFECT
# =====================================================================


def run_e1(args) -> Path:
    """Same models, same features, three ways of splitting the week."""
    run = Run("e1", args)
    log = run.log
    protocols = args.protocols or ["random", "blocked", "crossday"]
    models = args.models or ["xgb", "rf"]
    budgets, n_boot = budgets_of(args), n_boot_of(args)
    det_all, summ = [], []

    for protocol in protocols:
        for seed in seeds_of(args):
            bundle = make_bundle(args, log, protocol, seed)
            blocks = block_ids(bundle.meta_test, len(bundle.X_test))
            y = np.asarray(bundle.y_test_str, dtype=object)
            is_attack = y != BENIGN_LABEL
            known = np.isin(y, bundle.known_classes)
            for model in models:
                L = classifier_scores(bundle, model, seed, train_config(args),
                                      run.dir / "ckpt", log, quick_overrides(model, args))
                vb = np.asarray(bundle.y_val_str, dtype=object) == BENIGN_LABEL
                det = None
                for rule, (s_val, s_test) in classifier_rule_scores(L).items():
                    d = detection_table(y, s_test, np.asarray(s_val)[vb], budgets, blocks,
                                        n_boot, seed, detector=model)
                    d.insert(0, "protocol", protocol)
                    d["model"], d["seed"], d["score_rule"] = model, seed, rule
                    det_all.append(d)
                    det = d if det is None else det      # the primary rule, for the log

                pred = np.asarray(L.extra["y_pred_test"], dtype=object)
                rep = evaluate_predictions(
                    y[known], pred[known],
                    labels=sorted(set(y[known].tolist()) | set(pred[known].tolist())))
                pa = pred != BENIGN_LABEL
                # Attack-versus-benign decision of the arg-max, over ALL test
                # rows.  It is the one score that means the same thing under
                # every protocol.  Macro-F1 over the known classes does not:
                # on a cross-day test split the only known class present is
                # BENIGN, so it is a one-class number near 1 whatever happens
                # to the attacks -- ``known_classes_in_test`` says when.
                tp = int((pa & is_attack).sum())
                fp = int((pa & ~is_attack).sum())
                fn = int((~pa & is_attack).sum())
                prec = tp / (tp + fp) if tp + fp else np.nan
                rec_ = tp / (tp + fn) if tp + fn else np.nan
                f1 = (2 * prec * rec_ / (prec + rec_)
                      if tp and np.isfinite(prec) and np.isfinite(rec_) else
                      (0.0 if tp + fn else np.nan))
                summ.append({
                    "protocol": protocol, "model": model, "seed": seed,
                    "n_train": len(bundle.X_train), "n_test": len(y),
                    "n_test_attack": int(is_attack.sum()),
                    "unseen_classes_in_test": int(len(set(y[~known].tolist()))),
                    "known_classes_in_test": int(len(set(y[known].tolist()))),
                    "accuracy_known_rows": rep.metrics["accuracy"],
                    "macro_f1_known_rows": rep.metrics["macro_f1_present"],
                    "attack_precision_argmax": prec,
                    "attack_recall_argmax": rec_,
                    "attack_f1_argmax": f1,
                    "benign_fpr_argmax": float(pa[~is_attack].mean()),
                    "fit_seconds": float(L.extra["seconds"]),
                    "rows_removed_by_dedup": bundle.info.get("rows_removed_by_dedup"),
                })
                any_row = det[(det["class"] == ANY_ATTACK)]
                log(f"  {protocol:<9s} {model:<5s} seed {seed}: attack F1 (arg-max) {f1:.4f} | "
                    f"macro-F1 over {len(set(y[known].tolist()))} known class(es) "
                    f"{rep.metrics['macro_f1_present']:.4f} | any-attack detected "
                    + ", ".join(f"{r.detection_rate:.1%}@{100 * r.budget:g}%"
                                for r in any_row.itertuples()))
            del bundle
            gc.collect()

    det_df = run.table(pd.concat(det_all, ignore_index=True), "e1_detection")
    run.table(pd.DataFrame(summ), "e1_summary")

    from figures import fig_protocol_effect
    primary = min(budgets, key=lambda b: abs(b - PRIMARY_BUDGET))
    fig_df = det_df[(det_df["class"] == ANY_ATTACK) & (det_df["budget"] == primary)
                    & (det_df["score_rule"] == SCORE_RULES[0])]
    run.figure(fig_protocol_effect(
        fig_df, run.dir / "e1_protocol_effect",
        f"Attack flows detected at a {100 * primary:g}% false-alarm budget, by split protocol"),
        "e1_protocol_effect")
    return run.finish({"protocols": protocols, "models": models, "primary_budget": primary,
                       "seeds_used": seeds_of(args)})


# =====================================================================
# E2 -- SEEN VERSUS UNSEEN
# =====================================================================


def week_diagnostics(log) -> pd.DataFrame:
    """Model-free facts about each class, from the feature vectors alone.

    ``unique_vectors``         distinct feature vectors among the class's flows
    ``benign_collision_share`` share of the class's flows whose exact feature
                               vector ALSO occurs with the label BENIGN.  No
                               per-flow model can tell such a flow from the
                               benign flows that carry the same vector.
    ``benign_cost_share``      share of ALL benign flows that carry one of the
                               class's vectors (``benign_cost_flows`` is the
                               count).  This is the price: a per-flow model
                               that fires on the colliding attack flows must
                               fire on these benign flows too.  The collision
                               share alone is NOT a ceiling on detection --
                               a thousand attack flows that share a vector
                               with one benign flow are all "colliding" and
                               cost one false alarm to catch.  The two numbers
                               are read together.
    ``seen_before_share``      share of the class's test-block flows whose exact
                               vector is already in the training block.  High
                               values mean "seen" detection is recognition of a
                               repeated flow, not generalisation.
    ``collides_with_benign_<Day>``  the collision share again, against one
                               day's benign flows at a time.
    """
    df = load_clean(verbose=False)
    feat = feature_columns(df)
    h = row_hashes(df[feat])
    lab = df[LABEL_COL].astype(str).to_numpy()
    day = df[DAY_COL].astype(str).to_numpy()
    assign = blocked_assignment(df)
    del df
    gc.collect()

    is_benign = lab == BENIGN_LABEL
    benign_set = np.unique(h[is_benign])
    benign_by_day = {d: np.unique(h[is_benign & (day == d)]) for d in DAY_ORDER
                     if (is_benign & (day == d)).any()}
    train_set = np.unique(h[assign == 0])
    rows = []
    for c in attack_classes(lab):
        m = lab == c
        mt = m & (assign == 2)
        class_set = np.unique(h[m])
        cost = int(np.isin(h[is_benign], class_set).sum())
        rec = {
            "class": c, "n_total": int(m.sum()),
            "unique_vectors": int(class_set.size),
            "benign_collision_share": float(np.isin(h[m], benign_set).mean()),
            "benign_cost_flows": cost,
            "benign_cost_share": cost / max(int(is_benign.sum()), 1),
            "n_test_block": int(mt.sum()),
            "seen_before_share": float(np.isin(h[mt], train_set).mean()) if mt.any() else np.nan,
        }
        # WHICH day's "benign" traffic the class collides with.  A collision
        # spread evenly over the week is ordinary traffic that looks like the
        # attack.  A collision concentrated on one afternoon points at flows
        # of that afternoon carrying the wrong label.
        for d, hs in benign_by_day.items():
            rec[f"collides_with_benign_{d}"] = float(np.isin(h[m], hs).mean())
        rows.append(rec)
    out = pd.DataFrame(rows)
    log("\nModel-free class facts (whole week):")
    log(out.to_string(index=False, float_format=lambda v: f"{v:.4f}"))
    return out


def run_e2(args) -> Path:
    """Per attack class: detection when the class is in training (time-blocked
    split) and when every flow of it is held out (leave-one-class-out)."""
    run = Run("e2", args)
    log = run.log
    model = (args.models or ["xgb"])[0]
    seed = seeds_of(args)[0]
    budgets, n_boot = budgets_of(args), n_boot_of(args)
    cfg = train_config(args)

    diag = run.table(week_diagnostics(log), "e2_class_facts")

    # ---- seen: every class is in training ---------------------------------
    b = make_bundle(args, log, "blocked", seed)
    y = np.asarray(b.y_test_str, dtype=object)
    L = classifier_scores(b, model, seed, cfg, run.dir / "ckpt", log, quick_overrides(model, args))
    vb = np.asarray(b.y_val_str, dtype=object) == BENIGN_LABEL
    blocks_seen = block_ids(b.meta_test, len(y))
    seen_rules = []
    for rule, (s_val, s_test) in classifier_rule_scores(L).items():
        seen_rules.append(detection_table(y, s_test, np.asarray(s_val)[vb], budgets, blocks_seen,
                                          n_boot, seed, detector=model).assign(score_rule=rule))
    seen_all = pd.concat(seen_rules, ignore_index=True)
    run.table(seen_all.assign(condition="seen", seed=seed), "e2_seen_detection")
    classes = [c for c in (args.classes or attack_classes(b.y_train_str))]
    del b, L
    gc.collect()

    # ---- unseen: one class at a time is removed from training --------------
    unseen_rows, routing = [], []
    for c in classes:
        try:
            bc = make_bundle(args, log, "loco", seed, holdout_class=c)
        except ValueError as exc:
            log(f"  [skip] {c}: {exc}")
            continue
        yc = np.asarray(bc.y_test_str, dtype=object)
        Lc = classifier_scores(bc, model, seed, cfg, run.dir / "ckpt", log,
                               quick_overrides(model, args))
        vbc = np.asarray(bc.y_val_str, dtype=object) == BENIGN_LABEL
        blocks_c = block_ids(bc.meta_test, len(yc))
        det = None
        for rule, (s_val, s_test) in classifier_rule_scores(Lc).items():
            d = detection_table(yc, s_test, np.asarray(s_val)[vbc], budgets, blocks_c, n_boot,
                                seed, detector=model)
            unseen_rows.append(d[d["class"] == c].assign(condition="unseen", score_rule=rule,
                                                         seed=seed))
            det = d if det is None else det              # the primary rule
        pred = np.asarray(Lc.extra["y_pred_test"], dtype=object)[yc == c]
        vc = pd.Series(pred).value_counts(normalize=True)
        routing.append({"class": c, "pred_benign_share": float(vc.get(BENIGN_LABEL, 0.0)),
                        "top_predicted": str(vc.index[0]), "top_predicted_share": float(vc.iloc[0])})
        r1 = det[(det["class"] == c)]
        log(f"  held out {c:<26s} n={int(r1['n'].iloc[0]):>7,}  detected "
            + ", ".join(f"{r.detection_rate:.1%}@{100 * r.budget:g}%" for r in r1.itertuples())
            + f"  | argmax -> {routing[-1]['top_predicted']} ({routing[-1]['top_predicted_share']:.0%})")
        del bc, Lc
        gc.collect()

    unseen_all = pd.concat(unseen_rows, ignore_index=True) if unseen_rows else pd.DataFrame()
    run.table(unseen_all, "e2_unseen_detection")
    route_df = run.table(pd.DataFrame(routing), "e2_unseen_routing")

    # ---- the comparison table ---------------------------------------------
    # ``n_blocks`` travels with each interval: the seen and the unseen rate are
    # measured on different test sets, so each has its own block count.
    keep = ["class", "budget", "n", "n_blocks", "interval_counts", "detection_rate", "det_lo",
            "det_hi", "observed_fpr"]
    parts = []
    for rule in SCORE_RULES:
        sr = seen_all[(seen_all["score_rule"] == rule) & (seen_all["class"] != ANY_ATTACK)]
        t = sr[keep].rename(columns={
            "n": "n_seen_test", "n_blocks": "n_blocks_seen",
            "interval_counts": "interval_counts_seen", "detection_rate": "det_seen",
            "det_lo": "det_seen_lo", "det_hi": "det_seen_hi", "observed_fpr": "fpr_seen"})
        if len(unseen_all):
            ur = unseen_all[unseen_all["score_rule"] == rule]
            u = ur[keep].rename(columns={
                "n": "n_unseen_test", "n_blocks": "n_blocks_unseen",
                "interval_counts": "interval_counts_unseen", "detection_rate": "det_unseen",
                "det_lo": "det_unseen_lo", "det_hi": "det_unseen_hi",
                "observed_fpr": "fpr_unseen"})
            t = t.merge(u, on=["class", "budget"], how="outer")
        else:
            t = t.assign(det_unseen=np.nan)
        t.insert(0, "score_rule", rule)
        parts.append(t)
    cmp_df = pd.concat(parts, ignore_index=True).merge(diag, on="class", how="left")
    if len(route_df):
        cmp_df = cmp_df.merge(route_df, on="class", how="left")
    cmp_df["gap"] = cmp_df["det_seen"] - cmp_df["det_unseen"]
    cmp_df["seed"] = seed                      # one seed: the table carries no spread
    run.table(cmp_df.sort_values(["score_rule", "budget", "class"]), "e2_seen_unseen")

    from figures import fig_seen_unseen
    primary = min(budgets, key=lambda v: abs(v - PRIMARY_BUDGET))
    run.figure(fig_seen_unseen(
        cmp_df[(cmp_df["budget"] == primary) & (cmp_df["score_rule"] == SCORE_RULES[0])],
        run.dir / "e2_seen_unseen",
        f"Per-flow detection at a {100 * primary:g}% false-alarm budget ({model})"),
        "e2_seen_unseen")
    return run.finish({"model": model, "classes": classes, "primary_budget": primary,
                       "seed": seed, "seeds_used": [seed]})


# =====================================================================
# E3 -- NOVELTY LAYER
# =====================================================================


def run_e3(args) -> Path:
    """Detectors that have seen BENIGN only, scored per attack class."""
    run = Run("e3", args)
    log = run.log
    protocols = args.protocols or ["crossday", "blocked"]
    detectors = args.detectors or list(NOVELTY_DETECTORS)
    budgets, n_boot = budgets_of(args), n_boot_of(args)
    cfg = train_config(args)
    det_all, auc_all, pair_rows = [], [], []
    classical = [d for d in detectors if d not in DEEP_NOVELTY]

    for protocol in protocols:
        bundle = make_bundle(args, log, protocol, SEED)
        y = np.asarray(bundle.y_test_str, dtype=object)
        blocks = block_ids(bundle.meta_test, len(y))
        masks = {c: (y == c) for c in attack_classes(y)}
        masks[ANY_ATTACK] = y != BENIGN_LABEL
        masks[BENIGN_LABEL] = y == BENIGN_LABEL
        fired: Dict[tuple, np.ndarray] = {}          # (detector, budget) at the first seed

        def record(L: LayerScores, name: str, seed: int, bundle=bundle, y=y, blocks=blocks,
                   protocol=protocol, fired=fired) -> None:
            d = detection_table(y, L.test, L.val_benign(bundle), budgets, blocks, n_boot, seed,
                                detector=name)
            a = auc_table(y, L.test, detector=name)
            a["fit_seconds"] = float(L.extra.get("seconds", np.nan))
            for t in (d, a):
                t.insert(0, "protocol", protocol)
                t["seed"] = seed
            det_all.append(d)
            auc_all.append(a)
            if seed == SEED:
                for b in budgets:
                    fired[(name, b)] = fired_at_budget(L.test, L.val_benign(bundle), b)
            anyr = d[d["class"] == ANY_ATTACK]
            log(f"  {protocol:<9s} {name:<22s} seed {seed}: any-attack "
                + ", ".join(f"{r.detection_rate:.1%}@{100 * r.budget:g}% (FPR {r.observed_fpr:.2%})"
                            for r in anyr.itertuples()))

        # Every detector once per seed.  PCA and Mahalanobis look deterministic
        # but are fitted on a random subsample of the benign rows, so they
        # vary with the seed like the others.  The split itself is fixed.
        for seed in seeds_of(args):
            if not args.no_reference:
                ref = (args.models or ["xgb"])[0]
                record(classifier_scores(bundle, ref, seed, cfg, run.dir / "ckpt", log,
                                         quick_overrides(ref, args)), f"{ref} (supervised)", seed)
            for name in detectors:
                record(novelty_scores(bundle, name, seed, cfg, run.dir / "ckpt", log), name, seed)

        # Does a deep detector beat a simple one?  Paired, flow by flow, on the
        # same test rows -- the only comparison that can carry that claim.
        for deep in [d for d in detectors if d in DEEP_NOVELTY]:
            for base in classical:
                for b in budgets:
                    r = paired_difference(fired[(deep, b)], fired[(base, b)], masks, blocks,
                                          n_boot, SEED)
                    for c, v in r.items():
                        pair_rows.append({
                            "protocol": protocol, "budget": b, "class": c, "deep": deep,
                            "baseline": base, "diff": v["diff"], "diff_lo": v["lo"],
                            "diff_hi": v["hi"], "rate_deep": v["rate_a"],
                            "rate_baseline": v["rate_b"], "n": v["n"],
                            "n_blocks": v["n_blocks"],
                            "interval_counts": interval_counts(v["n_blocks"]),
                            "differs": difference_supported(v["lo"], v["hi"], v["n_blocks"]),
                            "seed": SEED})
        del bundle, fired
        gc.collect()

    det_df = run.table(pd.concat(det_all, ignore_index=True), "e3_detection")
    auc_df = run.table(pd.concat(auc_all, ignore_index=True), "e3_auc")
    keys = ["protocol", "detector", "budget", "class"]
    by_seed = [g for _, g in det_df.groupby("seed")]
    run.table(summarise_seeds(by_seed, keys, ["detection_rate", "observed_fpr"]), "e3_summary")
    pcol = [c for c in auc_df.columns if c.startswith("pauc@")]
    run.table(summarise_seeds([g for _, g in auc_df.groupby("seed")],
                              ["protocol", "detector", "class"], ["auroc", *pcol]),
              "e3_auc_summary")
    primary = min(budgets, key=lambda v: abs(v - PRIMARY_BUDGET))
    supported = supported_strict = rq3_supported(None)
    if pair_rows:
        pairs = run.table(pd.DataFrame(pair_rows), "e3_deep_vs_classical")
        view = pairs[(pairs["budget"] == primary) & (pairs["class"] != BENIGN_LABEL)]
        log(f"\nDeep minus classical, detection rate at a {100 * primary:g}% budget "
            "(paired block bootstrap, first seed).")
        log(reading_rule())
        log(view[["protocol", "class", "deep", "baseline", "diff", "diff_lo", "diff_hi",
                  "n_blocks", "differs"]]
            .to_string(index=False, float_format=lambda v: f"{v:+.4f}"))
        # The rule of PROTOCOL.md for RQ3, applied by code and not by eye.
        verdict = run.table(rq3_verdict(pairs, primary, deep=PRIMARY_NOVELTY, protocol="crossday",
                                        required_baselines=CLASSICAL_BASELINES),
                            "e3_rq3_verdict")
        supported = rq3_supported(verdict)
        supported_strict = rq3_supported(verdict, strict=True)
        log(f"\nRQ3 rule: does the {PRIMARY_NOVELTY} beat EVERY classical baseline, cross-day, "
            f"at {100 * primary:g}%, on a class spanning at least {MIN_BLOCKS_FOR_INTERVAL} blocks,"
            "\nat acceptable false alarms (within the budget on the test day, or an excess over "
            "each baseline is not shown)?")
        if len(verdict):
            log(verdict.to_string(index=False, float_format=lambda v: f"{v:+.4f}"))
        if len(verdict) and not bool(verdict["baselines_complete"].all()):
            log("  [note] not every classical baseline the protocol names was run "
                f"({', '.join(CLASSICAL_BASELINES)}), so RQ3 is not evaluated.")
        log(f"RQ3 supported: {supported}")
        log(f"The harsher reading (observed false-alarm rate not above any baseline's): "
            f"{supported_strict}")

    from figures import fig_auc_heatmap
    for protocol in protocols:
        run.figure(fig_auc_heatmap(
            auc_df[auc_df["protocol"] == protocol], run.dir / f"e3_auroc_{protocol}",
            f"Separability of each attack class from benign traffic ({protocol} split)"),
            f"e3_auroc_{protocol}")
    return run.finish({"protocols": protocols, "detectors": detectors,
                       "primary_budget": primary, "seeds_used": seeds_of(args),
                       "split_seed": SEED, "rq3_supported": supported,
                       "rq3_supported_strict": supported_strict})


# =====================================================================
# E4 -- BEHAVIOUR LAYER
# =====================================================================


def run_e4(args) -> Path:
    """The per-source window layer over the whole week: what it covers at the
    deployed thresholds, what it costs in alerts, and how it does at a
    validation-fitted false-alarm budget."""
    from behaviour import (behaviour_score, canonical_endpoints, replay_scan_tracker,
                           top_sources, window_counts)

    run = Run("e4", args)
    log = run.log
    budgets, n_boot = budgets_of(args), n_boot_of(args)
    W = float(args.window or BEHAVIOUR_WINDOW_SECONDS)
    PT = int(args.port_threshold or BEHAVIOUR_PORT_THRESHOLD)
    HT = int(args.host_threshold or BEHAVIOUR_HOST_THRESHOLD)

    meta, y, day = load_week_meta()
    require_identifiers(meta, "Experiment E4")
    benign = y == BENIGN_LABEL
    days = [d for d in DAY_ORDER if (day == d).any()]
    log(f"{len(y):,} flows over {len(days)} days; window {W:g}s, thresholds "
        f"{PT} ports / {HT} hosts")

    # ---- (a) deployed operating point, with and without the direction repair
    deployed, groups = [], []
    kept = {}
    for canonical in (True, False):
        ep = canonical_endpoints(meta, canonical=canonical)
        counts = window_counts(ep, W)
        score = behaviour_score(counts, PT, HT)
        flagged = score >= 1.0
        if canonical:
            kept = {"ep": ep, "counts": counts, "score": score}
        g = pd.DataFrame({"group": counts["group"].to_numpy(), "day": day,
                          "flag": flagged, "atk": (~benign).astype(float)})
        gg = g.groupby("group", sort=False).agg(day=("day", "first"), flag=("flag", "first"),
                                                atk=("atk", "mean"), flows=("atk", "size"))
        gg = gg[gg["flag"]]
        for d in days:
            sub = gg[gg["day"] == d]
            groups.append({"canonical": canonical, "day": d,
                           "flagged_source_windows": int(len(sub)),
                           "attack_majority": int((sub["atk"] >= 0.5).sum()),
                           "benign_labelled_only": int((sub["atk"] == 0).sum()),
                           "swapped_share": float(ep.swapped[day == d].mean())})
        for d in days + ["ALL"]:
            md = np.ones(len(y), dtype=bool) if d == "ALL" else (day == d)
            for c in [BENIGN_LABEL] + attack_classes(y[md]):
                m = md & (y == c)
                deployed.append({"canonical": canonical, "day": d, "class": c, "n": int(m.sum()),
                                 "flows_covered_share": float(flagged[m].mean())})
    run.table(pd.DataFrame(deployed), "e4_deployed_coverage")
    grp_df = run.table(pd.DataFrame(groups), "e4_flagged_source_windows")
    log("\nFlagged (source, window) groups per day, deployed thresholds:")
    log(grp_df.to_string(index=False))

    ep, counts, score = kept["ep"], kept["counts"], kept["score"]

    # ---- replay through the deployed tracker ------------------------------
    rep = replay_scan_tracker(meta, labels=y, canonical=True, window_seconds=W,
                              port_threshold=PT, host_threshold=HT)
    alerts = rep.alerts.copy()
    cap_day = {i: d for i, (d, _f) in enumerate(
        [(d, f) for d, files in DAY_FILES.items() for f in files])}
    if len(alerts):
        alerts["day"] = alerts["capture"].map(cap_day)
        alerts["time"] = pd.to_datetime(alerts["ts"], unit="s").astype(str)
        alerts["true_alert"] = alerts["attack_share"] >= 0.5
    run.table(alerts, "e4_alerts")
    alert_sum = []
    for d in days:
        sub = alerts[alerts["day"] == d] if len(alerts) else alerts
        alert_sum.append({"day": d, "alerts": int(len(sub)),
                          "vertical": int((sub["kind"] == "vertical").sum()) if len(sub) else 0,
                          "horizontal": int((sub["kind"] == "horizontal").sum()) if len(sub) else 0,
                          "on_attack_labelled_traffic": int(sub["true_alert"].sum()) if len(sub) else 0,
                          "on_benign_labelled_traffic": int((~sub["true_alert"]).sum()) if len(sub) else 0,
                          "distinct_sources": int(sub["src"].nunique()) if len(sub) else 0})
    alert_df = run.table(pd.DataFrame(alert_sum), "e4_alert_summary")
    online = [{"class": c, "n": int((y == c).sum()),
               "online_coverage": float(rep.in_alert[y == c].mean())}
              for c in [BENIGN_LABEL] + attack_classes(y)]
    run.table(pd.DataFrame(online), "e4_online_coverage")
    log("\nAlerts from the deployed tracker, per day:")
    log(alert_df.to_string(index=False))

    # ---- (b) calibrated budgets: Friday is the test day -------------------
    calib = []
    test_day = "Friday" if "Friday" in days else days[-1]
    fri = day == test_day
    blocks = block_ids(meta[fri].reset_index(drop=True), int(fri.sum()))
    refs = {
        "monday (benign-only day)": (day == "Monday") & benign,
        "thursday benign as labelled": (day == "Thursday") & benign,
        "all earlier days, benign as labelled": (day != test_day) & benign,
    }
    for name, mask in refs.items():
        if not mask.any():
            continue
        d = detection_table(y[fri], score[fri], score[mask], budgets, blocks, n_boot, SEED,
                            detector="behaviour")
        d.insert(0, "calibration_set", name)
        d["n_calibration"] = int(mask.sum())
        calib.append(d)
    run.table(pd.concat(calib, ignore_index=True) if calib else pd.DataFrame(),
              "e4_calibrated")

    # ---- (c) threshold sweep and window length -----------------------------
    sweep = []
    ports = counts["ports"].to_numpy()
    grp = counts["group"].to_numpy()
    n_days = max(len(days), 1)
    for T in (5, 10, 20, 50, 100, 200, 500, 1000, 2000, 5000):
        f = ports >= T
        rec = {"port_threshold": T, "benign_flow_share": float(f[benign].mean())}
        g = pd.DataFrame({"g": grp[f], "atk": (~benign[f]).astype(float)})
        gm = g.groupby("g")["atk"].mean() if len(g) else pd.Series(dtype=float)
        rec["benign_groups_per_day"] = float((gm == 0).sum()) / n_days
        rec["attack_groups_per_day"] = float((gm >= 0.5).sum()) / n_days
        for c in attack_classes(y):
            rec[f"det__{c}"] = float(f[y == c].mean())
        sweep.append(rec)
    sweep_df = pd.DataFrame(sweep)
    det_cols = [c for c in sweep_df.columns if c.startswith("det__")]
    at_deployed = sweep_df.loc[(sweep_df["port_threshold"] - PT).abs().idxmin(), det_cols]
    lead = at_deployed.sort_values(ascending=False).index[:3].tolist()
    sweep_df = sweep_df[[c for c in sweep_df.columns if c not in det_cols] + lead
                        + [c for c in det_cols if c not in lead]]
    run.table(sweep_df, "e4_threshold_sweep")

    win = []
    for w in (60.0, 120.0, 300.0):
        cw = counts if w == W else window_counts(ep, w)
        fw = behaviour_score(cw, PT, HT) >= 1.0
        rec = {"window_seconds": w, "benign_flow_share": float(fw[benign].mean())}
        for c in attack_classes(y):
            rec[f"covered__{c}"] = float(fw[y == c].mean())
        win.append(rec)
    run.table(pd.DataFrame(win), "e4_window_length")

    # ---- (d) who trips the counter -----------------------------------------
    top = top_sources(meta, counts, y, ep, k=20)
    run.table(top, "e4_top_source_windows")
    log("\nBusiest (source, window) groups of the week:")
    log(top.head(10).to_string(index=False))

    from figures import fig_threshold_sweep
    run.figure(fig_threshold_sweep(
        sweep_df, run.dir / "e4_threshold_sweep",
        "Behaviour layer: coverage and cost over the port threshold", deployed=PT),
        "e4_threshold_sweep")
    return run.finish({"window_seconds": W, "port_threshold": PT, "host_threshold": HT,
                       "test_day": test_day, "tracker_stats": rep.stats})


# =====================================================================
# E5 -- FUSION
# =====================================================================


def run_e5(args) -> Path:
    """Every subset of the three layers at the same total false-alarm budget,
    and which layer caught what.

    Run under two data conditions when the cache carries identifiers:

    ``as labelled``                     the dataset's labels, untouched
    ``scan-like benign flows removed``  see ``sanitise_bundle``

    The first is the result.  The second is an ablation that asks how much of
    the first is the product of scan traffic labelled BENIGN in the training
    and validation days.

    THE SCORE RULE.  The classifier layer is scored by the largest attack
    probability, the rule the pipeline adopted earlier with the test day in
    view.  An answer to RQ5 that held only under that rule would not be an
    answer, so everything that involves the classifier -- both fusion rules,
    the gains and the layer attribution -- is computed a second time with the
    same fitted classifier read through ``1 - P(benign)`` (tables
    ``*_other_score_rule``), and ``e5_rq5_verdict`` applies the protocol's
    rule to both.  ``e5_summary`` and the figure are the primary rule's.
    """
    from behaviour import week_behaviour

    run = Run("e5", args)
    log = run.log
    budgets, n_boot = budgets_of(args), n_boot_of(args)
    cfg = train_config(args)
    clf = (args.models or ["xgb"])[0]
    nov = args.novelty or PRIMARY_NOVELTY

    base = make_bundle(args, log, "crossday", SEED, val_mode=args.val_mode)
    y = np.asarray(base.y_test_str, dtype=object)
    blocks = block_ids(base.meta_test, len(y))
    masks = {c: (y == c) for c in attack_classes(y)}
    masks[ANY_ATTACK] = y != BENIGN_LABEL
    masks[BENIGN_LABEL] = y == BENIGN_LABEL

    with_behaviour = has_identifiers(base.meta_val) and has_identifiers(base.meta_test)
    week = None
    if with_behaviour:
        week_meta, _wy, _wd = load_week_meta()
        week = week_behaviour(week_meta)
        del week_meta
    else:
        log("[note] this cache has no source IP / timestamp: the behaviour layer is left out.\n"
            "       Fusion is reported for classifier + novelty only.  See `study.py doctor`.")

    conditions = [("as labelled", base)]
    if with_behaviour and not args.no_sanitise:
        conditions.append(("scan-like benign flows removed",
                           sanitise_bundle(base, week.scan_like(), log)))

    fusion_rows, attr_rows, cond_info, gain_rows = [], [], [], []
    alt_fusion_rows, alt_gain_rows, alt_attr_rows = [], [], []
    for cond, bundle in conditions:
        log(f"\n----- condition: {cond} -----")
        yv_benign = np.asarray(bundle.y_val_str, dtype=object) == BENIGN_LABEL
        Lb = behaviour_scores(bundle, week=week) if with_behaviour else None
        cond_info.append({"condition": cond, "n_train": len(bundle.X_train),
                          "n_val": len(bundle.X_val), "n_val_benign": int(yv_benign.sum()),
                          **{k: v for k, v in bundle.info.items() if k.startswith("sanitise")}})

        for seed in seeds_of(args):
            # Classifier and novelty model are both refitted per seed, so the
            # spread over seeds is a spread over retrainings of the whole
            # system.  The split is fixed, and the behaviour layer has nothing
            # random in it.
            Lc = classifier_scores(bundle, clf, seed, cfg, run.dir / "ckpt", log,
                                   quick_overrides(clf, args))
            Ln = novelty_scores(bundle, nov, seed, cfg, run.dir / "ckpt", log)
            layers = {"classifier": Lc, "novelty": Ln}
            if Lb is not None:
                layers["behaviour"] = Lb
            names = list(layers)
            test = {k: v.test for k, v in layers.items()}
            valb = {k: v.val[yv_benign] for k, v in layers.items()}

            full = "+".join(names)

            def score_all(valb_: Dict[str, np.ndarray], test_: Dict[str, np.ndarray],
                          score_rule: str, reuse: Optional[Dict[tuple, np.ndarray]] = None,
                          cond=cond, seed=seed, names=names):
                """Every subset of the layers under both fusion rules, for ONE way
                of scoring the classifier.

                ``reuse`` holds the min-p decisions of the subsets that do not
                contain the classifier.  They do not depend on its score rule,
                so under the second rule they are taken over instead of being
                computed and tabulated a second time.
                """
                f_rows, a_rows = [], []
                fired_by: Dict[tuple, np.ndarray] = {}
                for subset in layer_subsets(names):
                    tag = "+".join(subset)
                    if reuse is not None and "classifier" not in subset:
                        for b in budgets:
                            fired_by[(tag, b)] = reuse[(tag, b)]
                        continue
                    s_test = minp_scores(valb_, test_, subset)
                    s_ref = minp_scores(valb_, valb_, subset)
                    d = detection_table(y, s_test, s_ref, budgets, blocks, n_boot, seed,
                                        detector=tag)
                    d["rule"], d["layers"], d["n_layers"] = "min-p", tag, len(subset)
                    d["condition"], d["seed"], d["score_rule"] = cond, seed, score_rule
                    f_rows.append(d)
                    for b in budgets:
                        fired_by[(tag, b)] = fired_at_budget(s_test, s_ref, b)

                    for b in budgets:
                        fired, per = bonferroni_fired(valb_, test_, subset, b)
                        r = bootstrap_rates(fired, masks, blocks, n_boot, seed)
                        fpr = r[BENIGN_LABEL]
                        f_rows.append(pd.DataFrame([{
                            "detector": tag, "budget": b, "threshold": np.nan, "class": c,
                            "n": v["n"], "n_blocks": v["n_blocks"],
                            "interval_counts": interval_counts(v["n_blocks"]),
                            "detection_rate": v["rate"], "det_lo": v["lo"],
                            "det_hi": v["hi"], "observed_fpr": fpr["rate"],
                            "fpr_lo": fpr["lo"], "fpr_hi": fpr["hi"], "n_benign": fpr["n"],
                            "n_blocks_benign": fpr["n_blocks"], "rule": "bonferroni",
                            "layers": tag, "n_layers": len(subset), "condition": cond,
                            "seed": seed, "score_rule": score_rule}
                            for c, v in r.items() if c != BENIGN_LABEL]))
                        if len(subset) == len(names):
                            a = layer_attribution(per, y)
                            a["budget"], a["condition"], a["seed"] = b, cond, seed
                            a["score_rule"] = score_rule
                            a_rows.append(a)
                return f_rows, a_rows, fired_by

            def gains(fired_by: Dict[tuple, np.ndarray], score_rule: str, cond=cond, seed=seed,
                      names=names, full=full) -> List[dict]:
                """What the whole system adds over each smaller one, at the SAME
                total budget, paired flow by flow.  The BENIGN row is the
                difference in false alarms, which a fair comparison must show."""
                rows = []
                for subset in layer_subsets(names):
                    tag = "+".join(subset)
                    if tag == full:
                        continue
                    for b in budgets:
                        r = paired_difference(fired_by[(full, b)], fired_by[(tag, b)], masks,
                                              blocks, n_boot, seed)
                        for c, v in r.items():
                            rows.append({
                                "condition": cond, "seed": seed, "score_rule": score_rule,
                                "budget": b, "class": c, "system": full, "compared_with": tag,
                                "n_layers_compared": len(subset),
                                "diff": v["diff"], "diff_lo": v["lo"], "diff_hi": v["hi"],
                                "rate_system": v["rate_a"], "rate_compared": v["rate_b"],
                                "n": v["n"], "n_blocks": v["n_blocks"],
                                "interval_counts": interval_counts(v["n_blocks"]),
                                "differs": difference_supported(v["lo"], v["hi"], v["n_blocks"])})
                return rows

            f, a, fired_minp = score_all(valb, test, SCORE_RULES[0])
            fusion_rows += f
            attr_rows += a
            gain_rows += gains(fired_minp, SCORE_RULES[0])

            # Everything that involves the classifier, once more, with the SAME
            # fitted classifier read through the other score rule.  Nothing is
            # refitted: same model, same novelty and behaviour scores, same
            # budgets.
            test_alt = {**test, "classifier": np.asarray(Lc.extra["one_minus_benign_test"],
                                                         dtype=np.float64)}
            valb_alt = {**valb, "classifier": np.asarray(Lc.extra["one_minus_benign_val"],
                                                         dtype=np.float64)[yv_benign]}
            f, a, fired_alt = score_all(valb_alt, test_alt, SCORE_RULES[1], reuse=fired_minp)
            alt_fusion_rows += f
            alt_attr_rows += a
            alt_gain_rows += gains(fired_alt, SCORE_RULES[1])
            del fired_minp, fired_alt

    fus = run.table(pd.concat(fusion_rows, ignore_index=True), "e5_fusion")
    attr = run.table(pd.concat(attr_rows, ignore_index=True), "e5_attribution")
    run.table(pd.DataFrame(cond_info), "e5_conditions")
    keys = ["condition", "rule", "layers", "n_layers", "budget", "class"]
    summ = summarise_seeds([g for _, g in fus.groupby("seed")], keys,
                           ["detection_rate", "observed_fpr"])
    # What the measured rates would mean on a link where attacks are rare.
    # Only for the any-attack rows: a per-class precision needs a per-class
    # prevalence, which nobody knows.
    is_any = summ["class"] == ANY_ATTACK
    for prev in (0.001, 0.01):
        summ[f"precision_at_{100 * prev:g}pct_prevalence"] = [
            projected_precision(t, f, prev) if a else np.nan
            for t, f, a in zip(summ["detection_rate_mean"], summ["observed_fpr_mean"], is_any)]
    run.table(summ, "e5_summary")
    gain = run.table(pd.DataFrame(gain_rows), "e5_gain")
    alt_gain = run.table(pd.DataFrame(alt_gain_rows), "e5_gain_other_score_rule")
    run.table(pd.concat(alt_fusion_rows, ignore_index=True) if alt_fusion_rows else pd.DataFrame(),
              "e5_fusion_other_score_rule")
    run.table(pd.concat(alt_attr_rows, ignore_index=True) if alt_attr_rows else pd.DataFrame(),
              "e5_attribution_other_score_rule")

    primary = min(budgets, key=lambda v: abs(v - PRIMARY_BUDGET))
    first_seed = seeds_of(args)[0]
    view = summ[(summ["class"] == ANY_ATTACK) & (summ["budget"] == primary)
                & (summ["rule"] == "min-p")]
    log(f"\nAny-attack detection at a {100 * primary:g}% total budget (min-p fusion):")
    log(view[["condition", "layers", "detection_rate_mean", "observed_fpr_mean"]]
        .sort_values(["condition", "detection_rate_mean"]).to_string(index=False))
    g0 = gain[(gain["budget"] == primary) & (gain["class"].isin([ANY_ATTACK, BENIGN_LABEL]))
              & (gain["seed"] == first_seed)]
    log(f"\nFull system minus each smaller one at the same {100 * primary:g}% total budget "
        "(paired block bootstrap, first seed;\n the BENIGN rows are the difference in false "
        "alarms).")
    log(reading_rule())
    log(g0[["condition", "compared_with", "class", "diff", "diff_lo", "diff_hi", "n_blocks",
            "differs"]].to_string(index=False, float_format=lambda v: f"{v:+.4f}"))

    # The rule of PROTOCOL.md for RQ5, applied by code, under both score rules.
    verdict = run.table(rq5_verdict(pd.concat([gain, alt_gain], ignore_index=True), primary,
                                    required_layers=E5_LAYERS),
                        "e5_rq5_verdict")
    need = E5_LAYERS                       # the three-layer system the protocol asks about
    answer = rq5_supported(verdict, "as labelled", first_seed, SCORE_RULES[0],
                           required_layers=need)
    answer_alt = rq5_supported(verdict, "as labelled", first_seed, SCORE_RULES[1],
                               required_layers=need)
    answer_strict = rq5_supported(verdict, "as labelled", first_seed, SCORE_RULES[0], strict=True,
                                  required_layers=need)
    log(f"\nRQ5 rule: at {100 * primary:g}% under min-p, the full system must detect more "
        "attack flows than EACH single layer\n(paired interval above zero, on at least "
        f"{MIN_BLOCKS_FOR_INTERVAL} blocks) at acceptable false alarms: within the budget on "
        "the\ntest day, or an excess over that layer is not shown (BENIGN interval not above "
        "zero).")
    if not with_behaviour:
        log("  [note] the behaviour layer could not run, so this is a two-layer system and "
            "RQ5 -- a question\n         about the three-layer system -- is not evaluated.")
    if len(verdict):
        log(verdict[verdict["seed"] == first_seed].to_string(
            index=False, float_format=lambda v: f"{v:+.4f}"))
    log(f"RQ5 supported (as labelled, seed {first_seed}, {SCORE_RULES[0]}): {answer}")
    log(f"The same question with the classifier scored by {SCORE_RULES[1]}: {answer_alt}")
    log(f"The harsher reading (observed false-alarm rate not above any layer's, budget or "
        f"not): {answer_strict}")
    if answer != answer_alt:
        log("  -> the two score rules give DIFFERENT answers.  The paper must report RQ5 as "
            "not robust to the score rule.")

    from figures import fig_layer_attribution
    a0 = attr[(attr["budget"] == primary) & (attr["condition"] == "as labelled")
              & (attr["seed"] == first_seed)]
    run.figure(fig_layer_attribution(
        a0, run.dir / "e5_layer_attribution",
        f"Which layer fires on each unseen attack class ({100 * primary:g}% total budget)"),
        "e5_layer_attribution")
    return run.finish({"classifier": clf, "novelty": nov, "behaviour_layer": with_behaviour,
                       "val_mode": args.val_mode, "primary_budget": primary,
                       "conditions": [c for c, _ in conditions],
                       "seeds_used": seeds_of(args), "split_seed": SEED,
                       "rq5_supported": answer,
                       "rq5_supported_other_score_rule": answer_alt,
                       "rq5_supported_strict": answer_strict})


# =====================================================================
# E6 -- THRESHOLD TRANSFER
# =====================================================================


def lodo_benign_scores(bundle: SplitBundle, model: str, seed: int, log,
                       overrides: Optional[dict] = None) -> Dict[str, np.ndarray]:
    """Benign scores from models that never saw the day being scored.

    ``bundle`` must hold the training days AS RECORDED
    (``build_splits(..., dedup_train=False)``).  For each training day *d*:

      * the model is fitted on the OTHER days, de-duplicated the way a deployed
        model's training set is;
      * early stopping uses a random tenth of those rows -- not the bundle's
        validation split, which is Thursday traffic and would let the Thursday
        fold look at its own day;
      * the scored rows are day *d*'s benign flows as captured, duplicates
        included, because that is what a threshold meets in deployment.

    Returns ``{score rule: pooled scores}`` for both score rules.

    WHAT THIS SAMPLE IS NOT.  Each fold model is a different model from the
    one whose threshold is being set.  It has a day less of training data and,
    when the held-out day is the only one carrying some attack classes, fewer
    classes -- and the largest attack probability over seven classes is not
    the same quantity as over twelve.  The pooled scores answer "what does the
    benign traffic of an unseen day look like to a model of this kind", which
    is the closest thing available before the test day, not "to this model".
    """
    from sklearn.model_selection import train_test_split

    if bundle.meta_train is None or META_CAPTURE not in bundle.meta_train.columns:
        raise RuntimeError("leave-one-day-out calibration needs the cache's capture ids; "
                           "build the splits from the cache, not from a hand-made frame")
    cap_day = [d for d, files in DAY_FILES.items() for _f in files]
    day = np.asarray([cap_day[int(c)] for c in bundle.meta_train[META_CAPTURE].to_numpy()],
                     dtype=object)
    bi = benign_index(bundle.encoder)
    X, y_all = bundle.X_train, bundle.y_train
    h = row_hashes(pd.DataFrame(X))
    out: Dict[str, List[np.ndarray]] = {r: [] for r in SCORE_RULES}

    for d in [x for x in DAY_ORDER if (day == x).any()]:
        held = (day == d) & (y_all == bi)
        fit_idx = np.flatnonzero(day != d)
        if not held.any() or fit_idx.size == 0:
            continue
        fit_idx = fit_idx[~pd.Series(h[fit_idx]).duplicated().to_numpy()]
        classes, y_fit = np.unique(y_all[fit_idx], return_inverse=True)
        if bi not in classes or len(classes) < 2:
            log(f"  [lodo] skip {d}: the other days do not hold both benign and attack flows")
            continue
        remap = {int(c): i for i, c in enumerate(classes)}

        counts = np.bincount(y_fit)
        strat = y_fit if counts.min() >= 2 and fit_idx.size >= 10 * len(classes) else None
        try:
            tr, es = train_test_split(np.arange(fit_idx.size), test_size=0.1,
                                      random_state=seed, stratify=strat)
        except ValueError:                           # a class too small to stratify
            tr, es = train_test_split(np.arange(fit_idx.size), test_size=0.1, random_state=seed)
        if np.unique(y_fit[tr]).size < len(classes):  # the tenth took a whole class
            tr, es = np.arange(fit_idx.size), np.arange(0)
        est = fit_sklearn(model, X[fit_idx[tr]], y_fit[tr],
                          X[fit_idx[es]] if es.size else None,
                          y_fit[es] if es.size else None, seed=seed, overrides=overrides)
        proba = np.asarray(est.predict_proba(X[held]), dtype=np.float64)
        out[SCORE_RULES[0]].append(max_attack_probability(proba, remap[bi]))
        out[SCORE_RULES[1]].append(1.0 - proba[:, remap[bi]])
        log(f"  [lodo] held out {d:<10s}: {int(held.sum()):>9,} benign flows scored by a model "
            f"fitted on {fit_idx.size:,} de-duplicated flows of the other days "
            f"({len(classes)} classes)")
    if not out[SCORE_RULES[0]]:
        raise RuntimeError("leave-one-day-out produced no scores")
    return {r: np.concatenate(v) for r, v in out.items()}


E6_STRATEGIES = (
    "A random half of Thursday",
    "A2 half of the validation flows, uncalibrated",
    "E the same flows, per-class isotonic recalibration",
    "B later part of Thursday",
    "C leave-one-day-out",
    "D oracle (test-day benign)",
)


def run_e6(args) -> Path:
    """Does a threshold fitted before the test day hold on the test day?

    Six rows per model and score rule, each a way of obtaining the benign
    scores the threshold is fitted on:

    A   a random half of Thursday (the protocol's own validation split)
    A2  a random half of A's flows, as the reference for E
    E   the same flows as A2 after per-class isotonic recalibration fitted on
        the OTHER half of A's flows -- the remedy an earlier README proposed
    B   the later part of each Thursday class (a separate fit: the training
        split differs)
    C   leave-one-day-out over the training days (different models: see
        ``lodo_benign_scores``)
    D   the test day's own benign flows -- an oracle, used for no claim

    One seed.  A, A2, E and D share one fitted model, so differences between
    them are differences in the calibration sample alone.  B and C are not:
    the model differs too, and the table cannot separate the two effects.
    """
    from sklearn.model_selection import train_test_split

    from layers import isotonic_recalibration

    run = Run("e6", args)
    log = run.log
    models = args.models or ["xgb", "rf"]
    budgets, n_boot = budgets_of(args), n_boot_of(args)
    seed = seeds_of(args)[0]
    rows, shifts, cal_rows = [], [], []
    q = [0.5, 0.9, 0.99, 0.999]

    def add(strategy, rule, model, y, s_test, ref, blocks):
        d = detection_table(y, s_test, ref, budgets, blocks, n_boot, seed, detector=model)
        d.insert(0, "strategy", strategy)
        d["model"], d["score_rule"] = model, rule
        rows.append(d)
        shifts.append({"model": model, "score_rule": rule, "sample": strategy, "n": int(len(ref)),
                       **{f"q{100 * v:g}": float(np.quantile(ref, v)) for v in q}})

    for model in models:
        ov = quick_overrides(model, args)

        # ---- one fit serves A, A2, E and D --------------------------------
        bA = make_bundle(args, log, "crossday", seed, val_mode="random")
        y = np.asarray(bA.y_test_str, dtype=object)
        blocks = block_ids(bA.meta_test, len(y))
        masks = {c: (y == c) for c in attack_classes(y)}
        masks[ANY_ATTACK] = y != BENIGN_LABEL
        masks[BENIGN_LABEL] = y == BENIGN_LABEL
        LA = classifier_scores(bA, model, seed, train_config(args), run.dir / "ckpt", log, ov,
                               keep_proba=True)
        bi = int(LA.extra["benign_index"])
        vb = np.asarray(bA.y_val_str, dtype=object) == BENIGN_LABEL
        rules = classifier_rule_scores(LA)
        test_scores = {rule: np.asarray(st, dtype=np.float64) for rule, (_sv, st) in rules.items()}
        for rule, (s_val, s_test) in rules.items():
            add(E6_STRATEGIES[0], rule, model, y, s_test, np.asarray(s_val)[vb], blocks)
            add(E6_STRATEGIES[5], rule, model, y, s_test, s_test[y == BENIGN_LABEL], blocks)
            shifts.append({"model": model, "score_rule": rule, "sample": "test-day benign",
                           "n": int((y == BENIGN_LABEL).sum()),
                           **{f"q{100 * v:g}": float(np.quantile(s_test[y == BENIGN_LABEL], v))
                              for v in q}})

        # ---- E: recalibrate on one half of validation, threshold on the other
        yv = np.asarray(bA.y_val)
        idx = np.arange(len(yv))
        present = np.bincount(yv)
        strat = yv if present[present > 0].min() >= 2 else None
        try:
            cal_i, thr_i = train_test_split(idx, test_size=0.5, random_state=seed, stratify=strat)
        except ValueError:
            cal_i, thr_i = train_test_split(idx, test_size=0.5, random_state=seed)
        apply = isotonic_recalibration(LA.extra["proba_val"][cal_i], yv[cal_i])
        Pc_thr, Pc_test = apply(LA.extra["proba_val"][thr_i]), apply(LA.extra["proba_test"])
        thr_benign = vb[thr_i]
        calibrated = {
            SCORE_RULES[0]: (max_attack_probability(Pc_thr, bi), max_attack_probability(Pc_test, bi)),
            SCORE_RULES[1]: (1.0 - Pc_thr[:, bi], 1.0 - Pc_test[:, bi]),
        }
        for rule, (s_val, s_test) in rules.items():
            raw_ref = np.asarray(s_val)[thr_i][thr_benign]
            cal_ref, cal_test = calibrated[rule][0][thr_benign], calibrated[rule][1]
            add(E6_STRATEGIES[1], rule, model, y, s_test, raw_ref, blocks)
            add(E6_STRATEGIES[2], rule, model, y, cal_test, cal_ref, blocks)
            for b in budgets:
                r = paired_difference(fired_at_budget(cal_test, cal_ref, b),
                                      fired_at_budget(s_test, raw_ref, b),
                                      masks, blocks, n_boot, seed)
                for c, v in r.items():
                    cal_rows.append({
                        "model": model, "score_rule": rule, "budget": b, "class": c,
                        "diff": v["diff"], "diff_lo": v["lo"], "diff_hi": v["hi"],
                        "rate_recalibrated": v["rate_a"], "rate_uncalibrated": v["rate_b"],
                        "n": v["n"], "n_blocks": v["n_blocks"],
                        "interval_counts": interval_counts(v["n_blocks"]),
                        "differs": difference_supported(v["lo"], v["hi"], v["n_blocks"]),
                        "seed": seed})
        del bA, LA, Pc_thr, Pc_test, calibrated, rules
        gc.collect()

        # ---- C: leave-one-day-out, on the training days as recorded ---------
        try:
            bR = make_bundle(args, log, "crossday", seed, val_mode="random", dedup_train=False)
            lodo = lodo_benign_scores(bR, model, seed, log, ov)
            for rule in SCORE_RULES:
                add(E6_STRATEGIES[4], rule, model, y, test_scores[rule], lodo[rule], blocks)
            del bR, lodo
        except RuntimeError as exc:
            log(f"  [skip] leave-one-day-out: {exc}")
        gc.collect()

        # ---- B: its own fit, because the training split is different -------
        bB = make_bundle(args, log, "crossday", seed, val_mode="tail")
        yB = np.asarray(bB.y_test_str, dtype=object)
        LB = classifier_scores(bB, model, seed, train_config(args), run.dir / "ckpt", log, ov)
        vbB = np.asarray(bB.y_val_str, dtype=object) == BENIGN_LABEL
        blocksB = block_ids(bB.meta_test, len(yB))
        for rule, (s_val, s_test) in classifier_rule_scores(LB).items():
            add(E6_STRATEGIES[3], rule, model, yB, s_test, np.asarray(s_val)[vbB], blocksB)
        del bB, LB
        gc.collect()

    df = pd.concat(rows, ignore_index=True)
    df["fpr_ratio"] = df["observed_fpr"] / df["budget"]
    present = [o for o in E6_STRATEGIES if o in set(df["strategy"])]
    df["strategy"] = pd.Categorical(df["strategy"], present)
    df = df.sort_values(["model", "score_rule", "strategy", "budget", "class"])
    df["strategy"] = df["strategy"].astype(str)
    df["seed"] = seed                              # one seed: the table carries no spread
    run.table(df, "e6_threshold_transfer")
    run.table(pd.DataFrame(shifts), "e6_benign_score_quantiles")
    cal = run.table(pd.DataFrame(cal_rows), "e6_calibration_effect")

    primary_rule = df[df["score_rule"] == SCORE_RULES[0]]
    view = primary_rule[primary_rule["class"] == ANY_ATTACK][
        ["model", "strategy", "budget", "observed_fpr", "fpr_ratio", "detection_rate"]]
    log(f"\nObserved false-alarm rate on the test day against the budget aimed at "
        f"(score rule: {SCORE_RULES[0]}):")
    log(view.to_string(index=False, float_format=lambda v: f"{v:.4f}"))
    cv = cal[(cal["score_rule"] == SCORE_RULES[0]) & (cal["class"].isin([BENIGN_LABEL, ANY_ATTACK]))]
    log("\nPer-class isotonic recalibration minus no recalibration, same model, same reference "
        "flows\n(paired block bootstrap; BENIGN rows are false alarms).")
    log(reading_rule())
    log(cv[["model", "budget", "class", "rate_uncalibrated", "rate_recalibrated", "diff",
            "diff_lo", "diff_hi", "n_blocks", "differs"]]
        .to_string(index=False, float_format=lambda v: f"{v:+.4f}"))

    # The rule of PROTOCOL.md for RQ6's recalibration claim, applied by code.
    primary = min(budgets, key=lambda v: abs(v - PRIMARY_BUDGET))
    verdict = run.table(rq6_verdict(cal, primary), "e6_rq6_verdict")
    answer = rq6_supported(verdict, models[0], SCORE_RULES[0], "repairs")
    improves = rq6_supported(verdict, models[0], SCORE_RULES[0], "improves")
    lowers = rq6_supported(verdict, models[0], SCORE_RULES[0], "lowers_false_alarms")
    log(f"\nRQ6 rule: at {100 * primary:g}%, recalibration REPAIRS the threshold only if the "
        "uncalibrated one was over the budget\non the test day, the recalibrated one's "
        "false-alarm rate is lower (paired interval below zero), and that\nlower rate is "
        "within the budget.  Lower but still over is an improvement.  Lower when nothing was\n"
        "over the budget is a stricter threshold, not a repair.")
    if len(verdict):
        log(verdict.to_string(index=False, float_format=lambda v: f"{v:+.4f}"))
    log(f"RQ6 'recalibration repairs the threshold' ({models[0]}, {SCORE_RULES[0]}): {answer}"
        f"   [improves: {improves}; lowers false alarms: {lowers}]")

    from figures import fig_threshold_transfer
    run.figure(fig_threshold_transfer(
        primary_rule, run.dir / "e6_threshold_transfer",
        "Does a false-alarm budget fitted before the test day hold on it?"),
        "e6_threshold_transfer")
    return run.finish({"models": models, "seed": seed, "seeds_used": [seed],
                       "primary_budget": primary, "score_rule_in_figure": SCORE_RULES[0],
                       "rq6_recalibration_repairs": answer,
                       "rq6_recalibration_improves": improves,
                       "rq6_recalibration_lowers_false_alarms": lowers})


# =====================================================================
# REPORT
# =====================================================================


def _run_stamp(d: Path) -> str:
    """The ``YYYYMMDD-HHMMSS`` in a run directory's name.

    Run directories are named ``<git sha>-<date>-<time>-<tag>``.  Sorting the
    names sorts by commit hash first, so after a commit the "newest" run would
    be whichever hash happens to sort last.  The time stamp is what orders
    them."""
    import re

    m = re.search(r"-(\d{8}-\d{6})-", d.name)
    return m.group(1) if m else ""


def latest_runs(allow_quick: bool = False) -> Dict[str, Path]:
    """Newest finished run directory of each experiment."""
    found: Dict[str, Path] = {}
    for d in sorted(Path(RUNS_DIR).glob("*-study-*"), key=lambda x: (_run_stamp(x), x.name)):
        res = d / "results.json"
        if not res.exists():
            continue
        try:
            meta = json.loads(res.read_text())
        except Exception:
            continue
        if meta.get("quick") and not allow_quick:
            continue
        found[str(meta.get("experiment"))] = d          # sorted by time => newest wins
    return found


REPORT_MANIFEST = ".report_files.json"
REPORT_ORDER = ("doctor", "e1", "e2", "e3", "e4", "e5", "e6")


# ---------------------------------------------------------------------
# Did a run follow the plan?
# ---------------------------------------------------------------------


def plan_settings(experiment: str, a: Optional[dict]) -> Dict[str, object]:
    """The settings ``experiment`` actually reads, resolved to effective values.

    ``a`` is a run's saved command line (``config.json`` -> ``args``).  Two
    runs follow the same plan exactly when this returns the same thing for
    both.  It lists, per experiment, only what that experiment reads -- E4
    trains nothing, so ``--models`` cannot change it; E2 and E6 use one seed
    whatever ``--seeds`` says -- and it resolves "not given" to the value the
    code then uses, so ``--epochs 30`` and no ``--epochs`` are one setting.

    KEEP THIS IN STEP WITH THE ``run_eN`` FUNCTIONS.  A setting an experiment
    starts reading and this function does not list is a deviation the report
    cannot see.  ``tests/test_stats_fusion.py`` pins the current list.
    """
    g = (a or {}).get
    quick = bool(g("quick"))
    out: Dict[str, object] = {"quick": quick}
    if experiment == "doctor":
        out["data"] = str(Path(g("data") or DATA_DIR).resolve())
        return out
    out["budgets"] = [float(b) for b in (g("budgets") or STUDY_BUDGETS)]
    n_boot = g("n_boot")
    out["n_boot"] = int(n_boot) if n_boot is not None else (200 if quick else BOOTSTRAP_REPLICATES)
    if quick:
        out["quick_rows"] = g("quick_rows")
    seeds = max(int(g("seeds") or 1), 1)
    epochs = int(g("epochs")) if g("epochs") else (4 if quick else 30)
    models = g("models")
    if experiment == "e1":
        out.update(seeds=seeds, models=list(models or ["xgb", "rf"]),
                   protocols=list(g("protocols") or ["random", "blocked", "crossday"]))
    elif experiment == "e2":
        out.update(model=(models or ["xgb"])[0], classes=g("classes"))
    elif experiment == "e3":
        out.update(seeds=seeds, epochs=epochs,
                   reference=None if g("no_reference") else (models or ["xgb"])[0],
                   detectors=list(g("detectors") or NOVELTY_DETECTORS),
                   protocols=list(g("protocols") or ["crossday", "blocked"]))
    elif experiment == "e4":
        out.update(window=float(g("window") or BEHAVIOUR_WINDOW_SECONDS),
                   port_threshold=int(g("port_threshold") or BEHAVIOUR_PORT_THRESHOLD),
                   host_threshold=int(g("host_threshold") or BEHAVIOUR_HOST_THRESHOLD))
    elif experiment == "e5":
        out.update(seeds=seeds, epochs=epochs, classifier=(models or ["xgb"])[0],
                   novelty=g("novelty") or PRIMARY_NOVELTY,
                   val_mode=g("val_mode") or "random", ablation=not g("no_sanitise"))
    elif experiment == "e6":
        out.update(models=list(models or ["xgb", "rf"]))
    return out


def settings_changed(experiment: str, saved: Optional[dict]) -> Dict[str, object]:
    """What a run did differently from the plan.

    The command line's defaults ARE the plan: ``paper/PROTOCOL.md`` fixes the
    seeds, the budgets, the models, the epochs and the bootstrap replicates,
    and the defaults are those values.  Only ``--quick`` marks a run as "not a
    result"; a run started with ``--seeds 1`` or ``--models dt`` is a full run
    as far as that flag goes and would be collected like any other.  So
    ``report`` prints what such a run changed, next to its tables.

    Returns the settings of ``plan_settings`` whose value differs from the
    default command line's -- an empty dict when the run followed the plan.
    A run whose command line was not recorded returns ``{"settings":
    "unknown"}``: a run that cannot be shown to follow the plan is not
    treated as if it did.
    """
    if not isinstance(saved, dict) or not saved:
        return {"settings": "unknown"}
    try:
        defaults = vars(build_parser().parse_args([experiment]))
    except SystemExit:
        return {"settings": "unknown"}
    plan, ran = plan_settings(experiment, defaults), plan_settings(experiment, saved)
    missing = object()
    return {k: v for k, v in ran.items() if plan.get(k, missing) != v}


def _settings_of_run(experiment: str, d: Path) -> Dict[str, object]:
    try:
        saved = json.loads((d / "config.json").read_text()).get("args")
    except Exception:
        saved = None
    return settings_changed(experiment, saved)


def runs_for_report(allow_quick: bool = False) -> Dict[str, tuple]:
    """For each experiment, the run ``report`` copies:
    ``(directory, settings changed, newer runs passed over)``.

    Plain report: the newest finished full run THAT FOLLOWS THE PLAN.  A later
    look at something with other settings (``e5 --seeds 1``) must not replace
    the planned run's tables just by being newer.  Only when no run of an
    experiment follows the plan is the newest one taken, and it is then named
    as off-plan.

    With ``allow_quick`` (the smoke-test report): simply the newest run.
    """
    by_exp: Dict[str, List[Path]] = {}
    for d in sorted(Path(RUNS_DIR).glob("*-study-*"), key=lambda x: (_run_stamp(x), x.name)):
        res = d / "results.json"
        if not res.exists():
            continue
        try:
            meta = json.loads(res.read_text())
        except Exception:
            continue
        if meta.get("quick") and not allow_quick:
            continue
        by_exp.setdefault(str(meta.get("experiment")), []).append(d)
    out: Dict[str, tuple] = {}
    for exp, dirs in by_exp.items():
        newest = dirs[-1]
        pick = (newest, _settings_of_run(exp, newest), [])
        if pick[1] and not allow_quick:
            for i in range(len(dirs) - 2, -1, -1):
                if not _settings_of_run(exp, dirs[i]):
                    pick = (dirs[i], {}, [x.name for x in dirs[i + 1:]])
                    break
        out[exp] = pick
    return out


# ---------------------------------------------------------------------
# What a report may remove
# ---------------------------------------------------------------------


def _experiment_of(rel: str) -> str:
    """Which experiment a report file belongs to, from its name."""
    name = Path(rel).name
    if name == "doctor.json":
        return "doctor"
    m = re.match(r"^(e[1-6])_", name)
    return m.group(1) if m else "?"


def _load_report_list(dest: Path) -> Dict[str, dict]:
    """``{relative path: {"experiment", "run"}}`` written by the previous report.

    Empty when there is none or it cannot be read -- and then nothing is
    removed.  (An earlier version wrote a plain list of paths; it is still
    understood, with the experiment taken from the file name.)
    """
    path = dest / REPORT_MANIFEST
    if not path.exists():
        return {}
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return {}
    if isinstance(data, list):
        return {p: {"experiment": _experiment_of(p), "run": "?"}
                for p in data if isinstance(p, str)}
    if isinstance(data, dict):
        return {str(k): dict(v) for k, v in data.items() if isinstance(v, dict)}
    return {}


def _remove_report_files(dest: Path, rels: Sequence[str]) -> tuple:
    """Delete the named files -- and only inside ``tables/`` and ``figures/``.

    Returns ``(number removed, [what could not be removed])``.  A file that is
    open in another program (a table in a spreadsheet, on Windows) cannot be
    deleted; that is reported, not raised, so one locked file does not stop
    the report half-way.
    """
    allowed = {(dest / "tables").resolve(), (dest / "figures").resolve()}
    removed, failed = 0, []
    for rel in rels:
        f = (dest / str(rel)).resolve()
        if not (f.is_file() and f.parent in allowed):
            continue
        try:
            f.unlink()
            removed += 1
        except OSError as exc:
            failed.append(f"{rel}: could not be removed ({exc.__class__.__name__})")
    return removed, failed


def run_report(args, outcome: Optional[Dict[str, str]] = None,
               took: Optional[Dict[str, float]] = None) -> Path:
    """Copy the newest full run of each experiment into ``paper/``.

    Writes ``paper/tables/*.csv`` and ``.md``, ``paper/figures/*`` and
    ``paper/SOURCES.md``, which names the run every file came from.  A table
    in the paper can then be traced to a run directory, a git commit and a
    command line.  ``outcome`` (from ``all``) is appended, so the file also
    says which experiments did not finish.

    WHAT IT REMOVES, AND WHAT IT NEVER DOES.  ``report`` used to copy on top
    of whatever was there, so a table of an older run could stay in
    ``paper/tables`` with nothing to tell it from the new ones.  Now every
    report writes the list of files it vouches for (``.report_files.json``),
    and the next one works experiment by experiment:

      * an experiment it HAS a run for: the files the previous report wrote
        for that experiment are removed, then the new ones are copied;
      * an experiment it has NO run for: nothing is touched.  The run folder
        may have been deleted to free disk space, and then the tables in
        ``paper/`` are the only copy.  They are kept and listed as kept;
      * a file that is not on the list is never deleted, whatever it is
        called.  Such files are named in ``SOURCES.md`` as not written by the
        report, so the folders never hold anything the file does not explain.

    WHICH RUN IT TAKES: see ``runs_for_report`` -- the newest one that follows
    the plan.  ``SOURCES.md`` prints, per run, the settings that differ from
    the plan (``settings_changed``).

    QUICK RUNS NEVER REACH ``paper/tables``.  With ``--allow-quick`` (which
    ``all --quick`` sets) everything goes to ``paper/quick/`` instead: its own
    ``tables``, ``figures`` and ``SOURCES.md``.  A smoke run can then be
    looked at, and cannot be mistaken for -- or left behind among -- results.
    """
    quick = bool(args.allow_quick)
    picks = runs_for_report(allow_quick=quick)
    dest = PAPER_DIR / "quick" if quick else PAPER_DIR
    where = "paper/quick/" if quick else "paper/"
    tables, figures = dest / "tables", dest / "figures"
    tables.mkdir(parents=True, exist_ok=True)
    figures.mkdir(parents=True, exist_ok=True)
    previous = _load_report_list(dest)
    listed: Dict[str, dict] = {}                 # every file this report vouches for
    removed = 0
    trouble: List[str] = []                      # files that could not be removed or written
    off_plan: List[str] = []
    kept: List[str] = []
    passed_over: Dict[str, List[str]] = {}

    head = ("# QUICK RUNS -- a smoke test, not results" if quick
            else "# Where every table and figure came from")
    lines = [head, "",
             "Written by `python src/study.py report`.  Do not edit by hand.", ""]
    if quick:
        lines += ["Everything in this folder comes from `--quick` runs (or was collected with",
                  "`--allow-quick`).  They are subsampled and prove only that the code runs.",
                  "Nothing here may be quoted.  Results live in `paper/tables` and",
                  "`paper/figures`, which this command never writes quick runs into.", ""]
    lines += ["| experiment | run directory | git | quick | seconds | settings changed from "
              "the plan |",
              "| --- | --- | --- | --- | --- | --- |"]
    if not picks:
        print("No finished study runs found under runs/.  Run an experiment first"
              + ("" if quick else " (quick runs are ignored without --allow-quick).")
              + "\nNothing in " + where + " was removed.")

    for exp in REPORT_ORDER:
        mine = sorted(rel for rel, v in previous.items() if v.get("experiment") == exp)
        pick = picks.get(exp)
        if pick is None:
            still = [rel for rel in mine if (dest / rel).is_file()]
            if still:
                for rel in still:
                    listed[rel] = {**previous[rel], "kept": True}
                kept.append(exp)
                lines.append(f"| {exp} | *no run folder found: {len(still)} file(s) kept from "
                             f"`runs/{previous[still[0]].get('run', '?')}`* | | | | |")
            else:
                lines.append(f"| {exp} | *not run* | | | | |")
            continue

        d, changed, newer = pick
        n, failed = _remove_report_files(dest, mine)
        removed += n
        trouble += failed
        if newer:
            passed_over[exp] = newer
        meta = json.loads((d / "results.json").read_text())
        if changed and not meta.get("quick"):
            off_plan.append(exp)
        shown = ", ".join(f"`{k}={v}`" for k, v in sorted(changed.items())) or "none"
        lines.append(f"| {exp} | `runs/{d.name}` | {meta.get('environment', {}).get('git_sha', '?')} "
                     f"| {'YES' if meta.get('quick') else 'no'} | {meta.get('seconds', 0):.0f} "
                     f"| {shown} |")

        def put(src: Path, rel: str, exp=exp, d=d) -> bool:
            try:
                shutil.copy2(src, dest / rel)
            except OSError as exc:
                trouble.append(f"{rel}: could not be written ({exc.__class__.__name__})")
                return False
            listed[rel] = {"experiment": exp, "run": d.name}
            return True

        for csv in sorted(d.glob("*.csv")):
            if not put(csv, f"tables/{csv.name}"):
                continue
            try:
                frame = pd.read_csv(csv)
                if len(frame) <= 400:
                    (tables / (csv.stem + ".md")).write_text(md_table(frame), encoding="utf-8")
                    listed[f"tables/{csv.stem}.md"] = {"experiment": exp, "run": d.name}
            except Exception:
                pass
        for fig in sorted(list(d.glob("*.png")) + list(d.glob("*.pdf"))):
            put(fig, f"figures/{fig.name}")
        if exp == "doctor" and (d / "doctor.json").exists():
            put(d / "doctor.json", "tables/doctor.json")

    # What `doctor` said about the files.  The tables are only as good as the
    # CSVs they were computed from, so the report repeats the check's verdict
    # next to them instead of leaving it in a run folder nobody opens.
    check = None
    if picks.get("doctor"):
        try:
            rj = json.loads((picks["doctor"][0] / "results.json").read_text())
            if isinstance(rj.get("verdict"), dict) and rj["verdict"]:
                check = (picks["doctor"][0].name, str(rj.get("folder", "?")), rj["verdict"])
        except (OSError, ValueError):
            check = None
    check_problems = list(check[2].get("problems", [])) if check else []

    # Whatever else sits in the two folders: not written by this report.
    strays = sorted(f"{sub.name}/{f.name}" for sub in (tables, figures)
                    for f in sub.iterdir() if f.is_file() and f"{sub.name}/{f.name}" not in listed)

    if outcome:
        lines += ["", "## Last `study.py all`", "",
                  "| experiment | outcome | minutes |", "| --- | --- | --- |"]
        for exp, what in outcome.items():
            mins = f"{(took or {}).get(exp, float('nan')) / 60:.1f}" if exp in (took or {}) else ""
            lines.append(f"| {exp} | {what} | {mins} |")
    if check:
        name, folder, v = check
        try:
            folder = str(Path(folder).resolve().relative_to(PROJECT_ROOT.resolve())) + "/"
        except ValueError:
            pass                                   # outside the project: print it as recorded
        lines += ["", "## What `doctor` said about the CSVs", "",
                  f"Newest `doctor` run that looked at the folder the experiments read: "
                  f"`runs/{name}` (folder `{folder}`). It describes the files as they were when "
                  "it ran. A run made at another time may have read other files: compare the "
                  "dates in the run names.", "",
                  f"- all eight captures present: {v.get('all_files_present')}",
                  "- source IP and timestamp usable in every capture: "
                  f"{v.get('identifiers_in_every_capture')}",
                  f"- problems listed: {len(check_problems)}"]
        lines += [f"  - {t}" for t in check_problems]
        if check_problems:
            lines += ["", "**`doctor` listed a problem with the files.** Do not quote a table "
                      "computed from them until the problem is fixed, or explained under "
                      "*Deviations* in `paper/PROTOCOL.md`."]
    if off_plan:
        lines += ["", f"**Not run as planned: {', '.join(off_plan)}.** The settings in the last "
                  "column differ from the defaults, and the defaults are what "
                  "`paper/PROTOCOL.md` fixes. Before quoting these tables, add a dated line "
                  "under *Deviations* in that file saying what was changed and why."]
    if passed_over:
        lines += ["", "**Newer runs that were not used**, because their settings differ from the "
                  "plan and an older run follows it: "
                  + "; ".join(f"{exp}: " + ", ".join(f"`runs/{n}`" for n in names)
                              for exp, names in passed_over.items()) + "."]
    if kept:
        lines += ["", f"**Kept from an earlier report: {', '.join(kept)}.** No run folder of "
                  "these experiments was found, so their files were left in place. They were "
                  "not refreshed by this report."]
    if trouble:
        lines += ["", "**Could not be removed or written** (open in another program?): "
                  + "; ".join(trouble) + ". Close the file and run `report` again."]
    lines += ["", "## Every file this report vouches for, and the run it came from", "",
              "| file | run directory |", "| --- | --- |"]
    lines += [f"| `{rel}` | `runs/{v.get('run', '?')}`{' (kept)' if v.get('kept') else ''} |"
              for rel, v in sorted(listed.items())]
    if strays:
        lines += ["", "## Other files in `tables/` and `figures/`", "",
                  "Not written by this report, and not on the list of an earlier one. `report` "
                  "never deletes such files. They are **not** covered by the table above: check "
                  "where each came from before using it.", ""]
        lines += [f"- `{name}`" for name in strays]
    (dest / "SOURCES.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    (dest / REPORT_MANIFEST).write_text(json.dumps(listed, indent=0, sort_keys=True),
                                        encoding="utf-8")

    print(f"{where} updated from {len(picks)} run(s): {dest}"
          + (f"  ({removed} file(s) of an earlier report replaced)" if removed else "")
          + ("\n  QUICK RUNS: a smoke test, not results.  paper/tables and paper/figures "
             "were not touched." if quick else ""))
    if check_problems:
        print(f"  DATASET: the newest `doctor` run listed {len(check_problems)} problem(s) with "
              "the CSVs.  They are repeated in SOURCES.md;\n  do not quote a table computed from "
              "those files until they are fixed or explained.")
    if off_plan:
        print(f"  NOT RUN AS PLANNED: {', '.join(off_plan)} -- settings differ from the "
              "defaults paper/PROTOCOL.md fixes.\n  See SOURCES.md, and record the change under "
              "Deviations before quoting those tables.")
    if kept:
        print(f"  KEPT, NOT REFRESHED: {', '.join(kept)} -- no run folder found for them; "
              "their files from an earlier report were left in place.")
    if strays:
        print(f"  {len(strays)} file(s) in tables/ or figures/ were not written by this report "
              "(listed at the end of SOURCES.md); left alone.")
    for t in trouble:
        print(f"  [problem] {t}")
    return dest


# =====================================================================
# ALL
# =====================================================================


EXPERIMENTS = ("e1", "e2", "e3", "e4", "e5", "e6")


def common_argv(args) -> List[str]:
    """The shared command-line flags of ``args``, as text, for a child process."""
    out: List[str] = []
    if args.quick:
        out += ["--quick", "--quick-rows", str(int(args.quick_rows))]
    out += ["--seeds", str(int(args.seeds))]
    if args.budgets:
        out += ["--budgets", *[repr(float(b)) for b in args.budgets]]
    if args.n_boot is not None:
        out += ["--n-boot", str(int(args.n_boot))]
    if args.epochs:
        out += ["--epochs", str(int(args.epochs))]
    if args.device:
        out += ["--device", str(args.device)]
    if args.models:
        out += ["--models", *args.models]
    if args.quiet:
        out += ["--quiet"]
    return out


def run_all(args) -> int:
    """doctor, then E1..E6, then report.

    EACH EXPERIMENT RUNS IN ITS OWN PROCESS.  Two reasons.  A Python exception
    in one experiment can be caught and the rest can carry on -- but a process
    the operating system kills for running out of memory raises nothing: it is
    simply gone, and everything queued behind it with it.  On a laptop, with
    2.8 million flows and a 300-tree forest, that is the failure to plan for.
    And a fresh process starts with empty memory, so experiment six does not
    inherit whatever experiment one failed to release.

    The summary at the end says which experiments finished, which failed and
    how; ``paper/SOURCES.md`` records the same.  Re-run a failed one by name
    (``python src/study.py e3``) or with ``all --only e3``.

    ``all --quick`` collects its runs into ``paper/quick/`` and leaves
    ``paper/tables`` and ``paper/figures`` alone (see ``run_report``).

    IT STOPS BEFORE THE FIRST EXPERIMENT WHEN ``doctor`` LISTS A PROBLEM (exit
    code 2): a missing capture, a mixture of the two downloads, timestamps
    that cannot be read or fall on the wrong day or outside the working day.
    ``--despite-problems`` runs anyway; the problems are then repeated in
    ``SOURCES.md``.  A dataset with no identifiers in ANY capture is not a
    "problem" in this sense: E4 is skipped, E5 runs with two layers, and the
    summary says so.
    """
    import subprocess

    if getattr(args, "data", None) and Path(args.data).resolve() != Path(DATA_DIR).resolve():
        # The experiments read config.DATA_DIR.  Letting `doctor` describe some
        # other folder here would make the "skip E4" decision below, and the
        # doctor.json copied into paper/, be about files nothing was run on.
        print(f"[note] --data is for `doctor` alone.  `all` checks the folder the experiments "
              f"read: {DATA_DIR}")
        args.data = None
    doctor_dir = run_doctor(args)
    verdict = json.loads((doctor_dir / "results.json").read_text()).get("verdict", {})
    problems = list(verdict.get("problems", []))
    if problems and not getattr(args, "despite_problems", False):
        # Hours of experiments on files with a known fault cannot be used, and
        # their tables would sit in paper/ looking like results.  Stop here,
        # where it costs nothing.
        print(f"\n`all` stopped before any experiment: `doctor` listed {len(problems)} "
              "problem(s) with the files in data/ (the [problem] lines above).\n"
              "  Fix the files, then run `python src/study.py doctor` until no [problem] line "
              "is left.\n"
              "  If you have looked at each problem and it is not a fault in the files, run again "
              "with --despite-problems\n"
              "  and write the reason under Deviations in paper/PROTOCOL.md.  Nothing was run "
              "and nothing in paper/ was changed.")
        return 2
    if problems:
        print(f"\n[note] --despite-problems: `doctor` listed {len(problems)} problem(s) and the "
              "experiments are run anyway.  SOURCES.md will repeat them.")
    wanted = list(getattr(args, "only", None) or EXPERIMENTS)
    outcome: Dict[str, str] = {}
    took: Dict[str, float] = {}

    for name in [e for e in EXPERIMENTS if e in wanted]:
        if name == "e4" and not verdict.get("identifiers_in_every_capture", False):
            outcome[name] = "skipped: no source IP / timestamp in the CSVs"
            continue
        cmd = [sys.executable, str(Path(__file__).resolve()), name, *common_argv(args)]
        print(f"\n>>> python src/study.py {' '.join(cmd[2:])}", flush=True)
        t0 = time.time()
        rc = subprocess.run(cmd).returncode
        took[name] = time.time() - t0
        if rc == 0:
            outcome[name] = "ok"
        elif rc < 0:
            outcome[name] = (f"KILLED by signal {-rc} -- the operating system stopped it, "
                             "which nearly always means it ran out of memory")
        else:
            outcome[name] = f"FAILED (exit code {rc}); the error is printed above"

    # The summary first: hours of work must not end without it because the
    # report hit a file it could not write.
    print("\n========== STUDY SUMMARY ==========")
    for k, v in outcome.items():
        t = f"  [{took[k] / 60:.1f} min]" if k in took else ""
        print(f"  {k}: {v}{t}")
    bad = [k for k, v in outcome.items() if not (v == "ok" or v.startswith("skipped"))]
    args.allow_quick = bool(args.quick)
    report_failed = False
    try:
        run_report(args, outcome=outcome, took=took)
    except Exception as exc:                         # the runs are safe in runs/
        report_failed = True
        print(f"\n[problem] the report could not be written: {exc!r}\n"
              "  The runs themselves are complete in runs/.  Fix the cause and run "
              "`python src/study.py report`" + (" --allow-quick" if args.quick else "") + ".")
    if bad:
        print("\n  To retry only what did not finish:  python src/study.py all --only "
              + " ".join(bad))
        print("  If one was KILLED: close other programs and try again.  If it still dies, "
              "--models xgb (no Random Forest)\n  or --seeds 1 makes it lighter -- but both "
              "change the plan: note it under Deviations in paper/PROTOCOL.md.")
    return 1 if (bad or report_failed) else 0


# =====================================================================
# CLI
# =====================================================================


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="The experiments behind the paper",
                                formatter_class=argparse.RawDescriptionHelpFormatter,
                                epilog=__doc__)
    common = argparse.ArgumentParser(add_help=False)
    common.add_argument("--quick", action="store_true",
                        help="smoke run on a training subsample; never a result")
    common.add_argument("--quick-rows", dest="quick_rows", type=int, default=120_000,
                        help="training rows kept by --quick")
    common.add_argument("--seeds", type=int, default=3,
                        help="repeats for models whose fit is random (default 3)")
    common.add_argument("--budgets", type=float, nargs="+",
                        help="benign false-alarm budgets (default 0.001 0.005 0.01)")
    common.add_argument("--n-boot", dest="n_boot", type=int,
                        help="block-bootstrap replicates (default 1000)")
    common.add_argument("--epochs", type=int, help="epochs for the neural models (default 30)")
    common.add_argument("--device", help="cpu | cuda (default: auto)")
    common.add_argument("--models", nargs="+", help="classifiers: xgb rf lgbm mlp cnn lstm")
    common.add_argument("--quiet", action="store_true", help="less split-building output")
    common.add_argument("--data", help="folder holding the eight CSVs (doctor only)")
    common.set_defaults(protocols=None, detectors=None, classes=None, novelty=None,
                        val_mode="random", no_reference=False, window=None,
                        port_threshold=None, host_threshold=None, allow_quick=False,
                        no_sanitise=False)

    sub = p.add_subparsers(dest="command", required=True)
    sub.add_parser("doctor", parents=[common], help="describe the CSVs in data/").set_defaults(func=run_doctor)

    e1 = sub.add_parser("e1", parents=[common], help="RQ1 protocol effect")
    e1.add_argument("--protocols", nargs="+", choices=["random", "blocked", "crossday", "closedset"])
    e1.set_defaults(func=run_e1)

    e2 = sub.add_parser("e2", parents=[common], help="RQ2 seen versus unseen")
    e2.add_argument("--classes", nargs="+", help="attack classes to hold out (default: all)")
    e2.set_defaults(func=run_e2)

    e3 = sub.add_parser("e3", parents=[common], help="RQ3 novelty layer")
    e3.add_argument("--detectors", nargs="+", choices=list(NOVELTY_DETECTORS))
    e3.add_argument("--protocols", nargs="+", choices=["crossday", "blocked"])
    e3.add_argument("--no-reference", dest="no_reference", action="store_true",
                    help="do not add the supervised classifier as a reference row")
    e3.set_defaults(func=run_e3)

    e4 = sub.add_parser("e4", parents=[common], help="RQ4 behaviour layer")
    e4.add_argument("--window", type=float, help="window length in seconds (default 60)")
    e4.add_argument("--port-threshold", dest="port_threshold", type=int)
    e4.add_argument("--host-threshold", dest="host_threshold", type=int)
    e4.set_defaults(func=run_e4)

    e5 = sub.add_parser("e5", parents=[common], help="RQ5 fusion")
    e5.add_argument("--novelty", choices=list(NOVELTY_DETECTORS),
                    help="the novelty layer (default autoencoder, fixed before the runs)")
    e5.add_argument("--val-mode", dest="val_mode", choices=["random", "tail"], default="random")
    e5.add_argument("--no-sanitise", dest="no_sanitise", action="store_true",
                    help="skip the 'scan-like benign flows removed' ablation")
    e5.set_defaults(func=run_e5)

    e6 = sub.add_parser("e6", parents=[common], help="RQ6 threshold transfer")
    e6.set_defaults(func=run_e6)

    al = sub.add_parser("all", parents=[common], help="doctor, e1..e6, report")
    al.add_argument("--only", nargs="+", choices=list(EXPERIMENTS),
                    help="run just these experiments (doctor and report still run)")
    al.add_argument("--despite-problems", dest="despite_problems", action="store_true",
                    help="run the experiments although `doctor` lists a problem with the CSVs "
                         "(without this, `all` stops before the first experiment)")
    al.set_defaults(func=run_all)

    rp = sub.add_parser("report", help="collect the newest runs into paper/")
    rp.add_argument("--allow-quick", dest="allow_quick", action="store_true",
                    help="include --quick runs; the report then goes to paper/quick/, "
                         "never to paper/tables")
    rp.set_defaults(func=run_report)
    return p


def main(argv: Optional[List[str]] = None) -> int:
    args = build_parser().parse_args(argv)
    out = args.func(args)
    return out if isinstance(out, int) else 0


if __name__ == "__main__":
    raise SystemExit(main())
