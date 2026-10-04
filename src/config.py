"""
Central configuration for the ML-NIDS project.

Every literal that used to be scattered across twelve entry-point scripts lives
here.  Nothing else in the codebase may hard-code a path, a seed, a threshold,
an epoch count or a feature dimension.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Dict, List, Optional

# =====================================================================
# PATHS
# =====================================================================

PROJECT_ROOT = Path(__file__).resolve().parent.parent

# Each location can be redirected with an environment variable.  The tests use
# this to run the whole study against synthetic captures in a temporary
# directory, so that a test run can never read, or overwrite, a real cache or
# a real result.
DATA_DIR = Path(os.environ.get("NIDS_DATA_DIR", PROJECT_ROOT / "data"))
CACHE_DIR = Path(os.environ.get("NIDS_CACHE_DIR", PROJECT_ROOT / "cache"))
RUNS_DIR = Path(os.environ.get("NIDS_RUNS_DIR", PROJECT_ROOT / "runs"))
ARTIFACT_DIR = Path(os.environ.get("NIDS_ARTIFACT_DIR", PROJECT_ROOT / "saved_models"))

for _d in (CACHE_DIR, RUNS_DIR, ARTIFACT_DIR):
    _d.mkdir(parents=True, exist_ok=True)

CLEAN_PARQUET = CACHE_DIR / "clean.parquet"


# =====================================================================
# DATASET MANIFEST
# ---------------------------------------------------------------------
# BUG FIX: the old loader globbed data/*.csv, which silently ingested
# data/custom_sample.csv (115 rows derived from the Friday DDoS capture,
# tagged Day="Custom").  Dataset composition must not depend on which
# files happen to be sitting in a directory.
# =====================================================================

DAY_FILES: Dict[str, List[str]] = {
    "Monday": [
        "Monday-WorkingHours.pcap_ISCX.csv",
    ],
    "Tuesday": [
        "Tuesday-WorkingHours.pcap_ISCX.csv",
    ],
    "Wednesday": [
        "Wednesday-workingHours.pcap_ISCX.csv",
    ],
    "Thursday": [
        "Thursday-WorkingHours-Morning-WebAttacks.pcap_ISCX.csv",
        "Thursday-WorkingHours-Afternoon-Infilteration.pcap_ISCX.csv",
    ],
    "Friday": [
        "Friday-WorkingHours-Morning.pcap_ISCX.csv",
        "Friday-WorkingHours-Afternoon-PortScan.pcap_ISCX.csv",
        "Friday-WorkingHours-Afternoon-DDos.pcap_ISCX.csv",
    ],
}

DAY_ORDER = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday"]


# =====================================================================
# SPLIT PROTOCOL
# ---------------------------------------------------------------------
# Thursday is the validation day and Friday is the test day, touched
# exactly once.
#
# Subtlety that a naive Mon-Wed / Thu / Fri split gets wrong: Thursday is
# the ONLY day carrying Infiltration, Web Brute Force, XSS and SQL
# Injection.  Holding all of Thursday out therefore leaves a validation
# set whose only class is BENIGN -- macro-F1 is then 1.0 by construction
# and early stopping selects on noise.  (Verified: validation collapses
# to a single class.)
#
# So Thursday is stratified into a training half and a validation half.
# The validation half covers every known class, Friday is still never
# touched during training or model selection, and DDoS / PortScan / Bot
# remain genuinely unseen.
#
# Set THURSDAY_VAL_FRAC = 1.0 for the purist "whole day held out"
# variant, accepting the degenerate validation set.
# =====================================================================

TRAIN_DAYS = ["Monday", "Tuesday", "Wednesday"]
VAL_DAYS = ["Thursday"]
TEST_DAYS = ["Friday"]

THURSDAY_VAL_FRAC = 0.5


# =====================================================================
# COLUMNS
# =====================================================================

LABEL_COL = "Label"
DAY_COL = "Day"

# Network identifiers.  Destination Port in particular is the shortcut that
# lets a tree learn "port 21 -> FTP-Patator" without learning anything about
# traffic.  Keeping it removed is a deliberate methodological choice.
IDENTIFIER_COLUMNS = [
    "Flow ID",
    "Source IP",
    "Src IP",
    "Source Port",
    "Src Port",
    "Destination IP",
    "Dst IP",
    "Destination Port",
    "Dst Port",
    "Timestamp",
    "Protocol",
]

# Exactly-collinear duplicate features.  Verified byte-identical over 400,000
# rows of the project's own preprocessed dataset:
#
#   Fwd Header Length      == Fwd Header Length.1   (pandas auto-rename of a
#                                                    duplicated CSV header)
#   Total Fwd Packets      == Subflow Fwd Packets
#   Total Backward Packets == Subflow Bwd Packets
#   Total Length of Fwd..  == Subflow Fwd Bytes
#   Total Length of Bwd..  == Subflow Bwd Bytes
#   Fwd Packet Length Mean == Avg Fwd Segment Size
#   Bwd Packet Length Mean == Avg Bwd Segment Size
#
# Keeping both members of a pair splits tree feature-importance arbitrarily
# between the twins and wastes a dimension in every neural encoder.
DUPLICATE_FEATURES = [
    "Fwd Header Length.1",
    "Subflow Fwd Packets",
    "Subflow Bwd Packets",
    "Subflow Fwd Bytes",
    "Subflow Bwd Bytes",
    "Avg Fwd Segment Size",
    "Avg Bwd Segment Size",
]

# =====================================================================
# META COLUMNS  (carried beside the features, NEVER used as features)
# ---------------------------------------------------------------------
# The identifiers above are dropped from the feature set, and that stays
# true.  But two things in the study need them as *side information*:
#
#   * time ordering   -- a time-blocked split needs to know which flow came
#                        first; a row index is only a proxy for that
#   * the behaviour layer -- "one source touched 900 ports in a minute" is a
#                        statement about source IP, destination port and
#                        time, none of which may be a per-flow feature
#
# So ``build_cache`` copies them into columns with the ``meta__`` prefix
# before the identifiers are dropped.  Every place that builds a feature
# matrix goes through ``meta.feature_columns``, which excludes the prefix,
# and ``tests/test_meta.py`` asserts that no ``meta__`` column can reach a
# model.
#
# The MachineLearningCSV distribution of CIC-IDS2017 ships no source IP and
# no timestamp.  In that case the columns exist and are null, the pipeline
# still runs, and the behaviour layer refuses to start with a message that
# says which download carries them (GeneratedLabelledFlows.zip).
# =====================================================================

META_PREFIX = "meta__"
META_SRC_IP = "meta__src_ip"
META_DST_IP = "meta__dst_ip"
META_SRC_PORT = "meta__src_port"
META_DST_PORT = "meta__dst_port"
META_PROTO = "meta__proto"
META_TS = "meta__ts"            # seconds since the Unix epoch, capture-local clock
META_ROW = "meta__row"          # row index inside the capture file
META_CAPTURE = "meta__capture"  # index of the capture in manifest order

META_COLUMNS = [
    META_SRC_IP, META_DST_IP, META_SRC_PORT, META_DST_PORT,
    META_PROTO, META_TS, META_ROW, META_CAPTURE,
]

# raw header (after .str.strip()) -> meta column.  Both CICFlowMeter naming
# generations are listed, as in IDENTIFIER_COLUMNS.
RAW_META_ALIASES: Dict[str, List[str]] = {
    META_SRC_IP: ["Source IP", "Src IP"],
    META_DST_IP: ["Destination IP", "Dst IP"],
    META_SRC_PORT: ["Source Port", "Src Port"],
    META_DST_PORT: ["Destination Port", "Dst Port"],
    META_PROTO: ["Protocol"],
    META_TS: ["Timestamp"],
}

# Bump when the cache layout changes, so an old cache is rebuilt rather than
# silently read without the columns the new code expects.
CACHE_VERSION = 2

BENIGN_LABEL = "BENIGN"
UNKNOWN_LABEL = "Unknown_Attack"

LABEL_ALIASES = {
    "Web Attack - Brute Force": "WebAttack_BruteForce",
    "Web Attack - XSS": "WebAttack_XSS",
    "Web Attack - Sql Injection": "WebAttack_SQLInjection",
    "Web Attack \x96 Brute Force": "WebAttack_BruteForce",
    "Web Attack \x96 XSS": "WebAttack_XSS",
    "Web Attack \x96 Sql Injection": "WebAttack_SQLInjection",
}


# =====================================================================
# REPRODUCIBILITY
# =====================================================================

SEED = 42


# =====================================================================
# TRAINING CONFIG
# =====================================================================


@dataclass
class TrainConfig:
    """Hyper-parameters for the deep models."""

    model: str = "mlp"                       # mlp | cnn | lstm | autoencoder | deep_svdd
    protocol: str = "crossday"               # crossday | closedset

    epochs: int = 60
    batch_size: int = 1024
    eval_batch_size: int = 8192
    lr: float = 1e-3
    weight_decay: float = 1e-2               # AdamW decoupled decay
    warmup_epochs: int = 3
    min_lr_factor: float = 1e-2              # cosine floor = lr * this

    grad_clip: float = 1.0
    patience: int = 12                       # early-stopping patience (epochs)
    min_delta: float = 1e-5

    # imbalance handling: "focal" | "weighted_ce" | "none"
    loss: str = "focal"
    focal_gamma: float = 2.0
    class_weight_power: float = 0.5          # w_c proportional to (N/n_c)**power

    dropout: float = 0.2
    # LayerNorm, not BatchNorm. BN's running statistics are estimated from
    # ~80%-BENIGN batches and rescale OOD flows toward that benign reference,
    # which inverts every logit-based novelty score (measured: energy AUROC
    # 0.120 under BN vs 0.630 under LN on near-OOD). See models.make_norm.
    norm: str = "ln"                         # ln | bn | none
    # Class-rebalanced sampling. 0.0 = uniform, 0.5 = sqrt, 1.0 = fully balanced.
    #
    # Default OFF, deliberately. On a 12-class ablation at CIC-IDS2017-like
    # imbalance the sqrt sampler traded macro-F1 for rare-class recall rather
    # than improving both:
    #     uniform      macro-F1 0.857   rare-class recall 0.665
    #     sqrt sampler macro-F1 0.785   rare-class recall 0.751
    # That ablation was NOT collapsing to the majority class, so it cannot say
    # what the sampler does to a model that is. Turn it on when the per-class
    # validation table printed at the end of training shows classes with ZERO
    # recall -- that is the collapse the sampler exists to fix.
    sampler_power: float = 0.0
    hidden: int = 256
    depth: int = 3
    latent_dim: int = 32

    amp: bool = True                         # mixed precision when CUDA present
    num_workers: int = 4
    pin_memory: bool = True
    persistent_workers: bool = True

    # model selection metric on the validation day
    monitor: str = "macro_f1"                # macro_f1 | balanced_accuracy | loss
    monitor_mode: str = "max"                # max | min

    seed: int = SEED
    device: Optional[str] = None             # None -> auto

    # open-set
    target_benign_fpr: float = 0.05          # tau calibrated at this FPR on val
    # Primary rejection score. mahalanobis_embed measures distance from the
    # training manifold in feature space and is the only scorer that survives
    # near-OOD (AUROC 0.914 under BN / 0.930 under LN, vs 0.120 / 0.630 for
    # energy on the same models). msp and energy are still computed and
    # reported alongside so the choice is evidence, not assertion.
    scorer: str = "mahalanobis_embed"        # mahalanobis_embed | energy | msp

    tags: Dict[str, str] = field(default_factory=dict)

    def to_dict(self) -> dict:
        return asdict(self)


# =====================================================================
# SKLEARN BASELINE CONFIG
# =====================================================================

RF_PARAMS = dict(
    n_estimators=300,
    max_depth=24,                 # was unbounded -> 150 MB pickle, pure leaves
    min_samples_leaf=5,           # no single-sample leaves for n=11 classes
    max_features="sqrt",
    class_weight="balanced_subsample",
    n_jobs=-1,
    random_state=SEED,
)

XGB_PARAMS = dict(
    n_estimators=400,
    max_depth=8,
    learning_rate=0.05,
    subsample=0.8,
    colsample_bytree=0.8,
    reg_lambda=1.0,
    min_child_weight=5,
    tree_method="hist",           # 5-10x faster than the default path here
    objective="multi:softprob",   # was implicit; forked silently on binary data
    eval_metric="mlogloss",       # was "logloss" (the binary metric)
    random_state=SEED,
    n_jobs=-1,
)

LGBM_PARAMS = dict(
    objective="multiclass",
    n_estimators=400,
    learning_rate=0.05,
    num_leaves=63,                # was 255 with max_depth=-1
    max_depth=12,                 # was -1 (unbounded)
    min_child_samples=50,         # was 20 -> private leaves for rare classes
    subsample=0.8,
    subsample_freq=1,             # BUG FIX: subsample is a no-op without this
    colsample_bytree=0.8,
    reg_lambda=1.0,
    class_weight="balanced",
    random_state=SEED,
    force_col_wise=True,
    verbose=-1,
    n_jobs=-1,
)


# =====================================================================
# STUDY CONFIG  (src/study.py -- the experiments behind the paper)
# =====================================================================

# Benign false-alarm budgets every detector is compared at.  Thresholds are
# fitted on validation benign scores only; the rate observed on the test split
# is reported next to the target because the two differ.
STUDY_BUDGETS = (0.001, 0.005, 0.01)

# Time-blocked split: per (capture, class), earliest 60% of flows -> train,
# next 20% -> validation, last 20% -> test.
BLOCKED_FRACS = (0.6, 0.2, 0.2)

# Behaviour layer.  Same defaults as the live scan tracker, so the offline
# measurement describes the component that actually runs.  These are COPIES of
# backend/live/scan_tracker.py's WINDOW_SECONDS, VERTICAL_PORT_THRESHOLD and
# HORIZONTAL_HOST_THRESHOLD (src/ must not import the backend to read three
# numbers); tests/test_behaviour_layer.py fails if the two ever differ.
BEHAVIOUR_WINDOW_SECONDS = 60.0
BEHAVIOUR_PORT_THRESHOLD = 100
BEHAVIOUR_HOST_THRESHOLD = 50

# Ports below this are treated as service ports when repairing flow direction
# (see behaviour.canonical_endpoints).
SERVICE_PORT_MAX = 1024

# The dataset was captured 09:00-17:00 local time and its CSVs print a 12-hour
# clock with no AM/PM marker.  An hour below this value is read as afternoon.
TIMESTAMP_PM_BELOW_HOUR = 8

# A capture (or a split) "has identifiers" when at least this share of its rows
# carries a source IP and a timestamp that parses.  A round number, not a
# derived one: it lets a handful of broken rows through and nothing more.
# One constant so that `doctor`, the cache, the time-ordered splits and the
# bootstrap blocks all ask the same question (they used to carry four copies
# of the literal).
IDENTIFIER_COVERAGE = 0.99

# Block bootstrap: flows inside one burst are not independent, so confidence
# intervals resample time blocks, not single flows.
BOOTSTRAP_BLOCK_SECONDS = 60.0
BOOTSTRAP_BLOCK_ROWS = 2000      # fallback when a split has no timestamps
BOOTSTRAP_REPLICATES = 1000

# Isolation Forest baseline of the novelty experiment.  Written with the study
# and never adjusted: nothing in the study is tuned (paper/PROTOCOL.md).
IFOREST_PARAMS = dict(
    n_estimators=200,
    max_samples=4096,
    contamination="auto",
    n_jobs=-1,
    random_state=SEED,
)


def resolve_device(requested: Optional[str] = None) -> str:
    if requested:
        return requested
    env = os.environ.get("NIDS_DEVICE")
    if env:
        return env
    try:
        import torch

        if torch.cuda.is_available():
            return "cuda"
    except Exception:
        pass
    return "cpu"
