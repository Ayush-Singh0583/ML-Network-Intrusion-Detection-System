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

DATA_DIR = PROJECT_ROOT / "data"
CACHE_DIR = PROJECT_ROOT / "cache"
RUNS_DIR = PROJECT_ROOT / "runs"
ARTIFACT_DIR = PROJECT_ROOT / "saved_models"

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
