#!/usr/bin/env python
"""
Single entry point.

Replaces main.py, main_rf.py, main_dt.py, main_lr.py, main_xgb.py, main_lgbm.py,
main_binary.py, main_binary_crossday.py, main_attack.py, main_attack_crossday.py,
train_svdd.py, evaluate_svdd.py, find_svdd_radius.py, compare_models.py --
fourteen scripts that each re-ran the full pipeline from the raw CSVs at import
time, with no shared configuration and no way to reproduce which one produced
which artifact.

Usage
-----
    python src/run.py cache
    python src/run.py train --model mlp  --protocol crossday
    python src/run.py train --model cnn  --protocol closedset --epochs 40
    python src/run.py train --model deep_svdd --pretrain-ae
    python src/run.py train --model rf   --protocol crossday
    python src/run.py compare --protocol closedset
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Dict, List, Optional

# make `src/` importable regardless of the caller's working directory
SRC_DIR = Path(__file__).resolve().parent
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

import numpy as np
import pandas as pd

from config import (
    BENIGN_LABEL,
    CLEAN_PARQUET,
    TrainConfig,
    UNKNOWN_LABEL,
    resolve_device,
)
from evaluation import (
    EvalReport,
    decode_predictions,
    evaluate_predictions,
    feature_importance,
    new_run_dir,
    plot_confusion,
    plot_history,
    save_json,
)
from openset import (
    MahalanobisScorer,
    energy_score,
    make_open_set_predictions,
    msp_score,
    open_set_report,
    threshold_at_fpr,
    unknown_mask,
)
from preprocessing import benign_index, build_cache, build_splits, load_clean, open_set_truth
from seeding import set_seed
from training import SKLEARN_TRAINERS, save_bundle

SKLEARN_ALIASES = {
    "rf": "random_forest",
    "random_forest": "random_forest",
    "xgb": "xgboost",
    "xgboost": "xgboost",
    "lgbm": "lightgbm",
    "lightgbm": "lightgbm",
    "lr": "logistic",
    "logistic": "logistic",
    "dt": "decision_tree",
    "decision_tree": "decision_tree",
}
TORCH_MODELS = {"mlp", "cnn", "lstm", "autoencoder", "deep_svdd"}


class Tee:
    """Write the console log into the run directory as well as stdout."""

    def __init__(self, path: Path):
        self.f = open(path, "w", encoding="utf-8")

    def __call__(self, msg: str = "") -> None:
        print(msg)
        self.f.write(str(msg) + "\n")
        self.f.flush()

    def close(self) -> None:
        self.f.close()


# =====================================================================
# SHARED EVALUATION
# =====================================================================


def evaluate_classifier(
    y_test_str: np.ndarray,
    y_pred_str: np.ndarray,
    scores_test: np.ndarray,
    scores_val_known: np.ndarray,
    known_classes: List[str],
    cfg: TrainConfig,
    run_dir: Path,
    scorer_name: str,
    log,
) -> Dict[str, dict]:
    """
    Three reports, all from the same ground-truth arrays:

      closed_set  -- known-class rows only, no rejection
      open_set    -- every row, unseen classes folded into Unknown_Attack,
                     rejection applied at a validation-calibrated tau
      detection   -- AUROC / AUPR / TPR@5%FPR of the rejection score itself
    """
    is_unknown = unknown_mask(y_test_str, known_classes)
    results: Dict[str, dict] = {}

    # ---- closed-set on known rows -------------------------------------
    if (~is_unknown).any():
        labels = sorted(set(np.asarray(y_test_str, dtype=object)[~is_unknown].tolist())
                        | set(np.asarray(y_pred_str, dtype=object)[~is_unknown].tolist()))
        rep = evaluate_predictions(
            np.asarray(y_test_str, dtype=object)[~is_unknown],
            np.asarray(y_pred_str, dtype=object)[~is_unknown],
            tag="closed_set", labels=labels,
        )
        log(str(rep))
        rep.save(run_dir)
        plot_confusion(rep, run_dir / "closed_set_confusion.png")
        results["closed_set"] = rep.to_dict()

    # ---- open-set -----------------------------------------------------
    if is_unknown.any():
        tau = threshold_at_fpr(scores_val_known, cfg.target_benign_fpr)
        log(f"\ntau calibrated on VALIDATION only: {tau:.6f} "
            f"(target known-rejection FPR {cfg.target_benign_fpr:.0%}, scorer={scorer_name})")

        y_true_open = open_set_truth(y_test_str, known_classes)
        y_pred_open = make_open_set_predictions(y_pred_str, scores_test, tau)

        labels = sorted(set(y_true_open.tolist()) | set(y_pred_open.tolist()))
        rep = evaluate_predictions(y_true_open, y_pred_open, tag="open_set", labels=labels)
        log(str(rep))
        rep.save(run_dir)
        plot_confusion(rep, run_dir / "open_set_confusion.png")
        results["open_set"] = rep.to_dict()

        det = open_set_report(scores_test, is_unknown, tau, scorer=scorer_name)
        log(str(det))
        save_json(det.to_dict(), run_dir / "detection_metrics.json")
        results["detection"] = det.to_dict()
    else:
        log("\n[note] test split contains no unknown classes; open-set report skipped")

    return results


# =====================================================================
# SKLEARN PATH
# =====================================================================


def run_sklearn(name: str, cfg: TrainConfig, args, log) -> Dict[str, dict]:
    from training import train_lightgbm, train_xgboost

    bundle = build_splits(protocol=cfg.protocol, seed=cfg.seed)
    run_dir = Path(args.run_dir)

    t0 = time.time()
    trainer = SKLEARN_TRAINERS[name]
    if name in ("xgboost", "lightgbm"):
        model = trainer(bundle.X_train, bundle.y_train, bundle.X_val, bundle.y_val)
    else:
        model = trainer(bundle.X_train, bundle.y_train)
    log(f"\nTrained {name} in {time.time() - t0:.1f}s")

    proba = model.predict_proba(bundle.X_test)
    y_pred_idx = proba.argmax(axis=1)
    y_pred_str = decode_predictions(y_pred_idx, bundle.encoder)

    # Tree ensembles have no logits, so Mahalanobis on the scaled features is
    # the rejection score.  Max-softmax on an RF is degenerate (mostly 1.0).
    scorer = MahalanobisScorer().fit(bundle.X_train, bundle.y_train)
    scores_test = scorer.score(bundle.X_test)
    scores_val = scorer.score(bundle.X_val)
    scorer_name = "mahalanobis"

    results = evaluate_classifier(
        bundle.y_test_str, y_pred_str, scores_test, scores_val,
        bundle.known_classes, cfg, run_dir, scorer_name, log,
    )

    feature_importance(model, bundle.feature_names, run_dir, tag=name)
    save_bundle(model, bundle.scaler, bundle.encoder, bundle.cleaner,
                bundle.feature_names, f"{name}_{cfg.protocol}",
                extra_meta={"protocol": cfg.protocol, "run_dir": str(run_dir)})
    return results


# =====================================================================
# TORCH PATH
# =====================================================================


def run_torch(name: str, cfg: TrainConfig, args, log) -> Dict[str, dict]:
    import torch

    from datasets import build_loaders
    from engine import (
        embed,
        predict_logits,
        reconstruction_errors,
        svdd_distances,
        train_autoencoder,
        train_classifier,
        train_deep_svdd,
    )
    from losses import build_loss
    from models import Autoencoder, CLASSIFIERS, build_model, count_parameters, init_center

    device = resolve_device(cfg.device)
    log(f"Device : {device}")
    if device.startswith("cuda"):
        log(f"GPU    : {torch.cuda.get_device_name(0)}")

    bundle = build_splits(protocol=cfg.protocol, seed=cfg.seed)
    run_dir = Path(args.run_dir)
    ckpt = run_dir / f"{name}_best.pt"

    # ---------------- classifiers ----------------
    if name in CLASSIFIERS:
        model = build_model(name, bundle.n_features, bundle.n_classes, cfg)
        log(f"Model  : {name}  ({count_parameters(model):,} trainable parameters)")

        train_loader = build_loaders(bundle.X_train, bundle.y_train, cfg.batch_size, True, cfg, device)
        val_loader = build_loaders(bundle.X_val, bundle.y_val, cfg.eval_batch_size, False, cfg, device)
        test_loader = build_loaders(bundle.X_test, None, cfg.eval_batch_size, False, cfg, device)

        criterion = build_loss(cfg, bundle.y_train, bundle.n_classes)
        log(f"Loss   : {type(criterion).__name__}")

        history = train_classifier(
            model, train_loader, val_loader, bundle.y_val,
            criterion, cfg, device, ckpt, log=log,
        )
        history.save(run_dir / "history.json")
        plot_history(history.rows, run_dir / "history.png")

        logits_test = predict_logits(model, test_loader, device)
        logits_val = predict_logits(model, val_loader, device)

        y_pred_str = decode_predictions(logits_test.argmax(axis=1), bundle.encoder)

        # energy retains logit magnitude; msp is kept for comparison
        scores_test = energy_score(logits_test)
        scores_val = energy_score(logits_val)

        results = evaluate_classifier(
            bundle.y_test_str, y_pred_str, scores_test, scores_val,
            bundle.known_classes, cfg, run_dir, "energy", log,
        )

        # secondary scorers, reported side by side
        if unknown_mask(bundle.y_test_str, bundle.known_classes).any():
            extra = {}
            is_unk = unknown_mask(bundle.y_test_str, bundle.known_classes)

            msp_t, msp_v = msp_score(logits_test), msp_score(logits_val)
            extra["msp"] = open_set_report(
                msp_t, is_unk, threshold_at_fpr(msp_v, cfg.target_benign_fpr), "msp"
            ).to_dict()

            emb_tr = embed(model, build_loaders(bundle.X_train, None, cfg.eval_batch_size, False, cfg, device), device)
            maha = MahalanobisScorer().fit(emb_tr, bundle.y_train)
            m_t = maha.score(embed(model, test_loader, device))
            m_v = maha.score(embed(model, val_loader, device))
            extra["mahalanobis_embed"] = open_set_report(
                m_t, is_unk, threshold_at_fpr(m_v, cfg.target_benign_fpr), "mahalanobis_embed"
            ).to_dict()

            for k, v in extra.items():
                log(f"\n[alt scorer] {k}: AUROC={v['auroc']:.6f}  AUPR={v['aupr']:.6f}")
            save_json(extra, run_dir / "alt_scorers.json")
            results["alt_scorers"] = extra

        torch.save({"state_dict": model.state_dict(), "config": cfg.to_dict(),
                    "feature_names": bundle.feature_names,
                    "classes": list(map(str, bundle.encoder.classes_))},
                   run_dir / f"{name}_final.pt")
        return results

    # ---------------- one-class models ----------------
    b_idx = benign_index(bundle.encoder)
    Xtr_b = bundle.X_train[bundle.y_train == b_idx]
    Xva_b = bundle.X_val[bundle.y_val == b_idx]
    log(f"\nOne-class training on BENIGN only: "
        f"{len(Xtr_b):,} train / {len(Xva_b):,} val")
    if len(Xtr_b) == 0 or len(Xva_b) == 0:
        raise ValueError("no BENIGN rows in train or val split")

    tr_loader = build_loaders(Xtr_b, None, cfg.batch_size, True, cfg, device)
    va_loader = build_loaders(Xva_b, None, cfg.eval_batch_size, False, cfg, device)
    te_loader = build_loaders(bundle.X_test, None, cfg.eval_batch_size, False, cfg, device)

    if name == "autoencoder":
        model = build_model("autoencoder", bundle.n_features, bundle.n_classes, cfg)
        log(f"Model  : autoencoder ({count_parameters(model):,} parameters)")
        history = train_autoencoder(model, tr_loader, va_loader, cfg, device, ckpt, log=log)
        history.save(run_dir / "history.json")
        plot_history(history.rows, run_dir / "history.png")
        scores_test = reconstruction_errors(model, te_loader, device)
        scores_val = reconstruction_errors(model, va_loader, device)
        scorer_name = "ae_reconstruction"

    elif name == "deep_svdd":
        model = build_model("deep_svdd", bundle.n_features, bundle.n_classes, cfg).to(device)

        if args.pretrain_ae:
            log("\n---------- autoencoder pretraining ----------")
            ae = Autoencoder(bundle.n_features, latent_dim=cfg.latent_dim).to(device)
            ae_cfg = TrainConfig(**{**cfg.to_dict(), "epochs": max(cfg.epochs // 2, 10)})
            train_autoencoder(ae, tr_loader, va_loader, ae_cfg, device,
                              run_dir / "svdd_ae_pretrain.pt", log=log)
            model.load_pretrained_encoder(ae)
            log("Encoder warm-started from the pretrained autoencoder (weights only, no biases)")

        center = init_center(model, tr_loader, device, eps=0.1)
        log(f"Center initialised: ||c|| = {float(center.norm()):.6f}, "
            f"min |c_i| = {float(center.abs().min()):.4f}  (eps-guard applied)")

        history, center = train_deep_svdd(model, center, tr_loader, va_loader,
                                          cfg, device, ckpt, log=log)
        history.save(run_dir / "history.json")
        plot_history(history.rows, run_dir / "history.png")

        scores_test = svdd_distances(model, center, te_loader, device)
        scores_val = svdd_distances(model, center, va_loader, device)
        scorer_name = "svdd_distance"
        torch.save(center.cpu(), run_dir / "svdd_center.pt")
    else:
        raise ValueError(f"unhandled torch model {name!r}")

    # one-class evaluation: BENIGN vs ATTACK on the test day
    is_attack = np.asarray(bundle.y_test_str, dtype=object) != BENIGN_LABEL
    tau = threshold_at_fpr(scores_val, cfg.target_benign_fpr)
    log(f"\ntau calibrated on VALIDATION benign only: {tau:.6e}")

    det = open_set_report(scores_test, is_attack, tau, scorer=scorer_name)
    log(str(det))
    save_json(det.to_dict(), run_dir / "detection_metrics.json")

    y_true = np.where(is_attack, "ATTACK", BENIGN_LABEL).astype(object)
    y_pred = np.where(scores_test > tau, "ATTACK", BENIGN_LABEL).astype(object)
    rep = evaluate_predictions(y_true, y_pred, tag="one_class",
                               labels=[BENIGN_LABEL, "ATTACK"])
    log(str(rep))
    rep.save(run_dir)
    plot_confusion(rep, run_dir / "one_class_confusion.png")

    np.save(run_dir / "scores_test.npy", scores_test)
    return {"detection": det.to_dict(), "one_class": rep.to_dict()}


# =====================================================================
# CLI
# =====================================================================


def cfg_from_args(args) -> TrainConfig:
    cfg = TrainConfig()
    for field_name in cfg.to_dict():
        if hasattr(args, field_name) and getattr(args, field_name) is not None:
            setattr(cfg, field_name, getattr(args, field_name))
    cfg.model = args.model
    cfg.protocol = args.protocol
    return cfg


def cmd_cache(args) -> int:
    build_cache(out_path=CLEAN_PARQUET)
    return 0


def cmd_train(args) -> int:
    cfg = cfg_from_args(args)
    set_seed(cfg.seed)

    tag = f"{cfg.model}-{cfg.protocol}"
    run_dir = new_run_dir(tag)
    args.run_dir = run_dir
    log = Tee(run_dir / "train.log")

    try:
        log(f"========== RUN {run_dir.name} ==========")
        log(json.dumps(cfg.to_dict(), indent=2))
        save_json(cfg.to_dict(), run_dir / "config.json")

        name = args.model.lower()
        t0 = time.time()
        if name in SKLEARN_ALIASES:
            results = run_sklearn(SKLEARN_ALIASES[name], cfg, args, log)
        elif name in TORCH_MODELS:
            results = run_torch(name, cfg, args, log)
        else:
            raise ValueError(f"unknown model {name!r}")

        results["wall_clock_seconds"] = time.time() - t0
        save_json(results, run_dir / "results.json")
        log(f"\nRun complete in {results['wall_clock_seconds']:.1f}s")
        log(f"Artifacts: {run_dir}")
        return 0
    finally:
        log.close()


def cmd_compare(args) -> int:
    models = args.models or ["rf", "xgb", "mlp", "cnn", "lstm"]
    rows: List[dict] = []
    for m in models:
        print(f"\n\n########## {m} ##########")
        sub = argparse.Namespace(**vars(args))
        sub.model = m
        try:
            cmd_train(sub)
            runs = sorted(Path("runs").glob(f"*-{m}-{args.protocol}"))
            if runs:
                res = json.loads((runs[-1] / "results.json").read_text())
                cs = res.get("closed_set", {}).get("metrics", {})
                det = res.get("detection", {})
                rows.append({
                    "model": m,
                    "accuracy": cs.get("accuracy"),
                    "macro_f1": cs.get("macro_f1"),
                    "weighted_f1": cs.get("weighted_f1"),
                    "balanced_acc": cs.get("balanced_accuracy"),
                    "openset_auroc": det.get("auroc"),
                    "seconds": res.get("wall_clock_seconds"),
                })
        except Exception as exc:                        # keep the sweep going
            print(f"[error] {m} failed: {exc}")
            rows.append({"model": m, "error": str(exc)})

    df = pd.DataFrame(rows)
    if "macro_f1" in df.columns:
        df = df.sort_values("macro_f1", ascending=False, na_position="last")
    out = new_run_dir(f"compare-{args.protocol}") / "model_comparison.csv"
    df.to_csv(out, index=False)
    print("\n========== MODEL COMPARISON ==========")
    print(df.to_string(index=False))
    print(f"\nSaved: {out}")
    return 0


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="ML-NIDS pipeline")
    sub = p.add_subparsers(dest="command", required=True)

    sub.add_parser("cache", help="parse raw CSVs once into a float32 Parquet cache")

    t = sub.add_parser("train", help="train and evaluate one model")
    t.add_argument("--model", required=True,
                   choices=sorted(set(SKLEARN_ALIASES) | TORCH_MODELS))
    t.add_argument("--protocol", default="crossday", choices=["crossday", "closedset"])
    t.add_argument("--epochs", type=int)
    t.add_argument("--batch-size", dest="batch_size", type=int)
    t.add_argument("--lr", type=float)
    t.add_argument("--weight-decay", dest="weight_decay", type=float)
    t.add_argument("--dropout", type=float)
    t.add_argument("--hidden", type=int)
    t.add_argument("--depth", type=int)
    t.add_argument("--latent-dim", dest="latent_dim", type=int)
    t.add_argument("--loss", choices=["focal", "weighted_ce", "none"])
    t.add_argument("--patience", type=int)
    t.add_argument("--seed", type=int)
    t.add_argument("--device")
    t.add_argument("--num-workers", dest="num_workers", type=int)
    t.add_argument("--no-amp", dest="amp", action="store_false", default=None)
    t.add_argument("--target-benign-fpr", dest="target_benign_fpr", type=float)
    t.add_argument("--pretrain-ae", dest="pretrain_ae", action="store_true",
                   help="warm-start Deep SVDD from an autoencoder")

    c = sub.add_parser("compare", help="train several models and tabulate")
    c.add_argument("--models", nargs="+")
    c.add_argument("--protocol", default="closedset", choices=["crossday", "closedset"])
    c.add_argument("--epochs", type=int)
    c.add_argument("--seed", type=int)
    c.add_argument("--device")
    c.add_argument("--pretrain-ae", dest="pretrain_ae", action="store_true")
    for name in ("batch_size", "lr", "weight_decay", "dropout", "hidden", "depth",
                 "latent_dim", "loss", "patience", "num_workers", "amp",
                 "target_benign_fpr"):
        c.set_defaults(**{name: None})

    return p


def main(argv: Optional[List[str]] = None) -> int:
    args = build_parser().parse_args(argv)
    if not hasattr(args, "pretrain_ae"):
        args.pretrain_ae = False
    return {"cache": cmd_cache, "train": cmd_train, "compare": cmd_compare}[args.command](args)


if __name__ == "__main__":
    raise SystemExit(main())
