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
    detection_by_class,
    format_detection_table,
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
    format_scorer_table,
    make_open_set_predictions,
    msp_score,
    open_set_report,
    rank_scorers,
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
    p_attack_test: Optional[np.ndarray] = None,
    p_attack_val: Optional[np.ndarray] = None,
    y_val_str: Optional[np.ndarray] = None,
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

    # ---- PRIMARY REPORT: detection per class at a fixed false-alarm budget --
    # Printed first, deliberately.  It is the only report here that answers the
    # question a detector exists to answer: what do you catch, and how much
    # noise does that cost.  Accuracy is reported further down as a secondary
    # figure with BOTH readings stated, because on this split they differ by
    # eleven points and quoting one unqualified says nothing.
    if p_attack_test is not None and p_attack_val is not None and y_val_str is not None:
        det_df = detection_by_class(
            y_test_str, p_attack_test, p_attack_val, y_val_str,
            budgets=(0.001, 0.01, 0.05),
        )
        log(format_detection_table(det_df))
        det_df.to_csv(run_dir / "detection_by_class.csv", index=False)
        results["detection_by_class"] = det_df.to_dict(orient="records")
    else:
        log("\n[warn] no class probabilities supplied; the PRIMARY per-class "
            "detection report was skipped and only accuracy-style metrics follow.")

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

        # The crossday protocol's closed-set subset is Friday's KNOWN-class rows,
        # and Friday's only known class is BENIGN (DDoS / PortScan / Bot are all
        # novel). macro_f1 then averages one real F1 over the union of true and
        # predicted labels, so it reads as a catastrophic number when it is
        # arithmetic. Observed in the wild: accuracy 0.9847 with macro_f1 0.099,
        # which is 0.9923 / 10 labels. Say so rather than letting it be read as
        # a collapse.
        n_true = int(rep.metrics.get("n_classes_true", 0))
        n_union = int(rep.metrics.get("n_classes_union", 0))
        if n_true <= 1 < n_union:
            log(
                f"\n[note] this closed-set subset contains only {n_true} true class"
                f" but {n_union} labels appear in the union with the predictions.\n"
                f"       'macro_f1' = {rep.metrics['macro_f1']:.4f} is that one F1"
                f" divided by {n_union}.\n"
                f"       Quote 'macro_f1_present' = "
                f"{rep.metrics['macro_f1_present']:.4f} instead, or read the\n"
                f"       open_set / detection reports -- on the crossday protocol"
                f" those are the\n       meaningful ones."
            )

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

    # A single tunable knob for the operating point: P(attack) = 1 - P(BENIGN).
    # argmax alone gives no knob, so there is no way to ask "what do you catch
    # at 1% false alarms" -- which is the only question worth asking.
    #
    # SCORE CHOICE, MEASURED.  The obvious knob is ``1 - P(BENIGN)``.  It is
    # measurably the WRONG one.  At a 1% benign false-alarm budget on the
    # crossday split:
    #
    #     1 - P(BENIGN)        FPR 1.05%   any-attack 11.97%   DDoS 26.90%
    #     max P(attack class)  FPR 1.02%   any-attack 27.63%   DDoS 62.22%
    #
    # Over twice the recall at the same cost.  With 12 classes, ``1-P(BENIGN)``
    # sums probability mass across every attack class, so a flow that is
    # diffusely uncertain (0.08 spread over ten classes) scores as highly as one
    # a single class confidently claims.  Diffuse mass is noise here;
    # concentrated mass is signal.  ``max P(attack)`` requires some class to
    # actually claim the flow, and it also keeps the tunable knob that argmax
    # does not have.
    from preprocessing import benign_index
    bi = benign_index(bundle.encoder)

    def _p_attack(P):
        Q = P.copy()
        Q[:, bi] = -1.0
        return Q.max(axis=1)

    p_attack_test = _p_attack(proba)
    p_attack_val = _p_attack(model.predict_proba(bundle.X_val))

    # Tree ensembles have no logits, so Mahalanobis on the scaled features is
    # the rejection score.  Max-softmax on an RF is degenerate (mostly 1.0).
    scorer = MahalanobisScorer().fit(bundle.X_train, bundle.y_train)
    scores_test = scorer.score(bundle.X_test)
    scores_val = scorer.score(bundle.X_val)
    scorer_name = "mahalanobis"

    results = evaluate_classifier(
        bundle.y_test_str, y_pred_str, scores_test, scores_val,
        bundle.known_classes, cfg, run_dir, scorer_name, log,
        p_attack_test=p_attack_test, p_attack_val=p_attack_val,
        y_val_str=bundle.y_val_str,
    )

    feature_importance(model, bundle.feature_names, run_dir, tag=name)
    # tau is calibrated on the VALIDATION day only -- nothing from the test
    # day may enter it -- and ships with the model so that serving applies the
    # same rejection rule the evaluation measured.
    tau = threshold_at_fpr(scores_val, cfg.target_benign_fpr)
    save_bundle(model, bundle.scaler, bundle.encoder, bundle.cleaner,
                bundle.feature_names, f"{name}_{cfg.protocol}",
                extra_meta={"protocol": cfg.protocol, "run_dir": str(run_dir),
                            "scorer_name": scorer_name,
                            "target_benign_fpr": cfg.target_benign_fpr,
                            # The dashboard reads these. Previously it hardcoded
                            # "Validated Accuracy 99.83%", a closed-set
                            # random-split figure typed in by hand that nothing
                            # in the repo produced. A UI that states a metric
                            # must read it from the artifact that earned it.
                            "detection_by_class": results.get("detection_by_class"),
                            "detection": results.get("detection")},
                scorer=scorer, novelty_tau=tau)
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
        log(f"Model  : {name}  ({count_parameters(model):,} trainable parameters), norm={cfg.norm}")

        train_loader = build_loaders(
            bundle.X_train, bundle.y_train, cfg.batch_size, True, cfg, device,
            balanced=cfg.sampler_power > 0.0,
        )
        if cfg.sampler_power > 0.0:
            log(f"Sampler: class-rebalanced, power={cfg.sampler_power} "
                f"(0=uniform, 0.5=sqrt, 1=fully balanced)")
        val_loader = build_loaders(bundle.X_val, bundle.y_val, cfg.eval_batch_size, False, cfg, device)
        test_loader = build_loaders(bundle.X_test, None, cfg.eval_batch_size, False, cfg, device)

        criterion = build_loss(cfg, bundle.y_train, bundle.n_classes)
        log(f"Loss   : {type(criterion).__name__}")

        history = train_classifier(
            model, train_loader, val_loader, bundle.y_val,
            criterion, cfg, device, ckpt, log=log,
            class_names=[str(c) for c in bundle.encoder.classes_],
        )
        history.save(run_dir / "history.json")
        plot_history(history.rows, run_dir / "history.png")

        logits_test = predict_logits(model, test_loader, device)
        logits_val = predict_logits(model, val_loader, device)

        y_pred_str = decode_predictions(logits_test.argmax(axis=1), bundle.encoder)

        # ---- build EVERY rejection score, on test and on validation ----
        # The primary scorer is cfg.scorer (default mahalanobis_embed): it
        # measures distance from the training manifold in feature space rather
        # than classifier confidence, and is the only one of the three that
        # survives near-OOD. Measured on a 12-class ablation at CIC-IDS2017-like
        # imbalance, near-OOD drawn inside a known class's cloud:
        #     msp 0.112 | energy 0.120 | mahalanobis_embed 0.914   (BatchNorm)
        #     msp 0.578 | energy 0.630 | mahalanobis_embed 0.930   (LayerNorm)
        train_emb_loader = build_loaders(
            bundle.X_train, None, cfg.eval_batch_size, False, cfg, device
        )
        emb_tr = embed(model, train_emb_loader, device)
        emb_va = embed(model, val_loader, device)
        emb_te = embed(model, test_loader, device)
        maha = MahalanobisScorer().fit(emb_tr, bundle.y_train)

        test_scores = {
            "mahalanobis_embed": maha.score(emb_te),
            "energy": energy_score(logits_test),
            "msp": msp_score(logits_test),
        }
        val_scores = {
            "mahalanobis_embed": maha.score(emb_va),
            "energy": energy_score(logits_val),
            "msp": msp_score(logits_val),
        }

        primary = cfg.scorer if cfg.scorer in test_scores else "mahalanobis_embed"
        if cfg.scorer not in test_scores:
            log(f"[warn] unknown scorer {cfg.scorer!r}; using mahalanobis_embed")

        # Same operating-point knob as the sklearn path: P(attack) from the
        # softmax over logits, so the primary per-class detection report is
        # produced identically for neural and tree models.
        from preprocessing import benign_index

        def _p_attack(lg):
            # max P(attack class), not 1 - P(BENIGN).  See the note in
            # run_sklearn: the latter measures over twice as badly at a matched
            # false-alarm budget, because it rewards diffuse uncertainty.
            z = np.asarray(lg, dtype=np.float64)
            z = z - z.max(axis=1, keepdims=True)
            pr = np.exp(z)
            pr /= pr.sum(axis=1, keepdims=True)
            pr[:, benign_index(bundle.encoder)] = -1.0
            return pr.max(axis=1)

        results = evaluate_classifier(
            bundle.y_test_str, y_pred_str, test_scores[primary], val_scores[primary],
            bundle.known_classes, cfg, run_dir, primary, log,
            p_attack_test=_p_attack(logits_test),
            p_attack_val=_p_attack(logits_val),
            y_val_str=bundle.y_val_str,
        )

        is_unk = unknown_mask(bundle.y_test_str, bundle.known_classes)
        if is_unk.any():
            ranked = rank_scorers(test_scores, is_unk, val_scores, cfg.target_benign_fpr)
            log(format_scorer_table(ranked))
            table = {r.scorer: r.to_dict() for r in ranked}
            save_json(table, run_dir / "scorer_comparison.json")
            results["scorer_comparison"] = table
            best = ranked[0]
            if best.scorer != primary and best.auroc > results.get(
                "detection", {}
            ).get("auroc", -1.0):
                log(f"\n[note] {best.scorer} scored higher (AUROC {best.auroc:.4f}) than the")
                log(f"       primary scorer {primary}. Re-run with --scorer {best.scorer}")
                log("       ONLY if you selected it on validation or a held-out")
                log("       pseudo-unknown class -- picking it from this table is")
                log("       selection on the test set.")

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
    t.add_argument("--norm", choices=["ln", "bn", "none"],
                   help="ln (default) preserves the OOD signal; bn is the old behaviour")
    t.add_argument("--sampler-power", dest="sampler_power", type=float,
                   help="0=uniform, 0.5=sqrt-balanced (default), 1=fully balanced")
    t.add_argument("--scorer", choices=["mahalanobis_embed", "energy", "msp"])
    t.add_argument("--focal-gamma", dest="focal_gamma", type=float)
    t.add_argument("--class-weight-power", dest="class_weight_power", type=float)
    t.add_argument("--monitor", choices=["macro_f1", "balanced_accuracy", "loss"])
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
                 "target_benign_fpr", "norm", "sampler_power", "scorer",
                 "focal_gamma", "class_weight_power", "monitor"):
        c.set_defaults(**{name: None})

    return p


def main(argv: Optional[List[str]] = None) -> int:
    args = build_parser().parse_args(argv)
    if not hasattr(args, "pretrain_ae"):
        args.pretrain_ae = False
    return {"cache": cmd_cache, "train": cmd_train, "compare": cmd_compare}[args.command](args)


if __name__ == "__main__":
    raise SystemExit(main())
