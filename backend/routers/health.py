"""Health and model-introspection endpoints."""

from typing import Any, Dict

from fastapi import APIRouter

from backend.services.model_service import get_bundle, model_status

router = APIRouter()


@router.get("/")
def home():
    return {"message": "ML Network Intrusion Detection API is Running"}


@router.get("/health")
def health() -> Dict[str, Any]:
    """Liveness AND model readiness.

    This used to return ``{"status": "Healthy"}`` unconditionally -- it stayed
    green while the model failed to load and every /predict returned 500.  A
    health check that cannot go red is decoration.
    """
    st = model_status()
    return {
        "status": "healthy" if st.get("available") else "degraded",
        "model_loaded": bool(st.get("available")),
        "detail": st.get("error"),
    }


@router.get("/model")
def model_info() -> Dict[str, Any]:
    """What is ACTUALLY loaded, plus the metrics that artifact earned.

    The dashboard used to hardcode "Validated Accuracy 99.83%", "78
    Dimensions" and "15 Categories" -- none of which any artifact in this
    repository produces. Everything below is read from the loaded bundle's
    manifest, so the UI cannot drift from the model.
    """
    st = model_status()
    if not st.get("available"):
        return {**st, "metrics": None}

    meta = getattr(get_bundle(), "meta", {}) or {}
    rows = meta.get("detection_by_class") or []

    # headline = the tightest false-alarm budget that was evaluated
    headline = None
    if rows:
        budgets = sorted({r["fpr_budget"] for r in rows})
        b = budgets[0]
        at_b = [r for r in rows if r["fpr_budget"] == b]
        headline = {
            "fpr_budget": b,
            "observed_benign_fpr": next((r["observed_benign_fpr"] for r in at_b), None),
            "per_class": {
                r["class"]: r["detection_rate"]
                for r in at_b if r["class"] != "__ANY_ATTACK__"
            },
            "any_attack": next(
                (r["detection_rate"] for r in at_b if r["class"] == "__ANY_ATTACK__"), None
            ),
        }

    return {
        **st,
        "protocol": meta.get("protocol"),
        "scorer": meta.get("scorer_name"),
        "novelty_tau": meta.get("novelty_tau"),
        "has_rejector": bool(meta.get("has_scorer")),
        "smoke_test": bool(meta.get("SMOKE_TEST")),
        "note": meta.get("note"),
        "metrics": headline,
    }
