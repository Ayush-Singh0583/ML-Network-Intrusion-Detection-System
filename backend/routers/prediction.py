"""
Prediction endpoints.

CONTRACT NOTE -- this is the bug that lived in this file.

``model_service.predict()`` returns a DICT:

    {"prediction", "closed_set_confidence", "class_probabilities",
     "is_known", ...}

Both call sites in this repository unpacked it into two names:

    attack, confidence = predict(features)        # <- ValueError, always

A 5-key dict does not unpack into 2 names, so EVERY call raised
``ValueError: too many values to unpack (expected 2)``.  The refactor that
changed ``predict()``'s return type never updated its callers.  Nothing
caught it because nothing called this endpoint in a test.

Also removed: ``GET /download``.  It served ``predicted_output.csv`` from the
process working directory, but ``predict_dataframe`` deliberately stopped
writing that file (it was a race between concurrent requests and an unbounded
disk write).  The endpoint therefore returned a stale July artifact for every
request regardless of what the caller had just uploaded -- a silent wrong
answer, which is worse than a 404.  CSV export now happens in the browser from
the response the client already holds.
"""

from typing import Any, Dict

import pandas as pd
from fastapi import APIRouter, File, HTTPException, UploadFile

from backend.services.model_service import (
    FeatureMismatch,
    ModelUnavailable,
    predict,
    predict_dataframe,
)

router = APIRouter()


def _to_http(exc: Exception) -> HTTPException:
    """Map service-layer failures onto honest status codes.

    Previously every one of these surfaced as a bare 500, which tells the
    caller nothing about whether to retry, fix their payload, or page someone.
    """
    if isinstance(exc, ModelUnavailable):
        # The service is up, the model is not. Retryable; not the caller's fault.
        return HTTPException(status_code=503, detail=str(exc))
    if isinstance(exc, FeatureMismatch):
        # Well-formed request, semantically unusable payload.
        return HTTPException(status_code=422, detail=str(exc))
    raise exc


# ==========================================
# SINGLE FLOW PREDICTION
# ==========================================


@router.post("/predict")
def predict_attack(features: Dict[str, Any]) -> Dict[str, Any]:
    """Score one flow.

    Returns the full prediction record, not just a label: the caller needs
    ``is_known`` to tell "this is BENIGN" apart from "this resembles nothing
    I was trained on".  The old handler discarded both the confidence and the
    novelty flag and returned ``{"prediction": ...}`` alone.
    """
    try:
        return predict(features)
    except (ModelUnavailable, FeatureMismatch) as exc:
        raise _to_http(exc) from exc


# ==========================================
# CSV PREDICTION
# ==========================================


@router.post("/predict_csv")
async def predict_csv(file: UploadFile = File(...)) -> Dict[str, Any]:
    try:
        df = pd.read_csv(file.file)
    except Exception as exc:  # noqa: BLE001 - any parse failure is the caller's
        raise HTTPException(
            status_code=400, detail=f"could not parse uploaded CSV: {exc}"
        ) from exc

    try:
        return predict_dataframe(df)
    except (ModelUnavailable, FeatureMismatch) as exc:
        raise _to_http(exc) from exc
