from typing import Dict, Any

import pandas as pd

from fastapi import (
    APIRouter,
    UploadFile,
    File
)

from backend.services.model_service import (
    predict,
    predict_dataframe
)

router = APIRouter()


# ==========================================
# SINGLE FLOW PREDICTION
# ==========================================

@router.post("/predict")
def predict_attack(features: Dict[str, Any]):

    attack, confidence = predict(features)

    return {
        "prediction": attack
    }


# ==========================================
# CSV PREDICTION
# ==========================================

@router.post("/predict_csv")
async def predict_csv(file: UploadFile = File(...)):

    df = pd.read_csv(file.file)

    result = predict_dataframe(df)

    return result

from fastapi.responses import FileResponse


@router.get("/download")
def download_csv():

    return FileResponse(
        "predicted_output.csv",
        filename="predicted_output.csv",
        media_type="text/csv"
    )