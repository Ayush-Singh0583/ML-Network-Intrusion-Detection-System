from fastapi import APIRouter

router = APIRouter()


@router.get("/")
def home():
    return {
        "message": "ML Network Intrusion Detection API is Running"
    }


@router.get("/health")
def health():
    return {
        "status": "Healthy"
    }