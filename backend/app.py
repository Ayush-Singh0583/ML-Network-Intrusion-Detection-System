from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from backend.routers.live import router as live_router
from backend.routers.health import router as health_router
from backend.routers.prediction import router as prediction_router
from backend.database import initialize_database
from backend.live.cleanup import start_cleanup
app = FastAPI(
    title="ML Network Intrusion Detection System",
    description="FastAPI Backend for Network Intrusion Detection",
    version="1.0.0"
)

# Allow React Frontend
app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:5173",
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
@app.on_event("startup")
def startup_db():

    initialize_database()
@app.on_event("startup")
def startup_cleanup():

    start_cleanup()
app.include_router(health_router)
app.include_router(prediction_router)
app.include_router(live_router)