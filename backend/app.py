from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from backend.config import ALLOWED_ORIGINS
from backend.routers.live import router as live_router
from backend.routers.health import router as health_router
from backend.routers.prediction import router as prediction_router
from backend.database import initialize_database
from backend.live.cleanup import start_cleanup


@asynccontextmanager
async def lifespan(app: FastAPI):
    # Replaces two @app.on_event("startup") hooks. FastAPI deprecates
    # on_event (a DeprecationWarning on every start) in favour of lifespan.
    initialize_database()
    start_cleanup()
    yield


app = FastAPI(
    title="ML Network Intrusion Detection System",
    description="FastAPI Backend for Network Intrusion Detection",
    version="1.0.0",
    lifespan=lifespan
)

# Allow React Frontend. The origins come from backend.config
# (NIDS_CORS_ORIGINS); they used to be hardcoded here, so that setting did
# nothing and `npm run preview` (port 4173) was blocked.
app.add_middleware(
    CORSMiddleware,
    allow_origins=ALLOWED_ORIGINS,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
app.include_router(health_router)
app.include_router(prediction_router)
app.include_router(live_router)