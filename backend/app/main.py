"""
main.py – FastAPI entry point
Trains the ML pipeline on startup, then serves all routes.
"""

from contextlib import asynccontextmanager
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from .ml.pipeline import run_pipeline
from .routes.predict import router as predict_router


@asynccontextmanager
async def lifespan(app: FastAPI):
    # Train once at startup – cached in memory for all requests
    print("[startup] Running ML pipeline…")
    run_pipeline()
    yield
    print("[shutdown] Bye.")


app = FastAPI(
    title="Urban Anomaly Detection API",
    description="ML-powered urban anomaly detection for Bangalore",
    version="1.0.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(predict_router, prefix="/api", tags=["ML"])


@app.get("/health")
def health():
    return {"status": "ok", "service": "Urban Anomaly Detection API"}
