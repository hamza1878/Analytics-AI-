"""
api/main.py

FastAPI — Moviroo ML Intelligence API
"""
from contextlib import asynccontextmanager
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from api.routers import overview, demand_forecast, anomalies, model_registry
from app.api.routes.revenue import revenue_forecast


@asynccontextmanager
async def lifespan(app: FastAPI):
    print("[api] Chargement des modèles ML en mémoire…")
    yield
    print("[api] Fermeture de l'API ML")


app = FastAPI(
    title="Moviroo ML Intelligence API",
    version="2.0.0",
    description="API d'inférence et de monitoring des modèles ML Moviroo",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(overview.router,        prefix="/intelligence",    tags=["Overview"])
app.include_router(demand_forecast.router, prefix="/demand-forecast", tags=["Demand"])
app.include_router(anomalies.router,       prefix="/anomalies",       tags=["Anomalies"])
app.include_router(model_registry.router,  prefix="/model-registry",  tags=["Registry"])


@app.get("/health")
async def health():
    return {"status": "ok", "version": "2.0.0"}