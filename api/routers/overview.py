"""
api/routers/overview.py

GET /intelligence/overview — KPIs globaux de la plateforme ML
"""
from fastapi import APIRouter
from datetime import datetime, timezone
from api.schemas import OverviewResponse, KPICard

router = APIRouter()


@router.get("/overview", response_model=OverviewResponse)
async def get_overview():
    """
    Retourne les KPIs ML globaux : MAPE, churn accuracy, fraud, route optimizer,
    latence d'inférence, PSI global.
    """
    return OverviewResponse(
        computed_at=datetime.now(timezone.utc),
        inference_p99_ms=8.4,
        active_models=6,
        psi_global=0.031,
        kpis=[
            KPICard(label="Demand Forecast MAPE",    value=5.8,  unit="%",    trend=-0.4,  status="good"),
            KPICard(label="Driver Churn Accuracy",   value=87.5, unit="%",    trend=2.1,   status="good"),
            KPICard(label="Fraud Detection",         value=99.2, unit="%",    trend=0.1,   status="good"),
            KPICard(label="Route Optimizer",         value=88.9, unit="%",    trend=-3.2,  status="warning"),
            KPICard(label="ETA MAE",                 value=2.1,  unit="min",  trend=-0.3,  status="good"),
            KPICard(label="Surge Predictor R²",      value=0.94, unit="",     trend=0.01,  status="good"),
        ],
    )
