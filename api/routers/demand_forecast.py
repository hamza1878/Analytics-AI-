"""
api/routers/demand_forecast.py
"""
from fastapi import APIRouter, Query
from datetime import datetime, timezone, timedelta
import numpy as np
from api.schemas import DemandForecastResponse, ForecastPoint

router = APIRouter()


def _mock_forecast_points(horizon: int = 24) -> list[ForecastPoint]:
    rng = np.random.default_rng(42)
    base_demand = [
        12, 8, 5, 9, 38, 52, 44, 41,
        36, 55, 48, 22, 18, 14, 20, 32,
        41, 58, 50, 38, 28, 22, 18, 14,
    ]
    now = datetime.now(timezone.utc).replace(minute=0, second=0, microsecond=0)
    points = []
    for i in range(min(horizon, len(base_demand))):
        actual    = float(base_demand[i]) + rng.normal(0, 2)
        predicted = float(base_demand[i]) + rng.normal(0, 1.5)
        ci_width  = predicted * 0.12
        points.append(ForecastPoint(
            hour=(now + timedelta(hours=i)).strftime("%H:00"),
            actual=round(actual, 1),
            predicted=round(predicted, 1),
            confidence_low=round(predicted - ci_width, 1),
            confidence_high=round(predicted + ci_width, 1),
        ))
    return points


@router.get("", response_model=DemandForecastResponse)
async def get_demand_forecast(
    zone_lat: float = Query(36.80),
    zone_lon: float = Query(10.18),
    horizon:  int   = Query(24, ge=1, le=72),
):
    points = _mock_forecast_points(horizon)
    return DemandForecastResponse(
        zone_lat=zone_lat,
        zone_lon=zone_lon,
        horizon_h=horizon,
        mape=5.8,
        r2=0.97,
        points=points,
    )


@router.get("/surge")
async def get_surge_zones():
    """
    Retourne les zones de surge avec risque ML et potentiel revenu.
    En prod : appeler le modèle XGBoost avec les features de get_surge_features().
    """
    rng = np.random.default_rng()
    zones = [
        {"zone": "Centre-Ville", "risk": int(80 + rng.integers(-5, 10)), "revenue": 12400, "drivers": 8,  "demand": 34},
        {"zone": "Aéroport",     "risk": int(70 + rng.integers(-5, 10)), "revenue": 9800,  "drivers": 12, "demand": 28},
        {"zone": "La Marsa",     "risk": int(45 + rng.integers(-5, 10)), "revenue": 6200,  "drivers": 15, "demand": 18},
        {"zone": "Bardo",        "risk": int(30 + rng.integers(-5, 10)), "revenue": 4100,  "drivers": 20, "demand": 11},
        {"zone": "Ben Arous",    "risk": int(18 + rng.integers(-5, 10)), "revenue": 2800,  "drivers": 25, "demand": 7},
    ]
    return {"zones": zones}