"""
api/routers/anomalies.py
"""
from fastapi import APIRouter, Query
from datetime import datetime, timezone
from api.schemas import AnomaliesResponse, AnomalyItem, AlertSeverity, ChurnResponse, ChurnRiskItem

router = APIRouter()

_MOCK_ANOMALIES = [
    {
        "alert_type": "PAYMENT_SPIKE",
        "severity": "HIGH",
        "description": "Passager a3f2 — 5 trajets en 1h, spend total 410€",
        "affected_entity": "passenger:a3f2",
        "score": 0.91,
        "metadata": {"rides_last_hour": 5, "total_spend": 410.0, "z_score_rides": 4.2},
    },
    {
        "alert_type": "SURGE_MISMATCH",
        "severity": "MEDIUM",
        "description": "Ride 7b9c — surge déclaré 1.2× vs ratio réel 1.68× (Δ=0.48)",
        "affected_entity": "ride:7b9c",
        "score": 0.67,
        "metadata": {"price_estimate": 18.5, "price_final": 31.1, "surge_declared": 1.2, "actual_ratio": 1.68},
    },
    {
        "alert_type": "RATING_DRIFT",
        "severity": "LOW",
        "description": "Driver 2d1e — note 7j: 3.9 vs historique: 4.4 (Δ=0.5)",
        "affected_entity": "driver:2d1e",
        "score": 0.34,
        "metadata": {"recent_avg": 3.9, "historical_avg": 4.4, "drift": 0.5, "sample_count": 12},
    },
]

_MOCK_CHURN = [
    {"driver_id": "4a2f", "name": "Amir Ben Ali",    "churn_risk": 91, "total_trips": 142, "risk_score": 0.91, "label": "HIGH",   "top_factors": ["days_since_last_trip", "accept_rate", "rating_average"]},
    {"driver_id": "7b9c", "name": "Sami Trabelsi",   "churn_risk": 74, "total_trips": 89,  "risk_score": 0.74, "label": "MEDIUM", "top_factors": ["days_since_last_offer", "reject_rate", "total_trips"]},
    {"driver_id": "2d1e", "name": "Youssef Hamdi",   "churn_risk": 68, "total_trips": 210, "risk_score": 0.68, "label": "MEDIUM", "top_factors": ["recent_avg_rating", "days_since_last_login", "expire_rate"]},
    {"driver_id": "9f3a", "name": "Karim Mansouri",  "churn_risk": 31, "total_trips": 378, "risk_score": 0.31, "label": "LOW",    "top_factors": ["days_since_last_trip", "avg_dispatch_score", "is_online"]},
]


@router.get("", response_model=AnomaliesResponse)
async def get_anomalies(
    severity: str = Query(None),
    limit:    int = Query(50, ge=1, le=200),
):
    items = _MOCK_ANOMALIES
    if severity:
        items = [a for a in items if a["severity"] == severity.upper()]
    items = items[:limit]

    return AnomaliesResponse(
        total=len(items),
        anomalies=[
            AnomalyItem(
                **{k: v for k, v in a.items() if k != "severity"},
                severity=AlertSeverity(a["severity"]),
                detected_at=datetime.now(timezone.utc),
            )
            for a in items
        ],
    )


@router.get("/churn")
async def get_churn_risk(
    min_risk: float = Query(0.0, ge=0.0, le=1.0),
    limit:    int   = Query(20, ge=1, le=100),
):
    filtered = [d for d in _MOCK_CHURN if d["risk_score"] >= min_risk][:limit]
    return {
        "total_drivers_analyzed": len(_MOCK_CHURN),
        "high_risk_count": sum(1 for d in filtered if d["label"] == "HIGH"),
        "drivers": filtered,
    }


@router.post("/run")
async def run_anomaly_scan():
    return {
        "status": "triggered",
        "detectors": ["payment_spikes", "surge_mismatch", "rating_drift", "lstm_residuals"],
        "triggered_at": datetime.now(timezone.utc).isoformat(),
        "estimated_duration_s": 12,
    }