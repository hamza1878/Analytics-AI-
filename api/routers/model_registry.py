"""
api/routers/model_registry.py
"""
from fastapi import APIRouter, HTTPException
from datetime import datetime, timezone, timedelta
from api.schemas import (
    RegistryResponse, ModelInfo, ModelStatus,
    RetrainRequest, RetrainResponse,
)
import uuid

router = APIRouter()

_MODEL_REGISTRY: list[ModelInfo] = [
    ModelInfo(
        name="demand_forecast",   version="v4",
        status=ModelStatus.PRODUCTION,
        algorithm="LSTM + Prophet ensemble",
        source_tables=["rides", "classes"],
        primary_metric="MAPE",        metric_value=5.8,
        last_trained=datetime.now(timezone.utc) - timedelta(days=3),
        psi_score=0.023,
    ),
    ModelInfo(
        name="surge_predictor",   version="v2",
        status=ModelStatus.PRODUCTION,
        algorithm="XGBoost Regressor",
        source_tables=["rides", "classes"],
        primary_metric="R²",          metric_value=0.94,
        last_trained=datetime.now(timezone.utc) - timedelta(days=5),
        psi_score=0.028,
    ),
    ModelInfo(
        name="churn_classifier",  version="v3",
        status=ModelStatus.PRODUCTION,
        algorithm="Random Forest + SMOTE",
        source_tables=["drivers", "driver_locations", "dispatch_offers", "ride_ratings"],
        primary_metric="CV-Accuracy", metric_value=87.5,
        last_trained=datetime.now(timezone.utc) - timedelta(days=7),
        psi_score=0.031,
    ),
    ModelInfo(
        name="eta_estimator",     version="v5",
        status=ModelStatus.SHADOW,
        algorithm="LightGBM",
        source_tables=["rides", "trip_waypoints", "classes"],
        primary_metric="MAE (min)",   metric_value=2.1,
        last_trained=datetime.now(timezone.utc) - timedelta(days=1),
        psi_score=0.019,
    ),
    ModelInfo(
        name="fraud_detector",    version="v2",
        status=ModelStatus.PRODUCTION,
        algorithm="IsolationForest",
        source_tables=["rides", "passengers"],
        primary_metric="Precision",   metric_value=99.2,
        last_trained=datetime.now(timezone.utc) - timedelta(days=2),
        psi_score=0.021,
    ),
    ModelInfo(
        name="route_optimizer",   version="v1",
        status=ModelStatus.DEGRADED,
        algorithm="DQN (Reinforcement Learning)",
        source_tables=["dispatch_offers", "rides", "drivers", "driver_locations"],
        primary_metric="Accuracy",    metric_value=88.9,
        last_trained=datetime.now(timezone.utc) - timedelta(days=21),
        psi_score=0.087,
    ),
]


@router.get("", response_model=RegistryResponse)
async def list_models():
    return RegistryResponse(models=_MODEL_REGISTRY)


# ⚠️ Routes spécifiques AVANT /{model_name} pour éviter les conflits

@router.post("/retrain", response_model=RetrainResponse)
async def trigger_retrain(req: RetrainRequest):
    run_id = str(uuid.uuid4())[:8]

    # Cas spécial : retrain tous les modèles
    if req.model_name == "all":
        print(f"[model_registry] Full retrain déclenché | run_id={run_id}")
        return RetrainResponse(
            model_name="all",
            run_id=run_id,
            status="queued",
            triggered_at=datetime.now(timezone.utc),
        )

    valid_models = {m.name for m in _MODEL_REGISTRY}
    if req.model_name not in valid_models:
        raise HTTPException(
            status_code=400,
            detail=f"Modèle inconnu : {req.model_name}. Disponibles : {sorted(valid_models)}",
        )

    print(
        f"[model_registry] Retraining déclenché : {req.model_name} | "
        f"lookback={req.lookback_days}j | reason={req.reason} | run_id={run_id}"
    )
    return RetrainResponse(
        model_name=req.model_name,
        run_id=run_id,
        status="queued",
        triggered_at=datetime.now(timezone.utc),
    )


@router.get("/drift/report")
async def get_drift_report():
    return {
        "computed_at": datetime.now(timezone.utc).isoformat(),
        "psi_threshold_warning":  0.05,
        "psi_threshold_critical": 0.10,
        "infra": [
            {"title": "GPU Utilization",   "value": "67%",    "label": "A100 cluster",      "color": "#4CAF50", "detail": "avg over last 1h"},
            {"title": "Inference Latency", "value": "8.4ms",  "label": "p99 latency",       "color": "#4CAF50", "detail": "target < 10ms"},
            {"title": "Models in Prod",    "value": "4",      "label": "active endpoints",  "color": "#A855F7", "detail": "2 shadow, 1 degraded"},
            {"title": "PSI Global",        "value": "0.031",  "label": "data drift score",  "color": "#4CAF50", "detail": "threshold: 0.10"},
            {"title": "Retrain Queue",     "value": "1",      "label": "pending jobs",      "color": "#FF9500", "detail": "route_optimizer"},
            {"title": "Last Deploy",       "value": "3d ago", "label": "demand_forecast v4","color": "#A855F7", "detail": "MAPE improved 0.4%"},
        ],
        "models": [
            {
                "model":  m.name,
                "psi":    m.psi_score,
                "status": (
                    "stable"   if (m.psi_score or 0) < 0.05 else
                    "warning"  if (m.psi_score or 0) < 0.10 else
                    "critical"
                ),
            }
            for m in _MODEL_REGISTRY
            if m.psi_score is not None
        ],
    }


@router.get("/{model_name}", response_model=ModelInfo)
async def get_model(model_name: str):
    model = next((m for m in _MODEL_REGISTRY if m.name == model_name), None)
    if not model:
        raise HTTPException(status_code=404, detail=f"Modèle '{model_name}' introuvable")
    return model