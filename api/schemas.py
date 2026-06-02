"""
api/schemas.py — Modèles Pydantic pour l'API ML Moviroo
"""
from pydantic import BaseModel, Field
from typing import Optional, Any
from datetime import datetime
from enum import Enum


# ── Enums ──────────────────────────────────────────────────────────────────

class ModelStatus(str, Enum):
    PRODUCTION = "production"
    SHADOW     = "shadow"
    DEGRADED   = "degraded"
    ARCHIVED   = "archived"

class AlertSeverity(str, Enum):
    HIGH   = "HIGH"
    MEDIUM = "MEDIUM"
    LOW    = "LOW"


# ── Overview ───────────────────────────────────────────────────────────────

class KPICard(BaseModel):
    label:    str
    value:    float
    unit:     str
    trend:    Optional[float] = None   # % vs semaine précédente
    status:   str                      # "good" | "warning" | "critical"

class OverviewResponse(BaseModel):
    computed_at:         datetime
    inference_p99_ms:    float
    active_models:       int
    kpis:                list[KPICard]
    psi_global:          float


# ── Demand Forecast ────────────────────────────────────────────────────────

class ForecastPoint(BaseModel):
    hour:         str
    actual:       Optional[float] = None
    predicted:    float
    confidence_low:  float
    confidence_high: float

class DemandForecastResponse(BaseModel):
    zone_lat:    float
    zone_lon:    float
    horizon_h:   int
    mape:        float
    r2:          float
    points:      list[ForecastPoint]


# ── Surge Prediction ──────────────────────────────────────────────────────

class SurgePredictRequest(BaseModel):
    zone_lat:              float
    zone_lon:              float
    hour_of_day:           int = Field(ge=0, le=23)
    day_of_week:           int = Field(ge=0, le=6)
    concurrent_rides:      int = Field(ge=0)

class SurgePredictResponse(BaseModel):
    surge_multiplier:   float
    confidence:         float
    zone_lat:           float
    zone_lon:           float


# ── ETA ───────────────────────────────────────────────────────────────────

class ETARequest(BaseModel):
    pickup_lat:    float
    pickup_lon:    float
    dropoff_lat:   float
    dropoff_lon:   float
    class_name:    str = "Standard"
    hour_of_day:   int = Field(ge=0, le=23)

class ETAResponse(BaseModel):
    eta_minutes:          float
    confidence_interval:  list[float]   # [low, high]
    distance_km:          float


# ── Anomalies ─────────────────────────────────────────────────────────────

class AnomalyItem(BaseModel):
    alert_type:       str
    severity:         AlertSeverity
    description:      str
    affected_entity:  str
    score:            float
    detected_at:      datetime
    metadata:         dict[str, Any]

class AnomaliesResponse(BaseModel):
    total:      int
    anomalies:  list[AnomalyItem]


# ── Model Registry ────────────────────────────────────────────────────────

class ModelInfo(BaseModel):
    name:          str
    version:       str
    status:        ModelStatus
    algorithm:     str
    source_tables: list[str]
    primary_metric: str
    metric_value:  float
    last_trained:  Optional[datetime] = None
    psi_score:     Optional[float] = None

class RegistryResponse(BaseModel):
    models: list[ModelInfo]

class RetrainRequest(BaseModel):
    model_name:   str
    lookback_days: int = 60
    reason:       Optional[str] = None

class RetrainResponse(BaseModel):
    model_name:  str
    run_id:      str
    status:      str
    triggered_at: datetime


# ── Churn ─────────────────────────────────────────────────────────────────

class ChurnRiskItem(BaseModel):
    driver_id:   str
    risk_score:  float
    label:       str          # "HIGH" | "MEDIUM" | "LOW"
    top_factors: list[str]

class ChurnResponse(BaseModel):
    total_drivers_analyzed: int
    high_risk_count:        int
    drivers:                list[ChurnRiskItem]
