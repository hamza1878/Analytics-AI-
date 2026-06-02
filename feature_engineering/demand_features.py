"""
feature_engineering/demand_features.py

Extrait les features de demande depuis la table `rides`.
Agrège par zone géographique (grille H3) et heure.
"""
import pandas as pd
import numpy as np
from sqlalchemy import create_engine, text
from config import DB, ML


DEMAND_QUERY = """
SELECT
    r.id,
    r.pickup_lat,
    r.pickup_lon,
    r.dropoff_lat,
    r.dropoff_lon,
    r.status,
    r.distance_km,
    r.duration_min,
    r.price_final,
    r.price_estimate,
    r.surge_multiplier,
    r.created_at,
    r.class_id,
    c.name AS class_name,
    DATE_TRUNC('hour', r.created_at) AS hour_bucket,
    EXTRACT(DOW  FROM r.created_at) AS day_of_week,
    EXTRACT(HOUR FROM r.created_at) AS hour_of_day,
    EXTRACT(MONTH FROM r.created_at) AS month
FROM rides r
JOIN classes c ON c.id = r.class_id
WHERE r.created_at >= NOW() - INTERVAL ':lookback days'
  AND r.status != 'CANCELLED'
ORDER BY r.created_at
"""


def load_raw(lookback_days: int = ML.demand_lookback_days) -> pd.DataFrame:
    engine = create_engine(DB.url)
    with engine.connect() as conn:
        df = pd.read_sql(
            text(DEMAND_QUERY.replace(":lookback", str(lookback_days))),
            conn,
            parse_dates=["created_at", "hour_bucket"],
        )
    return df


def build_zone_hour_features(df: pd.DataFrame, resolution: float = 0.05) -> pd.DataFrame:
    """
    Discrétise pickup_lat/lon en cellules grille et agrège par (zone, heure).

    Returns DataFrame avec colonnes :
        zone_lat | zone_lon | hour_bucket | ride_count | avg_distance_km |
        avg_duration_min | avg_price | demand_trend_7d | demand_trend_1d
    """
    df = df.copy()

    # Grille simple (arrondi à ~5km) — remplacer par H3 en prod
    df["zone_lat"] = (df["pickup_lat"] // resolution) * resolution
    df["zone_lon"] = (df["pickup_lon"] // resolution) * resolution

    agg = (
        df.groupby(["zone_lat", "zone_lon", "hour_bucket", "day_of_week", "hour_of_day"])
        .agg(
            ride_count=("id", "count"),
            avg_distance_km=("distance_km", "mean"),
            avg_duration_min=("duration_min", "mean"),
            avg_price=("price_final", "mean"),
            cancelled_count=("status", lambda s: (s == "CANCELLED").sum()),
        )
        .reset_index()
    )

    agg = agg.sort_values(["zone_lat", "zone_lon", "hour_bucket"])

    # Trend features (rolling sur la même cellule)
    agg["demand_trend_1d"] = (
        agg.groupby(["zone_lat", "zone_lon"])["ride_count"]
        .transform(lambda x: x.rolling(24, min_periods=1).mean())
    )
    agg["demand_trend_7d"] = (
        agg.groupby(["zone_lat", "zone_lon"])["ride_count"]
        .transform(lambda x: x.rolling(24 * 7, min_periods=1).mean())
    )

    # Taux d'annulation par bucket
    agg["cancellation_rate"] = agg["cancelled_count"] / (agg["ride_count"] + 1)

    # Features cycliques pour heure et jour
    agg["hour_sin"] = np.sin(2 * np.pi * agg["hour_of_day"] / 24)
    agg["hour_cos"] = np.cos(2 * np.pi * agg["hour_of_day"] / 24)
    agg["dow_sin"] = np.sin(2 * np.pi * agg["day_of_week"] / 7)
    agg["dow_cos"] = np.cos(2 * np.pi * agg["day_of_week"] / 7)

    return agg


def build_lstm_sequences(
    zone_df: pd.DataFrame,
    seq_len: int = 24,
    target_col: str = "ride_count",
) -> tuple[np.ndarray, np.ndarray]:
    """
    Prépare les séquences (X, y) pour l'entraînement LSTM.

    X shape : (n_samples, seq_len, n_features)
    y shape : (n_samples,)
    """
    feature_cols = [
        "ride_count", "avg_distance_km", "avg_price",
        "demand_trend_1d", "demand_trend_7d",
        "hour_sin", "hour_cos", "dow_sin", "dow_cos",
    ]

    # Normalisation min-max
    values = zone_df[feature_cols].fillna(0).values
    mins = values.min(axis=0)
    maxs = values.max(axis=0) + 1e-8
    values_norm = (values - mins) / (maxs - mins)

    target_idx = feature_cols.index(target_col)

    X, y = [], []
    for i in range(seq_len, len(values_norm)):
        X.append(values_norm[i - seq_len : i])
        y.append(values_norm[i, target_idx])

    return np.array(X), np.array(y)


def run(lookback_days: int = ML.demand_lookback_days) -> dict:
    """Point d'entrée principal du pipeline de features demande."""
    print(f"[demand_features] Chargement des rides ({lookback_days}j)…")
    raw = load_raw(lookback_days)
    print(f"  → {len(raw):,} rides chargés")

    zone_hour = build_zone_hour_features(raw)
    print(f"  → {len(zone_hour):,} buckets zone×heure construits")

    # Séquences pour la zone la plus active (exemple)
    top_zone = (
        zone_hour.groupby(["zone_lat", "zone_lon"])["ride_count"]
        .sum()
        .idxmax()
    )
    zone_df = zone_hour[
        (zone_hour["zone_lat"] == top_zone[0]) & (zone_hour["zone_lon"] == top_zone[1])
    ]
    X, y = build_lstm_sequences(zone_df)
    print(f"  → Séquences LSTM : X={X.shape}, y={y.shape}")

    return {"zone_hour_df": zone_hour, "X": X, "y": y, "top_zone": top_zone}


if __name__ == "__main__":
    run()
