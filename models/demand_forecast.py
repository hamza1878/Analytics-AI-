"""
models/demand_forecast.py

Ensemble LSTM + Prophet pour la prédiction de la demande.
Métriques cibles : MAPE < 6%, R² > 0.95
"""
import numpy as np
import pandas as pd
import mlflow
import mlflow.sklearn
from sklearn.metrics import mean_absolute_percentage_error, r2_score
from sklearn.model_selection import train_test_split
from config import ML

# TensorFlow/Keras importés à la demande (optionnel si non dispo)
try:
    import tensorflow as tf
    import keras
    from keras import layers
    TF_AVAILABLE = True
except ImportError:
    TF_AVAILABLE = False


try:
    from prophet import Prophet
    PROPHET_AVAILABLE = True
except ImportError:
    PROPHET_AVAILABLE = False


# ─────────────────────────────────────────────
# LSTM
# ─────────────────────────────────────────────

def build_lstm(seq_len: int, n_features: int) -> "keras.Model":
    """Architecture LSTM bi-directionnel pour la demande."""
    if not TF_AVAILABLE:
        raise RuntimeError("TensorFlow non disponible. pip install tensorflow")

    inp = keras.Input(shape=(seq_len, n_features))
    x = layers.Bidirectional(layers.LSTM(128, return_sequences=True))(inp)
    x = layers.Dropout(0.2)(x)
    x = layers.LSTM(64)(x)
    x = layers.Dropout(0.2)(x)
    x = layers.Dense(32, activation="relu")(x)
    out = layers.Dense(1)(x)

    model = keras.Model(inp, out)
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=1e-3),
        loss="huber",
        metrics=["mae"],
    )
    return model


def train_lstm(X: np.ndarray, y: np.ndarray, epochs: int = 50, batch_size: int = 64) -> dict:
    """Entraîne le LSTM et retourne le modèle + métriques."""
    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.15, shuffle=False)

    model = build_lstm(X.shape[1], X.shape[2])

    callbacks = []
    if TF_AVAILABLE:
        callbacks = [
            keras.callbacks.EarlyStopping(patience=8, restore_best_weights=True),
            keras.callbacks.ReduceLROnPlateau(patience=4, factor=0.5),
        ]

    history = model.fit(
        X_train, y_train,
        validation_data=(X_val, y_val),
        epochs=epochs,
        batch_size=batch_size,
        callbacks=callbacks,
        verbose=1,
    )

    y_pred = model.predict(X_val).flatten()
    mape = mean_absolute_percentage_error(y_val + 1e-8, y_pred + 1e-8)
    r2   = r2_score(y_val, y_pred)

    return {
        "model": model,
        "history": history.history,
        "mape": mape,
        "r2": r2,
        "val_predictions": y_pred,
        "val_actuals": y_val,
    }


# ─────────────────────────────────────────────
# Prophet
# ─────────────────────────────────────────────

def train_prophet(zone_df: pd.DataFrame, target_col: str = "ride_count") -> dict:
    """Entraîne Prophet sur une série temporelle par zone."""
    if not PROPHET_AVAILABLE:
        raise RuntimeError("Prophet non disponible. pip install prophet")

    ts = zone_df[["hour_bucket", target_col]].rename(
        columns={"hour_bucket": "ds", target_col: "y"}
    )

    model = Prophet(
        yearly_seasonality=True,
        weekly_seasonality=True,
        daily_seasonality=True,
        changepoint_prior_scale=0.05,
        seasonality_prior_scale=10,
    )
    model.fit(ts)

    future = model.make_future_dataframe(periods=24, freq="h")
    forecast = model.predict(future)

    return {"model": model, "forecast": forecast}


# ─────────────────────────────────────────────
# Ensemble
# ─────────────────────────────────────────────

def ensemble_predict(
    lstm_pred: np.ndarray,
    prophet_pred: np.ndarray,
    lstm_weight: float = 0.65,
) -> np.ndarray:
    """Combine LSTM et Prophet par pondération linéaire."""
    prophet_weight = 1.0 - lstm_weight
    return lstm_weight * lstm_pred + prophet_weight * prophet_pred


# ─────────────────────────────────────────────
# MLflow training run
# ─────────────────────────────────────────────

def train_and_log(X: np.ndarray, y: np.ndarray, zone_df: pd.DataFrame) -> str:
    """
    Entraîne l'ensemble, loggue dans MLflow et retourne le run_id.
    """
    mlflow.set_tracking_uri(ML.mlflow_tracking_uri)
    mlflow.set_experiment(ML.mlflow_experiment)

    with mlflow.start_run(run_name="demand_forecast_v4") as run:
        # LSTM
        lstm_result = train_lstm(X, y)

        # Prophet
        prophet_result = train_prophet(zone_df)
        prophet_vals = prophet_result["forecast"]["yhat"].values[-len(lstm_result["val_actuals"]):]
        prophet_vals = np.clip(prophet_vals, 0, None)

        # Normalise prophet sur la même échelle que lstm (démo)
        if prophet_vals.max() > 0:
            prophet_vals = prophet_vals / prophet_vals.max()

        # Ensemble
        ensemble_pred = ensemble_predict(lstm_result["val_predictions"], prophet_vals)
        ensemble_mape = mean_absolute_percentage_error(
            lstm_result["val_actuals"] + 1e-8,
            ensemble_pred + 1e-8,
        )
        ensemble_r2 = r2_score(lstm_result["val_actuals"], ensemble_pred)

        # MLflow logging
        mlflow.log_params({
            "lstm_layers": "BiLSTM(128) → LSTM(64) → Dense(32)",
            "lstm_weight": 0.65,
            "prophet_changepoint_prior": 0.05,
            "seq_len": X.shape[1],
            "n_features": X.shape[2],
        })
        mlflow.log_metrics({
            "lstm_mape": lstm_result["mape"],
            "lstm_r2": lstm_result["r2"],
            "ensemble_mape": ensemble_mape,
            "ensemble_r2": ensemble_r2,
        })

        # Sauvegarde modèle Keras
        if TF_AVAILABLE:
            lstm_result["model"].save("/tmp/demand_lstm.h5")
            mlflow.log_artifact("/tmp/demand_lstm.h5")

        print(f"[demand_forecast] MAPE={ensemble_mape:.2%} | R²={ensemble_r2:.4f}")
        return run.info.run_id


if __name__ == "__main__":
    from feature_engineering.demand_features import run as fe_run
    data = fe_run()
    train_and_log(data["X"], data["y"], data["zone_hour_df"])
