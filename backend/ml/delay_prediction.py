"""
ml/delay_prediction.py
XGBoost-based delivery delay prediction model.
Trained on features from the logistics database.
"""
import os
import pickle
import numpy as np
import pandas as pd
from typing import Dict, Optional, Tuple
from sqlalchemy import text

from backend.database.db_connection import get_engine

MODEL_PATH = os.path.join(os.path.dirname(__file__), "delay_model.pkl")

TRAFFIC_ENCODING = {
    "Low": 1, "Medium": 2, "High": 3, "Very High": 4,
    "Standard Class": 2, "First Class": 1, "Second Class": 3, "Same Day": 4,
}

SHIPPING_MODE_ENCODING = {
    "Same Day": 1, "First Class": 2, "Second Class": 3, "Standard Class": 4,
}


def _load_training_data() -> pd.DataFrame:
    """Load features from PostgreSQL for training."""
    query = """
    SELECT
        d.distance_km,
        d.days_for_shipment_scheduled,
        d.late_delivery_risk,
        r.average_traffic_level,
        r.shipping_mode,
        dr.experience_years,
        dr.rating,
        EXTRACT(HOUR FROM d.pickup_time) AS hour_of_day,
        EXTRACT(DOW FROM d.pickup_time) AS day_of_week
    FROM deliveries d
    JOIN routes r ON d.route_id = r.route_id
    JOIN drivers dr ON d.driver_id = dr.driver_id
    WHERE d.distance_km IS NOT NULL
      AND d.late_delivery_risk IS NOT NULL
    LIMIT 50000
    """
    engine = get_engine()
    with engine.connect() as conn:
        df = pd.DataFrame(conn.execute(text(query)).fetchall(),
                          columns=["distance_km", "days_scheduled", "late_delivery_risk",
                                   "traffic_level", "shipping_mode", "experience_years",
                                   "rating", "hour_of_day", "day_of_week"])
    return df


def _prepare_features(df: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray]:
    """Encode categorical features and return X, y arrays."""
    df = df.copy()
    df["traffic_encoded"] = df["traffic_level"].map(TRAFFIC_ENCODING).fillna(2)
    df["shipping_encoded"] = df["shipping_mode"].map(SHIPPING_MODE_ENCODING).fillna(3)
    df = df.fillna(0)

    feature_cols = [
        "distance_km", "days_scheduled", "traffic_encoded",
        "shipping_encoded", "experience_years", "rating",
        "hour_of_day", "day_of_week"
    ]
    X = df[feature_cols].values
    y = df["late_delivery_risk"].values.astype(int)
    return X, y


def train_model() -> dict:
    """Train XGBoost delay prediction model and save to disk."""
    try:
        from xgboost import XGBClassifier
        from sklearn.model_selection import train_test_split
        from sklearn.metrics import classification_report, accuracy_score
        from sklearn.preprocessing import StandardScaler
    except ImportError as e:
        return {"error": f"Missing dependency: {e}"}

    print("📊 Loading training data...")
    try:
        df = _load_training_data()
    except Exception as e:
        return {"error": f"Failed to load data: {e}"}

    if len(df) < 100:
        return {"error": "Insufficient data for training (need at least 100 rows)."}

    X, y = _prepare_features(df)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    print("🤖 Training XGBoost model...")
    model = XGBClassifier(
        n_estimators=200,
        max_depth=6,
        learning_rate=0.1,
        use_label_encoder=False,
        eval_metric="logloss",
        random_state=42,
        n_jobs=-1,
    )
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    report = classification_report(y_test, y_pred, output_dict=True)

    # Save model + scaler
    with open(MODEL_PATH, "wb") as f:
        pickle.dump({"model": model, "scaler": scaler}, f)

    print(f"✅ Model trained. Accuracy: {acc:.4f}")
    return {
        "accuracy": round(acc, 4),
        "precision": round(report.get("1", {}).get("precision", 0), 4),
        "recall": round(report.get("1", {}).get("recall", 0), 4),
        "f1": round(report.get("1", {}).get("f1-score", 0), 4),
        "training_samples": len(X_train),
    }


def predict_delay(features: Dict) -> Dict:
    """
    Predict delay probability for given logistics features.
    features dict keys:
        distance_km, days_scheduled, traffic_level, shipping_mode,
        experience_years, rating, hour_of_day, day_of_week
    """
    if not os.path.exists(MODEL_PATH):
        return {"error": "Model not trained yet. Call /train-model first.", "probability": None}

    with open(MODEL_PATH, "rb") as f:
        bundle = pickle.load(f)
    model = bundle["model"]
    scaler = bundle["scaler"]

    traffic_enc = TRAFFIC_ENCODING.get(features.get("traffic_level", "Medium"), 2)
    shipping_enc = SHIPPING_MODE_ENCODING.get(features.get("shipping_mode", "Standard Class"), 3)

    X = np.array([[
        features.get("distance_km", 0),
        features.get("days_scheduled", 3),
        traffic_enc,
        shipping_enc,
        features.get("experience_years", 3),
        features.get("rating", 3.5),
        features.get("hour_of_day", 12),
        features.get("day_of_week", 1),
    ]])
    X = scaler.transform(X)
    prob = float(model.predict_proba(X)[0][1])
    label = "HIGH RISK" if prob >= 0.5 else "LOW RISK"
    return {
        "probability": round(prob, 4),
        "risk_label": label,
        "delay_chance_pct": round(prob * 100, 1),
    }


def model_is_trained() -> bool:
    return os.path.exists(MODEL_PATH)


if __name__ == "__main__":
    result = train_model()
    print(result)
