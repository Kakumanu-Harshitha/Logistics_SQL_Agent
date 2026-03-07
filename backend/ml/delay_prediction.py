"""
ml/delay_prediction.py
XGBoost-based delivery delay prediction model.
Trained on features from the logistics database.
"""
import os
import pickle
import numpy as np
import pandas as pd
from typing import Dict, Optional, Tuple, List
from sqlalchemy import text

from backend.database.db_connection import get_engine

MODEL_PATH = os.path.join(os.path.dirname(__file__), "delay_model.pkl")

# Global Cache for Model Bundle
_MODEL_CACHE = None

TRAFFIC_ENCODING = {
    "Low": 1, "Medium": 2, "High": 3, "Very High": 4,
    "Standard Class": 2, "First Class": 1, "Second Class": 3, "Same Day": 4,
}

SHIPPING_MODE_ENCODING = {
    "Same Day": 1, "First Class": 2, "Second Class": 3, "Standard Class": 4,
}

def _get_model_bundle() -> Optional[Dict]:
    """Lazy-load the model bundle from disk and cache it in memory."""
    global _MODEL_CACHE
    if _MODEL_CACHE is not None:
        return _MODEL_CACHE
    
    if not os.path.exists(MODEL_PATH):
        return None
        
    try:
        with open(MODEL_PATH, "rb") as f:
            _MODEL_CACHE = pickle.load(f)
        return _MODEL_CACHE
    except Exception as e:
        print(f"❌ Error loading model: {e}")
        return None


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
    
    # Force numerical columns to float to avoid Decimal vs float errors
    numeric_cols = ["distance_km", "days_scheduled", "late_delivery_risk", "experience_years", "rating", "hour_of_day", "day_of_week"]
    for col in numeric_cols:
        df[col] = pd.to_numeric(df[col], errors='coerce').fillna(0).astype(float)
        
    return df


def _prepare_features(df: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray, List[str]]:
    """Encode categorical features and return X, y arrays + feature names."""
    df = df.copy()
    df["traffic_encoded"] = df["traffic_level"].map(TRAFFIC_ENCODING).fillna(2)
    df["shipping_encoded"] = df["shipping_mode"].map(SHIPPING_MODE_ENCODING).fillna(3)
    
    # 1. Feature Engineering: Schedule Tightness
    dist = df["distance_km"].astype(float)
    days = df["days_scheduled"].astype(float).replace(0, 0.5)
    df["schedule_tightness"] = dist / days
    
    # 2. Distance Bins (Short, Medium, Long)
    df["dist_cat"] = pd.cut(df["distance_km"], bins=[0, 300, 800, 10000], labels=[1, 2, 3]).astype(float)
    
    # 3. Interaction: Traffic * Distance
    df["traffic_dist_interaction"] = df["traffic_encoded"] * df["distance_km"]
    
    # 4. Driver Experience Groups
    df["exp_group"] = pd.cut(df["experience_years"], bins=[0, 2, 5, 10, 100], labels=[1, 2, 3, 4]).astype(float)
    
    df = df.fillna(0)

    feature_cols = [
        "distance_km", "days_scheduled", "schedule_tightness",
        "dist_cat", "traffic_dist_interaction", "exp_group",
        "traffic_encoded", "shipping_encoded", "experience_years", "rating",
        "hour_of_day", "day_of_week"
    ]
    X = df[feature_cols].values
    y = df["late_delivery_risk"].values.astype(int)
    return X, y, feature_cols


def train_model() -> dict:
    """Train optimized XGBoost model with Tuning and Thresholding."""
    try:
        from xgboost import XGBClassifier
        from sklearn.model_selection import train_test_split, RandomizedSearchCV, StratifiedKFold
        from sklearn.metrics import classification_report, accuracy_score, precision_recall_curve, f1_score
        from sklearn.preprocessing import StandardScaler
    except ImportError as e:
        return {"error": f"Missing dependency: {e}"}

    print("📊 Loading training data...")
    try:
        df = _load_training_data()
    except Exception as e:
        return {"error": f"Failed to load data: {e}"}

    if len(df) < 100:
        return {"error": "Insufficient data for training."}

    X, y, feature_names = _prepare_features(df)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    print("🤖 Tuning XGBoost model...")
    num_neg = np.sum(y_train == 0)
    num_pos = np.sum(y_train == 1)
    pos_weight = num_neg / num_pos if num_pos > 0 else 1.0
    
    param_dist = {
        'n_estimators': [100, 200, 300, 500],
        'max_depth': [3, 4, 5, 6],
        'learning_rate': [0.01, 0.05, 0.1],
        'subsample': [0.7, 0.8, 0.9],
        'colsample_bytree': [0.7, 0.8, 0.9],
        'min_child_weight': [1, 3, 5]
    }

    base_model = XGBClassifier(
        scale_pos_weight=pos_weight,
        use_label_encoder=False,
        eval_metric="logloss",
        random_state=42,
        n_jobs=-1
    )

    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    random_search = RandomizedSearchCV(
        base_model, param_distributions=param_dist, 
        n_iter=10, cv=cv, scoring='f1', n_jobs=-1, random_state=42
    )
    random_search.fit(X_train, y_train)
    
    best_model = random_search.best_estimator_
    
    # Threshold Optimization to maximize F1 while keeping high Recall
    y_probs = best_model.predict_proba(X_test)[:, 1]
    precisions, recalls, thresholds = precision_recall_curve(y_test, y_probs)
    
    # Weight Recall higher in the F1-like score for selection
    # Or just find the threshold that gives highest F1
    f1_scores = (2 * precisions * recalls) / (precisions + recalls + 1e-9)
    best_idx = np.argmax(f1_scores)
    best_threshold = float(thresholds[best_idx]) if best_idx < len(thresholds) else 0.5

    # Final Evaluation with optimal threshold
    y_pred_optimal = (y_probs >= best_threshold).astype(int)
    acc = accuracy_score(y_test, y_pred_optimal)
    report = classification_report(y_test, y_pred_optimal, output_dict=True)
    
    # Importance Analysis
    importances = best_model.feature_importances_
    feat_imp = {name: round(float(imp), 4) for name, imp in zip(feature_names, importances)}

    # Save model + scaler + threshold
    with open(MODEL_PATH, "wb") as f:
        pickle.dump({
            "model": best_model, 
            "scaler": scaler, 
            "threshold": best_threshold,
            "feature_names": feature_names
        }, f)

    # Invalidate cache after training
    global _MODEL_CACHE
    _MODEL_CACHE = None

    return {
        "accuracy": round(acc, 4),
        "precision": round(report.get("1", {}).get("precision", 0), 4),
        "recall": round(report.get("1", {}).get("recall", 0), 4),
        "f1": round(report.get("1", {}).get("f1-score", 0), 4),
        "best_threshold": round(best_threshold, 4),
        "best_params": random_search.best_params_,
        "feature_importance": feat_imp,
        "training_samples": len(X_train),
    }


def predict_delay(features: Dict) -> Dict:
    """Predict delay with threshold optimization and in-memory caching."""
    bundle = _get_model_bundle()
    if not bundle:
        return {"error": "Model not trained yet.", "probability": None}
    
    model = bundle["model"]
    scaler = bundle["scaler"]
    threshold = bundle.get("threshold", 0.5)

    traffic_enc = TRAFFIC_ENCODING.get(features.get("traffic_level", "Medium"), 2)
    shipping_enc = SHIPPING_MODE_ENCODING.get(features.get("shipping_mode", "Standard Class"), 3)
    
    dist = float(features.get("distance_km", 0))
    days = float(features.get("days_scheduled", 3))
    tightness = dist / max(days, 0.5)
    
    # Replicate FE logic
    dist_cat = 1.0 if dist <= 300 else (2.0 if dist <= 800 else 3.0)
    traffic_dist = traffic_enc * dist
    exp = float(features.get("experience_years", 3))
    exp_group = 1.0 if exp <= 2 else (2.0 if exp <= 5 else (3.0 if exp <= 10 else 4.0))

    X = np.array([[
        dist,
        days,
        tightness,
        dist_cat,
        traffic_dist,
        exp_group,
        traffic_enc,
        shipping_enc,
        exp,
        float(features.get("rating", 3.5)),
        float(features.get("hour_of_day", 12)),
        float(features.get("day_of_week", 1)),
    ]])
    X = scaler.transform(X)
    prob = float(model.predict_proba(X)[0][1])
    
    label = "HIGH RISK" if prob >= threshold else "LOW RISK"
    
    return {
        "probability": round(prob, 4),
        "risk_label": label,
        "optimized_threshold": round(threshold, 4),
        "delay_chance_pct": round(prob * 100, 1),
    }


def model_is_trained() -> bool:
    return os.path.exists(MODEL_PATH)


if __name__ == "__main__":
    result = train_model()
    print(result)
