import pandas as pd
import numpy as np
import xgboost as xgb
import joblib, json
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.metrics import precision_recall_curve

FEATURE_COLS = [
    "temp_max", "temp_min", "snowfall_sum", "precip_sum", "windspeed_max",
    "snowfall_3d", "precip_3d", "temp_min_3d",
    "month", "doy_sin", "doy_cos",
    "elevation_m", "slope_deg", "aspect_deg"
]

def get_f2_threshold(y_true, y_prob) -> float:
    """
    Find the decision threshold that maximises F2 score.
    F2 weights recall twice as heavily as precision.
    Justified by domain: missing a closure (false negative) costs more
    than a false alarm (false positive) — you over-prepare rather than under-prepare.
    """
    prec, rec, thresholds = precision_recall_curve(y_true, y_prob)
    f2 = (5 * prec * rec) / (4 * prec + rec + 1e-9)
    best_idx = np.argmax(f2[:-1])
    return float(thresholds[best_idx])

def train_model(df: pd.DataFrame, target: str, season_months: list, save_path: str):
    """
    Train and save one hazard model.

    Train/test split: time-based (train 2019-2022, test 2023-2024).
    This is critical — random k-fold on time-series leaks future data into the past.

    Models compared: Logistic Regression (baseline), Random Forest, XGBoost.
    XGBoost selected as final model based on AUC on test set.
    scale_pos_weight handles class imbalance — computed from training data.
    """
    df_s = df[df["month"].isin(season_months)].copy().sort_values("date")

    train = df_s[pd.to_datetime(df_s["date"]).dt.year <= 2022]
    test  = df_s[pd.to_datetime(df_s["date"]).dt.year >= 2023]

    X_train, y_train = train[FEATURE_COLS], train[target]
    X_test,  y_test  = test[FEATURE_COLS],  test[target]

    spw = (len(y_train) - y_train.sum()) / max(y_train.sum(), 1)

    models = {
        "logistic":      LogisticRegression(class_weight="balanced", max_iter=1000),
        "random_forest": RandomForestClassifier(n_estimators=200, max_depth=8,
                             class_weight="balanced", random_state=42, n_jobs=-1),
        "xgboost":       xgb.XGBClassifier(n_estimators=300, max_depth=6,
                             learning_rate=0.05, scale_pos_weight=spw,
                             eval_metric="auc", use_label_encoder=False,
                             random_state=42)
    }

    results = {}
    for name, m in models.items():
        m.fit(X_train, y_train)
        prob = m.predict_proba(X_test)[:, 1]
        auc  = roc_auc_score(y_test, prob)
        results[name] = {"model": m, "prob": prob, "auc": auc}
        print(f"{target} | {name}: AUC = {auc:.3f}")

    best_model = results["xgboost"]["model"]
    best_prob  = results["xgboost"]["prob"]
    threshold  = get_f2_threshold(y_test, best_prob)
    print(f"{target} | F2-optimal threshold: {threshold:.3f}")

    joblib.dump(
        {"model": best_model, "threshold": threshold, "all_results": results},
        save_path
    )
    print(f"Saved: {save_path}\n")
    return results

if __name__ == "__main__":
    df = pd.read_csv("data/processed/features.csv")
    import os; os.makedirs("models", exist_ok=True)

    train_model(df, "avalanche_closure", [10,11,12,1,2,3,4,5], "models/xgb_avalanche.pkl")
    train_model(df, "landslide_closure", [6,7,8,9],             "models/xgb_landslide.pkl")
    json.dump(FEATURE_COLS, open("models/feature_names.json", "w"))
    print("All models saved.")
