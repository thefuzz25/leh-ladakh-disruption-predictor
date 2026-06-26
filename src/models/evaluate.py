import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import shap
import joblib, os
from sklearn.metrics import roc_curve, roc_auc_score, ConfusionMatrixDisplay

FEATURE_COLS = [
    "temp_max", "temp_min", "snowfall_sum", "precip_sum", "windspeed_max",
    "snowfall_3d", "precip_3d", "temp_min_3d",
    "month", "doy_sin", "doy_cos",
    "elevation_m", "slope_deg", "aspect_deg"
]

os.makedirs("notebooks/figures", exist_ok=True)

def evaluate_all():
    df = pd.read_csv("data/processed/features.csv")

    configs = [
        ("avalanche_closure", [10,11,12,1,2,3,4,5], "models/xgb_avalanche.pkl", "Avalanche"),
        ("landslide_closure", [6,7,8,9],             "models/xgb_landslide.pkl", "Landslide"),
    ]

    # --- Figure 1: AUC-ROC curves (both models on same axes) ---
    fig, ax = plt.subplots(figsize=(7, 5))
    for target, months, path, label in configs:
        df_s   = df[df["month"].isin(months)].copy().sort_values("date")
        test   = df_s[pd.to_datetime(df_s["date"]).dt.year >= 2023]
        X_test = test[FEATURE_COLS]
        y_test = test[target]
        bundle = joblib.load(path)
        prob   = bundle["model"].predict_proba(X_test)[:, 1]
        fpr, tpr, _ = roc_curve(y_test, prob)
        auc = roc_auc_score(y_test, prob)
        ax.plot(fpr, tpr, label=f"{label} (AUC = {auc:.3f})", linewidth=2)
    ax.plot([0,1],[0,1],"k--", linewidth=1)
    ax.set_xlabel("False Positive Rate"); ax.set_ylabel("True Positive Rate")
    ax.set_title("ROC Curves — Avalanche vs Landslide Model")
    ax.legend(); ax.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig("notebooks/figures/auc_curves.png", dpi=150)
    plt.close()
    print("Saved: auc_curves.png")

    # --- Figure 2: Confusion matrix at F2-optimal threshold (avalanche model) ---
    df_aval  = df[df["month"].isin([10,11,12,1,2,3,4,5])].copy().sort_values("date")
    test_a   = df_aval[pd.to_datetime(df_aval["date"]).dt.year >= 2023]
    bundle_a = joblib.load("models/xgb_avalanche.pkl")
    prob_a   = bundle_a["model"].predict_proba(test_a[FEATURE_COLS])[:, 1]
    y_pred   = (prob_a >= bundle_a["threshold"]).astype(int)
    fig, ax  = plt.subplots(figsize=(5, 4))
    ConfusionMatrixDisplay.from_predictions(
        test_a["avalanche_closure"], y_pred, ax=ax,
        display_labels=["No closure", "Closure"],
        cmap="Blues"
    )
    ax.set_title(f"Confusion Matrix — Avalanche Model\n(threshold = {bundle_a['threshold']:.2f}, F2-optimised)")
    plt.tight_layout()
    plt.savefig("notebooks/figures/confusion_matrix.png", dpi=150)
    plt.close()
    print("Saved: confusion_matrix.png")

    # --- Figure 3: SHAP beeswarm — avalanche model, top 10 features ---
    explainer   = shap.TreeExplainer(bundle_a["model"])
    shap_values = explainer.shap_values(test_a[FEATURE_COLS])
    fig, ax = plt.subplots(figsize=(8, 5))
    shap.summary_plot(shap_values, test_a[FEATURE_COLS],
                      max_display=10, show=False)
    plt.title("SHAP Feature Importance — Avalanche Model")
    plt.tight_layout()
    plt.savefig("notebooks/figures/shap_summary.png", dpi=150, bbox_inches="tight")
    plt.close()
    print("Saved: shap_summary.png")

    # --- Figure 4: SHAP waterfall for one high-risk prediction ---
    high_risk_idx = np.argmax(prob_a)
    shap_exp = shap.Explanation(
        values=shap_values[high_risk_idx],
        base_values=explainer.expected_value,
        data=test_a[FEATURE_COLS].iloc[high_risk_idx],
        feature_names=FEATURE_COLS
    )
    fig, ax = plt.subplots(figsize=(8, 5))
    shap.waterfall_plot(shap_exp, show=False)
    plt.title("SHAP Waterfall — Highest Risk Prediction")
    plt.tight_layout()
    plt.savefig("notebooks/figures/shap_waterfall.png", dpi=150, bbox_inches="tight")
    plt.close()
    print("Saved: shap_waterfall.png")

if __name__ == "__main__":
    evaluate_all()
