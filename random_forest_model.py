"""
random_forest_model.py
-----------------------
Random Forest classifier for layoff risk prediction.
Works with small datasets (39 rows) using cross-validation.
"""

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import joblib, os, logging

from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_val_score, StratifiedKFold
from sklearn.metrics import (
    accuracy_score, classification_report, confusion_matrix,
    roc_auc_score, roc_curve, ConfusionMatrixDisplay
)
from data_preprocessing import load_data, clean_data, engineer_features, encode_and_scale

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)

os.makedirs("models",  exist_ok=True)
os.makedirs("outputs", exist_ok=True)


def train_random_forest(X_train, y_train) -> RandomForestClassifier:
    import config as cfg
    max_depth = None if str(cfg.RF_MAX_DEPTH).lower() == "none" else int(cfg.RF_MAX_DEPTH)
    # For small datasets, use shallow trees to avoid overfitting
    n_samples = len(X_train)
    if n_samples < 50:
        max_depth  = 4
        n_est      = 100
        min_split  = 2
        min_leaf   = 1
        logger.info(f"Small dataset ({n_samples} rows) — using conservative RF params")
    else:
        n_est     = cfg.RF_N_ESTIMATORS
        min_split = cfg.RF_MIN_SAMPLES_SPLIT
        min_leaf  = cfg.RF_MIN_SAMPLES_LEAF

    model = RandomForestClassifier(
        n_estimators      = n_est,
        max_depth         = max_depth,
        min_samples_split = min_split,
        min_samples_leaf  = min_leaf,
        max_features      = "sqrt",
        class_weight      = "balanced",
        random_state      = 42,
        n_jobs            = -1
    )
    model.fit(X_train, y_train)
    logger.info("Random Forest trained successfully")
    return model


def evaluate_model(model, X_train, X_test, y_train, y_test, feature_names):
    # Use StratifiedKFold — safe for small datasets
    n_splits  = min(5, len(y_train) // 2)
    cv        = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)
    cv_scores = cross_val_score(model, X_train, y_train, cv=cv, scoring="f1")

    y_pred      = model.predict(X_test)
    y_pred_prob = model.predict_proba(X_test)[:, 1]

    accuracy = accuracy_score(y_test, y_pred)

    # ROC-AUC needs both classes in test set
    try:
        roc_auc = roc_auc_score(y_test, y_pred_prob)
    except ValueError:
        roc_auc = 0.5
        logger.warning("Only one class in test set — ROC-AUC set to 0.5")

    report = classification_report(y_test, y_pred,
                target_names=["No Layoff", "Layoff"], output_dict=True)

    logger.info(f"Accuracy : {accuracy:.4f}")
    logger.info(f"ROC-AUC  : {roc_auc:.4f}")
    logger.info(f"CV F1    : {cv_scores.mean():.4f} +/- {cv_scores.std():.4f}")
    logger.info(f"\n{classification_report(y_test, y_pred, target_names=['No Layoff','Layoff'])}")

    _plot_confusion_matrix(y_test, y_pred)
    _plot_feature_importance(model, feature_names)
    if len(np.unique(y_test)) > 1:
        _plot_roc_curve(y_test, y_pred_prob, roc_auc)

    return {
        "accuracy":   round(float(accuracy), 4),
        "roc_auc":    round(float(roc_auc),  4),
        "cv_f1_mean": round(float(cv_scores.mean()), 4),
        "cv_f1_std":  round(float(cv_scores.std()),  4),
        "classification_report": report,
    }


def _plot_confusion_matrix(y_test, y_pred):
    fig, ax = plt.subplots(figsize=(6, 5))
    ConfusionMatrixDisplay(confusion_matrix(y_test, y_pred),
        display_labels=["No Layoff", "Layoff"]).plot(ax=ax, colorbar=False, cmap="Blues")
    ax.set_title("Random Forest — Confusion Matrix", fontsize=13, fontweight="bold")
    plt.tight_layout()
    plt.savefig("outputs/rf_confusion_matrix.png", dpi=150)
    plt.close()
    logger.info("Saved → outputs/rf_confusion_matrix.png")


def _plot_roc_curve(y_test, y_pred_prob, roc_auc):
    fpr, tpr, _ = roc_curve(y_test, y_pred_prob)
    fig, ax = plt.subplots(figsize=(7, 5))
    ax.plot(fpr, tpr, color="#0077b6", lw=2, label=f"Random Forest (AUC = {roc_auc:.3f})")
    ax.plot([0,1],[0,1], "k--", lw=1, label="Random Baseline")
    ax.set_xlabel("False Positive Rate"); ax.set_ylabel("True Positive Rate")
    ax.set_title("ROC Curve — Random Forest", fontsize=13, fontweight="bold")
    ax.legend(loc="lower right"); ax.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig("outputs/rf_roc_curve.png", dpi=150)
    plt.close()
    logger.info("Saved → outputs/rf_roc_curve.png")


def _plot_feature_importance(model, feature_names, top_n=10):
    importances = model.feature_importances_
    indices     = np.argsort(importances)[::-1][:top_n]
    top_feat    = [feature_names[i] for i in indices]
    top_vals    = importances[indices]
    fig, ax = plt.subplots(figsize=(9, 6))
    bars = ax.barh(top_feat[::-1], top_vals[::-1], color="#0077b6")
    ax.set_xlabel("Importance Score")
    ax.set_title("Feature Importance — Random Forest", fontsize=13, fontweight="bold")
    ax.bar_label(bars, fmt="%.3f", padding=3, fontsize=9)
    ax.grid(alpha=0.3, axis="x")
    plt.tight_layout()
    plt.savefig("outputs/rf_feature_importance.png", dpi=150)
    plt.close()
    logger.info("Saved → outputs/rf_feature_importance.png")


def predict_single(model, scaler, encoders, feature_names, input_data: dict) -> dict:
    import pandas as pd, config as cfg
    from data_preprocessing import CATEGORICAL_COLS
    row = pd.DataFrame([input_data])
    for col in CATEGORICAL_COLS:
        if col in row.columns and col in encoders:
            try:
                row[col] = encoders[col].transform(row[col].astype(str))
            except ValueError:
                row[col] = 0
    for col in feature_names:
        if col not in row.columns:
            row[col] = 0
    row_scaled = scaler.transform(row[feature_names])
    prob  = model.predict_proba(row_scaled)[0][1]
    label = "High" if prob >= cfg.PRED_HIGH_PROB else ("Medium" if prob >= cfg.PRED_MEDIUM_PROB else "Low")
    return {"probability": round(float(prob)*100, 1), "risk_label": label, "prediction": int(prob >= 0.5)}


def save_model(model, scaler, encoders, feature_names):
    joblib.dump({"model": model, "scaler": scaler,
                 "encoders": encoders, "feature_names": feature_names},
                "models/random_forest.pkl")
    logger.info("Model saved → models/random_forest.pkl")


def load_model():
    p = joblib.load("models/random_forest.pkl")
    return p["model"], p["scaler"], p["encoders"], p["feature_names"]


if __name__ == "__main__":
    import sys
    path = sys.argv[1] if len(sys.argv) > 1 else "data/Layoffs_Dataset.csv"
    df = load_data(path); df = clean_data(df); df = engineer_features(df)
    X_train, X_test, y_train, y_test, scaler, features, encoders = encode_and_scale(df)
    model   = train_random_forest(X_train, y_train)
    metrics = evaluate_model(model, X_train, X_test, y_train, y_test, features)
    for k, v in metrics.items():
        if k != "classification_report": print(f"  {k}: {v}")
    save_model(model, scaler, encoders, features)