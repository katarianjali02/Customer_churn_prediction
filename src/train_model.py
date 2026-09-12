"""
train_model.py
---------------
Trains and compares several classification models for customer churn prediction,
selects the best performer, and saves it (plus evaluation artifacts) to disk.

Run with:
    python src/train_model.py
"""

import os
import json
import joblib
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")  # non-interactive backend, safe for headless runs
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, roc_curve, confusion_matrix, classification_report
)

from data_preprocessing import preprocess_pipeline

MODELS_DIR = "models"
OUTPUTS_DIR = "outputs"

MODEL_CANDIDATES = {
    "LogisticRegression": LogisticRegression(max_iter=1000, random_state=42),
    "DecisionTree": DecisionTreeClassifier(random_state=42, max_depth=6),
    "RandomForest": RandomForestClassifier(n_estimators=300, random_state=42),
    "GradientBoosting": GradientBoostingClassifier(random_state=42),
    "KNN": KNeighborsClassifier(n_neighbors=15),
    "SVM": SVC(probability=True, random_state=42),
}


def evaluate_model(model, X_test, y_test):
    """Compute standard classification metrics for a fitted model."""
    y_pred = model.predict(X_test)
    y_proba = model.predict_proba(X_test)[:, 1]

    return {
        "accuracy": accuracy_score(y_test, y_pred),
        "precision": precision_score(y_test, y_pred),
        "recall": recall_score(y_test, y_pred),
        "f1_score": f1_score(y_test, y_pred),
        "roc_auc": roc_auc_score(y_test, y_proba),
    }, y_pred, y_proba


def plot_confusion_matrix(y_test, y_pred, model_name, save_path):
    cm = confusion_matrix(y_test, y_pred)
    plt.figure(figsize=(5, 4))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
                xticklabels=["No Churn", "Churn"], yticklabels=["No Churn", "Churn"])
    plt.title(f"Confusion Matrix - {model_name}")
    plt.ylabel("Actual")
    plt.xlabel("Predicted")
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()


def plot_roc_curves(results_proba, y_test, save_path):
    plt.figure(figsize=(7, 6))
    for name, y_proba in results_proba.items():
        fpr, tpr, _ = roc_curve(y_test, y_proba)
        auc = roc_auc_score(y_test, y_proba)
        plt.plot(fpr, tpr, label=f"{name} (AUC = {auc:.3f})")
    plt.plot([0, 1], [0, 1], linestyle="--", color="gray", label="Random")
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("ROC Curves - Model Comparison")
    plt.legend(loc="lower right")
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()


def plot_feature_importance(model, feature_columns, save_path, top_n=15):
    if not hasattr(model, "feature_importances_"):
        return
    importances = model.feature_importances_
    idx = np.argsort(importances)[-top_n:]
    plt.figure(figsize=(8, 6))
    plt.barh(np.array(feature_columns)[idx], importances[idx], color="steelblue")
    plt.xlabel("Importance")
    plt.title("Top Feature Importances")
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()


def main():
    os.makedirs(MODELS_DIR, exist_ok=True)
    os.makedirs(OUTPUTS_DIR, exist_ok=True)

    print("Loading and preprocessing data...")
    X_train, X_test, y_train, y_test = preprocess_pipeline()
    feature_columns = X_train.columns.tolist()

    results = {}
    probas = {}
    fitted_models = {}

    for name, model in MODEL_CANDIDATES.items():
        print(f"Training {name}...")
        model.fit(X_train, y_train)
        metrics, y_pred, y_proba = evaluate_model(model, X_test, y_test)
        results[name] = metrics
        probas[name] = y_proba
        fitted_models[name] = model
        print(f"  {name}: accuracy={metrics['accuracy']:.3f}, "
              f"f1={metrics['f1_score']:.3f}, roc_auc={metrics['roc_auc']:.3f}")

    # Pick best model by ROC-AUC (a good overall measure for imbalanced churn data)
    best_name = max(results, key=lambda k: results[k]["roc_auc"])
    best_model = fitted_models[best_name]
    print(f"\nBest model: {best_name} (ROC-AUC = {results[best_name]['roc_auc']:.3f})")

    # Save best model + metadata needed for inference
    joblib.dump(best_model, os.path.join(MODELS_DIR, "churn_model.pkl"))
    joblib.dump(feature_columns, os.path.join(MODELS_DIR, "feature_columns.pkl"))

    with open(os.path.join(OUTPUTS_DIR, "model_comparison.json"), "w") as f:
        json.dump(results, f, indent=2)

    with open(os.path.join(OUTPUTS_DIR, "best_model.txt"), "w") as f:
        f.write(f"Best model: {best_name}\n")
        f.write(json.dumps(results[best_name], indent=2))

    # Classification report for best model
    y_pred_best = best_model.predict(X_test)
    report = classification_report(y_test, y_pred_best, target_names=["No Churn", "Churn"])
    with open(os.path.join(OUTPUTS_DIR, "classification_report.txt"), "w") as f:
        f.write(f"Best Model: {best_name}\n\n")
        f.write(report)

    # Plots
    plot_confusion_matrix(y_test, y_pred_best, best_name,
                           os.path.join(OUTPUTS_DIR, "confusion_matrix.png"))
    plot_roc_curves(probas, y_test, os.path.join(OUTPUTS_DIR, "roc_curves.png"))
    plot_feature_importance(best_model, feature_columns,
                             os.path.join(OUTPUTS_DIR, "feature_importance.png"))

    # Comparison bar chart across models
    comp_df = pd.DataFrame(results).T
    comp_df.to_csv(os.path.join(OUTPUTS_DIR, "model_comparison.csv"))
    comp_df[["accuracy", "f1_score", "roc_auc"]].plot(kind="bar", figsize=(9, 5))
    plt.title("Model Comparison")
    plt.ylabel("Score")
    plt.xticks(rotation=30, ha="right")
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUTS_DIR, "model_comparison.png"))
    plt.close()

    print("\nAll outputs saved to 'outputs/' and 'models/'.")


if __name__ == "__main__":
    main()
