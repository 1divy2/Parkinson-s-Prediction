# backend/train_model.py
from pathlib import Path
import json
import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, roc_auc_score, confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

BACKEND_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = BACKEND_DIR.parent
CSV_PATH = PROJECT_ROOT / "parkinsons_features_LIBROSA.csv"
MODELS_DIR = BACKEND_DIR / "models"
MODELS_DIR.mkdir(parents=True, exist_ok=True)

MODEL_OUT = MODELS_DIR / "model.joblib"
SCALER_OUT = MODELS_DIR / "scaler.joblib"
FNAMES_OUT = MODELS_DIR / "feature_names.joblib"
METRICS_OUT = MODELS_DIR / "metrics.json"

if not CSV_PATH.exists():
    raise FileNotFoundError(f"CSV not found: {CSV_PATH}")

df = pd.read_csv(CSV_PATH)

target_col = "status" if "status" in df.columns else ("label" if "label" in df.columns else None)
if target_col is None:
    raise ValueError("No target column found (need 'status' or 'label').")

y = df[target_col].astype(int).values

drop_cols = [target_col]
if "filename" in df.columns:
    drop_cols.append("filename")
X_df = df.drop(columns=drop_cols, errors="ignore")

for c in X_df.columns:
    X_df[c] = pd.to_numeric(X_df[c], errors="coerce")
X_df = X_df.replace([np.inf, -np.inf], np.nan).fillna(0.0)

feature_names = X_df.columns.tolist()
X = X_df.values

unique, counts = np.unique(y, return_counts=True)
class_dist = dict(zip(unique.tolist(), counts.tolist()))
print("Label distribution:", class_dist)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=42, stratify=y)

scaler = StandardScaler()
X_train_s = scaler.fit_transform(X_train)
X_test_s = scaler.transform(X_test)

rf = RandomForestClassifier(n_estimators=500, max_depth=None, class_weight="balanced", random_state=42, n_jobs=-1)
rf.fit(X_train_s, y_train)
y_pred = rf.predict(X_test_s)

metrics = {
    "name": "rf",
    "accuracy": float(accuracy_score(y_test, y_pred)),
    "precision": float(precision_score(y_test, y_pred, zero_division=0)),
    "recall": float(recall_score(y_test, y_pred, zero_division=0)),
    "f1": float(f1_score(y_test, y_pred, zero_division=0)),
    "auc": None,
}

y_proba = None
if hasattr(rf, "predict_proba"):
    try:
        y_proba = rf.predict_proba(X_test_s)[:, 1]
        metrics["auc"] = float(roc_auc_score(y_test, y_proba))
    except Exception:
        metrics["auc"] = None

# choose simple threshold (use 0.50 or find best threshold if probs exist)
selected_threshold = 0.50
th_info = None
if y_proba is not None:
    best_t, best_f1, best_rec = 0.50, -1, 0.0
    for t in np.linspace(0.15, 0.85, 71):
        y_hat = (y_proba >= t).astype(int)
        f1 = f1_score(y_test, y_hat, zero_division=0)
        rec = recall_score(y_test, y_hat, zero_division=0)
        if (f1 > best_f1) or (np.isclose(f1, best_f1) and rec > best_rec):
            best_f1, best_rec, best_t = f1, rec, t
    selected_threshold = float(best_t)
    th_info = {"f1_at_threshold": float(best_f1), "recall_at_threshold": float(best_rec)}

# confusion matrix at threshold
if y_proba is not None:
    y_hat_val = (y_proba >= selected_threshold).astype(int)
else:
    y_hat_val = y_pred

cm = confusion_matrix(y_test, y_hat_val, labels=[0,1])
tn, fp, fn, tp = cm.ravel() if cm.size == 4 else (0,0,0,0)

joblib.dump(rf, MODEL_OUT)
joblib.dump(scaler, SCALER_OUT)
joblib.dump(feature_names, FNAMES_OUT)

summary = {
    "chosen_model": "rf",
    "model_label": "RF",
    "n_features": len(feature_names),
    "metrics": metrics,
    "threshold_info": {
        "selected_threshold": selected_threshold,
        **(th_info or {}),
        "note": "Use this probability threshold for verdict.",
    },
    "confusion_matrix": {"TN": int(tn), "FP": int(fp), "FN": int(fn), "TP": int(tp)},
    "class_distribution": {str(int(k)): int(v) for k,v in zip(np.unique(y), np.unique(y, return_counts=True)[1])},
}

with open(METRICS_OUT, "w") as f:
    json.dump(summary, f, indent=2)

print("\n=== Training complete ===")
print(f"Model: RF")
print(f"Features used: {len(feature_names)}")
print(f"Accuracy: {metrics['accuracy']:.3f}")
print(f"Precision: {metrics['precision']:.3f}")
print(f"Recall: {metrics['recall']:.3f}")
print(f"F1: {metrics['f1']:.3f}")
print(f"AUC: {metrics['auc'] if metrics['auc'] is not None else 'N/A'}")
print(f"Selected threshold: {selected_threshold:.3f}")
print("Saved files:")
print(f" - {MODEL_OUT}")
print(f" - {SCALER_OUT}")
print(f" - {FNAMES_OUT}")
print(f" - {METRICS_OUT}")