from pathlib import Path
import joblib
import numpy as np
from feature_librosa import extract_vector
import json

ROOT = Path(__file__).resolve().parent
MODELS = ROOT / "models"
model = joblib.load(MODELS / "model.joblib")
scaler = joblib.load(MODELS / "scaler.joblib")
feature_names = joblib.load(MODELS / "feature_names.joblib")
metrics = json.loads((MODELS / "metrics.json").read_text())
thr = float(metrics.get("threshold_info", {}).get("selected_threshold", 0.5))

def prob_for(path):
    x = extract_vector(str(path), want_names=feature_names).reshape(1, -1)
    x = scaler.transform(x)
    p = model.predict_proba(x)[0,1] if hasattr(model, "predict_proba") else None
    y = model.predict(x)[0]
    return y, p

# change these two to actual files you know
HC = Path("../data/HC_AH").glob("*.wav")
PD = Path("../data/PD_AH").glob("*.wav")

hc_file = next(HC, None)
pd_file = next(PD, None)

for tag, f in [("HC", hc_file), ("PD", pd_file)]:
    if f is None:
        print(f"{tag}: no file found")
        continue
    y, p = prob_for(f)
    print(f"{tag}: {f.name}")
    print(f"  label_by_model = {int(y)}")
    print(f"  prob(PD)       = {None if p is None else round(float(p),3)}")
    if p is not None:
        print(f"  verdict@thr    = {'PD' if p>=thr else 'Healthy'} (thr={thr:.2f})")