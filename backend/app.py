# backend/app.py
from pathlib import Path
import tempfile, os, base64
import numpy as np
import librosa
import joblib
import json
from flask import Flask, request, jsonify
from flask_cors import CORS

from feature_librosa import extract_vector, safe_load

ROOT = Path(__file__).resolve().parent
MODELS_DIR = ROOT / "models"
MODEL_PATH = MODELS_DIR / "model.joblib"
SCALER_PATH = MODELS_DIR / "scaler.joblib"
FNAMES_PATH = MODELS_DIR / "feature_names.joblib"
METRICS_PATH = MODELS_DIR / "metrics.json"

def load_or_fail(p: Path):
    if not p.exists():
        raise FileNotFoundError(f"Missing file: {p}")
    return joblib.load(p)

model = load_or_fail(MODEL_PATH)
scaler = load_or_fail(SCALER_PATH)
FEATURE_NAMES = load_or_fail(FNAMES_PATH)
metrics = {}
if METRICS_PATH.exists():
    with open(METRICS_PATH, "r") as f:
        metrics = json.load(f)

app = Flask(__name__)
CORS(app, resources={r"/*": {"origins": ["http://127.0.0.1:5173","http://localhost:5173","http://127.0.0.1:8000","http://localhost:8000","*"]}})

def save_temp_bytes(b):
    f = tempfile.NamedTemporaryFile(suffix=".wav", delete=False)
    f.write(b); f.flush(); f.close()
    return f.name

@app.route("/meta")
def meta():
    out = {
        "model_label": metrics.get("model_label", "RF"),
        "chosen_model": metrics.get("chosen_model", "rf"),
        "accuracy": metrics.get("metrics", {}).get("accuracy"),
        "n_features": metrics.get("n_features", len(FEATURE_NAMES)),
        "class_distribution": metrics.get("class_distribution", {}),
        "threshold": metrics.get("threshold_info", {}).get("selected_threshold", 0.5),
    }
    return jsonify(out)

@app.route("/health")
def health():
    return jsonify(ok=True, model_loaded=True, n_features=len(FEATURE_NAMES))

@app.route("/predict", methods=["POST"])
def predict():
    path = None
    try:
        if request.files:
            fobj = next(iter(request.files.values()))
            if not fobj or fobj.filename == "":
                return jsonify(error="no file"), 400
            with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
                fobj.save(tmp.name)
                path = tmp.name
        elif request.data and (request.content_type or "").startswith(("audio/", "application/octet-stream")):
            path = save_temp_bytes(request.data)
        else:
            j = request.get_json(silent=True) or {}
            if "data" in j:
                b64 = j["data"].split(",")[-1]
                path = save_temp_bytes(base64.b64decode(b64))

        if not path:
            return jsonify(error="no file"), 400

        x = extract_vector(path, want_names=FEATURE_NAMES).reshape(1, -1)
        x = scaler.transform(x)
        pred = int(model.predict(x)[0])
        prob = None
        if hasattr(model, "predict_proba"):
            prob = float(model.predict_proba(x)[0, 1])

        used_threshold = metrics.get("threshold_info", {}).get("selected_threshold", 0.5)
        verdict = "Parkinsons" if (prob is not None and prob >= used_threshold) or (prob is None and pred == 1) else "Healthy"

        return jsonify(prediction=pred, probability=prob, verdict=verdict, n_features=len(FEATURE_NAMES), used_threshold=used_threshold)
    except Exception as e:
        return jsonify(error=str(e)), 500
    finally:
        try:
            if path and os.path.exists(path):
                os.remove(path)
        except:
            pass

@app.route("/features", methods=["POST"])
def features():
    path = None
    try:
        if request.files:
            fobj = next(iter(request.files.values()))
            if not fobj or fobj.filename == "":
                return jsonify(error="no file"), 400
            with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
                fobj.save(tmp.name)
                path = tmp.name
        else:
            return jsonify(error="no file"), 400

        y, sr = safe_load(path, sr=22050)

        n_fft = 2048
        hop = 256
        fmin = 75
        fmax = 600

        f0 = librosa.pyin(y, fmin=fmin, fmax=fmax, frame_length=n_fft, hop_length=hop, sr=sr)[0]
        f0 = np.nan_to_num(np.asarray(f0, dtype=float), nan=0.0).tolist()

        rms = librosa.feature.rms(y=y, frame_length=n_fft, hop_length=hop).flatten().tolist()
        zcr = librosa.feature.zero_crossing_rate(y=y, frame_length=n_fft, hop_length=hop).flatten().tolist()
        mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13, n_fft=n_fft, hop_length=hop).mean(axis=1).tolist()

        feats_vec = extract_vector(path, want_names=FEATURE_NAMES).tolist()
        feature_names = list(map(str, FEATURE_NAMES))

        return jsonify(f0=f0, rms=rms, zcr=zcr, mfcc_means=mfcc, features_vector=feats_vec, feature_names=feature_names)
    except Exception as e:
        return jsonify(error=str(e)), 500
    finally:
        try:
            if path and os.path.exists(path):
                os.remove(path)
        except:
            pass

if __name__ == "__main__":
    app.run(debug=True)