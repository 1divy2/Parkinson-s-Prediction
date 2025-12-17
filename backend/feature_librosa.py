import os, numpy as np, librosa, soundfile as sf
from scipy.stats import entropy

def _safe_load(path, sr=22050):
    try:
        y, sr = librosa.load(path, sr=sr, mono=True)
    except Exception:
        d, s = sf.read(path, always_2d=False)
        if hasattr(d, "ndim") and d.ndim > 1:
            d = np.mean(d, axis=1)
        y, sr = d.astype(np.float32), int(s)
    if y.size == 0:
        y = np.zeros(2048, dtype=np.float32)
    m = np.max(np.abs(y))
    if m > 0:
        y = y / m
    return y.astype(np.float32), sr

def _shannon_entropy(x):
    spec = np.abs(np.fft.rfft(x))**2
    spec = spec / (np.sum(spec) + 1e-12)
    spec = np.clip(spec, 1e-12, 1.0)
    return float(-(spec * np.log2(spec)).sum())

def _tkeo_mean(x):
    x = x.astype(np.float64)
    if x.size < 3: return 0.0
    t = x[1:-1]**2 - x[:-2]*x[2:]
    return float(np.mean(t))

def _hnr_db(y, sr):
    try:
        y_h = librosa.effects.harmonic(y=y)
        y_r = y - y_h
        Eh = np.sum(y_h**2) + 1e-12
        Er = np.sum(y_r**2) + 1e-12
        return float(10.0 * np.log10(Eh / Er))
    except Exception:
        return float("nan")

def _f0_block(y, sr, n_fft=1024, hop=256, fmin=75, fmax=600):
    try:
        f0, vf, vp = librosa.pyin(y, fmin=fmin, fmax=fmax, frame_length=n_fft, hop_length=hop, sr=sr)
    except Exception:
        f0 = None; vf = None; vp = None
    if f0 is None:
        try:
            f0 = librosa.yin(y, fmin=fmin, fmax=fmax, sr=sr, frame_length=n_fft, hop_length=hop)
            f0 = np.asarray(f0, dtype=float)
            en = librosa.feature.rms(y=y, frame_length=n_fft, hop_length=hop).flatten()
            thr = float(np.nanmean(en)) * 0.5
            vf = (en > thr)
            vp = (en - np.nanmin(en)) / (np.nanmax(en) - np.nanmin(en) + 1e-12)
        except Exception:
            f0 = np.array([np.nan]); vf = np.array([False]); vp = np.array([0.0])
    return np.asarray(f0, dtype=float), np.asarray(vf).astype(bool), np.asarray(vp, dtype=float)

def _safe_mean(v):
    v = np.asarray(v, dtype=float)
    v = v[np.isfinite(v)]
    return float(np.mean(v)) if v.size else float("nan")

def _safe_std(v):
    v = np.asarray(v, dtype=float)
    v = v[np.isfinite(v)]
    return float(np.std(v)) if v.size else float("nan")

def feature_dict_from_path(path, sr_target=22050):
    y, sr = _safe_load(path, sr=sr_target)
    n_fft = 1024
    hop = 256

    f0, vf, vp = _f0_block(y, sr, n_fft=n_fft, hop=hop)
    f0_finite = f0[np.isfinite(f0)]
    f0_mean = _safe_mean(f0_finite)
    f0_std  = _safe_std(f0_finite)
    voiced_fraction = float(np.mean(vf)) if vf.size else 0.0

    rms = librosa.feature.rms(y=y, frame_length=n_fft, hop_length=hop).flatten()
    zcr = librosa.feature.zero_crossing_rate(y=y, frame_length=n_fft, hop_length=hop).flatten()
    sc  = librosa.feature.spectral_centroid(y=y, sr=sr, n_fft=n_fft, hop_length=hop).flatten()
    sb  = librosa.feature.spectral_bandwidth(y=y, sr=sr, n_fft=n_fft, hop_length=hop).flatten()
    ro  = librosa.feature.spectral_rolloff(y=y, sr=sr, n_fft=n_fft, hop_length=hop, roll_percent=0.85).flatten()
    mf  = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13, n_fft=n_fft, hop_length=hop)

    feats = {
        "f0_mean": f0_mean,
        "f0_std": f0_std,
        "voiced_fraction": voiced_fraction,
        "rms_mean": _safe_mean(rms),
        "rms_std":  _safe_std(rms),
        "zcr_mean": _safe_mean(zcr),
        "zcr_std":  _safe_std(zcr),
        "centroid_mean": _safe_mean(sc),
        "centroid_std":  _safe_std(sc),
        "bandwidth_mean": _safe_mean(sb),
        "bandwidth_std":  _safe_std(sb),
        "rolloff_mean": _safe_mean(ro),
        "rolloff_std":  _safe_std(ro),
        "hnr_db": _hnr_db(y, sr),
        "shannon_entropy": _shannon_entropy(y),
        "tkeo_mean": _tkeo_mean(y),
    }
    for i in range(mf.shape[0]):
        feats[f"mfcc{i+1}_mean"] = _safe_mean(mf[i])
        feats[f"mfcc{i+1}_std"]  = _safe_std(mf[i])
    return feats

def extract_vector(path, want_names=None):
    feats = feature_dict_from_path(path)
    if want_names is None:
        return feats
    return np.array([feats.get(k, 0.0) for k in want_names], dtype=float)

def safe_load(path, sr=22050):
    return _safe_load(path, sr)