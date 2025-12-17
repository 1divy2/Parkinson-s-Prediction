from pathlib import Path
import pandas as pd, numpy as np, glob, os
from feature_librosa import feature_dict_from_path

ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT / "data"

HC_FOLDERS = [DATA_DIR / "HC_AH", DATA_DIR / "HC_AH_aug"]
PD_FOLDERS = [DATA_DIR / "PD_AH", DATA_DIR / "PD_AH_aug"]

OUT_CSV = ROOT / "parkinsons_features_LIBROSA.csv"

def list_wavs(folder: Path):
    return sorted([p for p in folder.glob("*.wav") if p.is_file()])

def process_many(folders, label):
    rows = []
    for fol in folders:
        files = list_wavs(fol)
        for p in files:
            try:
                feats = feature_dict_from_path(str(p))
                feats["filename"] = p.name
                feats["status"] = int(label)
                feats["folder"] = fol.name
                rows.append(feats)
            except Exception as e:
                print(f"[Feature Error] {p.name}: {e}")
    return rows

def main():
    print("Scanning data folders…")
    hc_rows = process_many(HC_FOLDERS, 0)
    pd_rows = process_many(PD_FOLDERS, 1)

    all_rows = hc_rows + pd_rows
    if not all_rows:
        print("No rows. Check your /data folders.")
        return

    df = pd.DataFrame(all_rows)
    cols = ["filename", "status", "folder"] + [c for c in df.columns if c not in ("filename","status","folder")]
    df = df[cols]
    df.to_csv(OUT_CSV, index=False)

    print(f"Saved {len(df)} rows -> {OUT_CSV}")
    c = df["status"].value_counts().to_dict()
    print("Counts by status:", c)
    for fol in HC_FOLDERS + PD_FOLDERS:
        print(f"{fol.name}: {len(list_wavs(fol))} files")

if __name__ == "__main__":
    main()