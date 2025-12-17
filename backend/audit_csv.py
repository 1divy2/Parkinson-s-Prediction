import re, sys
from pathlib import Path
import pandas as pd

ROOT = Path(__file__).resolve().parent
CSV = ROOT.parent / "parkinsons_features_LIBROSA.csv"

df = pd.read_csv(CSV)
assert "filename" in df.columns and ("status" in df.columns or "label" in df.columns)
ycol = "status" if "status" in df.columns else "label"
df["status"] = df[ycol].astype(int)

print("Counts by status:", df["status"].value_counts().to_dict())
mis_pd = []
mis_hc = []

for i,row in df.iterrows():
    fn = str(row.get("filename",""))
    st = int(row["status"])
    if re.search(r"PD_", fn, re.I) and st != 1:
        mis_pd.append(fn)
    if re.search(r"HC_", fn, re.I) and st != 0:
        mis_hc.append(fn)

print("Potential PD filename but status!=1:", len(mis_pd))
print("Potential HC filename but status!=0:", len(mis_hc))
if mis_pd[:5]: print("Example PD mismatch:", mis_pd[:5])
if mis_hc[:5]: print("Example HC mismatch:", mis_hc[:5])