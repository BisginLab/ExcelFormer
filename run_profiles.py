#!/usr/bin/env python3
# No-args runner for ExcelFormer → dumps JSONs into roc_dumps/compute_profiles

import sys, subprocess
from pathlib import Path

# ---- fixed paths ----
EF_ROOT   = Path(__file__).parent
THIRD_OUT = Path("/home/umflint.edu/koernerg/roc_dumps/compute_profiles")
INDICES   = "/home/umflint.edu/koernerg/xgboost/standardized_data"

FEATURE_SETS = ["MI-25", "FI-25"]
SIZES        = ["10000", "100000", "full"]

try:
    import torch
    DEVICES = ["cpu"] + (["cuda"] if torch.cuda.is_available() else [])
except Exception:
    DEVICES = ["cpu"]

THIRD_OUT.mkdir(parents=True, exist_ok=True)
script = EF_ROOT / "profile_excelformer.py"

def run(cmd):
    print("\n$ " + " ".join(cmd) + f"   (cwd={EF_ROOT})")
    return subprocess.call(cmd, cwd=str(EF_ROOT))

for fs in FEATURE_SETS:
    for dev in DEVICES:
        for size in SIZES:
            cmd = [
                sys.executable, "-u", str(script),
                "--feature_set", fs,
                "--device", dev,
                "--indices_dir", INDICES,
                "--out_dir", str(THIRD_OUT),
                "--sizes", size,          # safe even if accepted as list
            ]
            rc = run(cmd)
            if rc != 0:
                print(f"⚠️ EF {fs} {dev} {size} failed (rc={rc}); continuing.")

print("\n[EF] done → JSONs in", THIRD_OUT)
