import json
import pandas as pd
from scipy.stats import spearmanr

# --- Load your two JSONs ---
with open("/home/umflint.edu/koernerg/new-excelformer/ExcelFormer/output/mi/android_security/full/mi_top25_catenc(1)_norm(quantile).json") as f:
    mi_data = json.load(f)
with open("/home/umflint.edu/koernerg/new-excelformer/ExcelFormer/output/mi/android_security/full/xgboost_feature_importance_20250827_230123.json") as f:
    fi_data = json.load(f)

mi_top = mi_data["selected_names"]
fi_top = fi_data["selected_names"]

# --- Build full feature universe (union of both lists) ---
ALL_FEATURES = sorted(set(mi_top) | set(fi_top))
WORST = len(ALL_FEATURES) + 1

# --- Convert to rank dictionaries ---
mi_rank = {f: (mi_top.index(f)+1) if f in mi_top else WORST for f in ALL_FEATURES}
fi_rank = {f: (fi_top.index(f)+1) if f in fi_top else WORST for f in ALL_FEATURES}

# --- Compute Spearman correlation ---
mi = pd.Series(mi_rank)[ALL_FEATURES]
fi = pd.Series(fi_rank)[ALL_FEATURES]

rho, p = spearmanr(mi, fi)
print(f"Spearman ρ = {rho:.3f}, p = {p:.2g}, N = {len(ALL_FEATURES)} features")
