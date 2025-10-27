import json
import pandas as pd
from pathlib import Path
from scipy.stats import spearmanr

# --- Load your two JSONs ---
repo_root = Path(__file__).parent.parent.parent
mi_json = repo_root / "data/feature_regimes/mi_top25_catenc(1)_norm(quantile).json"
fi_json = repo_root / "data/feature_regimes/xgboost_feature_importance_20250827_230123.json"

with open(mi_json) as f:
    mi_data = json.load(f)
with open(fi_json) as f:
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
