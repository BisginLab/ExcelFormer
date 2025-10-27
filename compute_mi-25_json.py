#!/usr/bin/env python3
"""
Compute MI-25 JSON with fixed configuration.

This script computes Mutual Information on the training split with:
- sample_size: full
- normalization: quantile
- catenc: True (CatBoost encoding)
- k: 25 (top 25 features)
- seed: 42

Output: data/feature_regimes/mi_top25_catenc(1)_norm(quantile).json

Usage:
    python compute_mi-25_json.py
"""

import json
import os
import time
import random
from datetime import datetime
from typing import Dict, List
from pathlib import Path

import numpy as np
from sklearn.feature_selection import mutual_info_classif, mutual_info_regression
from category_encoders import CatBoostEncoder

# Import ExcelFormer libs
from lib import Transformations, build_dataset, DATA

def seed_everything(seed: int = 42):
    """Set random seed for reproducibility"""
    import torch
    random.seed(seed)
    np.random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

def main():
    # Fixed configuration
    dataset = "android_security"
    indices_dir = "/workspace/data/splits"
    sample_size = "full"
    normalization = "quantile"
    catenc = True
    k = 25
    seed = 42
    output_dir = "/workspace/data/feature_regimes"
    
    print("="*60)
    print("COMPUTING MI-25 JSON WITH FIXED CONFIGURATION")
    print("="*60)
    print(f"Dataset: {dataset}")
    print(f"Sample size: {sample_size}")
    print(f"Normalization: {normalization}")
    print(f"CatBoost encoding: {catenc}")
    print(f"Top-K features: {k}")
    print(f"Random seed: {seed}")
    print(f"Output directory: {output_dir}")
    print("="*60 + "\n")
    
    seed_everything(seed)

    # Build dataset with the committed indices + normalization
    transformation = Transformations(normalization=normalization)
    
    # Use the correct data path
    data_path = Path(__file__).parent.parent.parent / 'data' / 'raw'
    
    print(f"Loading data from: {data_path}")
    print(f"Using indices from: {indices_dir}")
    
    ds = build_dataset(
        str(data_path),
        transformation,
        cache=False,
        sample_size=sample_size,
        indices_dir=indices_dir
    )

    # Prepare numeric matrix to run MI on (AFTER CatBoost if requested)
    if ds.X_num["train"].dtype == np.float64:
        ds.X_num = {k: v.astype(np.float32) for k, v in ds.X_num.items()}

    if catenc and ds.X_cat is not None:
        print("\nApplying CatBoost encoding to categorical features...")
        card = ds.get_category_sizes("train")
        enc = CatBoostEncoder(
            cols=list(range(len(card))),
            return_df=False
        ).fit(ds.X_cat["train"], ds.y["train"])
        
        X_num_proc = {}
        for split in ["train", "val", "test"]:
            enc_cat = enc.transform(ds.X_cat[split]).astype(np.float32)
            X_num_proc[split] = np.concatenate([enc_cat, ds.X_num[split]], axis=1)
        
        # names are cat then num
        all_feature_names: List[str] = (ds.cat_feature_names or []) + (ds.num_feature_names or [])
    else:
        X_num_proc = ds.X_num
        all_feature_names = ds.num_feature_names or []

    X_train = X_num_proc["train"]
    y_train = ds.y["train"]
    is_reg = bool(ds.is_regression)
    mi_func = mutual_info_regression if is_reg else mutual_info_classif

    # Compute MI
    print(f"\n[MI] Computing on train split:")
    print(f"  X shape: {X_train.shape}")
    print(f"  y shape: {y_train.shape}")
    print(f"  CatBoost encoding: {catenc}")
    
    t0 = time.time()
    mi_scores = mi_func(X_train, y_train)
    dt = time.time() - t0
    
    print(f"[MI] Completed in {dt:.2f}s")
    print(f"[MI] Total features: {len(mi_scores)}")

    # Rank + pick top-K
    order = np.argsort(-mi_scores)
    keep_idx = order[:k].tolist()
    selected_names = [all_feature_names[i] for i in keep_idx]

    print(f"\nTop {k} features selected:")
    for i, (idx, name) in enumerate(zip(keep_idx[:10], selected_names[:10]), 1):
        print(f"  {i:2d}. {name} (MI: {mi_scores[idx]:.4f})")
    if k > 10:
        print(f"  ... ({k-10} more)")

    # Prepare JSON payload
    mi_by_name: Dict[str, float] = {
        all_feature_names[i]: float(mi_scores[i])
        for i in range(len(all_feature_names))
    }

    payload = {
        "timestamp": datetime.utcnow().isoformat() + "Z",
        "dataset": dataset,
        "sample_size": sample_size,
        "indices_dir": indices_dir,
        "normalization": normalization,
        "catenc": bool(catenc),
        "seed": seed,
        "k": k,
        "is_regression": is_reg,
        "feature_order": all_feature_names,        # full order used at MI time
        "mi_scores_aligned": [float(x) for x in mi_scores],  # aligned to feature_order
        "mi_scores_by_name": mi_by_name,           # convenience
        "selected_indices": keep_idx,              # indices in feature_order
        "selected_names": selected_names,          # same features by name
        "shapes": {k: list(v.shape) for k, v in X_num_proc.items()},
    }

    # Write JSON to output directory
    os.makedirs(output_dir, exist_ok=True)
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    json_name = f"mi-25_features_{timestamp}.json"
    out_path = os.path.join(output_dir, json_name)
    
    with open(out_path, "w") as f:
        json.dump(payload, f, indent=2)
    
    print(f"\n✅ Saved MI-25 JSON to: {out_path}")

    # Also emit a plain-text list for quick eyeballing
    txt_path = out_path.replace(".json", "_names.txt")
    with open(txt_path, "w") as f:
        f.write(f"Top {k} Features by Mutual Information\n")
        f.write("="*60 + "\n\n")
        for i, n in enumerate(selected_names, 1):
            mi_score = mi_by_name[n]
            f.write(f"{i:2d}. {n} (MI: {mi_score:.4f})\n")
    
    print(f"✅ Saved feature list to: {txt_path}")
    
    print("\n" + "="*60)
    print("MI-25 JSON GENERATION COMPLETE")
    print("="*60)

if __name__ == "__main__":
    main()

