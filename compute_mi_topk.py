#!/usr/bin/env python3
"""
Compute MI on the *training split* after optional CatBoost encoding,
pick top-K columns, and save everything needed (names + indices + MI scores)
to a JSON for reproducible training across models.

Matches ExcelFormer behavior:
- MI on X_num *after* CatBoost concat (if --catenc)
- training split only
- mutual_info_classif / mutual_info_regression with default params
- no caching

run with:

python compute_mi_topk.py \
  --dataset android_security \
  --indices_dir ./standardized_data \
  --sample_size full \
  --normalization quantile \
  --catenc \
  --k 25 \
  --seed 42 \
  --output_dir output/mi

"""

import argparse, json, os, time, random
from datetime import datetime
from typing import Dict, List

import numpy as np
from sklearn.feature_selection import mutual_info_classif, mutual_info_regression
from category_encoders import CatBoostEncoder

# import your project libs
from lib import Transformations, build_dataset, DATA

def seed_everything(seed: int = 42):
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

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--dataset", required=True, type=str)
    p.add_argument("--indices_dir", default="./standardized_data", type=str)
    p.add_argument("--sample_size", choices=["10000","50000","100000","full"], required=True)
    p.add_argument("--normalization", default="quantile", type=str)
    p.add_argument("--catenc", action="store_true", help="use CatBoostEncoder before MI (recommended for parity)")
    p.add_argument("--k", type=int, default=25, help="top-K features to keep")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--output_dir", default="output/mi", type=str,
                   help="base dir where JSON will be written")
    return p.parse_args()

def main():
    args = parse_args()
    seed_everything(args.seed)

    # Build dataset with the committed indices + normalization
    transformation = Transformations(normalization=(None if args.normalization == "__none__" else args.normalization))
    ds = build_dataset(DATA / args.dataset, transformation, cache=False,
                       sample_size=args.sample_size, indices_dir=args.indices_dir)

    # Prepare numeric matrix to run MI on (AFTER CatBoost if requested)
    if ds.X_num["train"].dtype == np.float64:
        ds.X_num = {k: v.astype(np.float32) for k, v in ds.X_num.items()}

    if args.catenc and ds.X_cat is not None:
        card = ds.get_category_sizes("train")
        enc = CatBoostEncoder(cols=list(range(len(card))), return_df=False).fit(ds.X_cat["train"], ds.y["train"])
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

    # EXACT call (no kwargs)
    print(f"[MI] computing on train: X={X_train.shape}, y={y_train.shape}, catenc={args.catenc}")
    t0 = time.time()
    mi_scores = mi_func(X_train, y_train)
    dt = time.time() - t0
    print(f"[MI] done in {dt:.2f}s. Features={len(mi_scores)}")

    # Rank + pick top-K
    order = np.argsort(-mi_scores)
    keep_idx = order[:args.k].tolist()
    selected_names = [all_feature_names[i] for i in keep_idx]

    # Prepare JSON payload
    # also include a per-name MI dict for easy downstream use
    mi_by_name: Dict[str, float] = {all_feature_names[i]: float(mi_scores[i]) for i in range(len(all_feature_names))}

    payload = {
        "timestamp": datetime.utcnow().isoformat() + "Z",
        "dataset": args.dataset,
        "sample_size": args.sample_size,
        "indices_dir": args.indices_dir,
        "normalization": args.normalization,
        "catenc": bool(args.catenc),
        "seed": args.seed,
        "k": args.k,
        "is_regression": is_reg,
        "feature_order": all_feature_names,        # full order used at MI time
        "mi_scores_aligned": [float(x) for x in mi_scores],  # aligned to feature_order
        "mi_scores_by_name": mi_by_name,           # convenience
        "selected_indices": keep_idx,              # indices in feature_order
        "selected_names": selected_names,          # same features by name
        "shapes": {k: list(v.shape) for k, v in X_num_proc.items()},
    }

    # Write JSON to a stable path
    out_dir = os.path.join(args.output_dir, args.dataset, args.sample_size)
    os.makedirs(out_dir, exist_ok=True)
    json_name = f"mi_top{args.k}_catenc({int(args.catenc)})_norm({args.normalization}).json"
    out_path = os.path.join(out_dir, json_name)
    with open(out_path, "w") as f:
        json.dump(payload, f, indent=2)
    print(f"[MI] wrote {out_path}")

    # Also emit a plain-text list for quick eyeballing
    txt_path = out_path.replace(".json", "_names.txt")
    with open(txt_path, "w") as f:
        for i, n in enumerate(selected_names, 1):
            f.write(f"{i:2d}. {n}\n")
    print(f"[MI] wrote {txt_path}")

if __name__ == "__main__":
    main()
