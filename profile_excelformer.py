# profile_excelformer.py
# Profiles ExcelFormer inference using the SAME checkpoints, indices, and
# preprocessing as excelformer_plot_pr.py. Writes one JSON per size to
# ./compute_profiles (for the external compute-table script).

import os, json, time, argparse, sys, csv
from pathlib import Path
from datetime import datetime

import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset

# training/eval utilities (same as PR script)
from lib import Transformations, build_dataset, DATA, prepare_tensors
from category_encoders import CatBoostEncoder
from bin import ExcelFormer

# Optional deps for memory stats
try:
    import psutil
except Exception:
    psutil = None

# ---------- Paths (updated to match new repo structure) ----------
MODEL_TYPE_BY_FEATURE = {"MI-25": "mi-25", "FI-25": "fi-25"}

# checkpoints produced by your training runs
def ckpt_path(model_type: str, size: str) -> Path:
    # results/excelformer/{model_type}/mixup(none)/android_security/42/{size}/pytorch_model.pt
    repo_root = Path(__file__).parent.parent.parent
    return repo_root / "results" / "excelformer" / model_type / "mixup(none)" / "android_security" / "42" / size / "pytorch_model.pt"

# feature lists used at train time
MI_JSON_PATHS = {
    "mi-25": "../../data/feature_regimes/mi_top25_catenc(1)_norm(quantile).json",
    "fi-25": "../../data/feature_regimes/xgboost_feature_importance_20250827_230123.json",
}

OUT_DIR_DEFAULT = "../../results/figures/compute_profiles"
CSV_NAME = "compute_profiles_summary.csv"


# ---------- helpers ----------
def _cpu_rss_mb() -> int:
    if psutil is None:
        return 0
    try:
        return int(psutil.Process(os.getpid()).memory_info().rss / (1024**2))
    except Exception:
        return 0


def _append_csv(row, csv_path):
    csv_path = Path(csv_path)
    write_header = not csv_path.exists()
    with open(csv_path, "a", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(row.keys()))
        if write_header: w.writeheader()
        w.writerow(row)


def _load_and_prepare(dataset_name: str, size: str, indices_dir: str, model_type: str):
    """
    Build dataset exactly like training, CatBoost-encode on train only,
    then slice to the selected 25 features from the JSON list.
    Returns dict X_num_processed, labels dict, and final feature count.
    """
    transformation = Transformations(normalization="quantile")
    mi_json_path = MI_JSON_PATHS[model_type]
    if not Path(mi_json_path).exists():
        raise FileNotFoundError(f"Missing feature JSON: {mi_json_path}")

    # Build dataset (same call signature used in PR)
    ds = build_dataset(
        DATA / dataset_name,
        transformation,
        cache=False,
        sample_size=size,
        indices_dir=indices_dir,
        mi_json_path=mi_json_path,  # still pass, then explicitly slice below like PR script
    )

    # CatBoost encode on TRAIN, apply to all splits, then concat with numeric
    if ds.X_cat is not None:
        card = ds.get_category_sizes("train")
        enc = CatBoostEncoder(cols=list(range(len(card))), return_df=False).fit(ds.X_cat["train"], ds.y["train"])
        X_num_processed = {}
        for k in ["train", "val", "test"]:
            enc_cat = enc.transform(ds.X_cat[k]).astype(np.float32)
            X_num_processed[k] = np.concatenate([enc_cat, ds.X_num[k].astype(np.float32)], axis=1)
        cat_names = ds.cat_feature_names or []
        num_names = ds.num_feature_names or []
        all_feature_names = cat_names + num_names
    else:
        X_num_processed = ds.X_num
        all_feature_names = ds.num_feature_names or []

    # Load selected feature names from JSON and slice columns to match training
    with open(mi_json_path, "r") as f:
        mi = json.load(f)
    sel = mi.get("selected_names") or mi.get("selected_features")
    if not sel:
        raise ValueError(f"No selected feature list in {mi_json_path}")

    keep_idx = []
    missing = []
    for name in sel:
        if name in all_feature_names:
            keep_idx.append(all_feature_names.index(name))
        else:
            missing.append(name)
    if missing:
        raise ValueError(f"Selected features not found in dataset columns: {missing[:5]}{'...' if len(missing)>5 else ''}")

    X_num_processed = {k: v[:, keep_idx] for k, v in X_num_processed.items()}

    ys = ds.y  # {'train','val','test'}
    return X_num_processed, ys, len(keep_idx), ds


def _build_model(n_features: int, d_out: int, device: torch.device) -> ExcelFormer:
    model = ExcelFormer(
        d_numerical=n_features,
        d_out=d_out,
        categories=None,
        prenormalization=True,
        token_bias=True,
        n_layers=3,
        n_heads=32,
        d_token=256,
        attention_dropout=0.3,
        ffn_dropout=0.0,
        residual_dropout=0.0,
        kv_compression=None,
        kv_compression_sharing=None,
        init_scale=0.01,
    ).to(device)
    return model


def _measure_inference(model: torch.nn.Module, X_test: np.ndarray, y_test: np.ndarray, device: torch.device, warmup_batches: int = 1):
    # Build loader
    bs = 8192 if X_test.shape[1] <= 100 else 512
    test_loader = DataLoader(TensorDataset(torch.from_numpy(X_test.astype(np.float32)), torch.from_numpy(y_test)), batch_size=bs, shuffle=False)

    model.eval()
    with torch.inference_mode():
        it = iter(test_loader)
        for _ in range(max(0, warmup_batches)):
            try:
                xb, _ = next(it)
            except StopIteration:
                break
            xb = xb[0].to(device, non_blocking=True) if isinstance(xb, (tuple, list)) else xb.to(device, non_blocking=True)
            _ = model(xb, None)

    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats()
        torch.cuda.synchronize()

    start = time.time()
    with torch.inference_mode():
        for xb, _ in test_loader:
            xb = xb.to(device, non_blocking=True)
            _ = model(xb, None)
    if device.type == "cuda":
        torch.cuda.synchronize()

    total_s = time.time() - start
    peak_vram_mb = int(torch.cuda.max_memory_allocated() / (1024**2)) if device.type == "cuda" else 0
    return total_s, peak_vram_mb


# ---------- main ----------
def main():
    ap = argparse.ArgumentParser(description="Profile ExcelFormer inference using standardized indices & training feature sets.")
    ap.add_argument("--feature_set", choices=["MI-25", "FI-25"], required=True,
                    help="Which ExcelFormer family to profile (maps to 'default' or 'xgbfi' folders).")
    ap.add_argument("--dataset", default="android_security", help="Dataset name under lib.DATA (default: android_security)")
    ap.add_argument("--device", choices=["cpu", "cuda"], default="cpu", help="Device for inference timing (default: cpu).")
    ap.add_argument("--indices_dir", default="../../data/splits", help="Directory with *val/test_indices_{size}.npy*.")  # updated to new structure
    ap.add_argument("--sizes", nargs="*", choices=["10000", "100000", "full"], help="Subset of sizes to run (default: all).")
    ap.add_argument("--warmup", type=int, default=1, help="Warmup batches before timing.")
    ap.add_argument("--out_dir", default=OUT_DIR_DEFAULT, help="Where to save JSONs (default: ./compute_profiles).")
    args = ap.parse_args()

    Path(args.out_dir).mkdir(parents=True, exist_ok=True)
    rows = []

    model_type = MODEL_TYPE_BY_FEATURE[args.feature_set]
    sizes = args.sizes or ["10000", "100000", "full"]

    # Device selection (default CPU for apples-to-apples with XGB)
    device = torch.device("cuda" if (args.device == "cuda" and torch.cuda.is_available()) else "cpu")
    if args.device == "cuda" and device.type != "cuda":
        print("[EF-PROFILER] CUDA requested but not available → using CPU")

    for size in sizes:
        ckpt = ckpt_path(model_type, size)
        if not ckpt.exists():
            print(f"⚠️  Missing checkpoint for {args.feature_set} {size}: {ckpt}")
            continue

        print(f"\n[EF-PROFILER] Loading data & features for {args.feature_set} • {size}")
        X_num_processed, ys, n_feats, ds = _load_and_prepare(args.dataset, size, args.indices_dir, model_type)

        # Build model with exact input dim & outputs
        if ds.is_binclass:
            d_out = 2
        elif ds.is_multiclass:
            d_out = ds.n_classes
        else:
            d_out = 1

        model = _build_model(n_feats, d_out, device)

        # Load weights (support old or new-style checkpoints)
        state = torch.load(str(ckpt), map_location=device)
        if isinstance(state, dict) and "model_state_dict" in state:
            model.load_state_dict(state["model_state_dict"])
        else:
            model.load_state_dict(state)
        model.eval()

        # Run timing on TEST split
        X_test = X_num_processed["test"]
        y_test = ys["test"].astype(np.int64)
        total_s, peak_vram_mb = _measure_inference(model, X_test, y_test, device, warmup_batches=args.warmup)

        n_test = len(y_test)
        throughput = n_test / total_s if total_s > 0 else None
        model_bytes = ckpt.stat().st_size if ckpt.exists() else 0
        cpu_rss_mb = _cpu_rss_mb()

        rec = {
            "model": "ExcelFormer",
            "feature_set": args.feature_set,           # MI-25 / FI-25
            "sample_size": size,
            "device": device.type,
            "test_size": int(n_test),
            "test_time_s": round(total_s, 3),
            "throughput_apps_per_s": None if throughput is None else round(throughput, 3),
            "test_auc": None,                          # not computed in profiler
            "peak_vram_mb": int(peak_vram_mb),
            "cpu_max_rss_mb": int(cpu_rss_mb),
            "model_bytes": int(model_bytes),
            "notes": {
                "ensemble_size": 1,
                "comparison_note": "CPU vs CPU for fair XGBoost comparison" if device.type == "cpu" else "GPU run for best performance",
            },
        }

        stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        out_json = Path(args.out_dir) / f"ExcelFormer-infer-{args.feature_set}-{size}_{stamp}.json"
        with open(out_json, "w") as f:
            json.dump(rec, f, indent=2)
        print(f"[EF-PROFILER] Wrote: {out_json}")
        rows.append(rec)

        # tidy
        del model

    # Optional CSV roll-up (same columns as your XGB profiler CSV block)
    if rows:
        csv_path = Path(args.out_dir) / CSV_NAME
        for r in rows:
            csv_row = {
                "run_label": f"EF-{r['feature_set']}-{r['sample_size']}",
                "dataset": args.dataset,
                "feature_set": r["feature_set"],
                "sample_size": r["sample_size"],
                "wall_train_s": None,
                "peak_ram_mb": r["cpu_max_rss_mb"],
                "peak_vram_mb": r["peak_vram_mb"],
                "val_auc_last": None,
                "test_auc_last": r["test_auc"],
                "test_time_s": r["test_time_s"],
                "test_size": r["test_size"],
                "throughput_apps_per_s": r["throughput_apps_per_s"],
                "checkpoint_bytes": r["model_bytes"],
                "checkpoint_path": "",  # EF checkpoints are loaded from result/… per size
                "started_at": None,
                "finished_at": None,
            }
            _append_csv(csv_row, csv_path)
        print(f"[EF-PROFILER] Appended {len(rows)} row(s) to {csv_path}")
    else:
        print("Nothing to do. No records generated.")


if __name__ == "__main__":
    main()
