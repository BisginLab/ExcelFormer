# profile_excelformer_infer.py
# 
# Device Strategy:
# - Default: CPU (--device cpu) for fair comparison with XGBoost
# - Optional: GPU (--device cuda) for best available performance
# - CPU vs CPU ensures apples-to-apples comparison
# - GPU results can be shown in separate "best available device" table
#
import os, json, time, argparse
from pathlib import Path
import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset

from bin import ExcelFormer
from lib import Transformations, build_dataset, prepare_tensors, DATA
from category_encoders import CatBoostEncoder

# Required deps for CPU memory monitoring
try:
    import psutil
except ImportError:
    print("WARNING: psutil not installed. CPU memory monitoring will be disabled.")
    print("Install with: pip install psutil")
    psutil = None

def load_data(dataset, sample_size, indices_dir, normalization, mi_json, use_catenc=True):
    T_cache = False
    transformation = Transformations(normalization=normalization)
    ds = build_dataset(DATA / dataset, transformation, T_cache,
                       sample_size=sample_size, indices_dir=indices_dir, mi_json_path=mi_json)

    # CatBoost encode (fit on train only) then concat with numerics — identical to training
    if use_catenc and ds.X_cat is not None:
        card = ds.get_category_sizes('train')
        enc = CatBoostEncoder(cols=list(range(len(card))), return_df=False).fit(ds.X_cat['train'], ds.y['train'])
        X_num_processed = {k: np.concatenate([enc.transform(ds.X_cat[k]).astype(np.float32),
                                              ds.X_num[k].astype(np.float32)], axis=1) for k in ['train','val','test']}
        cat_names = ds.cat_feature_names or []
        num_names = ds.num_feature_names or []
        all_feature_names = cat_names + num_names
    else:
        X_num_processed = ds.X_num
        all_feature_names = ds.num_feature_names or []

    # Build tensors (X_num_processed already sliced by mi_json inside build_dataset)
    X_num, X_cat, ys = prepare_tensors(
        type('D', (), {
            'X_num': X_num_processed, 'X_cat': None if use_catenc else ds.X_cat, 'y': ds.y,
            'n_classes': ds.n_classes, 'is_binclass': ds.is_binclass,
            'is_multiclass': ds.is_multiclass, 'is_regression': ds.is_regression,
            'calculate_metrics': ds.calculate_metrics, 'n_features': X_num_processed['train'].shape[1],
            'num_feature_names': ds.num_feature_names, 'cat_feature_names': ds.cat_feature_names,
            'get_category_sizes': ds.get_category_sizes,
        })(), device=torch.device('cpu')  # tensors moved to device later
    )
    return ds, X_num, ys, all_feature_names

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", required=True)
    ap.add_argument("--ckpt", required=True, help="Path to saved ExcelFormer checkpoint (.pt)")
    ap.add_argument("--mi_json", required=True)
    ap.add_argument("--sample_size", choices=["10000","100000","full"], default="full")
    ap.add_argument("--indices_dir", default="./standardized_data")
    ap.add_argument("--normalization", default="quantile")
    ap.add_argument("--device", choices=["cpu","cuda"], default="cpu", help="Device to run inference on. Use 'cpu' for fair comparison with XGBoost")
    ap.add_argument("--out_json", default=None)

    ap.add_argument("--warmup", type=int, default=1, help="Warmup batches before timing")
    args = ap.parse_args()

    device = torch.device("cuda" if args.device=="cuda" and torch.cuda.is_available() else "cpu")
    
    # Ensure CPU for fair comparison with XGBoost (which runs on CPU)
    if args.device == "cpu":
        device = torch.device("cpu")
        print("[EF-PROFILER] Running on CPU for fair comparison with XGBoost")
    elif args.device == "cuda" and not torch.cuda.is_available():
        print("[EF-PROFILER] CUDA requested but not available, falling back to CPU")
        device = torch.device("cpu")

    ds, X_num, ys, feat_names = load_data(
        dataset=args.dataset,
        sample_size=args.sample_size,
        indices_dir=args.indices_dir,
        normalization=args.normalization,
        mi_json=args.mi_json,
        use_catenc=True
    )

    # Dataloaders
    bs = 8192 if X_num['test'].shape[1] <= 100 else 512
    pin_memory = (device.type == "cuda")
    num_workers = 0
    persistent_workers = False
    test_loader = DataLoader(
        TensorDataset(X_num['test'], ys['test']), 
        batch_size=bs, 
        shuffle=False, 
        pin_memory=pin_memory,
        num_workers=num_workers, 
        persistent_workers=persistent_workers
    )

    # Rebuild model with correct dimensionality (n_features) & outputs
    n_num = X_num['test'].shape[1]
    if ds.is_binclass: d_out = 2
    elif ds.is_multiclass: d_out = ds.n_classes
    else: d_out = 1

    model = ExcelFormer(
        d_numerical=n_num, d_out=d_out, categories=None,  # catenc -> all numeric now
        prenormalization=True, token_bias=True,
        ffn_dropout=0., attention_dropout=0.3, residual_dropout=0.0,
        n_layers=3, n_heads=32, d_token=256, init_scale=0.01,
        kv_compression=None, kv_compression_sharing=None
    ).to(device)

    # Load checkpoint (expects 'model_state_dict')
    print("[EF-PROFILER] Loading checkpoint...")
    ckpt = torch.load(args.ckpt, map_location=device)
    print("[EF-PROFILER] Checkpoint loaded, size:", len(ckpt) if isinstance(ckpt, dict) else "single tensor")
    
    print("[EF-PROFILER] Loading state dict...")
    if isinstance(ckpt, dict) and 'model_state_dict' in ckpt:
        model.load_state_dict(ckpt['model_state_dict'])
        print("[EF-PROFILER] State dict loaded from 'model_state_dict' key")
    else:
        model.load_state_dict(ckpt)
        print("[EF-PROFILER] State dict loaded directly")
    
    print("[EF-PROFILER] Setting model to eval mode...")
    model.eval()
    print("[EF-PROFILER] Model ready for inference")

    # Warmup (not timed)
    with torch.inference_mode():
        it = iter(test_loader)
        for _ in range(max(0, args.warmup)):
            try:
                xb, _ = next(it)
            except StopIteration:
                break
            xb = xb.to(device, non_blocking=True)
            _ = model(xb, None)
    if device.type == "cuda":
        torch.cuda.synchronize()

    # Inference timing
    if device.type=="cuda":
        torch.cuda.reset_peak_memory_stats()
        torch.cuda.synchronize()
    start = time.time()
    preds = []
    with torch.inference_mode():
        for xb, yb in test_loader:
            xb = xb.to(device, non_blocking=True)
            logits = model(xb, None)
            if ds.is_binclass:
                p = torch.softmax(logits, dim=1)[:,1]
            elif ds.is_multiclass:
                p = torch.softmax(logits, dim=1).max(dim=1).values  # not used for AUC; dataset.calculate_metrics will.
            else:
                p = logits.squeeze()
            preds.append(p.detach().cpu())
    if device.type=="cuda":
        torch.cuda.synchronize()
    total_s = time.time() - start
    preds = torch.cat(preds).numpy()

    # Metrics - AUC computation removed for profiling
    test_auc = None

    n_test = len(ys['test'])
    throughput = n_test / total_s if total_s > 0 else None
    peak_vram_mb = int(torch.cuda.max_memory_allocated()/ (1024**2)) if device.type=="cuda" else 0
    ckpt_bytes = Path(args.ckpt).stat().st_size if Path(args.ckpt).exists() else 0

    # Get CPU memory usage
    cpu_max_rss_mb = 0
    if psutil is not None:
        try:
            proc = psutil.Process(os.getpid())
            cpu_max_rss_mb = int(proc.memory_info().rss / (1024**2))
        except Exception as e:
            print(f"[EF-PROFILER] Warning: Failed to get CPU memory usage: {e}")
            cpu_max_rss_mb = 0
    else:
        print("[EF-PROFILER] Warning: psutil not available. CPU memory will show as 0.")
        print("Install psutil with: pip install psutil")

    rec = {
        "model": "ExcelFormer",
        "sample_size": args.sample_size,
        "device": str(device),
        "test_size": n_test,
        "test_time_s": round(total_s, 3),
        "throughput_apps_per_s": None if throughput is None else round(throughput, 3),
        "test_auc": test_auc,                        # renamed from test_auc_last
        "peak_vram_mb": peak_vram_mb,
        "cpu_max_rss_mb": cpu_max_rss_mb,           # added CPU memory monitoring
        "model_bytes": int(ckpt_bytes),             # renamed from checkpoint_bytes
        "notes": {
            "ensemble_size": 1,                      # added ensemble info
            "comparison_note": "CPU vs CPU for fair XGBoost comparison" if device.type == "cpu" else "GPU run for best performance"
        }
    }

    out_path = args.out_json or f"compute_profiles/ExcelFormer-infer-{args.sample_size}_{int(time.time())}.json"
    Path(Path(out_path).parent).mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f: json.dump(rec, f, indent=2)
    print("[EF-PROFILER] Wrote:", out_path)
    print(rec)
    
    # Print comparison guidance
    if device.type == "cpu":
        print("\n[EF-PROFILER] CPU run completed - use this for fair comparison with XGBoost")
        print("For best performance comparison, run with --device cuda")
    else:
        print("\n[EF-PROFILER] GPU run completed - this shows best available performance")
        print("For fair XGBoost comparison, run with --device cpu")

if __name__ == "__main__":
    main()
