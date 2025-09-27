# scripts/make_compute_table_avg.py
import json, glob
import pandas as pd
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np

IN_DIR = Path("../compute_profiles")
OUT_CSV = IN_DIR / "compute_profiles_avg.csv"
OUT_LATEX = IN_DIR / "compute_profiles_avg.tex"

def load_records():
    rows = []
    for p in glob.glob(str(IN_DIR / "*.json")):
        try:
            with open(p, "r") as f:
                rec = json.load(f)
        except Exception:
            continue
        # require model + sample_size + device
        if not all(k in rec for k in ("model","sample_size","device")):
            continue
        # prefer tagged feature_set; allow missing (older runs)
        rec["feature_set"] = rec.get("feature_set")
        rows.append(rec)
    return pd.DataFrame(rows)

def create_averaged_compute_plot(df):
    """Create and save an averaged compute table visualization as PNG"""
    if df.empty:
        print("No data to visualize")
        return
    
    # Create figure with subplots
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(16, 12))
    fig.suptitle('Averaged Compute Profiles (MI-25 & FI-25)', fontsize=16, fontweight='bold')
    
    # Performance comparison by model and sample size
    models = df['model'].unique()
    sample_sizes = df['sample_size'].unique()
    
    # Test time comparison
    x = np.arange(len(sample_sizes))
    width = 0.35
    
    for i, model in enumerate(models):
        model_data = df[df['model'] == model]
        times = [model_data[model_data['sample_size'] == size]['test_time_s'].iloc[0] if len(model_data[model_data['sample_size'] == size]) > 0 else 0 for size in sample_sizes]
        ax1.bar(x + i*width, times, width, label=model, alpha=0.8)
    
    ax1.set_xlabel('Sample Size')
    ax1.set_ylabel('Test Time (s)')
    ax1.set_title('Test Time by Model and Sample Size')
    ax1.set_xticks(x + width/2)
    ax1.set_xticklabels(sample_sizes)
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # Throughput comparison
    for i, model in enumerate(models):
        model_data = df[df['model'] == model]
        throughput = [model_data[model_data['sample_size'] == size]['throughput_apps_per_s'].iloc[0] if len(model_data[model_data['sample_size'] == size]) > 0 else 0 for size in sample_sizes]
        ax2.bar(x + i*width, throughput, width, label=model, alpha=0.8)
    
    ax2.set_xlabel('Sample Size')
    ax2.set_ylabel('Throughput (apps/s)')
    ax2.set_title('Throughput by Model and Sample Size')
    ax2.set_xticks(x + width/2)
    ax2.set_xticklabels(sample_sizes)
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    # Memory usage comparison
    memory_metrics = ['cpu_max_rss_mb', 'peak_vram_mb']
    memory_labels = ['CPU RAM (MB)', 'GPU VRAM (MB)']
    
    for i, metric in enumerate(memory_metrics):
        if metric in df.columns:
            values = df[metric].values
            ax3.bar(f'{metric}_{i}', values, alpha=0.8)
    
    ax3.set_ylabel('Memory (MB)')
    ax3.set_title('Memory Usage Comparison')
    ax3.grid(True, alpha=0.3)
    
    # Model size comparison
    if 'model_bytes' in df.columns:
        model_sizes_mb = df['model_bytes'] / (1024 * 1024)
        ax4.bar(range(len(model_sizes_mb)), model_sizes_mb, alpha=0.8)
        ax4.set_xlabel('Model Index')
        ax4.set_ylabel('Model Size (MB)')
        ax4.set_title('Model Size Comparison')
        ax4.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    # Save PNG
    png_path = IN_DIR / "compute_profiles_avg.png"
    plt.savefig(png_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Generated averaged compute profile plot: {png_path}")

def main():
    df = load_records()
    if df.empty:
        print("No compute profile JSONs found in ./compute_profiles")
        return

    # Keep relevant columns if present
    keep = [
        "model","feature_set","sample_size","device",
        "test_time_s","throughput_apps_per_s",
        "cpu_max_rss_mb","peak_vram_mb","model_bytes"
    ]
    df = df[[c for c in keep if c in df.columns]].copy()

    # Aggregate MI-25 & FI-25 → 1 row per {model, sample_size, device}
    # Averaging policy:
    # - time/throughput: mean
    # - memory peaks (cpu_max_rss_mb, peak_vram_mb): max (conservative)
    # - model_bytes: mean
    grouped = df.groupby(["model","sample_size","device"], as_index=False).agg({
        "test_time_s": "mean",
        "throughput_apps_per_s": "mean",
        "cpu_max_rss_mb": "max",
        "peak_vram_mb": "max",
        "model_bytes": "mean",
    })

    # Also compute spread on time/throughput across MI/FI (optional)
    spread = (
        df.groupby(["model","sample_size","device"])
          .agg(test_time_s_std=("test_time_s","std"),
               throughput_std=("throughput_apps_per_s","std"))
          .reset_index()
    )
    out = grouped.merge(spread, on=["model","sample_size","device"], how="left")

    out.sort_values(["model","sample_size","device"], inplace=True)
    IN_DIR.mkdir(parents=True, exist_ok=True)
    out.to_csv(OUT_CSV, index=False)
    print(f"Wrote averaged CSV → {OUT_CSV}")

    # quick LaTeX (edit columns as needed)
    cols = ["model","sample_size","device","test_time_s","test_time_s_std",
            "throughput_apps_per_s","throughput_std","cpu_max_rss_mb",
            "peak_vram_mb","model_bytes"]
    have = [c for c in cols if c in out.columns]
    OUT_LATEX.write_text(out[have].to_latex(index=False, float_format="%.3f"))
    print(f"Wrote LaTeX table → {OUT_LATEX}")

    create_averaged_compute_plot(out)

if __name__ == "__main__":
    main()
