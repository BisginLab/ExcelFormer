#!/usr/bin/env python3
"""
Plot mutual information scores for all features in the android_security dataset.

This script uses the same dataset-building pipeline as compute_mi_topk.py
and calculates MI on the train split only, ensuring identical preprocessing
and feature ordering as the training pipeline.

Usage:
    python plot_mi_scores.py [--dataset android_security] [--sample_size full] [--catenc] [--normalization quantile] [--json_ref path/to/mi_top25.json]
"""

import argparse
import os
import sys
import time
import random
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import json
from pathlib import Path
from sklearn.feature_selection import mutual_info_classif, mutual_info_regression
from category_encoders import CatBoostEncoder

# Add project root to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from lib import Transformations, build_dataset, DATA

def seed_everything(seed=42):
    """Set random seeds for reproducibility."""
    import torch
    random.seed(seed)
    np.random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

def get_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description='Plot MI scores for all features')
    parser.add_argument('--dataset', type=str, default='android_security', 
                       help='Dataset name')
    parser.add_argument('--sample_size', type=str, default='full', 
                       choices=['10000', '50000', '100000', 'full'],
                       help='Sample size (use "full" for full dataset)')
    parser.add_argument('--catenc', action='store_true', 
                       help='Use CatBoost encoder for categorical features')
    parser.add_argument('--normalization', type=str, default='quantile',
                       choices=['quantile', 'standard', '__none__'],
                       help='Normalization method')
    parser.add_argument('--seed', type=int, default=42,
                       help='Random seed')
    parser.add_argument('--output', type=str, default='mi_scores_plot.png',
                       help='Output plot filename')
    parser.add_argument('--indices_dir', type=str, default='./standardized_data',
                       help='Directory containing train/val/test indices')
    parser.add_argument('--json_ref', type=str, default=None,
                       help='Path to reference JSON file for comparison')
    return parser.parse_args()

def main():
    args = get_args()
    seed_everything(args.seed)
    
    print(f"=== Loading Dataset: {args.dataset} ===")
    print(f"Sample size: {args.sample_size}")
    print(f"CatBoost encoding: {args.catenc}")
    print(f"Normalization: {args.normalization}")
    print(f"Seed: {args.seed}")
    
    # Build dataset with the same pipeline as compute_mi_topk.py
    transformation = Transformations(normalization=(None if args.normalization == "__none__" else args.normalization))
    ds = build_dataset(DATA / args.dataset, transformation, cache=False,
                       sample_size=args.sample_size, indices_dir=args.indices_dir)
    
    # Prepare numeric matrix to run MI on (AFTER CatBoost if requested)
    if ds.X_num["train"].dtype == np.float64:
        ds.X_num = {k: v.astype(np.float32) for k, v in ds.X_num.items()}

    if args.catenc and ds.X_cat is not None:
        print(f"\nApplying CatBoost encoding...")
        card = ds.get_category_sizes("train")
        enc = CatBoostEncoder(cols=list(range(len(card))), return_df=False).fit(ds.X_cat["train"], ds.y["train"])
        X_num_proc = {}
        for split in ["train", "val", "test"]:
            enc_cat = enc.transform(ds.X_cat[split]).astype(np.float32)
            X_num_proc[split] = np.concatenate([enc_cat, ds.X_num[split]], axis=1)
        # names are cat then num
        all_feature_names = (ds.cat_feature_names or []) + (ds.num_feature_names or [])
    else:
        X_num_proc = ds.X_num
        all_feature_names = ds.num_feature_names or []

    # Use ONLY the train split for MI calculation (exactly as in compute_mi_topk.py)
    X_train = X_num_proc["train"]
    y_train = ds.y["train"]
    is_reg = bool(ds.is_regression)
    mi_func = mutual_info_regression if is_reg else mutual_info_classif

    # EXACT call (no kwargs) - match compute_mi_topk.py exactly
    print(f"[MI] computing on train: X={X_train.shape}, y={y_train.shape}, catenc={args.catenc}")
    start_time = time.time()
    mi_scores = mi_func(X_train, y_train)
    calc_time = time.time() - start_time
    
    print(f"[MI] done in {calc_time:.2f}s. Features={len(mi_scores)}")
    
    # Sort features by MI score (descending)
    sorted_indices = np.argsort(-mi_scores)
    sorted_scores = mi_scores[sorted_indices]
    sorted_names = [all_feature_names[i] for i in sorted_indices]
    
    # Count features above 0.01 MI threshold
    above_threshold = np.sum(sorted_scores >= 0.01)
    below_threshold = np.sum(sorted_scores < 0.01)
    print(f"Features above 0.01 MI: {above_threshold}")
    print(f"Features below 0.01 MI: {below_threshold}")
    
    # JSON comparison if reference provided
    if args.json_ref and os.path.exists(args.json_ref):
        print(f"\n=== Comparing with reference JSON: {args.json_ref} ===")
        with open(args.json_ref, 'r') as f:
            ref_data = json.load(f)
        
        k = ref_data.get('k', 25)
        ref_selected = ref_data.get('selected_names', [])
        ref_scores = ref_data.get('mi_scores_by_name', {})
        
        # The selected_names array is already sorted by MI score (descending)
        # We need to get the scores for these selected features in the correct order
        ref_scores_sorted = [ref_scores.get(name, 0) for name in ref_selected]
        
        print(f"Reference k: {k}")
        print(f"Reference top-{k} features:")
        for i, name in enumerate(ref_selected[:k]):
            score = ref_scores_sorted[i]
            print(f"  {i+1:2d}. {name:30s} {score:.6f}")
        
        print(f"\nCurrent top-{k} features:")
        for i, name in enumerate(sorted_names[:k]):
            score = sorted_scores[i]
            print(f"  {i+1:2d}. {name:30s} {score:.6f}")
        
        print(f"\nRank differences (reference vs current):")
        print(f"{'Rank':<4} {'Ref Feature':<30} {'Cur Feature':<30} {'Match':<5}")
        print("-" * 75)
        for i in range(min(k, len(ref_selected), len(sorted_names))):
            ref_name = ref_selected[i] if i < len(ref_selected) else "N/A"
            cur_name = sorted_names[i] if i < len(sorted_names) else "N/A"
            match = "✓" if ref_name == cur_name else "✗"
            print(f"{i+1:<4} {ref_name:<30} {cur_name:<30} {match:<5}")
    
    # Create the plot
    print(f"\n=== Creating Plot ===")
    plt.figure(figsize=(20, 8))
    
    # Create bar plot with different colors: top 25 in normal blue, rest in pale blue
    bars = []
    for i in range(len(sorted_scores)):
        if i < 25:
            # Top 25 features: normal blue
            bar = plt.bar(i, sorted_scores[i], color='steelblue', alpha=0.7, 
                         edgecolor='black', linewidth=0.5)
        else:
            # Features beyond top 25: pale blue
            bar = plt.bar(i, sorted_scores[i], color='steelblue', alpha=0.2, 
                         edgecolor='black', linewidth=0.5)
        bars.extend(bar)
    
    # Customize the plot
    plt.xlabel('Features (sorted by MI score)', fontsize=12)
    plt.ylabel('Mutual Information Score', fontsize=12)
    plt.title('Mutual Information Scores', fontsize=14, fontweight='bold')
    
    # Add grid for better readability
    plt.grid(True, alpha=0.3, axis='y')
    
    # Show every feature name on x-axis
    plt.xticks(range(len(sorted_names)), sorted_names, rotation=45, ha='right', fontsize=8)
    
    # Adjust layout to prevent label cutoff
    plt.tight_layout()
    
    # Save the plot
    plt.savefig(args.output, dpi=300, bbox_inches='tight')
    print(f"Plot saved as: {args.output}")
    
    # Also save the data as CSV for further analysis
    csv_filename = args.output.replace('.png', '_data.csv')
    df = pd.DataFrame({
        'feature_name': sorted_names,
        'mi_score': sorted_scores,
        'rank': range(1, len(sorted_names) + 1)
    })
    df.to_csv(csv_filename, index=False)
    print(f"Data saved as: {csv_filename}")
    
    # Print summary statistics
    print(f"\n=== Summary Statistics ===")
    print(f"Total features: {len(sorted_scores)}")
    print(f"Mean MI score: {sorted_scores.mean():.6f}")
    print(f"Median MI score: {np.median(sorted_scores):.6f}")
    print(f"Std MI score: {sorted_scores.std():.6f}")
    print(f"Top 5 features:")
    for i in range(min(5, len(sorted_names))):
        print(f"  {i+1}. {sorted_names[i]:30s} {sorted_scores[i]:.6f}")
    
    # Show the plot
    plt.show()

if __name__ == '__main__':
    main()
