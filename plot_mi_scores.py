#!/usr/bin/env python3
"""
Plot mutual information scores for all features in the android_security dataset.

This script loads the original dataset WITHOUT doing any feature selection,
uses the full feature set, and plots a bar chart of mutual information scores
sorted by score. It calculates mutual information EXACTLY as the training script
would calculate it in the commented out code.

Usage:
    python plot_mi_scores.py [--dataset android_security] [--sample_size full] [--catenc] [--normalization quantile]
"""

import argparse
import os
import sys
import time
import random
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
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
    return parser.parse_args()

def main():
    args = get_args()
    seed_everything(args.seed)
    
    print(f"=== Loading Dataset: {args.dataset} ===")
    print(f"Sample size: {args.sample_size}")
    print(f"CatBoost encoding: {args.catenc}")
    print(f"Normalization: {args.normalization}")
    print(f"Seed: {args.seed}")
    
    # Build dataset with the same transformations as training script
    transformation = Transformations(
        normalization=args.normalization if args.normalization != '__none__' else None
    )
    
    # Load data directly from CSV (bypass the complex dataset loading)
    csv_path = Path('/home/umflint.edu/koernerg/android-security-comparison/AndroidSecurityComparison/data/raw/corrected_permacts.csv')
    print(f"Loading data from: {csv_path}")
    
    # Load and preprocess data exactly as in the training script
    df = pd.read_csv(csv_path)
    print(f"Initial DataFrame shape: {df.shape}")
    
    # Clean data exactly as in training script
    df = df.dropna()
    print(f"Shape after dropping NaNs: {df.shape}")
    
    if 'Unnamed: 0' in df.columns:
        df = df.drop('Unnamed: 0', axis=1)
        print(f"Shape after dropping 'Unnamed: 0': {df.shape}")
        
    df = df.drop(['pkgname'], axis=1)
    print(f"Shape after dropping pkgname: {df.shape}")
    
    # Separate features and target
    X = df.drop(['status'], axis=1)
    y = df['status']
    
    print(f"Final data shape: X={X.shape}, y={y.shape}")
    print(f"Target distribution: {y.value_counts().to_dict()}")
    
    # Identify numerical and categorical features
    num_features = X.select_dtypes(include=['int64', 'float64']).columns.tolist()
    cat_features = X.select_dtypes(include=['object']).columns.tolist()
    
    print(f"Numerical features: {len(num_features)}")
    print(f"Categorical features: {len(cat_features)}")
    print(f"Total features: {len(num_features) + len(cat_features)}")
    print(f"Expected features: 47 (49 columns - 2 dropped)")
    
    # Debug: print all column names and types
    print(f"\nAll columns and their types:")
    for col in X.columns:
        print(f"  {col}: {X[col].dtype}")
    
    # Convert to numpy arrays
    X_num = X.select_dtypes(include=['int64', 'float64']).values.astype(np.float32)
    X_cat = X.select_dtypes(include=['object']).values if cat_features else None
    
    # Apply CatBoost encoding if requested (exactly as in training script)
    if args.catenc and X_cat is not None:
        print(f"\nApplying CatBoost encoding...")
        # Get cardinalities for categorical features
        cardinalities = [len(set(X_cat[:, i])) for i in range(X_cat.shape[1])]
        
        enc = CatBoostEncoder(
            cols=list(range(len(cardinalities))), 
            return_df=False
        ).fit(X_cat, y)
        
        # Transform categorical features
        X_cat_encoded = enc.transform(X_cat).astype(np.float32)
        
        # Combine categorical and numerical features
        X_combined = np.concatenate([X_cat_encoded, X_num], axis=1)
        all_feature_names = cat_features + num_features
        
        print(f"  Shape after encoding: {X_combined.shape}")
        print(f"  Total features: {len(all_feature_names)}")
    else:
        X_combined = X_num
        all_feature_names = num_features
        print(f"  Using numerical features only: {len(all_feature_names)}")
    
    # Calculate mutual information (exactly as in training script)
    print(f"\n=== Calculating Mutual Information Scores ===")
    
    # Choose MI function based on task type (binary classification)
    mi_func = mutual_info_classif
    
    print(f"Computing MI on data: X={X_combined.shape}, y={y.shape}")
    print(f"Using function: {mi_func.__name__}")
    
    start_time = time.time()
    mi_scores = mi_func(X_combined, y)
    calc_time = time.time() - start_time
    
    print(f"MI calculation completed in {calc_time:.2f} seconds")
    print(f"Number of features: {len(mi_scores)}")
    print(f"MI scores range: [{mi_scores.min():.6f}, {mi_scores.max():.6f}]")
    
    # Sort features by MI score (descending)
    sorted_indices = np.argsort(-mi_scores)
    sorted_scores = mi_scores[sorted_indices]
    sorted_names = [all_feature_names[i] for i in sorted_indices]
    
    # Count features above 0.01 MI threshold
    above_threshold = np.sum(sorted_scores >= 0.01)
    below_threshold = np.sum(sorted_scores < 0.01)
    print(f"Features above 0.01 MI: {above_threshold}")
    print(f"Features below 0.01 MI: {below_threshold}")
    
    # Create the plot
    print(f"\n=== Creating Plot ===")
    plt.figure(figsize=(20, 8))
    
    # Create bar plot with different colors based on MI threshold (0.01)
    bars = []
    for i in range(len(sorted_scores)):
        if sorted_scores[i] >= 0.01:
            # Features above 0.01 MI: full blue
            bar = plt.bar(i, sorted_scores[i], color='steelblue', alpha=0.7, 
                         edgecolor='black', linewidth=0.5)
        else:
            # Features below 0.01 MI: very pale blue
            bar = plt.bar(i, sorted_scores[i], color='steelblue', alpha=0.2, 
                         edgecolor='black', linewidth=0.5)
        bars.extend(bar)
    
    # Customize the plot
    plt.xlabel('Features (sorted by MI score)', fontsize=12)
    plt.ylabel('Mutual Information Score', fontsize=12)
    plt.title('Mutual Information Scores', fontsize=14, fontweight='bold')
    
    # Add grid for better readability
    plt.grid(True, alpha=0.3, axis='y')
    
    # Add dashed red horizontal line at 0.01 MI
    plt.axhline(y=0.01, color='red', linestyle='--', linewidth=2, alpha=0.8, label='MI = 0.01')
    
    # Show every feature name on x-axis
    plt.xticks(range(len(sorted_names)), sorted_names, rotation=45, ha='right', fontsize=8)
    
    # Add legend for the horizontal line
    plt.legend(loc='upper right')
    
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
