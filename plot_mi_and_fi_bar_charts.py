#!/usr/bin/env python3
"""
Create side-by-side bar charts comparing XGBoost feature importance and mutual information scores.
Top chart: XGBoost feature importance
Bottom chart: Mutual information scores
"""

import json
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from pathlib import Path

def load_json_data(json_path):
    """Load JSON data and return feature names and scores."""
    with open(json_path, 'r') as f:
        data = json.load(f)
    
    # Get feature names and scores
    feature_names = data['selected_names']  # Already sorted by MI score (descending)
    scores_by_name = data['mi_scores_by_name']  # Dictionary mapping feature names to scores
    
    # Create a mapping of feature name to score using the correct order
    feature_scores = {}
    for name in feature_names:
        feature_scores[name] = scores_by_name[name]
    
    return feature_scores, feature_names

def main():
    # Load XGBoost feature importance data
    xgb_path = 'output/mi/android_security/full/xgboost_feature_importance_20250827_230123.json'
    xgb_scores, xgb_features = load_json_data(xgb_path)
    
    # Load MI data
    mi_path = 'output/mi/android_security/full/mi_top25_catenc(1)_norm(quantile).json'
    mi_scores, mi_features = load_json_data(mi_path)
    
    print(f"XGBoost features: {len(xgb_features)}")
    print(f"MI features: {len(mi_features)}")
    
    # Create figure with two subplots
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(20, 12))
    
    # Top chart: XGBoost Feature Importance
    xgb_values = [xgb_scores[f] for f in xgb_features]
    bars1 = ax1.bar(range(len(xgb_features)), xgb_values, 
                    color='steelblue', alpha=0.7, edgecolor='black', linewidth=0.5)
    ax1.set_title('Feature Importance', fontsize=16, fontweight='bold')
    ax1.set_ylabel('Feature Importance', fontsize=12)
    ax1.set_xticks(range(len(xgb_features)))
    ax1.set_xticklabels(xgb_features, rotation=45, ha='right', fontsize=8)
    ax1.grid(True, alpha=0.3, axis='y')
    
    # Bottom chart: Mutual Information Scores (sorted by score)
    # Sort MI features by their scores (descending)
    mi_feature_scores = [(f, mi_scores[f]) for f in mi_features]
    mi_feature_scores.sort(key=lambda x: x[1], reverse=True)
    mi_features_sorted = [f for f, _ in mi_feature_scores]
    mi_values_sorted = [s for _, s in mi_feature_scores]
    
    bars2 = ax2.bar(range(len(mi_features_sorted)), mi_values_sorted, 
                    color='darkorange', alpha=0.7, edgecolor='black', linewidth=0.5)
    ax2.set_title('Mutual Information', fontsize=16, fontweight='bold')
    ax2.set_ylabel('MI Score', fontsize=12)
    ax2.set_xlabel('Features', fontsize=12)
    ax2.set_xticks(range(len(mi_features_sorted)))
    ax2.set_xticklabels(mi_features_sorted, rotation=45, ha='right', fontsize=8)
    ax2.grid(True, alpha=0.3, axis='y')
    
    # Adjust layout
    plt.tight_layout()
    
    # Save the plot
    output_path = 'mi_and_fi_comparison.png'
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"Plot saved as: {output_path}")
    
    # Print summary statistics
    print(f"\n=== XGBoost Feature Importance Summary ===")
    print(f"Number of features: {len(xgb_features)}")
    print(f"Mean importance: {np.mean(xgb_values):.6f}")
    print(f"Max importance: {np.max(xgb_values):.6f}")
    print(f"Min importance: {np.min(xgb_values):.6f}")
    
    print(f"\n=== Mutual Information Summary ===")
    print(f"Number of features: {len(mi_features_sorted)}")
    print(f"Mean MI score: {np.mean(mi_values_sorted):.6f}")
    print(f"Max MI score: {np.max(mi_values_sorted):.6f}")
    print(f"Min MI score: {np.min(mi_values_sorted):.6f}")
    print(f"Features above 0.01 MI: {np.sum(np.array(mi_values_sorted) >= 0.01)}")
    
    # Show the plot
    plt.show()

if __name__ == '__main__':
    main()
