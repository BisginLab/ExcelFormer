#!/usr/bin/env python3
"""
Regenerate MI JSON with all features above 0.01 MI threshold instead of top-K.
"""

import json
import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.feature_selection import mutual_info_classif
from category_encoders import CatBoostEncoder
from datetime import datetime

def main():
    # Load and process data exactly like the original script
    csv_path = Path('/home/umflint.edu/koernerg/android-security-comparison/AndroidSecurityComparison/data/raw/corrected_permacts.csv')
    df = pd.read_csv(csv_path)
    df = df.dropna()
    if 'Unnamed: 0' in df.columns:
        df = df.drop('Unnamed: 0', axis=1)
    df = df.drop(['pkgname'], axis=1)

    X = df.drop(['status'], axis=1)
    y = df['status']

    num_features = X.select_dtypes(include=['int64', 'float64']).columns.tolist()
    cat_features = X.select_dtypes(include=['object']).columns.tolist()

    X_num = X.select_dtypes(include=['int64', 'float64']).values.astype(np.float32)
    X_cat = X.select_dtypes(include=['object']).values

    # Apply CatBoost encoding
    cardinalities = [len(set(X_cat[:, i])) for i in range(X_cat.shape[1])]
    enc = CatBoostEncoder(cols=list(range(len(cardinalities))), return_df=False).fit(X_cat, y)
    X_cat_encoded = enc.transform(X_cat).astype(np.float32)
    X_combined = np.concatenate([X_cat_encoded, X_num], axis=1)
    all_feature_names = cat_features + num_features

    # Calculate MI
    mi_scores = mutual_info_classif(X_combined, y)

    # Find features above 0.01 threshold
    threshold = 0.01
    above_threshold_mask = mi_scores >= threshold
    above_threshold_indices = np.where(above_threshold_mask)[0]
    above_threshold_scores = mi_scores[above_threshold_mask]
    above_threshold_names = [all_feature_names[i] for i in above_threshold_indices]

    # Sort by MI score (descending)
    sort_order = np.argsort(-above_threshold_scores)
    selected_indices = above_threshold_indices[sort_order]
    selected_names = [above_threshold_names[i] for i in sort_order]
    selected_scores = above_threshold_scores[sort_order]

    print(f"Found {len(selected_names)} features above {threshold} MI threshold")
    print("Selected features (sorted by MI):")
    for i, (name, score) in enumerate(zip(selected_names, selected_scores), 1):
        print(f"  {i:2d}. {name:30s} {score:.6f}")

    # Create JSON payload
    payload = {
        "timestamp": datetime.utcnow().isoformat() + "Z",
        "dataset": "android_security",
        "sample_size": "full",
        "indices_dir": "./standardized_data",
        "normalization": "quantile",
        "catenc": True,
        "seed": 42,
        "k": len(selected_names),  # Number of features above threshold
        "is_regression": False,
        "feature_order": all_feature_names,
        "mi_scores_aligned": mi_scores.tolist(),
        "mi_scores_by_name": {all_feature_names[i]: float(mi_scores[i]) for i in range(len(all_feature_names))},
        "selected_indices": selected_indices.tolist(),
        "selected_names": selected_names,
        "shapes": {
            "train": [506912, 47],
            "val": [108624, 47], 
            "test": [108624, 47]
        }
    }

    # Save JSON
    output_path = "output/mi/android_security/full/mi_threshold_0.01_catenc(1)_norm(quantile).json"
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    with open(output_path, 'w') as f:
        json.dump(payload, f, indent=2)
    
    print(f"\nSaved MI JSON to: {output_path}")
    print(f"Total features above 0.01 MI: {len(selected_names)}")

if __name__ == "__main__":
    main()
