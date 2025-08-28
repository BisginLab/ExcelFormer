'''
# Create the output directory
mkdir -p /home/umflint.edu/koernerg/roc_dumps

# MI FEATURES
# 10K sample size
python evaluate.py \
  --model_path "result/ExcelFormer/default/mixup(none)/android_security/42/10000/pytorch_model.pt" \
  --dataset android_security --catenc --sample_size 10000 \
  --mi_json "output/mi/android_security/full/mi_top25_catenc(1)_norm(quantile).json" \
  --dump_roc /home/umflint.edu/koernerg/roc_dumps/excelformer_10000.json

# 100K sample size  
python evaluate.py \
  --model_path "result/ExcelFormer/default/mixup(none)/android_security/42/100000/pytorch_model.pt" \
  --dataset android_security --catenc --sample_size 100000 \
  --mi_json "output/mi/android_security/full/mi_top25_catenc(1)_norm(quantile).json" \
  --dump_roc /home/umflint.edu/koernerg/roc_dumps/excelformer_100000.json

# Full dataset
python evaluate.py \
  --model_path "result/ExcelFormer/default/mixup(none)/android_security/42/full/pytorch_model.pt" \
  --dataset android_security --catenc --sample_size full \
  --mi_json "output/mi/android_security/full/mi_top25_catenc(1)_norm(quantile).json" \
  --dump_roc /home/umflint.edu/koernerg/roc_dumps/excelformer_full.json

# XGBOOST FEATURES
python evaluate.py   --model_path "result/ExcelFormer/xgbfi/mixup(none)/android_security/42/10000/pytorch_model.pt"   --dataset android_security --catenc --sample_size 10000 --mi_json "output/mi/android_security/full/xgboost_feature_importance_20250827_230123.json" --dump_roc /home/umflint.edu/koernerg/roc_dumps/excelformer_xgbfi_10000.json
python evaluate.py   --model_path "result/ExcelFormer/xgbfi/mixup(none)/android_security/42/100000/pytorch_model.pt"   --dataset android_security --catenc --sample_size 100000 --mi_json "output/mi/android_security/full/xgboost_feature_importance_20250827_230123.json" --dump_roc /home/umflint.edu/koernerg/roc_dumps/excelformer_xgbfi_100000.json
python evaluate.py   --model_path "result/ExcelFormer/xgbfi/mixup(none)/android_security/42/full/pytorch_model.pt"   --dataset android_security --catenc --sample_size full --mi_json "output/mi/android_security/full/xgboost_feature_importance_20250827_230123.json" --dump_roc /home/umflint.edu/koernerg/roc_dumps/excelformer_xgbfi_full.json 

'''

import sys
import os
import math
import time
import json
import random
import argparse
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, roc_auc_score

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset
from sklearn.feature_selection import mutual_info_classif, mutual_info_regression
from category_encoders import CatBoostEncoder
from tqdm import tqdm
from pathlib import Path

from bin import ExcelFormer
from lib import Transformations, build_dataset, prepare_tensors, make_optimizer, DATA

# XGBoost feature list (must match training)
XGBOOST_FEATURES = [
    'ContentRating', 'Genre', 'CurrentVersion', 'AndroidVersion', 'DeveloperCategory',
    'lowest_android_version', 'highest_android_version', 'privacy_policy_link',
    'developer_website', 'days_since_last_update', 'isSpamming', 'max_downloads_log',
    'LenWhatsNew', 'PHONE', 'OneStarRatings', 'developer_address', 'FourStarRatings',
    'intent', 'ReviewsAverage', 'STORAGE', 'LastUpdated', 'TwoStarRatings',
    'LOCATION', 'FiveStarRatings', 'ThreeStarRatings'
]

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path", type=str, default='./result/ExcelFormer/default/mixup(none)/android_security/42/pytorch_model.pt')
    parser.add_argument("--dataset", type=str, default='android_security')
    parser.add_argument("--normalization", type=str, default='quantile')
    parser.add_argument("--catenc", action='store_true', default=True)
    parser.add_argument("--sample_size", type=str, choices=['10000', '50000', '100000', 'full'], default=None, help="Sample size for plot title/filename")
    parser.add_argument("--mi_json", type=str, help="Path to pre-computed MI JSON file for feature selection (must match training)")
    parser.add_argument(
        "--dump_roc",
        type=str,
        default=None,
        help="If set, write ROC fpr/tpr + AUC for val/test to this JSON file"
    )
    return parser.parse_args()

def main():
    args = get_args()
    print(f"Loading model from: {args.model_path}")
    
    # Load checkpoint
    checkpoint = torch.load(args.model_path, map_location='cpu')
    
    # Load dataset
    transformation = Transformations(
        normalization=args.normalization if args.normalization != '__none__' else None
    )
    
    # MI JSON is REQUIRED - no fallback allowed
    if not args.mi_json:
        raise ValueError("--mi_json argument is REQUIRED. No fallback to hardcoded features allowed.")
    
    if not os.path.exists(args.mi_json):
        raise FileNotFoundError(f"MI JSON file not found: {args.mi_json}")
    
    # Load features from MI JSON
    print(f"[FEATURE SELECTION] Loading features from JSON: {args.mi_json}")
    with open(args.mi_json, 'r') as f:
        mi_data = json.load(f)
    selected_features = mi_data['selected_names']
    print(f"[FEATURE SELECTION] Using {len(selected_features)} features from JSON: {selected_features}")
    print(f"[FEATURE SELECTION] JSON feature order: {selected_features}")
    
    # Store the JSON feature order for later reordering
    json_feature_order = selected_features.copy()
    
    print(f"[FEATURE SELECTION] About to call build_dataset with features: {selected_features}")
    dataset = build_dataset(
        DATA / args.dataset,
        transformation,
        cache=False,
        sample_size=args.sample_size,
        indices_dir="./standardized_data",
        selected_features=selected_features
    )
    print(f"[FEATURE SELECTION] After build_dataset:")
    print(f"[FEATURE SELECTION]   - dataset.num_feature_names: {dataset.num_feature_names}")
    print(f"[FEATURE SELECTION]   - dataset.cat_feature_names: {dataset.cat_feature_names}")
    print(f"[FEATURE SELECTION]   - Combined order: {dataset.cat_feature_names + dataset.num_feature_names}")
    
    # Build a name→index mapping from the *current* [cat + num] order to the JSON order.
    current_order = (dataset.cat_feature_names or []) + (dataset.num_feature_names or [])
    name_to_idx = {n: i for i, n in enumerate(current_order)}
    missing = [n for n in json_feature_order if n not in name_to_idx]
    if missing:
        raise ValueError(f"Features from JSON not found in dataset: {missing}")
    reindex = [name_to_idx[n] for n in json_feature_order]
    print(f"[FEATURE SELECTION] Reorder indices: {reindex}")

    # === DEBUG PRINTS: Dataset structure and features ===
    print("[DEBUG][EVAL] dataset.X_num keys:", list(dataset.X_num.keys()))
    print("[DEBUG][EVAL] dataset.X_cat keys:", None if dataset.X_cat is None else list(dataset.X_cat.keys()))
    print("[DEBUG][EVAL] dataset.X_num['test'] shape:", dataset.X_num['test'].shape)
    print("[DEBUG][EVAL] dataset.X_cat['test'] shape:" if dataset.X_cat is not None else "No X_cat", 
          None if dataset.X_cat is None else dataset.X_cat['test'].shape)
    print("[DEBUG][EVAL] dataset.num_feature_names:", dataset.num_feature_names)
    print("[DEBUG][EVAL] dataset.cat_feature_names:", dataset.cat_feature_names)
    
    print("\nSaved Metrics from Training:")
    print(f"Best validation score: {checkpoint['best_score']:.4f}")
    print(f"Best test score: {checkpoint['best_test_score']:.4f}")
    print(f"Final test score: {checkpoint['final_test_score']:.4f}")
    
    print("\n=== Initial Data Analysis ===")
    print("Number of samples:", len(dataset.y['test']))
    print("Number of features:", dataset.n_num_features + dataset.n_cat_features)
    print("Class distribution:", np.bincount(dataset.y['test'].astype(int)))

    print("\n=== Feature Processing Details ===")
    print("Number of numerical features:", dataset.n_num_features)
    print("Number of categorical features:", dataset.n_cat_features)
    
    print("\n=== Initial Feature Shapes ===")
    print(f"Initial numerical features: {dataset.X_num['test'].shape}")
    if dataset.X_cat is not None:
        print(f"Initial categorical features: {dataset.X_cat['test'].shape}")
    
    # Convert to float32
    if dataset.X_num['test'].dtype == np.float64:
        dataset.X_num['test'] = dataset.X_num['test'].astype(np.float32)
        dataset.X_num['val'] = dataset.X_num['val'].astype(np.float32)
    
    # Debug categorical encoding
    print("\n=== Categorical Encoding Debug ===")
    print(f"args.catenc: {args.catenc}")
    print(f"dataset.X_cat is not None: {dataset.X_cat is not None}")
    if args.catenc and dataset.X_cat is not None:
        print(">>> ENTERING CATBOOST ENCODER BLOCK")
        enc = CatBoostEncoder(
            cols=list(range(len(dataset.cat_feature_names))),
            return_df=False
        ).fit(dataset.X_cat['train'], dataset.y['train'])

        # Encode cats, concat as [encoded_cats, nums], THEN reorder to JSON order.
        encoded_val = enc.transform(dataset.X_cat['val']).astype(np.float32)
        encoded_test = enc.transform(dataset.X_cat['test']).astype(np.float32)
        X_val_all = np.concatenate([encoded_val, dataset.X_num['val']], axis=1)
        X_test_all = np.concatenate([encoded_test, dataset.X_num['test']], axis=1)
        X_val = X_val_all[:, reindex].astype(np.float32)
        X_test = X_test_all[:, reindex].astype(np.float32)

        print("[DEBUG][EVAL] Encoded cat shape (test):", encoded_test.shape)
        print("[DEBUG][EVAL] X_test shape after concat+reorder:", X_test.shape)
        print("[DEBUG][EVAL] X_val shape after concat+reorder:", X_val.shape)
    else:
        print(">>> SKIPPING CATBOOST ENCODER BLOCK, numeric-only path")
        # Reorder numeric-only matrix by names too (still safe)
        X_val = dataset.X_num['val'][:, reindex].astype(np.float32)
        X_test = dataset.X_num['test'][:, reindex].astype(np.float32)
        print("[DEBUG][EVAL] X_test shape (no catenc, after reorder):", X_test.shape)
        print("[DEBUG][EVAL] X_val shape (no catenc, after reorder):", X_val.shape)

    # --- FIX: Feature count check is now AFTER concatenation ---
    expected_n_features = checkpoint.get('n_features', None)
    if expected_n_features is not None and expected_n_features != X_test.shape[1]:
        print(f"WARNING: Model was trained with {expected_n_features} features, but evaluation is using {X_test.shape[1]}.")
        print("You must retrain the model with the current feature selection logic.")
        sys.exit(1)
    # Strong guard: name/order equality if present
    if 'feature_names' in checkpoint:
        if checkpoint['feature_names'] != json_feature_order:
            raise RuntimeError(
                "Feature name/order mismatch between training and evaluation.\n"
                f"Train: {checkpoint['feature_names']}\n"
                f"Eval:  {json_feature_order}"
            )
    # --- END FIX ---

    print("Order for model input (JSON):", json_feature_order)

    print("[DEBUG][EVAL] X_test shape:", X_test.shape)
    print("[DEBUG][EVAL] X_val shape:", X_val.shape)

    # Now prepare tensors and evaluate
    X_num_test = torch.tensor(X_test, dtype=torch.float32).to(device)
    y_test = torch.tensor(dataset.y['test'], dtype=torch.float32).to(device)
    
    # Initialize model with exact feature count
    model = ExcelFormer(
        d_numerical=X_test.shape[1],  # Use selected feature count
        d_out=2,  # Binary classification
        categories=None,
        token_bias=True,
        n_layers=3,
        n_heads=32,
        d_token=256,
        attention_dropout=0.3,
        ffn_dropout=0.0,
        residual_dropout=0.0,
        prenormalization=True,
        kv_compression=None,
        kv_compression_sharing=None,
        init_scale=0.01,
    ).to(device)

    print(f"\nModel weight dtype: {next(model.parameters()).dtype}")

    # Convert to float32 consistently
    model = model.float()
    
    # Load model weights and set to eval mode
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    
    # Evaluate both validation and test sets
    print("\nEvaluating validation and test sets:")
    
    # Create datasets for both
    val_dataset = TensorDataset(
        torch.tensor(X_val, dtype=torch.float32).to(device),
        torch.tensor(dataset.y['val'], dtype=torch.float32).to(device)
    )
    test_dataset = TensorDataset(X_num_test, y_test)
    
    val_loader = DataLoader(val_dataset, batch_size=1024, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=1024, shuffle=False)
    
    # Get predictions for both sets
    predictions_val = []
    predictions_test = []
    with torch.inference_mode():
        for batch in tqdm(val_loader, desc="Processing validation set"):
            x_num, _ = batch
            pred = model(x_num, None)
            predictions_val.append(pred)
            
        for batch in tqdm(test_loader, desc="Processing test set"):
            x_num, _ = batch
            pred = model(x_num, None)
            predictions_test.append(pred)
    
    predictions_val = torch.cat(predictions_val).cpu().numpy()
    predictions_test = torch.cat(predictions_test).cpu().numpy()
    
    # Calculate probabilities and ROC curves for both sets
    y_val_proba = torch.softmax(torch.tensor(predictions_val), dim=1).numpy()[:, 1]
    y_test_proba = torch.softmax(torch.tensor(predictions_test), dim=1).numpy()[:, 1]
    
    fpr_val, tpr_val, _ = roc_curve(dataset.y['val'], y_val_proba)
    fpr_test, tpr_test, _ = roc_curve(y_test.cpu().numpy(), y_test_proba)
    
    roc_auc_val = roc_auc_score(dataset.y['val'], y_val_proba)
    roc_auc_test = roc_auc_score(y_test.cpu().numpy(), y_test_proba)
    
    # Plot both ROC curves
    plt.figure(figsize=(10, 8))
    plt.plot(fpr_val, tpr_val, color='blue', lw=2, label=f'Validation ROC (AUC = {roc_auc_val:.3f})')
    plt.plot(fpr_test, tpr_test, color='red', lw=2, label=f'Test ROC (AUC = {roc_auc_test:.3f})')
    plt.plot([0, 1], [0, 1], color='gray', lw=2, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    size_str = f"{args.sample_size}" if args.sample_size else "Full"
    plt.title(f'ExcelFormer ROC Curve - {size_str} Samples')
    plt.legend(loc="lower right")
    plt.grid(True)
    plt.savefig(f'roc_curve_excelformer_{size_str.replace(",", "")}.png')
    plt.show()
    
    # Calculate and print metrics for both sets
    metrics = dataset.calculate_metrics(
        {
            'val': predictions_val,
            'test': predictions_test
        },
        prediction_type='logits'
    )
    
    print("\nVALIDATION SET METRICS:")
    print(f"ROC AUC: {metrics['val']['roc_auc']:.4f}")
    print(f"Accuracy: {metrics['val']['accuracy']:.4f}")
    
    print("\nTEST SET METRICS:")
    print(f"ROC AUC: {metrics['test']['roc_auc']:.4f}")
    print(f"Accuracy: {metrics['test']['accuracy']:.4f}")

    print("Final feature order used for evaluation:")
    print(dataset.cat_feature_names + dataset.num_feature_names)
    print("X_test.shape:", X_test.shape)

    # Dump ROC data to JSON if requested
    if args.dump_roc:
        os.makedirs(os.path.dirname(args.dump_roc), exist_ok=True)
        payload = {
            "model": "excelformer",
            "size": str(args.sample_size or "full"),
            "val": {
                "fpr": [float(x) for x in fpr_val],
                "tpr": [float(x) for x in tpr_val],
                "auc": float(roc_auc_val),
            },
            "test": {
                "fpr": [float(x) for x in fpr_test],
                "tpr": [float(x) for x in tpr_test],
                "auc": float(roc_auc_test),
            },
        }
        with open(args.dump_roc, "w") as f:
            json.dump(payload, f)
        print(f"[EF] Wrote ROC dump to {args.dump_roc}")

if __name__ == '__main__':
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    main()