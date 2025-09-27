# excelformer_plot_pr.py
import pandas as pd
import numpy as np
import torch
import matplotlib.pyplot as plt
from datetime import datetime
import os
import glob
import sys
import json

# Import the same utilities used in training
from lib import Transformations, build_dataset, DATA, prepare_tensors
from category_encoders import CatBoostEncoder

# ---------- Logger (same pattern as your eval) ----------
class Logger:
    def __init__(self, filename):
        self.terminal = sys.stdout
        self.log = open(filename, 'w')
    def write(self, message):
        self.terminal.write(message)
        self.log.write(message); self.log.flush()
    def flush(self):
        self.terminal.flush(); self.log.flush()

timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
log_filename = f'excelformer_pr_evaluation_log_{timestamp}.txt'
sys.stdout = Logger(log_filename)

# ---------- MI JSON paths for each model type ----------
MI_JSON_PATHS = {
    'default': 'output/mi/android_security/full/mi_top25_catenc(1)_norm(quantile).json',
    'xgbfi': 'output/mi/android_security/full/xgboost_feature_importance_20250827_230123.json'
}

def load_excelformer_model(model_path, device='cpu'):
    """Load ExcelFormer model from checkpoint"""
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model not found: {model_path}")
    
    checkpoint = torch.load(model_path, map_location=device)
    
    # Check if this is a new-style checkpoint with full state
    if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
        # New-style checkpoint
        model_state_dict = checkpoint['model_state_dict']
        n_features = checkpoint.get('n_features', None)
        feature_names = checkpoint.get('feature_names', None)
        print(f"Loaded new-style checkpoint with {n_features} features")
        if feature_names:
            print(f"Feature names from checkpoint: {feature_names[:3]}...")
    else:
        # Old-style checkpoint (just state_dict)
        model_state_dict = checkpoint
        n_features = None
        feature_names = None
        print("Loaded old-style checkpoint")
    
    # Import ExcelFormer model
    from bin import ExcelFormer
    
    # Create model with appropriate parameters
    if n_features is None:
        n_features = 25
        print(f"Using default {n_features} features")
    
    model = ExcelFormer(
        d_numerical=n_features,
        d_out=2,  # Binary classification
        categories=None,  # No categorical features after encoding
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
    
    # Load the state dict
    model.load_state_dict(model_state_dict)
    model.eval()
    return model, n_features, feature_names

def plot_pr_subplot(ax, y_true, y_score, label, size, split_type, is_first_plot=False):
    from sklearn.metrics import precision_recall_curve, average_precision_score
    
    p, r, _ = precision_recall_curve(y_true, y_score)
    ap = average_precision_score(y_true, y_score)
    baseline = float(np.mean(y_true))  # positive prevalence

    # step plot is recommended for PR
    ax.step(r, p, where='post', linewidth=2, label=f'{label} (AP={ap:.4f})')
    
    # Only plot baseline once per subplot
    if is_first_plot:
        ax.hlines(baseline, 0, 1, linestyles='--', label=f'Baseline={baseline:.3f}')
        ax.set_xlim([0.0, 1.0])
        ax.set_ylim([0.0, 1.05])
        ax.set_xlabel('Recall')
        ax.set_ylabel('Precision')
        
        # Format size labels nicely
        if size == '10000':
            size_label = '10,000'
        elif size == '100000':
            size_label = '100,000'
        elif size == 'full':
            size_label = 'Full'
        else:
            size_label = str(size)
        
        ax.set_title(f'{size_label} Samples')
        ax.grid(True, alpha=0.3)
    
    ax.legend(loc='lower left')
    return p, r, ap, baseline

def predict_excelformer(model, X, device='cpu'):
    """Get predictions from ExcelFormer model"""
    model.eval()
    with torch.no_grad():
        # Convert numpy array to tensor, ensuring correct data type
        X_tensor = torch.tensor(X.astype(np.float32), device=device)
        
        # Process in smaller batches if dataset is large
        batch_size = 1000  # Adjust based on memory
        if len(X_tensor) > batch_size:
            print(f"Processing {len(X_tensor)} samples in batches of {batch_size}")
            predictions = []
            for i in range(0, len(X_tensor), batch_size):
                batch = X_tensor[i:i+batch_size]
                logits = model(batch, None)  # No categorical features
                probs = torch.softmax(logits, dim=1)
                predictions.append(probs[:, 1].cpu().numpy())
            return np.concatenate(predictions)
        else:
            logits = model(X_tensor, None)  # No categorical features
            probs = torch.softmax(logits, dim=1)
            return probs[:, 1].cpu().numpy()  # Return positive class probability

def check_model_exists(model_path):
    """Check if a model file exists and is accessible"""
    if os.path.exists(model_path):
        file_size = os.path.getsize(model_path)
        print(f"✅ Model found: {model_path} ({file_size / (1024*1024):.1f} MB)")
        return True
    else:
        print(f"❌ Model not found: {model_path}")
        return False

def find_available_models():
    """Find which models are actually available"""
    available = {}
    for model_type in ['default', 'xgbfi']:
        available[model_type] = {}
        for size in ['10000', '100000', 'full']:
            model_path = f'result/ExcelFormer/{model_type}/mixup(none)/android_security/42/{size}/pytorch_model.pt'
            if check_model_exists(model_path):
                available[model_type][size] = model_path
            else:
                available[model_type][size] = None
    return available

def main():
    print("=== EXCELFORMER PR EVALUATION WITH CORRECT PREPROCESSING ===")
    
    # output dirs
    pr_dump_dir = "/home/umflint.edu/koernerg/pr_dumps"
    os.makedirs(pr_dump_dir, exist_ok=True)
    os.makedirs("pr_plots", exist_ok=True)

    # ---------- Create matrix plot figures for each model type ----------
    # First check which models are available
    print("\n" + "="*60)
    print("CHECKING AVAILABLE MODELS")
    print("="*60)
    available_models = find_available_models()
    
    # Create single figure with two rows (one for each model type)
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    fig.suptitle('ExcelFormer Precision-Recall Curves', fontsize=16, fontweight='bold')
    
    # Row labels
    row_labels = ['Mutual Information', 'Feature Importance']
    
    for row_idx, model_type in enumerate(['default', 'xgbfi']):
        print(f"\n{'='*60}")
        print(f"EVALUATING {model_type.upper()} MODELS")
        print(f"{'='*60}")
        
        # Check if we have any models for this type
        if not any(available_models[model_type].values()):
            print(f"No models found for {model_type}, skipping...")
            continue
        
        # Get the correct MI JSON for this model type
        mi_json_path = MI_JSON_PATHS[model_type]
        if not os.path.exists(mi_json_path):
            print(f"❌ MI JSON not found: {mi_json_path}")
            continue
        print(f"Using MI JSON: {mi_json_path}")
        
        # Store results for JSON dumps
        all_results = {}

        # ---------- Evaluate each size ----------
        for i, size in enumerate(['10000', '100000', 'full']):
            print(f"\n== Size: {size} ==")
            model_path = available_models[model_type][size]
            
            if model_path is None:
                print(f"No model available for {size}")
                # Create empty subplot
                axes[row_idx, i].text(0.5, 0.5, f'No model\n{size}', 
                                   ha='center', va='center', transform=axes[row_idx, i].transAxes)
                continue
                
            print(f"Loading model: {model_path}")
            
            try:
                # Load the model
                device = 'cpu'  # Use CPU for evaluation
                model, n_features, feature_names = load_excelformer_model(model_path, device)
                
                # Load dataset using the SAME pipeline as training
                print(f"Loading dataset with correct preprocessing...")
                transformation = Transformations(normalization='quantile')  # Same as training
                dataset = build_dataset(
                    DATA / 'android_security',
                    transformation,
                    cache=False,
                    sample_size=size,
                    indices_dir='./standardized_data',
                    mi_json_path=mi_json_path
                )
                
                print(f"Dataset loaded - Features: {dataset.n_features}, Val: {dataset.size('val')}, Test: {dataset.size('test')}")
                
                # Apply CatBoost encoding EXACTLY like training does
                if dataset.X_cat is not None:
                    print("Applying CatBoost encoding (train-only fit)...")
                    cardinalities = dataset.get_category_sizes('train')
                    enc = CatBoostEncoder(
                        cols=list(range(len(cardinalities))), 
                        return_df=False
                    ).fit(dataset.X_cat['train'], dataset.y['train'])  # Fit on TRAIN ONLY
                    
                    # Process all splits
                    X_num_processed = {}
                    for k in ['train', 'val', 'test']:
                        encoded_cat = enc.transform(dataset.X_cat[k]).astype(np.float32)
                        X_num_processed[k] = np.concatenate([encoded_cat, dataset.X_num[k]], axis=1)
                    
                    print(f"CatBoost encoding complete - shapes: {X_num_processed['train'].shape}")
                else:
                    X_num_processed = dataset.X_num
                
                # CRITICAL: Apply feature selection from JSON (same as training)
                print("Applying feature selection from JSON...")
                with open(mi_json_path, 'r') as f:
                    mi_data = json.load(f)
                
                selected_feature_names = mi_data['selected_names']
                print(f"Selected features from JSON: {len(selected_feature_names)}")
                
                # Get feature order from dataset
                if dataset.X_cat is not None:
                    all_feature_names = (dataset.cat_feature_names or []) + (dataset.num_feature_names or [])
                else:
                    all_feature_names = dataset.num_feature_names or []
                
                # Find indices of selected features
                keep_idx = []
                missing_features = []
                for feature_name in selected_feature_names:
                    if feature_name in all_feature_names:
                        keep_idx.append(all_feature_names.index(feature_name))
                    else:
                        missing_features.append(feature_name)
                
                if missing_features:
                    raise ValueError(f"Features from JSON not found in dataset: {missing_features}")
                
                # Slice the data using the found indices
                X_num_processed = {k: v[:, keep_idx] for k, v in X_num_processed.items()}
                print(f"Feature selection complete - final shapes: {X_num_processed['train'].shape}")
                
                # Verify dimensions match checkpoint
                assert X_num_processed['val'].shape[1] == n_features, f"Feature mismatch: {X_num_processed['val'].shape[1]} != {n_features}"
                print(f"✅ Feature dimensions verified: {X_num_processed['val'].shape[1]} == {n_features}")
                
                # Get validation and test data
                X_val = X_num_processed['val']
                y_val = dataset.y['val']
                X_test = X_num_processed['test']
                y_test = dataset.y['test']
                
                print(f"Data shapes - Val: {X_val.shape}, Test: {X_test.shape}")
                print(f"Class balance - Val: {y_val.mean():.3f}, Test: {y_test.mean():.3f}")
                
                # Verify data quality
                assert np.isfinite(X_val).all(), "Non-finite values in validation data"
                assert np.isfinite(X_test).all(), "Non-finite values in test data"
                print("✅ Data quality verified (all finite values)")

                # VAL (first plot in subplot - sets up axes, baseline, etc.)
                print(f"Evaluating validation set ({len(X_val)} samples)...")
                y_val_proba = predict_excelformer(model, X_val, device)
                val_p, val_r, val_ap, val_base = plot_pr_subplot(
                    axes[row_idx, i], y_val, y_val_proba, 'Validation', size, 'Validation', is_first_plot=True
                )
                print(f"[VAL] AP={val_ap:.4f}, baseline={val_base:.4f}")

                # TEST (second plot in same subplot - just adds the curve)
                print(f"Evaluating test set ({len(X_test)} samples)...")
                y_test_proba = predict_excelformer(model, X_test, device)
                test_p, test_r, test_ap, test_base = plot_pr_subplot(
                    axes[row_idx, i], y_test, y_test_proba, 'Test', size, 'Test', is_first_plot=False
                )
                print(f"[TEST] AP={test_ap:.4f}, baseline={test_base:.4f}")

                # Set y-axis label based on row (model type)
                if row_idx == 0:  # Mutual Information row
                    axes[row_idx, i].set_ylabel('Mutual Information Precision')
                else:  # Feature Importance row
                    axes[row_idx, i].set_ylabel('Feature Importance Precision')

                # Store results for JSON dumps
                all_results[size] = {
                    "val": {
                        "precision": [float(x) for x in val_p],
                        "recall": [float(x) for x in val_r],
                        "average_precision": float(val_ap),
                        "baseline_prevalence": float(val_base)
                    },
                    "test": {
                        "precision": [float(x) for x in test_p],
                        "recall": [float(x) for x in test_r],
                        "average_precision": float(test_ap),
                        "baseline_prevalence": float(test_base)
                    }
                }
                
                # Clean up memory
                del model, X_val, X_test, y_val, y_test, dataset
                if device == 'cuda':
                    torch.cuda.empty_cache()
                
            except Exception as e:
                print(f"Error evaluating {model_type} {size}: {e}")
                import traceback
                traceback.print_exc()
                # Create error subplot
                axes[row_idx, i].text(0.5, 0.5, f'Error\n{size}\n{str(e)[:30]}...', 
                                   ha='center', va='center', transform=axes[row_idx, i].transAxes)

        # ---------- Save JSON dumps for this model type ----------
        for size, results in all_results.items():
            out_json = os.path.join(pr_dump_dir, f"excelformer_{model_type}_pr_{size}.json")
            payload = {
                "model": f"excelformer_{model_type}",
                "size": str(size),
                **results
            }
            with open(out_json, "w") as f:
                json.dump(payload, f)
            print(f"[{model_type.upper()}] Wrote PR dump to {out_json}")

    # ---------- Save the combined matrix plot ----------
    plt.tight_layout()
    matrix_plot_path = f'pr_plots/excelformer_combined_pr_matrix_{timestamp}.png'
    plt.savefig(matrix_plot_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"\n[PLOT] Saved combined matrix plot to: {matrix_plot_path}")

    # restore stdout
    sys.stdout = sys.stdout.terminal
    print(f"ExcelFormer PR evaluation log saved to: {log_filename}")

if __name__ == "__main__":
    main()
