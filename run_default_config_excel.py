print("Entered training script...")
'''
USAGE EXAMPLES:
---------------

Train with MI top 25 features (Mutual Information):
python run_default_config_excel.py   --dataset android_security   --output /workspace/results/excelformer/mi-25   --seed 42   --early_stop 20   --save   --catenc   --sample_size 10000   --features mi-25
python run_default_config_excel.py   --dataset android_security   --output /workspace/results/excelformer/mi-25   --seed 42   --early_stop 20   --save   --catenc   --sample_size 100000   --features mi-25
python run_default_config_excel.py   --dataset android_security   --output /workspace/results/excelformer/mi-25   --seed 42   --early_stop 20   --save   --catenc   --sample_size full   --features mi-25

Train with FI top 25 features (XGBoost Feature Importance):
python run_default_config_excel.py   --dataset android_security   --output /workspace/results/excelformer/fi-25   --seed 42   --early_stop 20   --save   --catenc   --sample_size 10000   --features fi-25
python run_default_config_excel.py   --dataset android_security   --output /workspace/results/excelformer/fi-25   --seed 42   --early_stop 20   --save   --catenc   --sample_size 100000   --features fi-25
python run_default_config_excel.py   --dataset android_security   --output /workspace/results/excelformer/fi-25   --seed 42   --early_stop 20   --save   --catenc   --sample_size full   --features fi-25

FEATURE MODES:
--------------
- mi-25: Top 25 features selected by Mutual Information
         (uses: ../../data/feature_regimes/mi_top25_catenc(1)_norm(quantile).json)
- fi-25: Top 25 features selected by XGBoost Feature Importance
         (uses: ../../data/feature_regimes/xgboost_feature_importance_20250827_230123.json)

OUTPUT:
-------
All outputs are saved with the feature mode in the directory structure:
- Models: /workspace/results/excelformer/{mi-25|fi-25}/mixup(none)/android_security/42/{sample_size}/pytorch_model.pt
- Results: /workspace/results/excelformer/{mi-25|fi-25}/mixup(none)/android_security/42/{sample_size}/results.json
- Logs: /workspace/results/excelformer/logs/excelformer_{features}_{sample_size}_training_log_{timestamp}.txt

NOTE: The --features argument is REQUIRED. Run this script from the models/excelformer directory.
'''
# Go to great lakes and test best model on test set.
# Figure out how to properly evaluate xgboost and excelformer. check paper - 70s%?
# push code to lab github
# draft email response to VT, post to slack
# retrain xgboost with dropped features put in slack results. does mutual information ranking coincide with xgboost feature importance ranking? 

# distribution difference between columns in corrected_permacts and earlier dataset
# confirm names havent changed
# numeric: averages, categorical: count
# make figures
# old dataset is with original paper
# we need to be 100% sure about making the training replicable - so be clear about all the input parameters that 
# lead to the results
# we should do a diff with original excelformer repo to justify every change we made
# we should also walk through a checklist of loading, preprocessing, dropping features, encoding them,
# then the main script's MI score and feature selection, ensure that model is trained with correct input
# we should possibly do loss graphs
# selected_features object trace - does the model load the correct features? Yes, it does.
import sys
import os
import math
import time
import json
import random
import argparse
import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset
from torch.optim.lr_scheduler import CosineAnnealingLR
from sklearn.feature_selection import mutual_info_classif, mutual_info_regression
from category_encoders import CatBoostEncoder

from bin import ExcelFormer
from lib import Transformations, build_dataset, prepare_tensors, make_optimizer, DATA
from datetime import datetime

# Logger class for saving training output to file
class Logger:
    def __init__(self, filename):
        self.terminal = sys.stdout
        self.log = open(filename, 'w')

    def write(self, message):
        self.terminal.write(message)
        self.log.write(message)
        self.log.flush()

    def flush(self):
        self.terminal.flush()
        self.log.flush()

# Diagnostic information
print("Python executable:", sys.executable)
print("Python path:", sys.path)
print("LD_LIBRARY_PATH:", os.environ.get('LD_LIBRARY_PATH', 'Not set'))
print("CUDA_HOME:", os.environ.get('CUDA_HOME', 'Not set'))

# Torch and CUDA details
print("Torch version:", torch.__version__)
print("Torch CUDA version:", torch.version.cuda)
print("CUDA available:", torch.cuda.is_available())
print("CUDA device count:", torch.cuda.device_count())

# Check for CUDA devices
import subprocess
try:
    nvidia_smi_output = subprocess.check_output(['nvidia-smi']).decode('utf-8')
    print("nvidia-smi output:\n", nvidia_smi_output)
except Exception as e:
    print("Error running nvidia-smi:", e)

print("Passed imports...")
print("CUDA available:", torch.cuda.is_available())
print("CUDA device count:", torch.cuda.device_count())
if torch.cuda.is_available():
    print("Current CUDA device:", torch.cuda.current_device())
    print("CUDA device name:", torch.cuda.get_device_name(0))


DATASETS = [
    'analcatdata_supreme', 'isolet', 'cpu_act', 'visualizing_soil', 'yprop_4_1', 'gesture', 'churn', 'sulfur', 'bank-marketing', 'Brazilian_houses'
    'eye', 'MagicTelescope', 'Ailerons', 'pol', 'polv2', 'credit', 'california', 'house_sales', 'house', 'diamonds', 'helena', 'jannis', 'higgs-small',
    'road-safety', 'medical_charges', 'SGEMM_GPU_kernel_performance', 'covtype', 'nyc-taxi-green-dec-2016', 'android_security'
]

# === XGBoost feature list, in order ===
XGBOOST_FEATURES = [
    'ContentRating', 'Genre', 'CurrentVersion', 'AndroidVersion', 'DeveloperCategory',
    'lowest_android_version', 'highest_android_version', 'privacy_policy_link',
    'developer_website', 'days_since_last_update', 'isSpamming', 'max_downloads_log',
    'LenWhatsNew', 'PHONE', 'OneStarRatings', 'developer_address', 'FourStarRatings',
    'intent', 'ReviewsAverage', 'STORAGE', 'LastUpdated', 'TwoStarRatings',
    'LOCATION', 'FiveStarRatings', 'ThreeStarRatings'
]

def get_training_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=str, default='/workspace/results/excelformer/mi-25')
    parser.add_argument("--dataset", type=str)
    parser.add_argument("--normalization", type=str, default='quantile')
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--early_stop", type=int, default=20)
    parser.add_argument("--min_delta", type=float, default=0.001)
    parser.add_argument("--beta", type=float, default=0.5, help='hyper-parameter of Beta Distribution in mixup, we choose 0.5 for all datasets in default config')
    parser.add_argument("--mix_type", type=str, default='none', choices=['niave_mix', 'feat_mix', 'hidden_mix', 'none'], help='mixup type, set to "niave_mix" for naive mixup, set to "none" if no mixup')
    parser.add_argument("--save", action='store_true', help='whether to save model')
    parser.add_argument("--catenc", action='store_true', help='whether to use catboost encoder for categorical features')
    parser.add_argument("--resume", type=str, default=None, help='path to checkpoint to resume from')
    parser.add_argument("--sample_size", type=str, choices=['10000', '50000', '100000', 'full'], default=None,
                        help="Subset size for training/validation/test (matches XGBoost splits). Use 'full' for full dataset.")
    parser.add_argument("--features", type=str, choices=['mi-25', 'fi-25'], required=True,
                        help="Feature set to use: mi-25 (Mutual Information top 25), fi-25 (Feature Importance top 25)")
    args = parser.parse_args()
    
    # Map feature selection to JSON paths
    FEATURE_JSON_PATHS = {
        'mi-25': '../../data/feature_regimes/mi_top25_catenc(1)_norm(quantile).json',
        'fi-25': '../../data/feature_regimes/xgboost_feature_importance_20250827_230123.json'
    }
    args.feature_json_path = FEATURE_JSON_PATHS[args.features]

    args.output = f'{args.output}/mixup({args.mix_type})/{args.dataset}/{args.seed}'
    if not os.path.isdir(args.output):
        os.makedirs(args.output, exist_ok=True)
    if args.sample_size is not None:
        args.output += f"/{args.sample_size}"
        if not os.path.isdir(args.output):
            os.makedirs(args.output, exist_ok=True)
    # Note: Full dataset models are saved directly in the seed directory
    # some basic model configuration
    cfg = {
        "model": {
            "prenormalization": True, # true or false, perform BETTER on a few datasets with no prenormalization 

            'kv_compression': None,
            'kv_compression_sharing': None,
            'token_bias': True
        },
        "training": {
            "max_epoch": 500,
            "optimizer": "adamw",
        }
    }
    
    return args, cfg

def record_exp(args, final_score, best_score, **kwargs):
    # 'best': the best test score during running
    # 'final': the final test score acquired by validation set
    results = {'config': args, 'final': final_score, 'best': best_score, **kwargs}
    with open(f"{args['output']}/results.json", 'w', encoding='utf8') as f:
        json.dump(results, f, ensure_ascii=False, indent=4)

def seed_everything(seed=42):
    '''
    Sets the seed of the entire notebook so results are the same every time we run.
    This is for REPRODUCIBILITY.
    '''
    random.seed(seed)
    # Set a fixed value for the hash seed
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        # When running on the CuDNN backend, two further options must be set
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


"""args"""
# device = torch.device('cuda')

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

args, cfg = get_training_args()

# Set up logging to file
timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
os.makedirs('/workspace/results/excelformer/logs', exist_ok=True)
log_filename = f'/workspace/results/excelformer/logs/excelformer_{args.features}_{args.sample_size if args.sample_size else "full"}_training_log_{timestamp}.txt'
sys.stdout = Logger(log_filename)

print("="*60)
print(f"EXCELFORMER TRAINING SCRIPT - FEATURE MODE: {args.features.upper()}")
print("="*60)
print(f"Selected feature mode: {args.features}")
print(f"Sample size: {args.sample_size if args.sample_size else 'full'}")
print(f"Output directory: {args.output}")
print(f"Log file: {log_filename}")
print("="*60)

seed_everything(args.seed)

""" prepare Datasets and Dataloaders """
assert args.dataset in DATASETS
T_cache = False # save data preprocessing cache
normalization = args.normalization if args.normalization != '__none__' else None
transformation = Transformations(normalization=normalization)

# Load dataset using standardized preprocessing
# NOTE: All data loading, cleaning, and index application happens in lib/data.py
# This includes loading cleaned_data.pkl and applying pre-computed indices
indices_dir = '../../data/splits'  # Path to standardized data created by master_preprocessing.py
print(f"\n{'='*60}")
print(f"LOADING STANDARDIZED DATA")
print(f"{'='*60}")
print(f"Data directory: {indices_dir}")
print(f"Feature set: {args.features}")
print(f"Sample size: {args.sample_size if args.sample_size else 'full'}")
print(f"Feature JSON: {args.feature_json_path}")
print(f"\nData loading (lib/data.py):")
print(f"  1. Loads cleaned_data.pkl with ALL features")
print(f"  2. Applies pre-computed train/val/test indices")
print(f"\nFeature selection (this script, AFTER encoding):")
print(f"  3. CatBoost encodes categorical features")
print(f"  4. Selects features from JSON after encoding")
print(f"{'='*60}\n")

dataset = build_dataset(
    DATA / args.dataset, transformation, T_cache,
    sample_size=args.sample_size, 
    indices_dir=indices_dir,
    feature_json_path=args.feature_json_path
)

if dataset.X_num['train'].dtype == np.float64:
    dataset.X_num = {k: v.astype(np.float32) for k, v in dataset.X_num.items()}
# convert categorical features to numerical features with CatBoostEncoder
if args.catenc and dataset.X_cat is not None:
    cardinalities = dataset.get_category_sizes('train')
    enc = CatBoostEncoder(
        cols=list(range(len(cardinalities))), 
        return_df=False
    ).fit(dataset.X_cat['train'], dataset.y['train'])
    # Use new variables for encoded+concatenated features
    X_num_processed = {}
    for k in ['train', 'val', 'test']:
        encoded_cat = enc.transform(dataset.X_cat[k]).astype(np.float32)
        X_num_processed[k] = np.concatenate([encoded_cat, dataset.X_num[k]], axis=1)
    print("Shape of X_num_train after concat:", X_num_processed['train'].shape)
else:
    X_num_processed = dataset.X_num  # Use as-is if no catenc

#############################
# FEATURE SELECTION (AFTER CATBOOST ENCODING)
#############################
# NOTE: This is where actual feature selection happens!
# data.py loaded ALL features - we select here AFTER encoding
# because CatBoost encoding changes the feature structure

print(f"[FEATURE SELECTION] Using feature mode: {args.features}")
print(f"[FEATURE SELECTION] Loading features from JSON: {args.feature_json_path}")

if not os.path.exists(args.feature_json_path):
    raise FileNotFoundError(f"Feature JSON file not found: {args.feature_json_path}")

# Load pre-computed feature selection from JSON
with open(args.feature_json_path, 'r') as f:
    feature_data = json.load(f)

# Extract selected features by name
selected_feature_names = feature_data['selected_names']
K = len(selected_feature_names)

print(f"[FEATURE SELECTION] JSON selected features: {selected_feature_names}")

# Get the full feature order from the dataset (after CatBoost encoding if applicable)
if args.catenc and dataset.X_cat is not None:
    cat_feature_names = dataset.cat_feature_names or []
    num_feature_names = dataset.num_feature_names or []
    all_feature_names = cat_feature_names + num_feature_names
else:
    all_feature_names = dataset.num_feature_names or []

print(f"[FEATURE SELECTION] Dataset feature order: {all_feature_names}")

# Find indices of selected features in the current feature order
keep_idx = []
missing_features = []
for feature_name in selected_feature_names:
    if feature_name in all_feature_names:
        keep_idx.append(all_feature_names.index(feature_name))
    else:
        missing_features.append(feature_name)

if missing_features:
    raise ValueError(f"Features from JSON not found in dataset: {missing_features}")

if len(keep_idx) != K:
    raise ValueError(f"Expected {K} features from JSON, but only found {len(keep_idx)}")

print(f"[FEATURE SELECTION] Found indices: {keep_idx}")
print(f"[FEATURE SELECTION] Reordered features: {[all_feature_names[i] for i in keep_idx]}")

# Slice the data using the found indices
X_num_processed = {k: v[:, keep_idx] for k, v in X_num_processed.items()}

# Get feature importance scores for the selected features (for feat_mix if needed)
# NOTE: Both MI-25 and FI-25 JSONs have 'mi_scores_aligned' key for compatibility
feature_scores = np.array(feature_data['mi_scores_aligned'])
selected_feature_scores = feature_scores[keep_idx]

print(f"[FEATURE SELECTION] Loaded {K} features from JSON. Shapes:",
      {k: v.shape for k, v in X_num_processed.items()})
print(f"[FEATURE SELECTION] Selected feature names: {selected_feature_names}")
print(f"[FEATURE SELECTION] Found indices: {keep_idx}")
print(f"[FEATURE SELECTION] Using pre-computed feature scores from JSON")

# Normalized feature scores for feat_mix (stable if sum == 0)
den = selected_feature_scores.sum()
sorted_feature_scores_np = (selected_feature_scores / den).astype(np.float32) if den > 0 else np.ones(K, np.float32) / K
sorted_feature_scores = torch.from_numpy(sorted_feature_scores_np).float().to(device)

# Now build tensors from the sliced matrices
X_num, X_cat, ys = prepare_tensors(
    # Use processed features
    type('DatasetObj', (), {
        'X_num': X_num_processed,
        'X_cat': None if args.catenc else dataset.X_cat,
        'y': dataset.y,
        'n_classes': dataset.n_classes,
        'is_binclass': dataset.is_binclass,
        'is_multiclass': dataset.is_multiclass,
        'is_regression': dataset.is_regression,
        'calculate_metrics': dataset.calculate_metrics,
        'n_features': X_num_processed['train'].shape[1],
        'num_feature_names': dataset.num_feature_names,
        'cat_feature_names': dataset.cat_feature_names,
        'get_category_sizes': dataset.get_category_sizes,
    })(),
    device=device
)
print(f"X_num shape after prepare_tensors: {X_num['train'].shape}")
if args.catenc:
    X_cat = None

# Set d_out and d_numerical AFTER slicing
n_num_features = X_num['train'].shape[1]
if dataset.is_binclass:
    d_out = 2
elif dataset.is_multiclass:
    d_out = dataset.n_classes
else:
    d_out = 1
print(f"n_num_features (top-{K}): {n_num_features}")

print("\nStarting training...")

# Print what features we're actually training with
print(f"\n=== FEATURES USED FOR TRAINING ===")
print(f"Feature set: {args.features}")
print(f"Total features: {n_num_features}")
print(f"X_num train shape: {X_num['train'].shape}")
print(f"X_num val shape: {X_num['val'].shape}")
print(f"X_num test shape: {X_num['test'].shape}")
print(f"These are the selected {n_num_features} features:")
for i, feature_name in enumerate(selected_feature_names, 1):
    print(f"  {i:2d}. {feature_name}")
print(f"=== END FEATURE INFO ===\n")





# After MI selection but before model creation (around line 213)
# print("\n=== Final Feature List for Training ===")
# print(f"Total selected features: {len(mi_ranks)}")

# Get feature names in order
# all_features = (dataset.cat_feature_names or []) + (dataset.num_feature_names or [])
# selected_features = [all_features[i] for i in mi_ranks]

# Count feature types
# num_features = len([f for f in selected_features if f in dataset.num_feature_names]) if dataset.num_feature_names else 0
# cat_features = len([f for f in selected_features if f in dataset.cat_feature_names]) if dataset.cat_feature_names else 0

# print("\nFeature counts:")
# print(f"Total selected features: {len(selected_features)}")
# print(f"Numerical features: {num_features}")
# print(f"Categorical features: {cat_features}")

# print("\nFeatures in order:")
# for i, feature in enumerate(selected_features, 1):
#     print(f"{i}. {feature}")

print("\nStarting training...")

# Remove the sys.exit() line if you want to continue with training
# sys.exit()  # Comment this out to proceed with training

# set batch size
batch_size_dict = {
    'churn': 128, 'eye': 128, 'gesture': 128, 'california': 256, 'house': 256,
    'higgs-small': 512, 'helena': 512, 'jannis': 512, 'covtype': 1024
} # batch size settings for datasets in FT-Transformer(Borisov et al., 2021)
if args.dataset in batch_size_dict:
    batch_size = batch_size_dict[args.dataset]
    val_batch_size = 512
else:
    # batch size settings for datasets in (Grinsztajn et al., 2022)
    if dataset.n_features <= 32:
        batch_size = 512
        val_batch_size = 8192
    elif dataset.n_features <= 100:
        batch_size = 128
        val_batch_size = 512
    elif dataset.n_features <= 1000:
        batch_size = 32
        val_batch_size = 64
    else:
        batch_size = 16
        val_batch_size = 16

# update training config
cfg['training'].update({
    "batch_size": batch_size, 
    "eval_batch_size": val_batch_size, 
    "patience": args.early_stop
})

# data loaders
print(f"X_num shape in data loaders: {X_num['train'].shape}")
data_list = [X_num, ys] if X_cat is None else [X_num, X_cat, ys]
print(f"data_list contents and shapes: {[{k: v.shape for k, v in d.items()} for d in data_list]}")
train_dataset = TensorDataset(*(d['train'] for d in data_list))
train_loader = DataLoader(
    dataset=train_dataset,
    batch_size=batch_size,
    shuffle=True,
)
val_dataset = TensorDataset(*(d['val'] for d in data_list))
val_loader = DataLoader(
    dataset=val_dataset,
    batch_size=val_batch_size,
    shuffle=False,
)
test_dataset = TensorDataset(*(d['test'] for d in data_list))
test_loader = DataLoader(
    dataset=test_dataset,
    batch_size=val_batch_size,
    shuffle=False,
)
dataloaders = {'train': train_loader, 'val': val_loader, 'test': test_loader}

""" Prepare Model """
# datset specific params
cardinalities = dataset.get_category_sizes('train')
n_categories = len(cardinalities)
if args.catenc:
    n_categories = 0 # all categorical features are converted to numerical ones
cardinalities = None if n_categories == 0 else cardinalities # drop category features

""" All default configs: model and training hyper-parameters """
# kwargs: model configs
kwargs = {
    'd_numerical': n_num_features,
    'd_out': d_out,
    'categories': cardinalities,
    **cfg['model']
}
default_model_configs = {
    'ffn_dropout': 0., 'attention_dropout': 0.3, 'residual_dropout': 0.0,
    'n_layers': 3, 'n_heads': 32, 'd_token': 256,
    'init_scale': 0.01, # param for the Attenuated Initialization
}
default_training_configs = {
    'lr': 1e-3,  # Much lower than current 2e-3
    'weight_decay': 1e-5,
}
kwargs.update(default_model_configs) # update model configs
cfg['training'].update(default_training_configs) # update training configs

# build model
model = ExcelFormer(**kwargs).to(device)

# optimizer
def needs_wd(name):
    return all(x not in name for x in ['tokenizer', '.norm', '.bias'])

parameters_with_wd = [v for k, v in model.named_parameters() if needs_wd(k)]
parameters_without_wd = [v for k, v in model.named_parameters() if not needs_wd(k)]
optimizer = make_optimizer(
    cfg['training']['optimizer'],
    (
        [
            {'params': parameters_with_wd},
            {'params': parameters_without_wd, 'weight_decay': 0.0},
        ]
    ),
    cfg['training']['lr'],
    cfg['training']['weight_decay'],
)

# parallelization
if torch.cuda.device_count() > 1:
    print('Using nn.DataParallel')
    model = nn.DataParallel(model)

"""Loss Function"""
loss_fn = (
    F.binary_cross_entropy_with_logits
    if dataset.is_binclass
    else F.cross_entropy
    if dataset.is_multiclass
    else F.mse_loss
)

"""Utils Function"""
def apply_model(x_num, x_cat=None, mixup=False):
    # print(f"apply_model x_num shape: {x_num.shape}")
    if mixup:
        return model(x_num, x_cat, mixup=True, beta=args.beta, mtype=args.mix_type)
    return model(x_num, x_cat)

@torch.inference_mode()
def evaluate(parts):
    model.eval()
    predictions = {}
    for part in parts:
        assert part in ['train', 'val', 'test']
        infer_time = 0.
        predictions[part] = []
        for batch in dataloaders[part]:
            x_num, x_cat, y = (
                (batch[0], None, batch[1])
                if len(batch) == 2
                else batch
            )
            start = time.time()
            # print(f"x_num shape in evaluate: {x_num.shape}")
            predictions[part].append(apply_model(x_num, x_cat))
            infer_time += time.time() - start
        predictions[part] = torch.cat(predictions[part]).cpu().numpy()
        if part == 'test':
            print('test time: ', infer_time)
    prediction_type = None if dataset.is_regression else 'logits'
    return dataset.calculate_metrics(predictions, prediction_type)

class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


"""Training"""
# we use AUC for binary classification, Accuracy for multi-class classification, RMSE for regression
metric = 'roc_auc' if dataset.is_binclass else 'score'
init_score = evaluate(['test'])['test'][metric] # test before training
print(f'Test score before training: {init_score: .4f}')

losses, val_metric, test_metric = [], [], []
n_epochs = 500 # default max training epoch #try 10-50 first, save local model
# upload to group repo, maybe frank can help 

# warmup and lr scheduler
warm_up = 10  # Longer warmup than current 5
scheduler = CosineAnnealingLR(
    optimizer=optimizer, 
    T_max=n_epochs - warm_up,
    eta_min=0
)
max_lr = cfg['training']['lr']
# report_frequency = len(ys['train']) // batch_size // 3
report_frequency = len(train_loader)  # This will make it print only at the end of each epoch
# metric containers
loss_holder = AverageMeter()

# Initialize variables before checkpoint loading
running_time = 0.0
start_epoch = 1
best_score = -np.inf
final_test_score = -np.inf
best_test_score = -np.inf
no_improvement = 0
losses = []
val_metric = []
test_metric = []
old_style = False

# Load checkpoint if resuming
if args.resume is not None and os.path.exists(args.resume):
    print(f"Loading checkpoint from {args.resume}")
    checkpoint = torch.load(args.resume)
    
    # Add debug prints
    print(f"Checkpoint type: {type(checkpoint)}")
    if isinstance(checkpoint, dict):
        print(f"Checkpoint keys: {checkpoint.keys()}")
    
    # Check if this is a state_dict (old-style) or full checkpoint (new-style)
    if isinstance(checkpoint, dict) and 'epoch' in checkpoint:  # New-style has 'epoch' key
        # New-style checkpoint with full state
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        scheduler.load_state_dict(checkpoint.get('scheduler_state_dict'))
        start_epoch = checkpoint['epoch'] + 1
        best_score = checkpoint.get('best_score', -np.inf)
        best_test_score = checkpoint.get('best_test_score', -np.inf)
        final_test_score = checkpoint.get('final_test_score', -np.inf)
        no_improvement = checkpoint.get('no_improvement', 0)
        losses = checkpoint.get('losses', losses)
        val_metric = checkpoint.get('val_metric', val_metric)
        test_metric = checkpoint.get('test_metric', test_metric)
        running_time = checkpoint.get('running_time', 0.0)
        print(f"Resuming from epoch {start_epoch} with full state")
    else:
        # Old-style checkpoint is just the model's state_dict
        model.load_state_dict(checkpoint)
        old_style = True

        # Initialize other state variables
        best_score = -np.inf
        best_test_score = -np.inf
        final_test_score = -np.inf
        no_improvement = 0
        losses = []
        val_metric = []
        test_metric = []
        running_time = 0.0
        
        print(f"Resuming from old-style checkpoint (model weights only)")
        print(f"Approximated learning rate state for epoch {start_epoch}")
        print(f"Current learning rate: {optimizer.param_groups[0]['lr']}")


if old_style:
    # Get the current epoch from command line or environment variable
    # You might want to add this as an argument if not already present
    current_epoch = 50  # Replace with actual current epoch
    
    # Approximate the learning rate state
    if current_epoch <= warm_up:
        # If still in warmup, set appropriate warmup lr
        lr = max_lr * current_epoch / warm_up
        print(f"Approximating warmup learning rate: {lr}")
        for param_group in optimizer.param_groups:
            param_group['lr'] = lr
    else:
        # Fast-forward scheduler to current epoch
        print(f"Fast-forwarding scheduler for {current_epoch - warm_up} steps")
        for _ in range(current_epoch - warm_up):
            scheduler.step()
        
    start_epoch = current_epoch + 1
    
print(f"Starting training from epoch {start_epoch}")
for epoch in range(start_epoch, n_epochs + 1):
    model.train()
    epoch_start = time.time()
    
    # warm up lr
    if warm_up > 0 and epoch <= warm_up:
        lr = max_lr * epoch / warm_up
        # print(f'warm up ({epoch}/{warm_up})')
        for param_group in optimizer.param_groups:
            param_group['lr'] = lr
    else:
        scheduler.step()
    for iteration, batch in enumerate(train_loader):
        x_num, x_cat, y = (
            (batch[0], None, batch[1])
            if len(batch) == 2
            else batch
        )
        # print(f"x_num shape in training block: {x_num.shape}")

        start = time.time()
        optimizer.zero_grad()
        if args.mix_type == 'none': # no mixup
            preds = apply_model(x_num, x_cat)
            if dataset.is_binclass:
                loss = loss_fn(preds[:, 1], y.float())  # Only use logit for positive class
            else:
                loss = loss_fn(preds, y)
        else:
            preds, feat_masks, shuffled_ids = apply_model(x_num, x_cat, mixup=True)
            if args.mix_type == 'feat_mix':
                lambdas = (sorted_feature_scores * feat_masks).sum(1) # bs
                lambdas2 = 1 - lambdas
            elif args.mix_type == 'hidden_mix':
                lambdas = feat_masks
                lambdas2 = 1 - lambdas
            elif args.mix_type == 'niave_mix':
                lambdas = feat_masks
                lambdas2 = 1 - lambdas
            
            if dataset.is_regression:
                mix_y = lambdas * y + lambdas2 * y[shuffled_ids]
                loss = loss_fn(preds, mix_y)
            else:
                if dataset.is_binclass:
                    loss = lambdas * loss_fn(preds[:, 1], y.float(), reduction='none') + \
                           lambdas2 * loss_fn(preds[:, 1], y[shuffled_ids].float(), reduction='none')
                else:
                    loss = lambdas * loss_fn(preds, y, reduction='none') + \
                           lambdas2 * loss_fn(preds, y[shuffled_ids], reduction='none')
                loss = loss.mean()
        loss.backward()
        # Add gradient clipping before optimizer step
        max_grad_norm = 1.0  # Add this near other hyperparameters
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)
        optimizer.step()
        running_time += time.time() - start
        loss_holder.update(loss.item(), len(ys))
        # Add progress prints
        if iteration % report_frequency == 0:
            print(f'Epoch {epoch:03d} | Batch {iteration} | Loss: {loss_holder.val:.4f} | LR: {optimizer.param_groups[0]["lr"]:.6f}')

    # Print epoch summary
    epoch_time = time.time() - epoch_start
    print(f'\nEpoch {epoch:03d} Summary:')
    print(f'Average Loss: {loss_holder.avg:.4f}')
    print(f'Learning Rate: {optimizer.param_groups[0]["lr"]:.6f}')
    print(f'Time: {epoch_time:.2f}s')
    
    # Evaluate and print metrics
    scores = evaluate(['val', 'test'])
    val_score, test_score = scores['val'][metric], scores['test'][metric]
    print(f'Validation score: {val_score:.4f}')
    print(f'Test score: {test_score:.4f}')
    print('-' * 50)

    losses.append(loss_holder.avg)
    loss_holder.reset()
    val_metric.append(val_score), test_metric.append(test_score)
    if val_score > (best_score + args.min_delta):
        best_score = val_score
        final_test_score = test_score
        print(f' <<< BEST VALIDATION EPOCH: {val_score:.4f}')
        no_improvement = 0
        if args.save:
            checkpoint = {
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'scheduler_state_dict': scheduler.state_dict(),
                'best_score': best_score,
                'best_test_score': best_test_score,
                'final_test_score': final_test_score,
                'no_improvement': no_improvement,
                'losses': losses,
                'val_metric': val_metric,
                'test_metric': test_metric,
                'running_time': running_time,
                'n_features': n_num_features,
                'feature_names': selected_feature_names  # exact JSON order used for training
            }
            model_path = f"{args.output}/pytorch_model.pt"
            torch.save(checkpoint, model_path)
    else:
        no_improvement += 1
    if test_score > best_test_score:
        best_test_score = test_score

    if no_improvement == args.early_stop:
        break
        
"""Record Exp Results"""
record_exp(
    vars(args), final_test_score, best_test_score,
    losses=str(losses), val_score=str(val_metric), test_score=str(test_metric),
    cfg=cfg, time=running_time,
)

print("Features used for ExcelFormer training (should match XGBoost):")
print(dataset.num_feature_names)
print(dataset.cat_feature_names)

print(f"=== TRAINING DATASET SIZE: {args.sample_size if args.sample_size else 'Full'} ===")

# After all preprocessing, just before model creation

print("\n[DEBUG][TRAIN] Final feature names (cat + num):")
if args.catenc and dataset.X_cat is not None:
    print("Cat features (encoded):", dataset.cat_feature_names)
    print("Num features:", dataset.num_feature_names)
    print("Order for model input:", dataset.cat_feature_names + dataset.num_feature_names)
else:
    print("Num features:", dataset.num_feature_names)
    print("Order for model input:", dataset.num_feature_names)

print("[DEBUG][TRAIN] X_num['train'] shape:", X_num['train'].shape)
print("[DEBUG][TRAIN] X_num['val'] shape:", X_num['val'].shape)
print("[DEBUG][TRAIN] X_num['test'] shape:", X_num['test'].shape)

# Restore normal stdout and print final message
sys.stdout = sys.stdout.terminal
print(f"\n{'='*60}")
print(f"EXCELFORMER TRAINING COMPLETE")
print(f"{'='*60}")
print(f"✅ Training log saved to: {log_filename}")
print(f"✅ Feature mode: {args.features.upper()}")
print(f"✅ Sample size: {args.sample_size if args.sample_size else 'full'}")
print(f"✅ Model saved to: {args.output}/pytorch_model.pt")
print(f"✅ Results saved to: {args.output}/results.json")
print(f"{'='*60}")
