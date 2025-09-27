# ExcelFormer

This repository will include the original implementation and experiment codes of [**ExcelFormer**](https://arxiv.org/abs/2301.02819). TabFormer is a pioneering neural network can surpass extensively-tuned XGboost, Catboost, and most tuned previous deep learning approaches on most of tabular data prediction tasks, in the supervised learning manner. It can be a go-to choice on tabualr dataset prediction competitions (e.g., Kaggle).

Even without hyper-parameter tuning, TabFormer performs comparable to tuned models. After hyper-parameter tuning, TabFormer typically outperforms them.

The implementation of TabFormer in the original paper is `bin/excel_former.py`.


## How to test your model

You can test your models by adding them to `bin` directory and `bin/__init__.py`. Keep the same API we used in other models, and write your own evaluation script (`run_default_config_excel.py` as a reference).

## Datasets:

The datasets (96 small tabular datasets + 21 large tabular datasets) are available at: https://huggingface.co/datasets/jyansir/excelformer.

## Future work

We will organize our previous works on **tabular prediction** into [Tabular AI Research](https://github.com/pytabular-ai) group for industrial use (e.g. further architecture optimization or acceleration / compilation). If you want to include our model as a baseline in your paper, please use the version in this repository rather than the industrial one in the group repository.

Run training script with

'''python
python -u run_default_config_excel.py --dataset android_security --normalization quantile --seed 42 --early_stop 20 --mix_type none --save --catenc 2>&1 | tee training_log_$(date +%Y%m%d_%H%M%S).txt
'''

## Averaging MI-25 and FI-25 for compute tables

When you have two runs per model (ExcelFormer, XGBoost)—one trained with MI-25 and one with FI-25—aggregate compute metrics by averaging across the two feature regimes so the table shows a single line per {model, sample_size, device}.

### Example runs for ExcelFormer:

```bash
# ExcelFormer (CPU fair-comparison), per size:
python profile_excelformer.py \
  --dataset android_security \
  --sample_size 10000 \
  --indices_dir ./standardized_data \
  --normalization quantile \
  --device cpu \
  --ckpt result/ExcelFormer/default/mixup(none)/android_security/42/10000/pytorch_model.pt \
  --mi_json output/mi/android_security/full/mi_top25_catenc(1)_norm(quantile).json \
  --feature_set MI-25

python profile_excelformer.py \
  --dataset android_security \
  --sample_size 10000 \
  --indices_dir ./standardized_data \
  --normalization quantile \
  --device cpu \
  --ckpt result/ExcelFormer/xgbfi/mixup(none)/android_security/42/10000/pytorch_model.pt \
  --mi_json output/mi/android_security/full/xgboost_feature_importance_20250827_230123.json \
  --feature_set FI-25
```

### Example runs for XGBoost:

```bash
python profile_xgboost.py \
  --models '{"10000":"/home/umflint.edu/koernerg/xgboost/saved_models/xgboost_ensemble_standardized_10000_run_20250825_160615.joblib"}' \
  --df_path ./content/sample_data/corrected_permacts.csv \
  --indices_dir ./standardized_data \
  --device cpu \
  --single_model \
  --feature_set MI-25

python profile_xgboost.py \
  --models '{"10000":"saved_models/xgboost_ensemble_fi_features_10000_run_20250828_061856.joblib"}' \
  --df_path ./content/sample_data/corrected_permacts.csv \
  --indices_dir ./standardized_data \
  --device cpu \
  --single_model \
  --feature_set FI-25
```

### Then aggregate:

```bash
python scripts/make_compute_table_avg.py
```

**Outputs:**
- `compute_profiles/compute_profiles_avg.csv`
- `compute_profiles/compute_profiles_avg.tex`