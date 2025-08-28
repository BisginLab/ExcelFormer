# standardized_data/scripts/derive_features_mi.py
import json, hashlib, argparse
from pathlib import Path
import numpy as np, pandas as pd
from sklearn.preprocessing import OrdinalEncoder
from sklearn.feature_selection import mutual_info_classif

def sha1_bytes(b: bytes) -> str:
    h=hashlib.sha1(); h.update(b); return h.hexdigest()

def file_sha1(p: Path) -> str:
    return sha1_bytes(p.read_bytes())

def df_sha1(df: pd.DataFrame) -> str:
    # stable CSV for hashing (no index, sorted columns)
    csv = df.sort_index(axis=1).to_csv(index=False).encode("utf-8")
    return sha1_bytes(csv)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="standardized_data/corrected_permacts.csv")
    ap.add_argument("--indices_dir", default="standardized_data")  # use FULL train for MI
    ap.add_argument("--out", default="standardized_data/meta/features_v1.json")
    ap.add_argument("--random_state", type=int, default=42)
    ap.add_argument("--n_neighbors", type=int, default=3)
    args = ap.parse_args()

    data_path = Path(args.data)
    idx_dir = Path(args.indices_dir)
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    # load and clean exactly like training
    df = pd.read_csv(data_path)
    df = df.dropna()
    if "Unnamed: 0" in df.columns: df = df.drop(columns=["Unnamed: 0"])
    if "pkgname" in df.columns: df = df.drop(columns=["pkgname"])

    # candidate features = all non-target columns
    target = "status"
    features_all = [c for c in df.columns if c != target]
    # split: MI ONLY on TRAIN of FULL
    train_idx = np.load(idx_dir/"train_indices_full.npy")
    X_train = df.loc[train_idx, features_all].copy()
    y_train = df.loc[train_idx, target].to_numpy()

    # identify categoricals
    cat_cols = [c for c in X_train.columns if X_train[c].dtype == "object"]
    num_cols = [c for c in X_train.columns if c not in cat_cols]

    # ordinal-encode categoricals (keeps “discrete_features” semantics)
    if cat_cols:
        enc = OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1)
        X_train[cat_cols] = enc.fit_transform(X_train[cat_cols])

    X_np = X_train.to_numpy()
    discrete_mask = np.isin(X_train.columns, cat_cols)

    mi = mutual_info_classif(
        X_np, y_train,
        discrete_features=discrete_mask,
        random_state=args.random_state,
        n_neighbors=args.n_neighbors
    )
    # rank (tie-break lexicographically for determinism)
    ranked = sorted(zip(X_train.columns.tolist(), mi),
                    key=lambda x: (-x[1], x[0]))
    full_ranking = [f for f,_ in ranked]
    top_25 = full_ranking[:25]

    # checksums
    dataset_checksum = df_sha1(df)
    split_sha = {
        "train_sha1": file_sha1(idx_dir/"train_indices_full.npy"),
        "val_sha1": file_sha1(idx_dir/"val_indices_full.npy"),
        "test_sha1": file_sha1(idx_dir/"test_indices_full.npy"),
    }

    meta = {
        "method": "mutual_info_classif",
        "random_state": args.random_state,
        "n_neighbors": args.n_neighbors,
        "using_split": "full/train",
        "dataset_checksum": dataset_checksum,
        "splits_sha1": {"full": split_sha},
        "categorical_features": cat_cols,
        "numerical_features": num_cols,
        "full_ranking": full_ranking,
        "canonical_feature_order": top_25,
    }
    # embed content SHA
    tmp = json.dumps(meta, sort_keys=True).encode("utf-8")
    meta["sha1"] = sha1_bytes(tmp)

    out_path.write_text(json.dumps(meta, indent=2))
    print(f"Wrote {out_path} with sha1={meta['sha1']}")
    print("Top-25:", top_25)

if __name__ == "__main__":
    main()
