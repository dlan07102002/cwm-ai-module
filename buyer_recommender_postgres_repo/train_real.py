import argparse
import os
import joblib
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from typing import Dict, Any
from lightgbm import early_stopping, log_evaluation, LGBMRanker

from model_utils import build_features_from_candidates, build_user_profiles_proper

# --- Constants ---
DEFAULT_TRAIN_PATH = 'data/train.parquet'
DEFAULT_MODEL_PATH = 'model.joblib'
RANDOM_STATE = 42


# ==============================================================
# Load embeddings helper
# ==============================================================
def load_emb_map_from_parquet(path: str, id_col: str, emb_col: str) -> Dict[str, np.ndarray]:
    """
    Loads a parquet file with an ID column and embedding column (array-like or stringified list).
    Returns a dictionary mapping id -> np.ndarray
    """
    df = pd.read_parquet(path)
    mapping = {}

    for _, row in df.iterrows():
        try:
            if isinstance(row[emb_col], str):
                emb = np.array(eval(row[emb_col]), dtype=float)
            else:
                emb = np.array(row[emb_col], dtype=float)
            mapping[str(row[id_col])] = emb
        except Exception as e:
            print(f"[WARN] Failed to parse embedding for {row[id_col]}: {e}")
    print(f"[INFO] Loaded {len(mapping)} embeddings from {path}")
    return mapping


# ==============================================================
# Training pipeline
# ==============================================================
def run_training(
    parquet_path: str,
    model_out: str,
    test_size: float,
    lgbm_params: Dict[str, Any],
    num_boost_round: int,
    early_stopping_rounds: int
):
    print(f"[INFO] Loading data from {parquet_path}...")
    df = pd.read_parquet(parquet_path)
    print(f"[INFO] Loaded {len(df)} rows.")

    if 'matched' not in df.columns:
        raise ValueError("Training data must contain a 'matched' column (label).")

    # --- Split Train/Test by user to avoid leakage ---
    print("[INFO] Splitting users into train/test...")
    unique_users = df['user_id'].unique()
    train_users, test_users = train_test_split(unique_users, test_size=test_size, random_state=RANDOM_STATE)

    train_df = df[df['user_id'].isin(train_users)].reset_index(drop=True)
    test_df = df[df['user_id'].isin(test_users)].reset_index(drop=True)

    print(f"[INFO] Train users: {len(train_users)}, Test users: {len(test_users)}")
    print(f"[INFO] Train rows: {len(train_df)}, Test rows: {len(test_df)}")

    # --- Historical matches (only from train set) ---
    historical_matches = train_df[train_df['matched'] == 1].reset_index(drop=True)
    print(f"[INFO] Historical matches used for user profiles: {len(historical_matches)}")

    # --- Load optional user verify embeddings ---
    user_verify_emb_path = "data/user_verify_embeddings.parquet"
    user_verify_emb_map = {}

    if os.path.exists(user_verify_emb_path):
        print(f"[INFO] Loading user verification embeddings from {user_verify_emb_path}")
        user_verify_emb_map = load_emb_map_from_parquet(
            user_verify_emb_path,
            id_col="user_id",
            emb_col="user_verify_emb"
        )
    else:
        print("[WARN] No user_verify_embeddings file found; continuing without verify embeddings.")

    # --- Feature Engineering ---
    print("[INFO] Building features for train set...")
    user_profiles = build_user_profiles_proper(historical_matches, verify_emb_map=user_verify_emb_map)
    X_train = build_features_from_candidates(train_df, user_profiles=user_profiles)
    y_train = train_df['matched'].astype(int)
    user_train = train_df['user_id']

    print("[INFO] Building features for test set (profiles built from train historical matches only)...")
    X_test = build_features_from_candidates(test_df, user_profiles=user_profiles)
    y_test = test_df['matched'].astype(int)
    user_test = test_df['user_id']

    # --- Sanity checks ---
    if X_train.empty or X_test.empty:
        raise ValueError("Feature matrices are empty. Check your ETL and feature functions.")

    print(f"[INFO] Feature dimensions: train={X_train.shape}, test={X_test.shape}")
    print("[INFO] Example feature columns:", list(X_train.columns)[:10])

    # --- Compute group sizes for LightGBM ---
    def groups_from_ordered_user_series(user_series: pd.Series):
        return user_series.groupby(user_series, sort=False).size().tolist()

    group_train = groups_from_ordered_user_series(user_train)
    group_test = groups_from_ordered_user_series(user_test)

    print(f"[INFO] Groups (train): {len(group_train)}, Groups (test): {len(group_test)}")

    # --- Train Model ---
    print("[INFO] Training LightGBM model (LGBMRanker, objective=lambdarank)...")
    model = LGBMRanker(
        **lgbm_params,
        n_estimators=num_boost_round,
        random_state=RANDOM_STATE
    )

    model.fit(
        X_train,
        y_train,
        group=group_train,
        eval_set=[(X_test, y_test)],
        eval_group=[group_test],
        eval_at=[5, 10],
        callbacks=[
            early_stopping(early_stopping_rounds),
            log_evaluation(50)
        ]
    )

    print("[INFO] ✅ Training complete!")

    # --- Save model ---
    os.makedirs(os.path.dirname(model_out) or ".", exist_ok=True)
    joblib.dump(model, model_out)
    print(f"[INFO] 💾 Model saved to {model_out}")


# ==============================================================
# CLI Entry Point
# ==============================================================
def main():
    parser = argparse.ArgumentParser(description="Train a buyer–vehicle ranking model with LambdaRank.")
    parser.add_argument('--train_path', default=DEFAULT_TRAIN_PATH, help="Path to the training data (parquet).")
    parser.add_argument('--model_path', default=DEFAULT_MODEL_PATH, help="Path to save trained model.")
    parser.add_argument('--test_size', type=float, default=0.2, help="Proportion of dataset for testing.")
    parser.add_argument('--lr', type=float, default=0.05, help="Learning rate for LightGBM.")
    parser.add_argument('--num_leaves', type=int, default=31, help="Number of leaves.")
    parser.add_argument('--num_rounds', type=int, default=1000, help="Number of boosting rounds.")
    parser.add_argument('--early_stopping', type=int, default=50, help="Early stopping rounds.")
    args = parser.parse_args()

    lgbm_params = {
        'objective': 'lambdarank',
        'metric': 'ndcg',
        'learning_rate': args.lr,
        'num_leaves': args.num_leaves,
        'verbose': -1
    }

    run_training(
        parquet_path=args.train_path,
        model_out=args.model_path,
        test_size=args.test_size,
        lgbm_params=lgbm_params,
        num_boost_round=args.num_rounds,
        early_stopping_rounds=args.early_stopping
    )


if __name__ == '__main__':
    main()
