import argparse
import joblib
import pandas as pd
from lightgbm import LGBMRanker
from sklearn.model_selection import train_test_split
from typing import Dict, Any
from lightgbm import early_stopping, log_evaluation

from model_utils import build_features_from_candidates

# --- Constants ---
DEFAULT_TRAIN_PATH = 'data/train.parquet'
DEFAULT_MODEL_PATH = 'model.joblib'
RANDOM_STATE = 42


def run_training(
    parquet_path: str,
    model_out: str,
    test_size: float,
    lgbm_params: Dict[str, Any],
    num_boost_round: int,
    early_stopping_rounds: int
):
    print(f"Loading data from {parquet_path}...")
    df = pd.read_parquet(parquet_path)
    print(f"Loaded {len(df)} rows.")

    # --- Feature Engineering ---
    print("Building features...")
    X = build_features_from_candidates(df)
    y = df['matched'].astype(int)
    user_ids = df['user_id']

    # --- Sort by user to keep groups contiguous ---
    df_sorted = df.sort_values('user_id').reset_index(drop=True)
    X = X.loc[df_sorted.index]
    y = y.loc[df_sorted.index]
    user_ids = user_ids.loc[df_sorted.index]

    # --- Compute group sizes ---
    group_sizes = user_ids.value_counts(sort=False).sort_index().to_list()
    print(f"Number of groups (buyers): {len(group_sizes)}")

    # --- Split Train/Test ---
    # Optional: split by unique users to prevent leakage
    unique_users = user_ids.unique()
    train_users, test_users = train_test_split(unique_users, test_size=test_size, random_state=RANDOM_STATE)

    train_idx = user_ids.isin(train_users)
    test_idx = user_ids.isin(test_users)

    X_train, X_test = X[train_idx], X[test_idx]
    y_train, y_test = y[train_idx], y[test_idx]
    user_train, user_test = user_ids[train_idx], user_ids[test_idx]
    print("Feature means:", X_train.mean())
    print("Feature stds:", X_train.std())
    # Group info for train/val
    group_train = user_train.value_counts(sort=False).sort_index().to_list()
    group_test = user_test.value_counts(sort=False).sort_index().to_list()

    # --- Train Model ---
    print("Training LightGBM model (LGBMRanker, objective=lambdarank)...")
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

    print("Training complete!")

    # --- Save Model ---
    print(f"Saving model to {model_out}...")
    joblib.dump(model, model_out)
    print(f"Model saved successfully to {model_out}")


def main():
    parser = argparse.ArgumentParser(description="Train a buyer–vehicle ranking model with LambdaRank.")
    parser.add_argument('--train_path', default=DEFAULT_TRAIN_PATH, help="Path to the training data in parquet format.")
    parser.add_argument('--model_path', default=DEFAULT_MODEL_PATH, help="Path to save the trained model.")
    parser.add_argument('--test_size', type=float, default=0.2, help="Proportion of the dataset for testing.")
    parser.add_argument('--lr', type=float, default=0.05, help="Learning rate for LightGBM.")
    parser.add_argument('--num_leaves', type=int, default=31, help="Number of leaves for LightGBM.")
    parser.add_argument('--num_rounds', type=int, default=1000, help="Maximum number of boosting rounds.")
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
