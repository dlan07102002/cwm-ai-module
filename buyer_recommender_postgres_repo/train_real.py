import argparse
import joblib
import pandas as pd
from sklearn.model_selection import train_test_split
from typing import Dict, Any
from lightgbm import early_stopping, log_evaluation, LGBMRanker

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

    # --- Split Train/Test by users FIRST to avoid leakage ---
    print("Splitting users into train/test to avoid leakage...")
    unique_users = df['user_id'].unique()
    train_users, test_users = train_test_split(unique_users, test_size=test_size, random_state=RANDOM_STATE)

    train_df = df[df['user_id'].isin(train_users)].reset_index(drop=True)
    test_df = df[df['user_id'].isin(test_users)].reset_index(drop=True)
    print(f"Train users: {len(train_users)}, Test users: {len(test_users)}")
    print(f"Train rows: {len(train_df)}, Test rows: {len(test_df)}")

    # --- Build historical matches from train only (to compute user profiles) ---
    historical_matches = train_df[train_df['matched'] == 1].reset_index(drop=True)
    print(f"Historical (train) matched records used for profiles: {len(historical_matches)}")

    # --- Feature Engineering: build features separately using historical matches only ---
    print("Building features for train set (no leakage)...")
    X_train = build_features_from_candidates(train_df, historical_matches=historical_matches)
    y_train = train_df['matched'].astype(int)
    user_train = train_df['user_id']

    print("Building features for test set (profiles built from train historical matches only)...")
    X_test = build_features_from_candidates(test_df, historical_matches=historical_matches)
    y_test = test_df['matched'].astype(int)
    user_test = test_df['user_id']

    # --- Print some basic stats ---
    if not X_train.empty:
        print("Feature means (train):", X_train.mean().to_dict())
        print("Feature stds (train):", X_train.std().to_dict())

    # --- Compute group sizes preserving the order of rows ---
    # LightGBM expects group sizes to match the order of rows passed in.
    def groups_from_ordered_user_series(user_series: pd.Series):
        # user_series is aligned with X rows and in the order passed to fit
        return user_series.groupby(user_series, sort=False).size().tolist()

    group_train = groups_from_ordered_user_series(user_train)
    group_test = groups_from_ordered_user_series(user_test)
    print(f"Number of groups (train buyers): {len(group_train)}, (test buyers): {len(group_test)}")

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
