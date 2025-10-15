import argparse
import joblib
import pandas as pd
from lightgbm import LGBMClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score, classification_report
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
    """
    Loads data, engineers features, splits the dataset, trains a LightGBM model
    with early stopping, evaluates its performance, and saves the final artifact.
    """
    # 1. Load and Prepare Data
    print(f"Loading data from {parquet_path}...")
    df = pd.read_parquet(parquet_path)
    print(f"Loaded {len(df)} rows.")

    # 2. Feature Engineering
    print("Building features...")
    X = build_features_from_candidates(df)
    y = df['matched'].astype(int)

    # 3. Data Splitting (Train / Validation / Test)
    print("Splitting data into training, validation, and test sets...")
    X_train_val, X_test, y_train_val, y_test = train_test_split(
        X, y, test_size=test_size, random_state=RANDOM_STATE, stratify=y
    )

    val_size_adjusted = test_size / (1 - test_size)
    X_train, X_val, y_train, y_val = train_test_split(
        X_train_val, y_train_val, test_size=val_size_adjusted, random_state=RANDOM_STATE, stratify=y_train_val
    )

    print(f"Train size: {len(X_train)}, Validation size: {len(X_val)}, Test size: {len(X_test)}")

    # 4. Model Training with Early Stopping
    print("Training LightGBM model (LGBMClassifier)...")
    model = LGBMClassifier(
        **lgbm_params,
        n_estimators=num_boost_round,
        random_state=RANDOM_STATE
    )

 
    model.fit(
        X_train,
        y_train,
        eval_set=[(X_val, y_val)],
        eval_metric='auc',
        callbacks=[
            early_stopping(early_stopping_rounds),
            log_evaluation(50)
        ]
    )

    # 5. Evaluation on Test Set
    print("\nEvaluating model on the test set...")
    y_pred_proba = model.predict_proba(X_test)[:, 1]
    y_pred_class = (y_pred_proba > 0.5).astype(int)

    auc = roc_auc_score(y_test, y_pred_proba)
    print(f"Test AUC: {auc:.4f}")
    print("Classification Report:")
    print(classification_report(y_test, y_pred_class))

    # 6. Save Model
    print(f"Saving model to {model_out}...")
    joblib.dump(model, model_out)
    print(f"Model saved successfully to {model_out}")


def main():
    parser = argparse.ArgumentParser(description="Train a buyer recommendation model.")
    parser.add_argument('--train_path', default=DEFAULT_TRAIN_PATH, help="Path to the training data in parquet format.")
    parser.add_argument('--model_path', default=DEFAULT_MODEL_PATH, help="Path to save the trained model.")
    parser.add_argument('--test_size', type=float, default=0.2, help="Proportion of the dataset for testing and validation.")
    parser.add_argument('--lr', type=float, default=0.05, help="Learning rate for LightGBM.")
    parser.add_argument('--num_leaves', type=int, default=31, help="Number of leaves for LightGBM.")
    parser.add_argument('--num_rounds', type=int, default=1000, help="Maximum number of boosting rounds.")
    parser.add_argument('--early_stopping', type=int, default=50, help="Early stopping rounds.")
    args = parser.parse_args()

    lgbm_params = {
        'objective': 'binary',
        'metric': 'auc',
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
