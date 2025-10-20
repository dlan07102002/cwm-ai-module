"""
Enhanced LightGBM LambdaRank training with:
- Better data validation
- Hyperparameter tuning
- Model evaluation metrics
- Feature importance analysis
- Model versioning
"""

import lightgbm as lgb
import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.model_selection import GroupShuffleSplit
from datetime import datetime
import json
import logging
from .feature_builder import FeatureBuilder

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class VehicleRankerTrainer:
    """Enhanced trainer with model evaluation and versioning."""
    
    def __init__(self, output_dir: str = "models"):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True, parents=True)
        self.feature_builder = FeatureBuilder()
        
        # Define feature columns
        self.features = [
            'price_diff', 'price_ratio', 'price_in_budget', 'price_percentile', 'price_optimality',
            'vehicle_age', 'year_diff', 
            'mileage_ratio', 'mileage_acceptable',
            'brand_match', 'color_match', 'fuel_match', 'transmission_match',
            'preference_score', 'urgency_score', 'is_same_location', 'overall_match_score'
        ]
        
        self.categorical_features = [
            'price_in_budget', 'mileage_acceptable', 'brand_match', 'color_match',
            'fuel_match', 'transmission_match', 'is_same_location'
        ]
    
    def validate_data(self, df: pd.DataFrame) -> bool:
        """Validate dataset before training."""
        
        logger.info("Validating dataset...")
        
        # Check for required columns
        required_cols = self.features + ["user_id", "label"]
        missing_cols = [col for col in required_cols if col not in df.columns]
        
        if missing_cols:
            logger.error(f"Missing required columns: {missing_cols}")
            return False
        
        # Check for sufficient data
        min_samples = 20  # Reduced for small datasets
        if len(df) < min_samples:
            logger.error(f"Insufficient data: only {len(df)} samples (minimum: {min_samples})")
            return False
        
        if len(df) < 100:
            logger.warning(f"Small dataset: {len(df)} samples. Consider adding more data for better model performance.")
        
        # Check for label distribution
        positive_count = (df['label'] > 0).sum()
        negative_count = (df['label'] == 0).sum()
        
        logger.info(f"Label distribution - Positive: {positive_count}, Negative: {negative_count}")
        
        if positive_count < 10:
            logger.warning(f"Very few positive labels ({positive_count}). Model may not learn well.")
        
        # Check for NaN values in features
        nan_counts = df[self.features].isna().sum()
        if nan_counts.sum() > 0:
            logger.warning(f"NaN values found in features:\n{nan_counts[nan_counts > 0]}")
        
        # Check for user distribution
        users_with_labels = df[df['label'] > 0]['user_id'].nunique()
        total_users = df['user_id'].nunique()
        
        logger.info(f"Users: {total_users} total, {users_with_labels} with positive labels")
        
        return True
    
    def split_data(self, df: pd.DataFrame, test_size: float = 0.2):
        """Split data by user groups."""
        
        logger.info(f"Splitting data (test_size={test_size})...")
        
        gss = GroupShuffleSplit(n_splits=1, test_size=test_size, random_state=42)
        train_idx, val_idx = next(gss.split(df, groups=df["user_id"]))
        
        train_df = df.iloc[train_idx].copy()
        val_df = df.iloc[val_idx].copy()
        
        # Ensure we have groups (users) with multiple items for ranking
        train_group_counts = train_df.groupby("user_id").size()
        val_group_counts = val_df.groupby("user_id").size()
        
        # Filter out users with only 1 item (can't rank)
        train_users_valid = train_group_counts[train_group_counts > 1].index
        val_users_valid = val_group_counts[val_group_counts > 1].index
        
        train_df = train_df[train_df['user_id'].isin(train_users_valid)]
        val_df = val_df[val_df['user_id'].isin(val_users_valid)]
        
        # Recompute group counts
        train_group_counts = train_df.groupby("user_id").size().tolist()
        val_group_counts = val_df.groupby("user_id").size().tolist()
        
        logger.info(f"Training: {len(train_df)} samples, {len(train_group_counts)} users")
        logger.info(f"Validation: {len(val_df)} samples, {len(val_group_counts)} users")
        
        return train_df, val_df, train_group_counts, val_group_counts
    
    def prepare_datasets(self, train_df, val_df, train_group_counts, val_group_counts):
        """Prepare LightGBM datasets."""
        
        logger.info("Preparing LightGBM datasets...")
        
        train_data = lgb.Dataset(
            train_df[self.features],
            label=train_df["label"],
            group=train_group_counts,
            feature_name=self.features,
            categorical_feature=self.categorical_features,
            free_raw_data=False
        )
        
        val_data = lgb.Dataset(
            val_df[self.features],
            label=val_df["label"],
            group=val_group_counts,
            reference=train_data,
            feature_name=self.features,
            categorical_feature=self.categorical_features,
            free_raw_data=False
        )
        
        return train_data, val_data
    
    def get_params(self, hyperparameter_tuning: bool = False):
        """Get training parameters."""
        
        if hyperparameter_tuning:
            # For hyperparameter search - more aggressive
            params = {
                "objective": "lambdarank",
                "metric": "ndcg",
                "eval_at": [3, 5, 10],
                "learning_rate": 0.1,
                "num_leaves": 63,
                "min_data_in_leaf": 10,
                "feature_fraction": 0.9,
                "bagging_fraction": 0.8,
                "bagging_freq": 5,
                "lambda_l1": 0.1,
                "lambda_l2": 0.1,
                "max_depth": 7,
                "verbosity": -1
            }
        else:
            # Conservative defaults for production
            params = {
                "objective": "lambdarank",
                "metric": "ndcg",
                "eval_at": [3, 5, 10],
                "learning_rate": 0.05,
                "num_leaves": 31,
                "min_data_in_leaf": 20,
                "feature_fraction": 0.9,
                "bagging_fraction": 0.8,
                "bagging_freq": 5,
                "lambda_l1": 0.05,
                "lambda_l2": 0.05,
                "max_depth": 6,
                "verbosity": -1
            }
        
        return params
    
    def train(self, params, train_data, val_data, num_boost_round=1000):
        """Train the model with early stopping."""
        
        logger.info("Training LightGBM LambdaRank model...")
        
        callbacks = [
            lgb.early_stopping(stopping_rounds=100, verbose=True),
            lgb.log_evaluation(period=50)
        ]
        
        model = lgb.train(
            params,
            train_data,
            num_boost_round=num_boost_round,
            valid_sets=[train_data, val_data],
            valid_names=['train', 'val'],
            callbacks=callbacks
        )
        
        logger.info(f"Best iteration: {model.best_iteration}")
        logger.info(f"Best score: {model.best_score}")
        
        return model
    
    def evaluate_model(self, model, val_df, val_group_counts):
        """Comprehensive model evaluation."""
        
        logger.info("\n" + "="*60)
        logger.info("MODEL EVALUATION")
        logger.info("="*60)
        
        # Predict on validation set
        X_val = val_df[self.features]
        preds = model.predict(X_val)
        val_df_copy = val_df.copy()
        val_df_copy['pred_score'] = preds
        
        # Calculate metrics per user
        metrics = []
        
        for user_id, group in val_df_copy.groupby('user_id'):
            if len(group) < 2:
                continue
            
            # Sort by prediction
            sorted_group = group.sort_values('pred_score', ascending=False)
            
            # Calculate NDCG@5 manually
            actual_labels = sorted_group['label'].values[:5]
            ideal_labels = sorted(actual_labels, reverse=True)
            
            dcg = sum((2**rel - 1) / np.log2(i + 2) for i, rel in enumerate(actual_labels))
            idcg = sum((2**rel - 1) / np.log2(i + 2) for i, rel in enumerate(ideal_labels))
            
            ndcg = dcg / idcg if idcg > 0 else 0
            
            # Precision@5
            precision = (actual_labels > 0).sum() / min(5, len(actual_labels))
            
            # MRR (Mean Reciprocal Rank)
            try:
                first_relevant = np.where(sorted_group['label'].values > 0)[0][0] + 1
                mrr = 1.0 / first_relevant
            except IndexError:
                mrr = 0.0
            
            metrics.append({
                'user_id': user_id,
                'ndcg@5': ndcg,
                'precision@5': precision,
                'mrr': mrr
            })
        
        metrics_df = pd.DataFrame(metrics)
        
        logger.info(f"\nAverage Metrics:")
        logger.info(f"  NDCG@5: {metrics_df['ndcg@5'].mean():.4f}")
        logger.info(f"  Precision@5: {metrics_df['precision@5'].mean():.4f}")
        logger.info(f"  MRR: {metrics_df['mrr'].mean():.4f}")
        
        # Feature importance
        logger.info(f"\n{'='*60}")
        logger.info("FEATURE IMPORTANCE (Top 10)")
        logger.info(f"{'='*60}")
        
        importance_df = pd.DataFrame({
            'feature': model.feature_name(),
            'importance': model.feature_importance(importance_type='gain')
        }).sort_values('importance', ascending=False)
        
        for idx, row in importance_df.head(10).iterrows():
            logger.info(f"  {row['feature']:30s}: {row['importance']:10.0f}")
        
        return metrics_df, importance_df
    
    def save_model(self, model, metrics_df, importance_df):
        """Save model with metadata and versioning."""
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # Save model
        model_path = self.output_dir / "vehicle_ranker.txt"
        versioned_path = self.output_dir / f"vehicle_ranker_{timestamp}.txt"
        
        model.save_model(str(model_path))
        model.save_model(str(versioned_path))
        
        # Save metadata
        metadata = {
            'timestamp': timestamp,
            'best_iteration': model.best_iteration,
            'best_score': model.best_score,
            'features': self.features,
            'categorical_features': self.categorical_features,
            'metrics': {
                'ndcg@5': float(metrics_df['ndcg@5'].mean()),
                'precision@5': float(metrics_df['precision@5'].mean()),
                'mrr': float(metrics_df['mrr'].mean())
            }
        }
        
        metadata_path = self.output_dir / "model_metadata.json"
        with open(metadata_path, 'w') as f:
            json.dump(metadata, f, indent=2)
        
        # Save feature importance
        importance_path = self.output_dir / f"feature_importance_{timestamp}.csv"
        importance_df.to_csv(importance_path, index=False)
        
        logger.info(f"\n✅ Model saved to: {model_path}")
        logger.info(f"✅ Versioned model: {versioned_path}")
        logger.info(f"✅ Metadata saved to: {metadata_path}")
        logger.info(f"✅ Feature importance: {importance_path}")
    
    def run(self, hyperparameter_tuning: bool = False):
        """Complete training pipeline."""
        
        logger.info("="*60)
        logger.info("🚀 VEHICLE RANKER TRAINING PIPELINE")
        logger.info("="*60)
        
        # 1. Build features
        logger.info("\n[1/6] Building features...")
        df = self.feature_builder.build_features(
            include_interactions=True,
            max_candidates_per_user=100
        )
        
        if df.empty:
            logger.error("❌ No data available for training!")
            return
        
        logger.info(f"✅ Dataset: {df.shape[0]} rows, {df.shape[1]} columns")
        
        # 2. Validate data
        logger.info("\n[2/6] Validating data...")
        if not self.validate_data(df):
            logger.error("❌ Data validation failed!")
            return
        
        logger.info("✅ Data validation passed")
        
        # 3. Split data
        logger.info("\n[3/6] Splitting data...")
        train_df, val_df, train_group_counts, val_group_counts = self.split_data(df)
        
        # 4. Prepare datasets
        logger.info("\n[4/6] Preparing LightGBM datasets...")
        train_data, val_data = self.prepare_datasets(
            train_df, val_df, train_group_counts, val_group_counts
        )
        
        # 5. Train model
        logger.info("\n[5/6] Training model...")
        params = self.get_params(hyperparameter_tuning)
        model = self.train(params, train_data, val_data)
        
        # 6. Evaluate and save
        logger.info("\n[6/6] Evaluating model...")
        metrics_df, importance_df = self.evaluate_model(model, val_df, val_group_counts)
        
        logger.info("\nSaving model...")
        self.save_model(model, metrics_df, importance_df)
        
        logger.info("\n" + "="*60)
        logger.info("✅ TRAINING PIPELINE COMPLETED SUCCESSFULLY")
        logger.info("="*60)
        
        # Cleanup
        self.feature_builder.close()


def main():
    """Entry point for training."""
    import argparse
    
    parser = argparse.ArgumentParser(description='Train Vehicle Ranker Model')
    parser.add_argument(
        '--tune', 
        action='store_true', 
        help='Enable hyperparameter tuning mode'
    )
    
    args = parser.parse_args()
    
    trainer = VehicleRankerTrainer()
    trainer.run(hyperparameter_tuning=args.tune)


if __name__ == "__main__":
    main()