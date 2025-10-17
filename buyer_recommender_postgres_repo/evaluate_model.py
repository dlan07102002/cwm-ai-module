"""
Comprehensive evaluation framework for recommendation system.
Measures both ML metrics and business metrics.
"""

import numpy as np
import pandas as pd
from typing import Dict, List
from sklearn.metrics import ndcg_score, precision_score, recall_score
import joblib

from model_utils import build_features_from_candidates

class RecommenderEvaluator:
    """Evaluate recommender performance on multiple dimensions."""
    
    def __init__(self, model_path: str):
        checkpoint = joblib.load(model_path)
        self.model = checkpoint['model']
        self.user_profiles = checkpoint.get('user_profiles', None)
    
    def evaluate_full(self, test_df: pd.DataFrame, k_values: List[int] = [5, 10, 20]) -> Dict:
        """
        Comprehensive evaluation on test set.
        
        Args:
            test_df: Test DataFrame with candidate pairs and labels
            k_values: List of k values for top-k metrics
        
        Returns:
            Dictionary with all evaluation metrics
        """
        print("=== Starting Comprehensive Evaluation ===\n")
        
        # Build features
        X = build_features_from_candidates(test_df, user_profiles=self.user_profiles)
        y_true = test_df['matched'].values
        groups = test_df['user_id'].values
        
        # Get predictions
        y_scores = self.model.predict(X)
        
        metrics = {}
        
        # --- Ranking Metrics ---
        print("Computing ranking metrics...")
        metrics['ranking'] = self._compute_ranking_metrics(
            y_true, y_scores, groups, k_values
        )
        
        # --- Business Metrics ---
        print("Computing business metrics...")
        metrics['business'] = self._compute_business_metrics(
            test_df, y_scores, groups, k_values
        )
        
        # --- Segment Analysis ---
        print("Computing segment analysis...")
        metrics['segments'] = self._compute_segment_analysis(
            test_df, y_scores, y_true
        )
        
        # --- Cold Start Performance ---
        print("Computing cold start metrics...")
        metrics['cold_start'] = self._compute_cold_start_metrics(
            test_df, y_scores, y_true
        )
        
        self._print_report(metrics)
        return metrics
    
    def _compute_ranking_metrics(
        self, 
        y_true: np.ndarray, 
        y_scores: np.ndarray,
        groups: np.ndarray,
        k_values: List[int]
    ) -> Dict:
        """Compute NDCG, Precision@K, Recall@K, MRR."""
        metrics = {}
        
        # Group-wise evaluation
        unique_groups = np.unique(groups)
        
        for k in k_values:
            ndcg_scores = []
            precision_scores = []
            recall_scores = []
            mrr_scores = []
            
            for group in unique_groups:
                mask = groups == group
                group_true = y_true[mask]
                group_scores = y_scores[mask]
                
                if len(group_true) == 0 or group_true.sum() == 0:
                    continue
                
                # NDCG@k
                ndcg = ndcg_score([group_true], [group_scores], k=k)
                ndcg_scores.append(ndcg)
                
                # Precision@k and Recall@k
                top_k_idx = np.argsort(group_scores)[-k:]
                top_k_true = group_true[top_k_idx]
                
                precision = top_k_true.sum() / k if k > 0 else 0
                recall = top_k_true.sum() / group_true.sum() if group_true.sum() > 0 else 0
                
                precision_scores.append(precision)
                recall_scores.append(recall)
                
                # MRR (Mean Reciprocal Rank)
                sorted_idx = np.argsort(group_scores)[::-1]
                sorted_true = group_true[sorted_idx]
                first_relevant = np.where(sorted_true == 1)[0]
                if len(first_relevant) > 0:
                    mrr_scores.append(1.0 / (first_relevant[0] + 1))
            
            metrics[f'ndcg@{k}'] = np.mean(ndcg_scores)
            metrics[f'precision@{k}'] = np.mean(precision_scores)
            metrics[f'recall@{k}'] = np.mean(recall_scores)
            metrics[f'mrr@{k}'] = np.mean(mrr_scores)
        
        return metrics
    
    def _compute_business_metrics(
        self,
        test_df: pd.DataFrame,
        y_scores: np.ndarray,
        groups: np.ndarray,
        k_values: List[int]
    ) -> Dict:
        """Compute business-relevant metrics."""
        metrics = {}
        
        # Coverage: What % of vehicles get at least one good recommendation?
        unique_groups = np.unique(groups)
        vehicles_with_match = 0
        
        for group in unique_groups:
            mask = groups == group
            group_scores = y_scores[mask]
            group_true = test_df[mask]['matched'].values
            
            # Check if any top-5 recommendation is a match
            top_5_idx = np.argsort(group_scores)[-5:]
            if group_true[top_5_idx].sum() > 0:
                vehicles_with_match += 1
        
        metrics['coverage@5'] = vehicles_with_match / len(unique_groups)
        
        # Price accuracy: Are recommendations within budget?
        if all(col in test_df.columns for col in ['price', 'price_min', 'price_max']):
            within_budget = (
                (test_df['price'] >= test_df['price_min']) &
                (test_df['price'] <= test_df['price_max'])
            ).mean()
            metrics['budget_compliance'] = within_budget
        
        # Brand match rate in top-k
        if 'pre_brand' in test_df.columns and 'veh_brand' in test_df.columns:
            brand_matches_in_topk = []
            for k in [5, 10]:
                matches = []
                for group in unique_groups:
                    mask = groups == group
                    group_scores = y_scores[mask]
                    top_k_idx = np.argsort(group_scores)[-k:]
                    
                    group_df = test_df[mask].iloc[top_k_idx]
                    brand_match_rate = (
                        group_df['pre_brand'] == group_df['veh_brand']
                    ).mean()
                    matches.append(brand_match_rate)
                
                metrics[f'brand_match_rate@{k}'] = np.mean(matches)
        
        return metrics
    
    def _compute_segment_analysis(
        self,
        test_df: pd.DataFrame,
        y_scores: np.ndarray,
        y_true: np.ndarray
    ) -> Dict:
        """Analyze performance across different segments."""
        segments = {}
        
        # By price range
        if 'price' in test_df.columns:
            price_bins = pd.qcut(test_df['price'], q=3, labels=['low', 'mid', 'high'], duplicates='drop')
            for segment in ['low', 'mid', 'high']:
                mask = price_bins == segment
                if mask.sum() > 0:
                    seg_true = y_true[mask]
                    seg_scores = y_scores[mask]
                    if len(seg_true) > 0:
                        segments[f'ndcg_price_{segment}'] = ndcg_score(
                            [seg_true], [seg_scores]
                        )
        
        # By vehicle age
        if 'veh_year' in test_df.columns:
            current_year = 2025
            test_df['veh_age'] = current_year - test_df['veh_year']
            age_bins = pd.cut(test_df['veh_age'], bins=[0, 3, 7, 100], labels=['new', 'mid', 'old'])
            
            for segment in ['new', 'mid', 'old']:
                mask = age_bins == segment
                if mask.sum() > 0:
                    seg_true = y_true[mask]
                    seg_scores = y_scores[mask]
                    if len(seg_true) > 0:
                        segments[f'ndcg_age_{segment}'] = ndcg_score(
                            [seg_true], [seg_scores]
                        )
        
        return segments
    
    def _compute_cold_start_metrics(
        self,
        test_df: pd.DataFrame,
        y_scores: np.ndarray,
        y_true: np.ndarray
    ) -> Dict:
        """Measure performance on cold start users."""
        metrics = {}
        
        if self.user_profiles is None:
            metrics['note'] = 'No user profiles available'
            return metrics
        
        # Identify cold start users (not in user profiles)
        cold_start_mask = ~test_df['user_id'].isin(self.user_profiles.keys())
        warm_start_mask = test_df['user_id'].isin(self.user_profiles.keys())
        
        if cold_start_mask.sum() > 0:
            metrics['cold_start_ndcg'] = ndcg_score(
                [y_true[cold_start_mask]], 
                [y_scores[cold_start_mask]]
            )
            metrics['cold_start_samples'] = cold_start_mask.sum()
        
        if warm_start_mask.sum() > 0:
            metrics['warm_start_ndcg'] = ndcg_score(
                [y_true[warm_start_mask]],
                [y_scores[warm_start_mask]]
            )
            metrics['warm_start_samples'] = warm_start_mask.sum()
        
        return metrics
    
    def _print_report(self, metrics: Dict):
        """Print formatted evaluation report."""
        print("\n" + "="*60)
        print("EVALUATION REPORT")
        print("="*60)
        
        print("\n📊 RANKING METRICS:")
        for key, value in metrics['ranking'].items():
            print(f"  {key:20s}: {value:.4f}")
        
        print("\n💼 BUSINESS METRICS:")
        for key, value in metrics['business'].items():
            print(f"  {key:20s}: {value:.4f}")
        
        print("\n🎯 SEGMENT ANALYSIS:")
        for key, value in metrics['segments'].items():
            print(f"  {key:20s}: {value:.4f}")
        
        print("\n❄️  COLD START PERFORMANCE:")
        for key, value in metrics['cold_start'].items():
            print(f"  {key:20s}: {value}")
        
        print("\n" + "="*60)
        
        # Performance Assessment
        ndcg_10 = metrics['ranking'].get('ndcg@10', 0)
        coverage = metrics['business'].get('coverage@5', 0)
        
        print("\n🎓 PRODUCTION READINESS ASSESSMENT:")
        if ndcg_10 > 0.7 and coverage > 0.6:
            print("  ✅ EXCELLENT - Ready for production")
        elif ndcg_10 > 0.5 and coverage > 0.4:
            print("  ⚠️  ACCEPTABLE - Consider improvements")
        else:
            print("  ❌ POOR - Needs significant improvement")
        
        print(f"  • NDCG@10: {ndcg_10:.3f} (target: >0.70)")
        print(f"  • Coverage@5: {coverage:.3f} (target: >0.60)")
        print("="*60 + "\n")


# === Usage Example ===
if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', default='model.joblib')
    parser.add_argument('--test_data', default='data/test.parquet')
    args = parser.parse_args()
    
    # Load test data
    print(f"Loading test data from {args.test_data}...")
    test_df = pd.read_parquet(args.test_data)
    
    # Evaluate
    evaluator = RecommenderEvaluator(args.model)
    metrics = evaluator.evaluate_full(test_df, k_values=[5, 10, 20])
    
    # Save results
    import json
    with open('evaluation_results.json', 'w') as f:
        json.dump(metrics, f, indent=2, default=str)
    
    print("\n✓ Results saved to evaluation_results.json")