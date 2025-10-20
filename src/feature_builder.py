"""
Production-ready feature builder with:
- Pre-filtering to avoid Cartesian explosion
- Rich feature engineering
- Proper null handling
- Performance optimization
- Caching support
- Comprehensive logging
"""

import pandas as pd
import numpy as np
from sqlalchemy import create_engine, text
from typing import Tuple, Optional, List
from datetime import datetime, timedelta
import logging
from functools import lru_cache
from .db_config import DB_CONFIG

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class FeatureBuilder:
    """Enhanced feature builder with caching and performance optimizations."""
    
    def __init__(self, cache_ttl_minutes: int = 5):
        self.engine = self._get_connection()
        self.cache_ttl = timedelta(minutes=cache_ttl_minutes)
        self._cache = {}
        self._cache_timestamps = {}
    
    def _get_connection(self):
        """Create a SQLAlchemy connection engine to PostgreSQL."""
        url = (
            f"postgresql+psycopg2://{DB_CONFIG['user']}:{DB_CONFIG['password']}"
            f"@{DB_CONFIG['host']}:{DB_CONFIG['port']}/{DB_CONFIG['database']}"
        )
        return create_engine(
            url, 
            pool_pre_ping=True, 
            pool_size=10, 
            max_overflow=20,
            pool_recycle=3600  # Recycle connections after 1 hour
        )
    
    def _is_cache_valid(self, key: str) -> bool:
        """Check if cached data is still valid."""
        if key not in self._cache_timestamps:
            return False
        return datetime.now() - self._cache_timestamps[key] < self.cache_ttl
    
    def _get_cached(self, key: str) -> Optional[pd.DataFrame]:
        """Get cached data if valid."""
        if self._is_cache_valid(key):
            logger.info(f"Cache hit for: {key}")
            return self._cache[key].copy()
        return None
    
    def _set_cache(self, key: str, data: pd.DataFrame):
        """Store data in cache."""
        self._cache[key] = data.copy()
        self._cache_timestamps[key] = datetime.now()
    
    @staticmethod
    def calculate_haversine_distance(lat1, lon1, lat2, lon2):
        """Calculate distance in km between two coordinates."""
        R = 6371  # Earth's radius in km
        
        lat1, lon1, lat2, lon2 = map(np.radians, [lat1, lon1, lat2, lon2])
        dlat = lat2 - lat1
        dlon = lon2 - lon1
        
        a = np.sin(dlat/2)**2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon/2)**2
        c = 2 * np.arcsin(np.sqrt(a))
        
        return R * c
    
    def load_user_preferences(self) -> pd.DataFrame:
        """Extract user preferences from verification questions with caching."""
        
        cache_key = "user_preferences"
        cached = self._get_cached(cache_key)
        if cached is not None:
            return cached
        
        logger.info("Loading user preferences from database...")
        
        # Load questions
        questions_query = "SELECT question_id, question_text FROM bs_verify_question"
        questions = pd.read_sql(questions_query, self.engine)
        
        # Map question types to IDs - more robust matching
        question_map = {}
        question_keywords = {
            'brand': ['Thương hiệu', 'brand', 'hãng'],
            'color': ['Màu', 'color', 'màu sắc'],
            'fuel_type': ['Nhiên liệu', 'fuel', 'xăng', 'dầu'],
            'transmission': ['Hộp số', 'transmission', 'số'],
        }
        
        for col_name, keywords in question_keywords.items():
            for keyword in keywords:
                q = questions[questions['question_text'].str.contains(keyword, case=False, na=False)]
                if not q.empty:
                    question_map[col_name] = q['question_id'].iloc[0]
                    break
        
        # Load all answers
        answers_query = "SELECT user_id, question_id, answer_value FROM bs_verify_answer"
        answers = pd.read_sql(answers_query, self.engine)
        
        if answers.empty:
            logger.warning("No user answers found in bs_verify_answer table")
            return pd.DataFrame(columns=['user_id'])
        
        # Pivot to get user preferences
        user_prefs = pd.DataFrame()
        user_prefs['user_id'] = answers['user_id'].unique()
        
        for col_name, qid in question_map.items():
            pref_data = answers[answers['question_id'] == qid][['user_id', 'answer_value']]
            if not pref_data.empty:
                pref_data = pref_data.rename(columns={'answer_value': f'{col_name}_pref'})
                user_prefs = user_prefs.merge(pref_data, on='user_id', how='left')
        
        self._set_cache(cache_key, user_prefs)
        logger.info(f"Loaded preferences for {len(user_prefs)} users")
        
        return user_prefs
    
    def generate_candidates_for_user(self, user_id: str, max_candidates: int = 100) -> pd.DataFrame:
        """Generate candidates for a specific user (for inference)."""
        
        query = text("""
        WITH user_order AS (
            SELECT
                user_id,
                pre_order_id,
                brand,
                model,
                year,
                price_min,
                price_max,
                mileage_max,
                fuel_type,
                transmission,
                color_preference,
                time_want_to_buy,
                location
            FROM bs_pre_order
            WHERE user_id = :user_id
              AND status = 'pending'
              AND price_min IS NOT NULL
              AND price_max IS NOT NULL
            ORDER BY created_at DESC
            LIMIT 1
        ),
        active_vehicles AS (
            SELECT 
                vehicle_id,
                seller_id,
                brand,
                model,
                year,
                color,
                mileage,
                transmission,
                fuel_type,
                price,
                location,
                seats,
                created_at
            FROM bs_vehicle
            WHERE price > 0
              AND price BETWEEN (SELECT price_min FROM user_order) 
                            AND (SELECT price_max FROM user_order)
        )
        SELECT 
            uo.user_id,
            uo.pre_order_id,
            v.vehicle_id,
            v.brand as vehicle_brand,
            v.model as vehicle_model,
            v.year as vehicle_year,
            v.color as vehicle_color,
            v.mileage as vehicle_mileage,
            v.transmission as vehicle_transmission,
            v.fuel_type as vehicle_fuel_type,
            v.price as vehicle_price,
            v.location as vehicle_location,
            v.seats as vehicle_seats,
            uo.brand as buyer_brand,
            uo.year as buyer_year,
            uo.price_min,
            uo.price_max,
            uo.mileage_max,
            uo.fuel_type as buyer_fuel_type,
            uo.transmission as buyer_transmission,
            uo.color_preference as buyer_color,
            uo.time_want_to_buy,
            uo.location as buyer_location
        FROM user_order uo
        CROSS JOIN active_vehicles v
        WHERE 
            -- Optional: brand match (if specified)
            (uo.brand IS NULL OR LOWER(v.brand) = LOWER(uo.brand))
            -- Optional: mileage constraint (if specified)
            AND (uo.mileage_max IS NULL OR v.mileage <= uo.mileage_max)
        ORDER BY v.created_at DESC
        LIMIT :max_candidates
        """)
        
        df = pd.read_sql(query, self.engine, params={
            'user_id': user_id,
            'max_candidates': max_candidates
        })
        
        return df
    
    def generate_candidates_for_vehicle(self, vehicle_id: str, max_candidates: int = 100) -> pd.DataFrame:
        """Generate potential buyer candidates for a specific vehicle (for inference)."""
        
        query = text("""
        WITH vehicle_info AS (
            SELECT 
                vehicle_id,
                brand,
                model,
                year,
                color,
                mileage,
                transmission,
                fuel_type,
                price,
                location,
                seats
            FROM bs_vehicle
            WHERE vehicle_id = :vehicle_id
        ),
        matching_orders AS (
            SELECT
                po.user_id,
                po.pre_order_id,
                po.brand,
                po.model,
                po.year,
                po.price_min,
                po.price_max,
                po.mileage_max,
                po.fuel_type,
                po.transmission,
                po.color_preference,
                po.time_want_to_buy,
                po.location,
                po.created_at
            FROM bs_pre_order po
            WHERE po.status = 'pending'
              AND po.price_min IS NOT NULL
              AND po.price_max IS NOT NULL
              AND (SELECT price FROM vehicle_info) BETWEEN po.price_min AND po.price_max
              AND (po.mileage_max IS NULL OR (SELECT mileage FROM vehicle_info) <= po.mileage_max)
        )
        SELECT 
            mo.user_id,
            mo.pre_order_id,
            v.vehicle_id,
            v.brand as vehicle_brand,
            v.model as vehicle_model,
            v.year as vehicle_year,
            v.color as vehicle_color,
            v.mileage as vehicle_mileage,
            v.transmission as vehicle_transmission,
            v.fuel_type as vehicle_fuel_type,
            v.price as vehicle_price,
            v.location as vehicle_location,
            v.seats as vehicle_seats,
            mo.brand as buyer_brand,
            mo.year as buyer_year,
            mo.price_min,
            mo.price_max,
            mo.mileage_max,
            mo.fuel_type as buyer_fuel_type,
            mo.transmission as buyer_transmission,
            mo.color_preference as buyer_color,
            mo.time_want_to_buy,
            mo.location as buyer_location
        FROM vehicle_info v
        CROSS JOIN matching_orders mo
        ORDER BY mo.created_at DESC
        LIMIT :max_candidates
        """)
        
        df = pd.read_sql(query, self.engine, params={
            'vehicle_id': vehicle_id,
            'max_candidates': max_candidates
        })
        
        return df
    
    def generate_candidates_sql(self, max_candidates_per_user: int = 100) -> pd.DataFrame:
        """Generate candidates using SQL for training (batch processing)."""
        
        logger.info(f"Generating training candidates (max {max_candidates_per_user} per user)...")
        
        query = text("""
        WITH active_vehicles AS (
            SELECT 
                vehicle_id,
                seller_id,
                brand,
                model,
                year,
                color,
                mileage,
                transmission,
                fuel_type,
                price,
                location,
                seats,
                created_at
            FROM bs_vehicle
            WHERE price > 0
        ),
        pending_orders AS (
            SELECT
                pre_order_id,
                user_id,
                brand,
                model,
                year,
                price_min,
                price_max,
                mileage_max,
                fuel_type,
                transmission,
                color_preference,
                time_want_to_buy,
                location
            FROM bs_pre_order
            WHERE status = 'pending'
              AND price_min IS NOT NULL
              AND price_max IS NOT NULL
        ),
        candidates AS (
            SELECT 
                po.user_id,
                po.pre_order_id,
                v.vehicle_id,
                v.brand as vehicle_brand,
                v.model as vehicle_model,
                v.year as vehicle_year,
                v.color as vehicle_color,
                v.mileage as vehicle_mileage,
                v.transmission as vehicle_transmission,
                v.fuel_type as vehicle_fuel_type,
                v.price as vehicle_price,
                v.location as vehicle_location,
                v.seats as vehicle_seats,
                po.brand as buyer_brand,
                po.year as buyer_year,
                po.price_min,
                po.price_max,
                po.mileage_max,
                po.fuel_type as buyer_fuel_type,
                po.transmission as buyer_transmission,
                po.color_preference as buyer_color,
                po.time_want_to_buy,
                po.location as buyer_location,
                ROW_NUMBER() OVER (PARTITION BY po.user_id ORDER BY v.created_at DESC) as rn
            FROM pending_orders po
            CROSS JOIN active_vehicles v
            WHERE 
                v.price BETWEEN po.price_min AND po.price_max
                AND (po.brand IS NULL OR LOWER(v.brand) = LOWER(po.brand))
                AND (po.mileage_max IS NULL OR v.mileage <= po.mileage_max)
        )
        SELECT * 
        FROM candidates
        WHERE rn <= :max_candidates
        """)
        
        df = pd.read_sql(query, self.engine, params={'max_candidates': max_candidates_per_user})
        
        logger.info(f"Generated {len(df)} candidate pairs for {df['user_id'].nunique()} users")
        
        return df
    
    def engineer_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Apply feature engineering to candidate pairs."""
        
        if df.empty:
            logger.warning("Empty dataframe passed to engineer_features")
            return pd.DataFrame()
        
        logger.info("Engineering features...")
        
        # Convert UUIDs to strings
        df['user_id'] = df['user_id'].astype(str)
        df['vehicle_id'] = df['vehicle_id'].astype(str)
        
        # === Price Features ===
        df['price_diff'] = (df['price_min'] - df['vehicle_price']).abs()
        df['price_ratio'] = df['vehicle_price'] / df['price_max'].clip(lower=1)
        df['price_in_budget'] = (
            (df['vehicle_price'] >= df['price_min']) & 
            (df['vehicle_price'] <= df['price_max'])
        ).astype(float)
        df['price_percentile'] = (df['vehicle_price'] - df['price_min']) / (
            df['price_max'] - df['price_min']
        ).clip(lower=1)
        
        # Price optimality (closer to mid-range is better)
        price_midpoint = (df['price_min'] + df['price_max']) / 2
        df['price_optimality'] = 1 - (df['vehicle_price'] - price_midpoint).abs() / (
            df['price_max'] - df['price_min']
        ).clip(lower=1)
        
        # === Vehicle Age Features ===
        current_year = pd.Timestamp.now().year
        df['vehicle_age'] = current_year - df['vehicle_year']
        df['year_diff'] = (df['buyer_year'] - df['vehicle_year']).abs() if 'buyer_year' in df.columns else 0
        
        # === Mileage Features ===
        df['mileage_ratio'] = np.where(
            df['mileage_max'].notna() & (df['mileage_max'] > 0),
            df['vehicle_mileage'] / df['mileage_max'],
            0.5
        )
        df['mileage_acceptable'] = (
            df['vehicle_mileage'] <= df['mileage_max'].fillna(999999)
        ).astype(float)
        
        # === Exact Match Features ===
        df['brand_match'] = (
            df.get('brand_pref', pd.Series()).fillna('').str.lower() == 
            df['vehicle_brand'].fillna('').str.lower()
        ).astype(float)
        
        df['color_match'] = (
            df.get('color_pref', pd.Series()).fillna('').str.lower() == 
            df['vehicle_color'].fillna('').str.lower()
        ).astype(float)
        
        df['fuel_match'] = (
            df.get('fuel_type_pref', pd.Series()).fillna('') == 
            df['vehicle_fuel_type'].fillna('')
        ).astype(float)
        
        df['transmission_match'] = (
            df.get('transmission_pref', pd.Series()).fillna('') == 
            df['vehicle_transmission'].fillna('')
        ).astype(float)
        
        # === Combined Preference Score (weighted) ===
        df['preference_score'] = (
            df['brand_match'] * 3.0 +      # Brand most important
            df['color_match'] * 0.5 +
            df['fuel_match'] * 1.5 +
            df['transmission_match'] * 1.0
        ) / 6.0  # Normalize
        
        # === Urgency Features ===
        urgency_map = {
            1: 1.0,   # IMMEDIATE
            2: 0.8,   # THIS_MONTH
            3: 0.5,   # THIS_QUARTER
            4: 0.3,   # THIS_YEAR
        }
        df['urgency_score'] = df['time_want_to_buy'].map(urgency_map).fillna(0.3)
        
        # === Location Features ===
        df['is_same_location'] = (
            df['buyer_location'].fillna('').str.lower() == 
            df['vehicle_location'].fillna('').str.lower()
        ).astype(float)
        
        # === Composite Score ===
        df['overall_match_score'] = (
            df['price_in_budget'] * 0.3 +
            df['preference_score'] * 0.3 +
            df['mileage_acceptable'] * 0.2 +
            df['urgency_score'] * 0.1 +
            df['is_same_location'] * 0.1
        )
        
        return df
    
    def load_interaction_labels(self, df: pd.DataFrame) -> pd.DataFrame:
        """Load interaction data for training labels (if table exists)."""
        
        logger.info("Attempting to load interaction data...")
        
        # Check if interaction table exists
        check_query = """
        SELECT EXISTS (
            SELECT FROM information_schema.tables 
            WHERE table_schema = 'public' 
            AND table_name = 'bs_interaction'
        )
        """
        
        try:
            table_exists = pd.read_sql(check_query, self.engine).iloc[0, 0]
            
            if not table_exists:
                logger.warning("bs_interaction table does not exist. Using zero labels.")
                df['label'] = 0
                return df
            
            interactions_query = """
            SELECT 
                user_id::text as user_id,
                vehicle_id::text as vehicle_id,
                MAX(interaction_score) as max_interaction_score,
                COUNT(*) as interaction_count
            FROM bs_interaction
            WHERE created_at >= NOW() - INTERVAL '90 days'
            GROUP BY user_id, vehicle_id
            """
            
            interactions = pd.read_sql(interactions_query, self.engine)
            
            if interactions.empty:
                logger.warning("No interactions found in last 90 days")
                df['label'] = 0
                return df
            
            df = df.merge(
                interactions,
                on=['user_id', 'vehicle_id'],
                how='left'
            )
            
            df['label'] = df['max_interaction_score'].fillna(0).astype(int)
            logger.info(f"Loaded {len(interactions)} interaction records")
            
        except Exception as e:
            logger.error(f"Error loading interactions: {e}")
            df['label'] = 0
        
        return df
    
    def build_features(
        self, 
        include_interactions: bool = True, 
        max_candidates_per_user: int = 100,
        user_id: Optional[str] = None,
        vehicle_id: Optional[str] = None
    ) -> pd.DataFrame:
        """
        Build features for training or inference.
        
        Args:
            include_interactions: Include interaction labels (for training)
            max_candidates_per_user: Max candidates per user
            user_id: If provided, build features only for this user
            vehicle_id: If provided, build features only for this vehicle
        """
        
        # Generate candidates based on mode
        if user_id:
            df = self.generate_candidates_for_user(user_id, max_candidates_per_user)
        elif vehicle_id:
            df = self.generate_candidates_for_vehicle(vehicle_id, max_candidates_per_user)
        else:
            df = self.generate_candidates_sql(max_candidates_per_user)
        
        if df.empty:
            logger.warning("No candidate pairs generated!")
            return pd.DataFrame()
        
        # Load user preferences and merge
        user_prefs = self.load_user_preferences()
        df = df.merge(user_prefs, on='user_id', how='left')
        
        # Ensure preference columns exist
        for pref_col in ['brand_pref', 'color_pref', 'fuel_type_pref', 'transmission_pref']:
            if pref_col not in df.columns:
                df[pref_col] = None
        
        # Engineer features
        df = self.engineer_features(df)
        
        # Load interaction labels if requested
        if include_interactions:
            df = self.load_interaction_labels(df)
        else:
            df['label'] = 0
        
        # Select final features
        feature_cols = [
            'user_id',
            'vehicle_id',
            # Price features
            'price_diff',
            'price_ratio',
            'price_in_budget',
            'price_percentile',
            'price_optimality',
            # Age features
            'vehicle_age',
            'year_diff',
            # Mileage features
            'mileage_ratio',
            'mileage_acceptable',
            # Match features
            'brand_match',
            'color_match',
            'fuel_match',
            'transmission_match',
            'preference_score',
            # Other
            'urgency_score',
            'is_same_location',
            'overall_match_score',
            # Label
            'label'
        ]
        
        output = df[feature_cols].copy()
        output = output.fillna(0)
        
        logger.info(f"Final dataset: {output.shape}")
        logger.info(f"Positive labels: {(output['label'] > 0).sum()}")
        logger.info(f"Negative labels: {(output['label'] == 0).sum()}")
        
        return output
    
    def close(self):
        """Cleanup database connections."""
        self.engine.dispose()


# Convenience function for backward compatibility
def build_features(include_interactions: bool = True, max_candidates_per_user: int = 100) -> pd.DataFrame:
    """Legacy function wrapper."""
    builder = FeatureBuilder()
    df = builder.build_features(include_interactions, max_candidates_per_user)
    builder.close()
    return df


if __name__ == "__main__":
    # Test the feature builder
    builder = FeatureBuilder()
    df = builder.build_features(include_interactions=True, max_candidates_per_user=50)
    print("\n=== Sample Features ===")
    print(df.head())
    print("\n=== Feature Statistics ===")
    print(df.describe())
    builder.close()