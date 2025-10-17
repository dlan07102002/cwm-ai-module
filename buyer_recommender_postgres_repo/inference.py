import numpy as np
import pandas as pd
from sentence_transformers import SentenceTransformer
from typing import Any, Dict, Optional
from datetime import datetime

# =============================
# Constants
# =============================
NUMERIC_COLS = ['price_min', 'price_max', 'price', 'pre_year', 'veh_year', 
                'mileage', 'bubble_score', 'user_verify_score']

FINAL_FEATURE_COLS = [
    'brand_match', 'model_match', 'within_budget', 'price_ratio',
    'year_diff', 'text_similarity', 'profile_similarity',
    'price_percentile_in_budget', 'budget_flexibility_score',
    'pre_order_age_days', 'mileage_normalized', 'location_match',
    'vehicle_popularity_score', 'user_credibility_score'
]

# =============================
# Global Model
# =============================
_sbert_model = SentenceTransformer('all-MiniLM-L6-v2')

# =============================
# Helper Functions
# =============================
def compute_text_embeddings(texts):
    """Compute normalized Sentence-BERT embeddings."""
    embeddings = _sbert_model.encode(
        [t if isinstance(t, str) else "" for t in texts],
        normalize_embeddings=True
    )
    return embeddings

def cosine_similarity(vec_a, vec_b):
    """Compute cosine similarity between normalized vectors."""
    return np.sum(vec_a * vec_b, axis=1)

def build_user_profiles_proper(df_historical: pd.DataFrame) -> Dict[str, np.ndarray]:
    """
    Build user profiles from HISTORICAL data only (not from training set).
    This should be called on a separate historical dataset to avoid leakage.
    
    Args:
        df_historical: DataFrame with columns [user_id, veh_brand, veh_model]
                       from past confirmed matches only.
    
    Returns:
        Dictionary mapping user_id to embedding vector
    """
    if df_historical.empty:
        return {}
    
    df_historical["veh_text"] = (
        df_historical["veh_brand"].fillna('') + " " + 
        df_historical["veh_model"].fillna('')
    )
    veh_emb = compute_text_embeddings(df_historical["veh_text"].tolist())
    df_historical["veh_emb"] = list(veh_emb)
    
    # Average embeddings per user
    user_profiles = (
        df_historical.groupby("user_id")["veh_emb"]
        .apply(lambda x: np.mean(np.stack(x), axis=0))
        .to_dict()
    )
    return user_profiles

def haversine_distance(lat1, lon1, lat2, lon2):
    """Calculate distance in km between two lat/lon points."""
    R = 6371  # Earth radius in km
    lat1, lon1, lat2, lon2 = map(np.radians, [lat1, lon1, lat2, lon2])
    dlat = lat2 - lat1
    dlon = lon2 - lon1
    a = np.sin(dlat/2)**2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon/2)**2
    c = 2 * np.arcsin(np.sqrt(a))
    return R * c

# =============================
# Main Feature Builder
# =============================
def build_features_from_candidates(
    df: pd.DataFrame,
    user_profiles: Optional[Dict[str, np.ndarray]] = None,
    user_verify_emb_map: Optional[Dict[Any, np.ndarray]] = None,
    current_date: Optional[datetime] = None
) -> pd.DataFrame:
    """
    Engineers comprehensive features from candidate pairs.
    
    Args:
        df: DataFrame with candidate pairs
        user_profiles: Pre-built user profiles from historical data (to avoid leakage)
        current_date: Reference date for computing recency features
    
    Returns:
        DataFrame with engineered features
    """
    data = df.copy()
    
    if current_date is None:
        current_date = datetime.now()
    
    # ---- Type conversion ----
    for col in NUMERIC_COLS:
        if col in data.columns:
            data[col] = pd.to_numeric(data[col], errors='coerce')
    
    features = {}
    
    # ---- Basic Match Features ----
    features['brand_match'] = (
        data['pre_brand'].fillna('').str.lower() == 
        data['veh_brand'].fillna('').str.lower()
    ).astype(int)
    
    features['model_match'] = (
        data['pre_model'].fillna('').str.lower() == 
        data['veh_model'].fillna('').str.lower()
    ).astype(int)
    
    # ---- Improved Price Features ----
    price = data['price'].fillna(0)
    price_min = data['price_min'].fillna(0)
    price_max = data['price_max'].fillna(np.inf)
    
    features['within_budget'] = (
        (price >= price_min) & (price <= price_max)
    ).astype(int)
    
    # Price ratio (normalized by budget range)
    budget_range = (price_max - price_min).replace(0, 1)
    price_mid = (price_min + price_max) / 2.0
    features['price_ratio'] = np.clip((price - price_mid) / budget_range, -2, 2)
    
    # Where does this price fall within buyer's budget? (0 to 1)
    features['price_percentile_in_budget'] = np.clip(
        (price - price_min) / budget_range, 0, 1
    )
    
    # Budget flexibility (wider budget = more flexible buyer)
    features['budget_flexibility_score'] = np.clip(
        budget_range / price_mid, 0, 2
    )
    
    # ---- Year Features ----
    features['year_diff'] = (
        data['veh_year'].fillna(0) - data['pre_year'].fillna(0)
    ).abs()
    
    # ---- Recency Features ----
    if 'pre_order_created_at' in data.columns:
        pre_order_dates = pd.to_datetime(data['pre_order_created_at'], errors='coerce')
        features['pre_order_age_days'] = (
            current_date - pre_order_dates
        ).dt.days.fillna(999)
    else:
        features['pre_order_age_days'] = 0
    
    # ---- Mileage Features ----
    if 'mileage' in data.columns:
        # Normalize by vehicle age
        veh_age = current_date.year - data['veh_year'].fillna(current_date.year)
        avg_mileage_per_year = data['mileage'].fillna(0) / veh_age.replace(0, 1)
        features['mileage_normalized'] = np.clip(avg_mileage_per_year / 15000, 0, 3)
    else:
        features['mileage_normalized'] = 0
    
    # ---- Location Features ----
    if all(col in data.columns for col in ['buyer_lat', 'buyer_lon', 'vehicle_lat', 'vehicle_lon']):
        distances = haversine_distance(
            data['buyer_lat'].fillna(0), data['buyer_lon'].fillna(0),
            data['vehicle_lat'].fillna(0), data['vehicle_lon'].fillna(0)
        )
        # Convert to similarity score (closer = better)
        features['location_match'] = np.exp(-distances / 50)  # 50km decay
    else:
        features['location_match'] = 0.5
    
    # ---- Semantic Text Features ----
    buyer_text = (
        data['pre_brand'].fillna('') + " " + data['pre_model'].fillna('')
    )
    vehicle_text = (
        data['veh_brand'].fillna('') + " " + data['veh_model'].fillna('')
    )
    
    buyer_emb = compute_text_embeddings(buyer_text.tolist())
    vehicle_emb = compute_text_embeddings(vehicle_text.tolist())
    
    features['text_similarity'] = cosine_similarity(buyer_emb, vehicle_emb)
    
    # ---- User Profile Similarity (NO LEAKAGE) ----
    if user_profiles is None:
        # For cold start: use text similarity as fallback
        features['profile_similarity'] = features['text_similarity']
    else:
        profile_sims = []
        for uid, veh_vec in zip(data['user_id'], vehicle_emb):
            if uid in user_profiles:
                sim = np.dot(user_profiles[uid], veh_vec)
            else:
                # Cold start: blend text similarity with default
                sim = 0.3 + 0.4 * features['text_similarity'][len(profile_sims)]
            profile_sims.append(sim)
        features['profile_similarity'] = profile_sims
    
    # ---- Vehicle Quality Signals ----
    if 'bubble_score' in data.columns:
        features['vehicle_popularity_score'] = data['bubble_score'].fillna(0) / 100.0
    else:
        features['vehicle_popularity_score'] = 0.5
    
    # ---- User Credibility ----
    if 'user_verify_score' in data.columns:
        features['user_credibility_score'] = data['user_verify_score'].fillna(0) / 100.0
    else:
        features['user_credibility_score'] = 0.5
    
    # ---- Assemble Final DataFrame ----
    X = pd.DataFrame(features)
    
    # Ensure all expected columns exist
    for col in FINAL_FEATURE_COLS:
        if col not in X.columns:
            X[col] = 0.0
    
    return X[FINAL_FEATURE_COLS]