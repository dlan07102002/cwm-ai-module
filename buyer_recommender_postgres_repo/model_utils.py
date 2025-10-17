import numpy as np
import pandas as pd
from sentence_transformers import SentenceTransformer
from typing import Dict, Any

# ==============================================================
# Constants
# ==============================================================
NUMERIC_COLS = ['price_min', 'price_max', 'price', 'pre_year', 'veh_year']

FINAL_FEATURE_COLS = [
    'brand_match',
    'model_match',
    'within_budget',
    'price_diff_abs',
    'year_diff',
    'user_pref_sim',          # similarity to user profile (historical matches)
    'user_verify_sim'         # similarity to user verification embedding
]

# Load a lightweight model only once
_embedder = SentenceTransformer('all-MiniLM-L6-v2')


# ==============================================================
# Helper functions
# ==============================================================
def safe_cosine_sim(v1: np.ndarray, v2: np.ndarray) -> float:
    """Compute cosine similarity between two vectors safely."""
    if v1 is None or v2 is None:
        return 0.0
    if np.linalg.norm(v1) == 0 or np.linalg.norm(v2) == 0:
        return 0.0
    return float(np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2)))


# ==============================================================
# User profile builder
# ==============================================================
def build_user_profiles_proper(historical_matches: pd.DataFrame, verify_emb_map: Dict[str, np.ndarray] = None) -> Dict[str, Dict[str, Any]]:
    """
    Builds user profiles based on their historically matched vehicles.
    Adds verify_emb_map if available.

    Returns:
        user_profiles: {
            user_id: {
                'mean_price': float,
                'mean_year': float,
                'pref_emb': np.ndarray,       # aggregated preference embedding
                'verify_emb': np.ndarray      # from verify_emb_map if available
            }
        }
    """
    user_profiles = {}
    if historical_matches.empty:
        print("[WARN] No historical matches found, returning empty profiles.")
        return user_profiles

    print(f"[INFO] Building user profiles for {historical_matches['user_id'].nunique()} users...")

    # Example: assume we have textual columns like 'brand', 'model', 'description'
    for user_id, group in historical_matches.groupby('user_id'):
        profile = {}

        # Numeric preferences
        profile['mean_price'] = group['price'].mean() if 'price' in group else 0.0
        profile['mean_year'] = group['veh_year'].mean() if 'veh_year' in group else 0.0

        # Text preference (use brand + model + description)
        text_parts = []
        for col in ['brand', 'model', 'description']:
            if col in group.columns:
                text_parts.extend(group[col].astype(str).tolist())
        text_blob = " ".join(text_parts).strip()
        if text_blob:
            profile['pref_emb'] = _embedder.encode(text_blob)
        else:
            profile['pref_emb'] = np.zeros(384, dtype=float)

        # Verification embedding (optional)
        profile['verify_emb'] = verify_emb_map.get(str(user_id)) if verify_emb_map else None

        user_profiles[str(user_id)] = profile

    print(f"[INFO] Built {len(user_profiles)} user profiles.")
    return user_profiles


# ==============================================================
# Feature builder
# ==============================================================
def build_features_from_candidates(df: pd.DataFrame, user_profiles: Dict[str, Dict[str, Any]]) -> pd.DataFrame:
    """
    Given candidate buyer-vehicle pairs, compute features including:
      - numeric comparisons (price diff, year diff)
      - string matches (brand/model)
      - similarity to user historical profile
      - similarity to verify embedding (if exists)
    """
    features = []

    for idx, row in df.iterrows():
        user_id = str(row['user_id'])
        prof = user_profiles.get(user_id, {})

        # Basic matching
        brand_match = int(str(row.get('pre_brand', '')).lower() == str(row.get('veh_brand', '')).lower())
        model_match = int(str(row.get('pre_model', '')).lower() == str(row.get('veh_model', '')).lower())

        # Numeric features
        price_diff_abs = abs(float(row.get('price', 0)) - float(prof.get('mean_price', 0)))
        year_diff = abs(float(row.get('veh_year', 0)) - float(prof.get('mean_year', 0)))

        # Within budget
        within_budget = 0
        if 'price_min' in row and 'price_max' in row and 'price' in row:
            try:
                within_budget = int(row['price_min'] <= row['price'] <= row['price_max'])
            except Exception:
                within_budget = 0

        # Text similarity (preference)
        veh_text = " ".join([str(row.get('veh_brand', '')), str(row.get('veh_model', '')), str(row.get('veh_desc', ''))])
        veh_emb = _embedder.encode(veh_text)
        user_pref_emb = prof.get('pref_emb')
        user_pref_sim = safe_cosine_sim(user_pref_emb, veh_emb)

        # Verify similarity
        verify_emb = prof.get('verify_emb')
        user_verify_sim = safe_cosine_sim(verify_emb, veh_emb)

        features.append([
            brand_match,
            model_match,
            within_budget,
            price_diff_abs,
            year_diff,
            user_pref_sim,
            user_verify_sim
        ])

    feat_df = pd.DataFrame(features, columns=FINAL_FEATURE_COLS, index=df.index)
    return feat_df
