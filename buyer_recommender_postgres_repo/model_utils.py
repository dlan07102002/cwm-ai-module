import numpy as np
import pandas as pd
import numpy as np
import pandas as pd
from sentence_transformers import SentenceTransformer

# =============================
# Constants
# =============================
NUMERIC_COLS = ['price_min', 'price_max', 'price', 'pre_year', 'veh_year']

FINAL_FEATURE_COLS = [
    'brand_match', 'model_match', 'within_budget', 'price_diff_abs',
    'year_diff', 'same_location', 'user_verify_score', 'bubble_score',
    'text_similarity', 'profile_similarity'
]

# =============================
# Global Model Initialization
# =============================
# Load once (fast and memory efficient)
_sbert_model = SentenceTransformer('all-MiniLM-L6-v2')


# =============================
# Helper Functions
# =============================
def compute_text_embeddings(texts):
    """Compute normalized Sentence-BERT embeddings for a list of texts."""
    embeddings = _sbert_model.encode(
        [t if isinstance(t, str) else "" for t in texts],
        normalize_embeddings=True
    )
    return embeddings


def cosine_similarity(vec_a, vec_b):
    """Compute cosine similarity between two sets of normalized vectors."""
    return np.sum(vec_a * vec_b, axis=1)


def build_user_profiles(df):
    """
    Build user profile embeddings from historically matched vehicles.
    Returns a mapping {user_id: np.array(vector)}.
    """
    if "matched" in df.columns:
        df_matched = df[df["matched"] == 1].copy()
    else:
        df_matched = df.copy()
        
    if df_matched.empty:
        return {}

    # Combine textual attributes
    df_matched["veh_text"] = (
        df_matched["veh_brand"].fillna('') + " " + df_matched["veh_model"].fillna('')
    )
    veh_emb = compute_text_embeddings(df_matched["veh_text"].tolist())
    df_matched["veh_emb"] = list(veh_emb)

    # Average embeddings per user
    user_profiles = (
        df_matched.groupby("user_id")["veh_emb"]
        .apply(lambda x: np.mean(np.stack(x), axis=0))
        .to_dict()
    )
    return user_profiles


# =============================
# Main Feature Builder
# =============================
def build_features_from_candidates(df: pd.DataFrame) -> pd.DataFrame:
    """
    Engineers both traditional and AI-augmented features
    from a DataFrame of candidate pairs.
    """

    data = df.copy()

    # ---- Step 1: Type conversion ----
    for col in NUMERIC_COLS:
        if col in data.columns:
            data[col] = pd.to_numeric(data[col], errors='coerce')

    # ---- Step 2: Basic features ----
    features = {}
    features['brand_match'] = (data['pre_brand'].fillna('') == data['veh_brand'].fillna('')).astype(int)
    features['model_match'] = (data['pre_model'].fillna('') == data['veh_model'].fillna('')).astype(int)

    price = data['price']
    price_min = data['price_min']
    price_max = data['price_max']

    features['within_budget'] = (
        (price >= price_min.fillna(0)) & (price <= price_max.fillna(np.inf))
    ).astype(int)

    price_mid = (price_min.fillna(price) + price_max.fillna(price)) / 2.0
    features['price_diff_abs'] = (price - price_mid).abs()
    features['year_diff'] = (data['veh_year'].fillna(0) - data['pre_year'].fillna(0)).abs()
    features['same_location'] = (
        data.get('pre_location', '').fillna('') == data.get('veh_location', '').fillna('')
    ).astype(int)

    if 'user_verify_score' in df.columns:
        features['user_verify_score'] = df['user_verify_score'].fillna(0.0)
    else:
        features['user_verify_score'] = 0.0  # assign a scalar
    if 'bubble_score' in df.columns:
        features['bubble_score'] = df['bubble_score'].fillna(0.0)
    else:
        features['bubble_score'] = 0.0

    # ---- Step 3: Semantic Text Features ----
    buyer_text = data['pre_brand'].fillna('') + " " + data['pre_model'].fillna('')
    vehicle_text = data['veh_brand'].fillna('') + " " + data['veh_model'].fillna('')

    buyer_emb = compute_text_embeddings(buyer_text.tolist())
    vehicle_emb = compute_text_embeddings(vehicle_text.tolist())

    features['text_similarity'] = cosine_similarity(buyer_emb, vehicle_emb)

    # ---- Step 4: User Profile Similarity ----
    # Build user profiles only once per dataset
    user_profiles = build_user_profiles(data)
    profile_sims = []
    for uid, veh_vec in zip(data['user_id'], vehicle_emb):
        if uid in user_profiles:
            sim = np.dot(user_profiles[uid], veh_vec)
        else:
            sim = 0.0
        profile_sims.append(sim)
    features['profile_similarity'] = profile_sims

    # ---- Step 5: Assemble Final Feature Frame ----
    X = pd.DataFrame(features)

    # Ensure correct order & fill missing
    for col in FINAL_FEATURE_COLS:
        if col not in X.columns:
            X[col] = 0.0

    return X[FINAL_FEATURE_COLS]
