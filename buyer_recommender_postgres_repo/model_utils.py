import numpy as np
import pandas as pd
from sentence_transformers import SentenceTransformer
from typing import Dict, Any, Optional

# =============================
# Constants
# =============================
NUMERIC_COLS = ['price_min', 'price_max', 'price', 'pre_year', 'veh_year']

FINAL_FEATURE_COLS = [
    'brand_match', 'model_match', 'within_budget', 'price_diff_abs',
    'year_diff', 'text_similarity', 'profile_similarity'
]

# =============================
# Global Model Initialization
# =============================
# You can change model name or device in a later pass if you want GPU usage
_sbert_model = SentenceTransformer('all-MiniLM-L6-v2')

# =============================
# Helper Functions
# =============================

def compute_text_embeddings(texts, batch_size: int = 64):
    """Compute normalized Sentence-BERT embeddings for a list of texts."""
    # ensure strings
    clean_texts = [t if isinstance(t, str) else "" for t in texts]
    embeddings = _sbert_model.encode(
        clean_texts,
        normalize_embeddings=True,
        batch_size=batch_size,
        show_progress_bar=False
    )
    return embeddings


def _ensure_2d(vec):
    """Ensure vector is 2D numpy array."""
    arr = np.asarray(vec)
    if arr.ndim == 1:
        return arr.reshape(1, -1)
    return arr


def cosine_similarity(vec_a, vec_b):
    """
    Compute cosine similarity between two sets of normalized vectors.
    vec_a: (N, D) or (N,) ; vec_b: (N, D) or (N,)
    returns an array of length N with pairwise cosine.
    """
    A = _ensure_2d(vec_a)
    B = _ensure_2d(vec_b)
    if A.shape != B.shape:
        # broadcast last row if needed (e.g., profile vector compared to many veh vectors)
        if A.shape[1] != B.shape[1]:
            raise ValueError("Vector dimensions do not match for cosine similarity.")
    # both assumed normalized; safe fallback normalization
    def safe_norm(X):
        norms = np.linalg.norm(X, axis=1, keepdims=True)
        norms[norms == 0] = 1.0
        return X / norms
    A_n = safe_norm(A)
    B_n = safe_norm(B)
    return np.sum(A_n * B_n, axis=1)


def build_user_profiles(df: pd.DataFrame) -> Dict[Any, np.ndarray]:
    """
    Build user profile embeddings from historically matched vehicles.
    Returns a mapping {user_id: np.array(vector)}.

    Expects df to contain columns 'user_id', 'veh_brand', 'veh_model' and
    should only contain *historical matches* (no leakage).
    """
    if df is None or df.empty:
        return {}

    # Combine textual attributes
    df_local = df.copy()
    df_local["veh_text"] = (
        df_local["veh_brand"].fillna('') + " " + df_local["veh_model"].fillna('')
    )

    unique_texts = df_local["veh_text"].tolist()
    emb = compute_text_embeddings(unique_texts)
    # Add embeddings as list objects so we can groupby
    df_local = df_local.reset_index(drop=True)
    df_local["veh_emb"] = list(emb)

    # Average embeddings per user (mean). If user has a single record, that's the vector.
    user_profiles_raw = (
        df_local.groupby("user_id")["veh_emb"]
        .apply(lambda x: np.mean(np.stack(list(x)), axis=0))
        .to_dict()
    )

    # Normalize all user profile vectors and compute population mean vector as fallback
    profiles = {}
    all_vecs = []
    for uid, vec in user_profiles_raw.items():
        v = np.asarray(vec, dtype=float)
        norm = np.linalg.norm(v)
        if norm == 0:
            v = v
        else:
            v = v / norm
        profiles[uid] = v
        all_vecs.append(v)

    if len(all_vecs) > 0:
        population_mean = np.mean(np.stack(all_vecs), axis=0)
        # normalize population mean
        pm_norm = np.linalg.norm(population_mean)
        if pm_norm > 0:
            population_mean = population_mean / pm_norm
    else:
        population_mean = None

    # Store population mean under special key to return along with profiles
    # but since signature expects only dict mapping, we'll return profiles and set
    # a reserved key '__pop_mean__' for fallback usage
    if population_mean is not None:
        profiles["__pop_mean__"] = population_mean

    return profiles


# =============================
# Main Feature Builder
# =============================

def build_features_from_candidates(
    df: pd.DataFrame,
    historical_matches: Optional[pd.DataFrame] = None,
    batch_size: int = 64
) -> pd.DataFrame:
    """
    Engineers features from a DataFrame of candidate pairs.

    Important: pass historical_matches containing *only* prior matches (no leakage).
    If historical_matches is None, no user profile similarities will be computed
    (profile_similarity will use a population fallback or zeros).

    The function will compute text embeddings for the rows in `df` provided.
    For scale, later you should precompute unique embeddings and pass them in
    instead of re-encoding repeated texts.
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

    price = data['price'].fillna(0.0)
    price_min = data.get('price_min', pd.Series([np.nan] * len(data))).fillna(np.nan)
    price_max = data.get('price_max', pd.Series([np.nan] * len(data))).fillna(np.nan)

    # within_budget: if price_min or price_max missing treat as not bounding (loose)
    price_min_f = price_min.fillna(0.0)
    price_max_f = price_max.fillna(np.inf)

    features['within_budget'] = ((price >= price_min_f) & (price <= price_max_f)).astype(int)

    price_mid = (price_min.fillna(price) + price_max.fillna(price)) / 2.0
    features['price_diff_abs'] = (price - price_mid).abs()
    features['year_diff'] = (data.get('veh_year', pd.Series([0]*len(data))).fillna(0) - data.get('pre_year', pd.Series([0]*len(data))).fillna(0)).abs()

    # ---- Step 3: Semantic Text Features ----
    buyer_text = data['pre_brand'].fillna('') + " " + data['pre_model'].fillna('')
    vehicle_text = data['veh_brand'].fillna('') + " " + data['veh_model'].fillna('')

    buyer_emb = compute_text_embeddings(buyer_text.tolist(), batch_size=batch_size)
    vehicle_emb = compute_text_embeddings(vehicle_text.tolist(), batch_size=batch_size)

    # ensure shapes
    buyer_emb = np.asarray(buyer_emb)
    vehicle_emb = np.asarray(vehicle_emb)

    # text similarity: dot product because embeddings are normalized by compute_text_embeddings
    features['text_similarity'] = cosine_similarity(buyer_emb, vehicle_emb)

    # ---- Step 4: User Profile Similarity ----
    # Build user profiles from historical_matches (to avoid leakage)
    if historical_matches is not None and not historical_matches.empty:
        profiles = build_user_profiles(historical_matches)
    else:
        profiles = {}

    pop_mean = profiles.get("__pop_mean__") if profiles is not None else None

    profile_sims = []
    # ensure vehicle_emb rows correspond to order of data
    for uid, veh_vec in zip(data['user_id'], vehicle_emb):
        if uid in profiles:
            user_vec = profiles[uid]
            # be defensive: ensure shapes match
            try:
                sim = np.dot(user_vec, veh_vec) / (np.linalg.norm(user_vec) * (np.linalg.norm(veh_vec) + 1e-12))
            except Exception:
                # fallback to cosine_similarity util (single-pair)
                sim = cosine_similarity(user_vec.reshape(1, -1), veh_vec.reshape(1, -1))[0]
        else:
            if pop_mean is not None:
                # use population mean similarity
                try:
                    sim = np.dot(pop_mean, veh_vec) / (np.linalg.norm(pop_mean) * (np.linalg.norm(veh_vec) + 1e-12))
                except Exception:
                    sim = cosine_similarity(pop_mean.reshape(1, -1), veh_vec.reshape(1, -1))[0]
            else:
                sim = 0.0  # conservative fallback
        # clip to [-1,1]
        sim = float(np.clip(sim, -1.0, 1.0))
        profile_sims.append(sim)

    features['profile_similarity'] = profile_sims

    # ---- Step 5: Assemble Final Feature Frame ----
    X = pd.DataFrame(features)

    # Ensure correct order & fill missing
    for col in FINAL_FEATURE_COLS:
        if col not in X.columns:
            X[col] = 0.0

    # Ensure deterministic column order
    X = X[FINAL_FEATURE_COLS]

    return X
