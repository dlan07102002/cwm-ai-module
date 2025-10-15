import numpy as np
import pandas as pd

# Define column constants for clarity and maintainability
NUMERIC_COLS = ['price_min', 'price_max', 'price', 'pre_year', 'veh_year']
FINAL_FEATURE_COLS = [
    'brand_match', 'model_match', 'within_budget', 'price_diff_abs',
    'year_diff', 'same_location', 'user_verify_score', 'bubble_score'
]

def build_features_from_candidates(df: pd.DataFrame) -> pd.DataFrame:
    """
    Engineers features from a DataFrame of candidate pairs for a recommendation model.

    This function takes a DataFrame of pre-order and vehicle data and creates
    a feature set suitable for a machine learning model. It handles missing
    values, performs type conversions, and creates several new features based
    on the comparison between pre-orders and vehicles.

    Args:
        df: A pandas DataFrame with candidate data. Expected columns include:
            pre_brand, pre_model, pre_year, price_min, price_max, pre_location,
            veh_brand, veh_model, veh_year, price, veh_location.
            Optional columns: user_verify_score, bubble_score.

    Returns:
        A new pandas DataFrame containing the engineered features in a specific order.
    """
    # 1. Initialization & Type Conversion
    # Work on a copy to avoid modifying the original DataFrame
    data = df.copy()

    for col in NUMERIC_COLS:
        if col in data.columns:
            # Coerce errors will turn non-numeric values into NaT/NaN
            data[col] = pd.to_numeric(data[col], errors='coerce')

    # 2. Feature Engineering
    features = {}

    # Match features
    features['brand_match'] = (data['pre_brand'].fillna('') == data['veh_brand'].fillna('')).astype(int)
    features['model_match'] = (data['pre_model'].fillna('') == data['veh_model'].fillna('')).astype(int)

    # Budget and price features
    price = data['price']
    price_min = data['price_min']
    price_max = data['price_max']
    
    features['within_budget'] = (
        (price >= price_min.fillna(0)) & (price <= price_max.fillna(np.inf))
    ).astype(int)
    
    # Calculate the middle of the user's desired price range.
    # If a bound is missing, use the vehicle's price as a substitute.
    price_mid = (price_min.fillna(price) + price_max.fillna(price)) / 2.0
    features['price_diff_abs'] = (price - price_mid).abs()

    # Other numeric difference features
    features['year_diff'] = (data['veh_year'].fillna(0) - data['pre_year'].fillna(0)).abs()

    # Location feature
    features['same_location'] = (data.get('pre_location', '').fillna('') == data.get('veh_location', '').fillna('')).astype(int)

    # Placeholder/optional features
    if 'user_verify_score' in df.columns:
        features['user_verify_score'] = df['user_verify_score'].fillna(0.0)
    else:
        features['user_verify_score'] = 0.0  # assign a scalar
    if 'bubble_score' in df.columns:
        features['bubble_score'] = df['bubble_score'].fillna(0.0)
    else:
        features['bubble_score'] = 0.0

    # 3. Final DataFrame construction
    X = pd.DataFrame(features)

    # Ensure final feature set and order, filling missing features with 0.0
    for col in FINAL_FEATURE_COLS:
        if col not in X.columns:
            X[col] = 0.0
            
    return X[FINAL_FEATURE_COLS]