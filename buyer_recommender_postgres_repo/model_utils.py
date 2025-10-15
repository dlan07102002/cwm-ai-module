import numpy as np
import pandas as pd

def build_features_from_candidates(df):
    # df expected to have columns: pre_brand, pre_model, pre_year, price_min, price_max, pre_location,
    # veh_brand, veh_model, veh_year, price, veh_location
    X = pd.DataFrame()
    X['brand_match'] = (df['pre_brand'].fillna('') == df['veh_brand'].fillna('')).astype(int)
    X['model_match'] = (df['pre_model'].fillna('') == df['veh_model'].fillna('')).astype(int)
    X['within_budget'] = ((df['price'] >= df['price_min'].fillna(0)) & (df['price'] <= df['price_max'].fillna(1e18))).astype(int)
    X['price_mid'] = (df['price_min'].fillna(df['price']) + df['price_max'].fillna(df['price']))/2.0
    X['price_diff_abs'] = (df['price'] - X['price_mid']).abs()
    X['year_diff'] = (df['veh_year'].fillna(0) - df['pre_year'].fillna(0)).abs()
    X['same_location'] = (df['pre_location'].fillna('') == df['veh_location'].fillna('')).astype(int)
    # Placeholder features for verification/bubble - if tables available, join and compute real values
    if 'user_verify_score' in df.columns:
        X['user_verify_score'] = df['user_verify_score'].fillna(0.0)
    else:
        X['user_verify_score'] = 0.0
    if 'bubble_score' in df.columns:
        X['bubble_score'] = df['bubble_score'].fillna(0.0)
    else:
        X['bubble_score'] = 0.0
    # final features order
    feat_cols = ['brand_match','model_match','within_budget','price_diff_abs','year_diff','same_location','user_verify_score','bubble_score']
    return X[feat_cols]
