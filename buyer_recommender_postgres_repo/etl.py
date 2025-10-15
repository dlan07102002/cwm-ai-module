"""ETL helper: extract candidate pairs from Postgres joining bs_pre_order and bs_vehicle.
Generates a training table with label 'matched' (1/0).

Usage:
    python etl.py --mode build_train --out data/train.parquet
"""
import os
import argparse
from db import create_engine_from_env, query_to_df
import pandas as pd
import numpy as np

SAMPLE_LIMIT = 500000  # limit rows to avoid huge exports in demo

BASE_CANDIDATE_SQL = '''
-- candidate generation: simple lateral join with filters (tune for your data)
SELECT p.pre_order_id, p.user_id, p.brand AS pre_brand, p.model AS pre_model,
       p.year AS pre_year, p.price_min, p.price_max, NULL::varchar AS pre_location,
       v.vehicle_id, v.brand AS veh_brand, v.model AS veh_model,
       v.year AS veh_year, v.price, NULL::varchar AS veh_location,
       p.matched_vehicle_id
FROM public.bs_pre_order p
CROSS JOIN LATERAL (
    SELECT * FROM public.bs_vehicle v
    WHERE (p.brand IS NULL OR v.brand = p.brand)
      AND (p.model IS NULL OR v.model = p.model)
      -- price filter: vehicle.price within buyer's desired range +/- 20%
      AND v.price BETWEEN COALESCE(p.price_min, 0) * 0.8 AND COALESCE(p.price_max, 999999999999)
    LIMIT 500
) v
LIMIT :limit;
'''

def build_candidate_table(engine, limit=SAMPLE_LIMIT):
    print('Querying candidates from DB...')
    df = query_to_df(engine, BASE_CANDIDATE_SQL, params={'limit': limit})
    print('Retrieved', len(df), 'rows')
    return df

def label_candidates(df):
    # label matched = 1 if matched_vehicle_id == vehicle_id
    df['matched'] = (df['matched_vehicle_id'].astype(str) == df['vehicle_id'].astype(str)).astype(int)
    return df

def main_build_train(out_path):
    engine = create_engine_from_env()
    df = build_candidate_table(engine)
    df = label_candidates(df)
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    df.to_parquet(out_path, index=False)
    print('Saved train data to', out_path)

def generate_synthetic(out_path, n_users=2000, n_listings=200):
    # Fallback generator (simple) if DB not available
    import numpy as np, pandas as pd
    np.random.seed(42)
    BRANDS = ['Toyota','Honda','Ford','BMW','Mazda']
    users = pd.DataFrame({'user_id':[f'u{i}' for i in range(n_users)],
                          'brand': np.random.choice(BRANDS, size=n_users)})
    listings = pd.DataFrame({'vehicle_id':[f'l{i}' for i in range(n_listings)],
                             'brand': np.random.choice(BRANDS, size=n_listings),
                             'price': np.random.randint(300000,2500000,size=n_listings)})
    pairs = []
    for u in users.itertuples(index=False):
        sample = listings.sample(10)
        for v in sample.itertuples(index=False):
            pairs.append({'pre_order_id': f'po_{u.user_id}',
                          'user_id': u.user_id,
                          'pre_brand': u.brand,
                          'vehicle_id': v.vehicle_id,
                          'veh_brand': v.brand,
                          'price': v.price,
                          'matched_vehicle_id': None})
    df = pd.DataFrame(pairs)
    df['matched'] = 0
    df.to_parquet(out_path, index=False)
    print('Saved synthetic train to', out_path)

if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--mode', choices=['build_train','generate_synth'], default='build_train')
    p.add_argument('--out', default='data/train.parquet')
    args = p.parse_args()
    if args.mode == 'build_train':
        try:
            eng = create_engine_from_env()
            main_build_train(args.out)
        except Exception as e:
            print('DB error:', e)
            print('Falling back to generating synthetic data...')
            generate_synthetic(args.out)
    else:
        generate_synthetic(args.out)
