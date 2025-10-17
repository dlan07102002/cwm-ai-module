"""ETL helper: extract candidate pairs from Postgres joining bs_pre_order and bs_vehicle.
Generates a training table with label 'matched' (1/0).

Usage:
    python etl.py --out data/train.parquet --limit 100000
"""
import argparse
import os
from pathlib import Path
from typing import Any, Dict

import pandas as pd
from sqlalchemy.engine import Engine

from db import create_engine_from_env, query_to_df

DEFAULT_LIMIT = 500000
SQL_QUERY_DIR = Path(__file__).parent / "queries"


def get_sql(name: str) -> str:
    """Reads an SQL query from the 'queries' directory."""
    query_path = SQL_QUERY_DIR / f"{name}.sql"
    try:
        with open(query_path, "r") as f:
            return f.read()
    except FileNotFoundError:
        print(f"Error: Query file not found at {query_path}")
        raise


def build_candidate_table(engine: Engine, limit: int = DEFAULT_LIMIT) -> pd.DataFrame:
    """Queries the database to build a table of candidate pre-order/vehicle pairs."""
    print("Querying candidates from DB...")
    base_candidate_sql = get_sql("base_candidate")
    df = query_to_df(engine, base_candidate_sql, params={'limit': limit})
    print(f"Retrieved {len(df)} rows")
    return df


def label_candidates(df: pd.DataFrame) -> pd.DataFrame:
    """Adds the 'matched' label to the DataFrame."""
    # label matched = 1 if matched_vehicle_id == vehicle_id
    # Coerce to strings, but fill NA with distinct values to prevent NaN == NaN matching
    s1 = df['matched_vehicle_id'].fillna('__na_1__').astype(str)
    s2 = df['vehicle_id'].fillna('__na_2__').astype(str)
    df['matched'] = (s1 == s2).astype(int)
    return df


def run_etl(output_path: str, limit: int) -> None:
    """
    Main ETL process to build and save the training data.
    """
    engine = create_engine_from_env()
    df = build_candidate_table(engine, limit)
    df = label_candidates(df)

    # Ensure output directory exists
    output_dir = os.path.dirname(output_path)
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)

    df.to_parquet(output_path, index=False)
    print(f"Saved training data to {output_path}")


def main():
    """Main entry point for the script."""
    parser = argparse.ArgumentParser(description="Build training data for buyer recommender.")
    parser.add_argument(
        "--out",
        default="data/train.parquet",
        help="Output path for the training data parquet file."
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=DEFAULT_LIMIT,
        help="Maximum number of candidate pairs to generate."
    )
    args = parser.parse_args()

    try:
        run_etl(args.out, args.limit)
    except Exception as e:
        print(f"An error occurred during the ETL process: {e}")


if __name__ == "__main__":
    main()