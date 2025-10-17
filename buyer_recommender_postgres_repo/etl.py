"""
ETL + Feature Engineering pipeline:
Extract candidate pairs from Postgres, enrich with bs_verify_answer + bs_verify_question,
and generate full model-ready features (LightGBM input).

Usage:
    python etl.py --out data/train_features.parquet --limit 100000
"""

import argparse
import os
from pathlib import Path
from typing import Any, Dict
from datetime import datetime

import pandas as pd
from sqlalchemy.engine import Engine

from db import create_engine_from_env, query_to_df
from model_utils import build_features_from_candidates, compute_text_embeddings  # ✅ import from your existing feature file


# ============================================================== #
# Constants
# ============================================================== #
DEFAULT_LIMIT = 500000
SQL_QUERY_DIR = Path(__file__).parent / "queries"


# ============================================================== #
# Utility to read SQL queries
# ============================================================== #
def get_sql(name: str) -> str:
    """Reads an SQL query from the 'queries' directory."""
    query_path = SQL_QUERY_DIR / f"{name}.sql"
    if not query_path.exists():
        raise FileNotFoundError(f"Error: Query file not found at {query_path}")
    with open(query_path, "r", encoding="utf-8") as f:
        return f.read()


# ============================================================== #
# Load and aggregate user verification answers
# ============================================================== #
def fetch_and_aggregate_user_verifications(engine: Engine) -> pd.DataFrame:
    """
    Return DataFrame with columns: user_id (string) and user_verify_text (concatenated Q:V).
    Each user gets one concatenated text block of all their answers.
    """
    sql = """
        SELECT 
            a.user_id::text AS user_id, 
            q.question_text, 
            a.answer_value
        FROM public.bs_verify_answer a
        JOIN public.bs_verify_question q 
            ON a.question_id = q.question_id
        WHERE a.answer_value IS NOT NULL
    """

    try:
        with engine.connect() as conn:
            df_ans = pd.read_sql_query(sql, con=conn)
    except Exception as e:
        print(f"[ETL] Failed to read verification answers: {e}")
        return pd.DataFrame(columns=["user_id", "user_verify_text"])

    if df_ans.empty:
        print("[ETL] No verification answers found.")
        return pd.DataFrame(columns=["user_id", "user_verify_text"])

    # Clean and concatenate Q&A
    df_ans["qv"] = (
        df_ans["question_text"].astype(str).str.strip() + ": " +
        df_ans["answer_value"].astype(str).str.strip()
    )
    user_verify = (
        df_ans.groupby("user_id")["qv"]
        .apply(lambda arr: " ; ".join(arr))
        .reset_index()
        .rename(columns={"qv": "user_verify_text"})
    )

    print(f"[ETL] Aggregated verification text for {len(user_verify)} users")
    return user_verify


# ============================================================== #
# Build candidate pairs
# ============================================================== #
def build_candidate_table(engine: Engine, limit: int = DEFAULT_LIMIT) -> pd.DataFrame:
    """Queries the database to build a table of candidate pre-order/vehicle pairs."""
    print("[ETL] Querying candidates from DB...")
    base_candidate_sql = get_sql("base_candidate")

    params = {"limit": limit}
    df = query_to_df(engine, base_candidate_sql, params=params)

    print(f"[ETL] Retrieved {len(df)} candidate rows")
    return df


# ============================================================== #
# Label candidates
# ============================================================== #
def label_candidates(df: pd.DataFrame) -> pd.DataFrame:
    """Adds the 'matched' label to the DataFrame."""
    if "matched_vehicle_id" not in df.columns or "vehicle_id" not in df.columns:
        raise ValueError("Expected columns 'matched_vehicle_id' and 'vehicle_id' not found.")

    s1 = df["matched_vehicle_id"].fillna("__na_1__").astype(str)
    s2 = df["vehicle_id"].fillna("__na_2__").astype(str)
    df["matched"] = (s1 == s2).astype(int)

    return df


# ============================================================== #
# Run full ETL + Feature Engineering
# ============================================================== #
def run_etl(output_path: str, limit: int) -> None:
    """Main ETL process to build and save the training data."""
    print("[ETL] Creating DB engine...")
    engine = create_engine_from_env()

    # 1️⃣ Extract candidate pairs
    df = build_candidate_table(engine, limit)
    df = label_candidates(df)

    # 2️⃣ Fetch and merge user verification answers
    print("[ETL] Fetching user verification answers...")
    user_verify_df = fetch_and_aggregate_user_verifications(engine)

    df["user_id"] = df["user_id"].astype(str)
    user_verify_df["user_id"] = user_verify_df["user_id"].astype(str)
    df = df.merge(user_verify_df, on="user_id", how="left")
    df["user_verify_text"] = df["user_verify_text"].fillna("")

    # 3️⃣ Compute user verification embeddings
    print("[ETL] Computing user verification embeddings...")
    user_verify_embs = compute_text_embeddings(df["user_verify_text"].tolist())
    user_verify_emb_map = {
        uid: emb for uid, emb in zip(df["user_id"], user_verify_embs)
    }

    # 4️⃣ Build ML features (using the improved feature builder)
    print("[ETL] Building model features...")
    feature_df = build_features_from_candidates(
        df,
        user_profiles=None,  # historical data optional
        user_verify_emb_map=user_verify_emb_map,
        current_date=datetime.now()
    )

    # 5️⃣ Combine with label
    final_df = pd.concat([df[["user_id", "vehicle_id", "matched"]], feature_df], axis=1)

    # 6️⃣ Save
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    final_df.to_parquet(output_path, index=False)
    print(f"[ETL] ✅ Saved feature dataset to {output_path}")
    print(f"[ETL] Final shape: {final_df.shape}")


# ============================================================== #
# CLI entry point
# ============================================================== #
def main():
    """Main entry point for the script."""
    parser = argparse.ArgumentParser(description="Build training features for buyer recommender.")
    parser.add_argument("--out", default="data/train_features.parquet", help="Output parquet file.")
    parser.add_argument("--limit", type=int, default=DEFAULT_LIMIT, help="Candidate limit.")
    args = parser.parse_args()

    try:
        run_etl(args.out, args.limit)
    except Exception as e:
        print(f"[ETL] ❌ ETL process failed: {e}")
        raise


if __name__ == "__main__":
    main()
