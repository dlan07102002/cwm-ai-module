import os
from pathlib import Path
from typing import List

import joblib
from fastapi import FastAPI, HTTPException, Request
from pydantic import BaseModel
from db import create_engine_from_env, query_to_df
from model_utils import build_features_from_candidates

# --- Constants ---
MODEL_PATH = "models/lgbm_ranker.joblib"
CANDIDATE_LIMIT = 1000
SQL_QUERY_DIR = Path(__file__).parent / "queries"

# --- FastAPI App Initialization ---
app = FastAPI(title="Buyer Recommendation - Postgres")

# --- Pydantic Models for API ---
class VehicleRequest(BaseModel):
    brand: str = None
    model: str = None
    price: float = None
    location: str = None
    year: int = None

class Recommendation(BaseModel):
    pre_order_id: str
    user_id: str
    score: float

# --- Helper Functions ---
def get_sql(name: str) -> str:
    """Reads an SQL query from the 'queries' directory."""
    query_path = SQL_QUERY_DIR / f"{name}.sql"
    try:
        with open(query_path, "r") as f:
            return f.read()
    except FileNotFoundError as e:
        raise RuntimeError(f"Query file not found at {query_path}") from e

# --- Application Events ---
@app.on_event("startup")
def startup():
    """Load resources on startup: ML model and database engine."""
    if not os.path.exists(MODEL_PATH):
        raise RuntimeError(f"{MODEL_PATH} not found. Run training first.")
    
    # For production, consider using an async DB driver (e.g., asyncpg)
    # to avoid blocking the event loop.
    app.state.engine = create_engine_from_env()
    app.state.model = joblib.load(MODEL_PATH)
    print("Application startup complete. Model and DB engine loaded.")

# --- API Endpoint ---
@app.post("/recommend_buyers_for_vehicle", response_model=List[Recommendation])
def recommend_buyers_for_vehicle(v: VehicleRequest, request: Request, top_n: int = 10):
    """
    Recommends potential buyers for a given vehicle.

    1.  Finds candidate pre-orders from the database.
    2.  Scores candidates using a pre-trained ML model.
    3.  Returns the top N recommendations.
    """
    sql = get_sql("recommend_buyers")
    params = {
        'brand': v.brand,
        'model': v.model,
        'price': v.price,
        'limit': CANDIDATE_LIMIT
    }

    try:
        candidates = query_to_df(request.app.state.engine, sql, params=params)
    except Exception as e:
        # In a real app, log the error properly.
        print(f"Database query failed: {e}")
        raise HTTPException(status_code=500, detail="Error querying the database.")

    if candidates.empty:
        return []

    # Attach vehicle data to candidates for feature engineering
    candidates['veh_brand'] = v.brand
    candidates['veh_model'] = v.model
    candidates['price'] = v.price
    candidates['veh_location'] = v.location
    candidates['veh_year'] = v.year

    # Score candidates
    features = build_features_from_candidates(candidates)
    # Use predict to get a probability score (0.0 to 1.0) for the 'matched' class
    scores = request.app.state.model.predict(features)
    candidates['score'] = scores

    # Get top N results
    top_results = candidates.sort_values('score', ascending=False).head(top_n)

    return top_results[['pre_order_id', 'user_id', 'score']].to_dict(orient='records')