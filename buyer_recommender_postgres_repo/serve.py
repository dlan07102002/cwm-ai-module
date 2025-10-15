from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import joblib, os
import pandas as pd
from db import create_engine_from_env, query_to_df
from model_utils import build_features_from_candidates

app = FastAPI(title='Buyer Recommendation - Postgres')

class VehicleRequest(BaseModel):
    brand: str = None
    model: str = None
    price: float = None
    location: str = None
    year: int = None

@app.on_event('startup')
def startup():
    global model, engine
    if not os.path.exists('model.joblib'):
        raise RuntimeError('model.joblib not found. Run training first.')
    model = joblib.load('model.joblib')
    engine = create_engine_from_env()

@app.post('/recommend_buyers_for_vehicle')
def recommend_buyers_for_vehicle(v: VehicleRequest, top_n: int = 10):
    # Candidate SQL: find pre_orders that could match this vehicle
    sql = """
    SELECT  p.pre_order_id, p.user_id, p.brand AS pre_brand, p.model AS pre_model, p.year AS pre_year,
            p.price_min, p.price_max, NULL::varchar AS pre_location, p.matched_vehicle_id
    FROM public.bs_pre_order p
    WHERE (p.brand IS NULL OR p.brand = :brand)
            AND (p.model IS NULL OR p.model = :model)
            AND ( (:price IS NULL) OR (p.price_min IS NULL OR p.price_max IS NULL OR (:price BETWEEN p.price_min AND p.price_max)) )
    LIMIT 1000
    """
    params = {'brand': v.brand, 'model': v.model, 'price': v.price}
    try:
        candidates = query_to_df(engine, sql, params=params)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
    if len(candidates) == 0:
        return []
    # attach vehicle columns
    candidates['veh_brand'] = v.brand
    candidates['veh_model'] = v.model
    candidates['price'] = v.price
    candidates['veh_location'] = v.location
    candidates['veh_year'] = v.year
    feats = build_features_from_candidates(candidates)
    scores = model.predict(feats)
    candidates['score'] = scores
    top = candidates.sort_values('score', ascending=False).head(top_n)
    return top[['pre_order_id','user_id','score']].to_dict(orient='records')
