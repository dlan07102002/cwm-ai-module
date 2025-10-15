# Buyer Recommender (Postgres integration)

This repo extends the prototype to connect to PostgreSQL and run an SQL -> ML pipeline using real tables:
`bs_pre_order` (buyer intents) and `bs_vehicle` (listings). It includes ETL helpers, a LightGBM training script,
and a FastAPI serving script that can query candidates from Postgres and rank them.

## Files
- `.env.example` - example env vars for DB connection
- `db.py` - SQLAlchemy engine & helper functions
- `etl.py` - extract candidate pairs and build training table
- `model_utils.py` - feature engineering helpers
- `train_real.py` - training script (LightGBM) to produce `model.joblib`
- `serve.py` - FastAPI app that uses DB to generate candidates and rank them
- `requirements.txt` - Python deps
- `Dockerfile` - container for service
- `scripts/run_etl.sh` - helper to run ETL and train locally

## How it works (high level)
1. ETL (`etl.py`) connects to Postgres and extracts joined user pre-orders and vehicles to produce training data.
   - It selects candidates with simple SQL filters (brand, price range, location)
   - Labels are derived from `bs_pre_order.matched_vehicle_id` (1 if matched to that vehicle)
2. Feature engineering (`model_utils.py`) converts joined rows to ML features (brand/model match, price gap, verification score if exists, bubble embedding placeholders).
3. Training (`train_real.py`) trains a LightGBM classifier on the features and saves `model.joblib`.
4. Serving (`serve.py`) exposes `/recommend_buyers_for_vehicle` endpoint which:
   - queries candidate pre_orders from Postgres (fast filter)
   - builds features, runs model, and returns top-N buyers to the caller.

## Quickstart (local)
1. Copy `.env.example` to `.env` and fill DB credentials.
2. Create virtualenv and install deps:
   ```bash
   python -m venv venv
   source venv/bin/activate
   pip install -r requirements.txt
   ```
3. Run ETL & train:
   ```bash
   python etl.py --mode build_train --out data/train.parquet
   python train_real.py --train data/train.parquet --model model.joblib
   ```
4. Start server (after model.joblib exists):
   ```bash
   uvicorn serve:app --host 0.0.0.0 --port 8000
   ```

## Notes
- The code includes fallbacks: if Postgres isn't reachable, `etl.py` can generate synthetic data for local experimentation.
- Adapt SQL in `etl.py` to include more join columns (user verify answers, bubble answers) when those tables exist.
- For production, add candidate generation with ANN (Faiss/Milvus) and a feature store (Feast/Redis).
