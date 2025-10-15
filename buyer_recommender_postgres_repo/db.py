from dotenv import load_dotenv
import os
from sqlalchemy import create_engine, text
from sqlalchemy.engine import Engine
from sqlalchemy.pool import QueuePool
from urllib.parse import quote_plus
from pathlib import Path

# path to parent-level .env
dotenv_path = Path(__file__).resolve().parent.parent / '.env'
load_dotenv(dotenv_path)

def get_db_url_from_env():
    host = os.getenv('DB_HOST','localhost')
    port = os.getenv('DB_PORT','5432')
    db = os.getenv('DB_NAME','postgres')
    user = os.getenv('DB_USER','postgres')
    pw = os.getenv('DB_PASSWORD','')
    return f"postgresql+psycopg2://{quote_plus(user)}:{quote_plus(pw)}@{host}:{port}/{db}"

def create_engine_from_env() -> Engine:
    db_url = get_db_url_from_env()
    engine = create_engine(db_url, poolclass=QueuePool,  connect_args={"client_encoding": "utf8"}, pool_pre_ping=True)
    return engine

# Helper to run query and return DataFrame
def query_to_df(engine: Engine, sql: str, params: dict=None):
    import pandas as pd
    with engine.connect() as conn:
        result = conn.execute(text(sql), params or {})
        df = pd.DataFrame(result.fetchall(), columns=result.keys())
    return df
