import pandas as pd
from sqlalchemy import create_engine
from sklearn.preprocessing import OneHotEncoder, MinMaxScaler
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
import numpy as np
import os

# ======================
# 1. CONFIG
# ======================

engine = create_engine(
    'postgresql+psycopg2://',
    connect_args={
        'host': os.getenv('DB_HOST', '127.0.0.1'),
        'port': int(os.getenv('DB_PORT', 5432)),
        'dbname': os.getenv('DB_NAME', 'postgres'),
        'user': os.getenv('DB_USER', 'postgres'),
        'password': os.getenv('DB_PASSWORD', 'postgres')
    }
)
# ======================
# 2. LOAD DATA
# ======================

query = """
SELECT vehicle_id, brand, model, year, color, mileage, transmission,
       fuel_type, condition, price, seats, position, description
FROM bs_vehicle
WHERE price IS NOT NULL AND brand IS NOT NULL;
"""

df = pd.read_sql(query, engine)
print(f"Loaded {len(df)} vehicles")

# ======================
# 3. FEATURE ENGINEERING
# ======================
categorical_cols = ['brand', 'model', 'color', 'transmission', 'fuel_type', 'condition', 'position']
numeric_cols = ['year', 'price', 'mileage', 'seats']

# Fill missing values
df[categorical_cols] = df[categorical_cols].fillna('Unknown')
df[numeric_cols] = df[numeric_cols].fillna(df[numeric_cols].median())

# ======================
# 4. BUILD PIPELINE
# ======================
preprocessor = ColumnTransformer([
    ('cat', OneHotEncoder(handle_unknown='ignore'), categorical_cols),
    ('num', MinMaxScaler(), numeric_cols)
], remainder='drop')

pipeline = Pipeline(steps=[('preprocessor', preprocessor)])

# Transform data
vehicle_features = pipeline.fit_transform(df)

# ======================
# 5. SIMILARITY FUNCTION
# ======================
def recommend_similar(vehicle_id, top_n=5):
    if vehicle_id not in df['vehicle_id'].values:
        raise ValueError("Vehicle ID không tồn tại trong dữ liệu!")

    idx = df.index[df['vehicle_id'] == vehicle_id][0]

    # Tính cosine similarity
    sim_matrix = cosine_similarity(vehicle_features[idx:idx+1], vehicle_features)[0]

    # Lấy top-n tương tự nhất
    similar_indices = sim_matrix.argsort()[::-1][1:top_n+1]

    return df.iloc[similar_indices][['vehicle_id', 'brand', 'model', 'price', 'year', 'fuel_type', 'transmission']]

# ======================
# 6. DEMO
# ======================
example_vehicle_id = df['vehicle_id'].iloc[6]
print(f"\nXe được chọn: {example_vehicle_id}")
recommendations = recommend_similar(example_vehicle_id, top_n=5)
print("\nTop 5 xe tương tự:")
print(recommendations)
