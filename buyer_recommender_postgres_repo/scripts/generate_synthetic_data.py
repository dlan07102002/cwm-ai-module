import pandas as pd
import numpy as np
import uuid
from tqdm import tqdm

# ==============================
# Config
# ==============================
NUM_PRE_ORDERS = 5000
NUM_VEHICLES = 1000
CANDIDATES_PER_ORDER = 20
MATCH_PROB = 0.2  # 20% candidates are positive matches

BRANDS_MODELS = {
    "Toyota": ["Corolla", "Camry", "Fortuner", "Vios"],
    "Honda": ["Civic", "Accord", "CR-V", "City"],
    "Ford": ["Focus", "Escape", "Ranger", "Mustang"],
    "Mazda": ["3", "6", "CX-5", "CX-8"],
    "VinFast": ["VF8", "VF9", "Lux A2.0", "Lux SA2.0"]
}

LOCATIONS = ["Hanoi", "Ho Chi Minh", "Da Nang", "Hai Phong", "Can Tho"]
CONDITIONS = ["new", "used", "damaged"]
TRANSMISSIONS = ["manual", "automatic", "semi-automatic"]
FUEL_TYPES = ["petrol", "diesel", "hybrid", "electric", "other"]

# ==============================
# Helper Functions
# ==============================
def random_price_for_model(brand, model):
    """Return realistic price range for brand/model (in VND millions)."""
    base = {
        "Toyota": 500, "Honda": 500, "Ford": 400, "Mazda": 450, "VinFast": 800
    }
    price = base.get(brand, 500) + np.random.randint(-50, 50)
    return max(100, price) * 1e6  # VND

def random_year():
    return np.random.randint(2010, 2025)

def generate_pre_orders(num_orders):
    orders = []
    for _ in range(num_orders):
        brand = np.random.choice(list(BRANDS_MODELS.keys()))
        model = np.random.choice(BRANDS_MODELS[brand])
        price_min = random_price_for_model(brand, model) * 0.8
        price_max = random_price_for_model(brand, model) * 1.2
        orders.append({
            "pre_order_id": str(uuid.uuid4()),
            "user_id": str(uuid.uuid4()),
            "brand": brand,
            "model": model,
            "year": random_year(),
            "price_min": price_min,
            "price_max": price_max,
            "location": np.random.choice(LOCATIONS),
            "condition": np.random.choice(CONDITIONS),
            "transmission": np.random.choice(TRANSMISSIONS),
            "fuel_type": np.random.choice(FUEL_TYPES)
        })
    return pd.DataFrame(orders)

def generate_vehicles(num_vehicles):
    vehicles = []
    for _ in range(num_vehicles):
        brand = np.random.choice(list(BRANDS_MODELS.keys()))
        model = np.random.choice(BRANDS_MODELS[brand])
        price = random_price_for_model(brand, model)
        vehicles.append({
            "vehicle_id": str(uuid.uuid4()),
            "brand": brand,
            "model": model,
            "year": random_year(),
            "price": price,
            "location": np.random.choice(LOCATIONS),
            "condition": np.random.choice(CONDITIONS),
            "transmission": np.random.choice(TRANSMISSIONS),
            "fuel_type": np.random.choice(FUEL_TYPES)
        })
    return pd.DataFrame(vehicles)

def build_training_table(pre_orders, vehicles):
    rows = []
    for _, order in tqdm(pre_orders.iterrows(), total=len(pre_orders)):
        candidate_vehicles = vehicles.sample(n=CANDIDATES_PER_ORDER, replace=True)
        for _, veh in candidate_vehicles.iterrows():
            # Logic match
            matched = (
                (veh['brand'] == order['brand']) and
                (veh['model'] == order['model']) and
                (order['price_min'] <= veh['price'] <= order['price_max'])
            )
            # Randomly assign some matches to match MATCH_PROB
            if not matched and np.random.rand() < MATCH_PROB:
                matched = True  # add some positive noise
            rows.append({
                "pre_order_id": order["pre_order_id"],
                "user_id": order["user_id"],
                "pre_brand": order["brand"],
                "pre_model": order["model"],
                "pre_year": order["year"],
                "price_min": order["price_min"],
                "price_max": order["price_max"],
                "pre_location": order["location"],
                "veh_brand": veh["brand"],
                "veh_model": veh["model"],
                "veh_year": veh["year"],
                "price": veh["price"],
                "veh_location": veh["location"],
                "user_verify_score": np.random.rand(),
                "bubble_score": np.random.rand(),
                "matched": int(matched)
            })
    return pd.DataFrame(rows)

# ==============================
# Generate Data
# ==============================
print("Generating pre_orders...")
pre_orders = generate_pre_orders(NUM_PRE_ORDERS)
print("Generating vehicles...")
vehicles = generate_vehicles(NUM_VEHICLES)

print("Building training table...")
train_df = build_training_table(pre_orders, vehicles)

# Shuffle rows
train_df = train_df.sample(frac=1).reset_index(drop=True)

# Save parquet
train_df.to_parquet("data/train.parquet", index=False)
print("Saved synthetic training data to data/train.parquet")
