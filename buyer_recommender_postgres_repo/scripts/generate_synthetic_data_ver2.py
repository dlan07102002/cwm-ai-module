import pandas as pd
import numpy as np
import uuid
from tqdm import tqdm
from sentence_transformers import SentenceTransformer
from sklearn.preprocessing import MinMaxScaler

# ==============================
# Config
# ==============================
NUM_PRE_ORDERS = 5000
NUM_VEHICLES = 1000
CANDIDATES_PER_ORDER = 20
MATCH_PROB = 0.2  # Some positive noise

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

# Initialize SBERT model
model_sbert = SentenceTransformer('all-MiniLM-L6-v2')

# ==============================
# Helper Functions
# ==============================
def random_price_for_model(brand, model):
    base = {"Toyota": 500, "Honda": 500, "Ford": 400, "Mazda": 450, "VinFast": 800}
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

def compute_text_similarity(pre_texts, veh_texts):
    embeddings_pre = model_sbert.encode(pre_texts, convert_to_numpy=True)
    embeddings_veh = model_sbert.encode(veh_texts, convert_to_numpy=True)
    # Cosine similarity
    sim = np.einsum('ij,ij->i', embeddings_pre, embeddings_veh)
    sim /= (np.linalg.norm(embeddings_pre, axis=1) * np.linalg.norm(embeddings_veh, axis=1) + 1e-8)
    return sim

def build_training_table(pre_orders, vehicles):
    rows = []
    for _, order in tqdm(pre_orders.iterrows(), total=len(pre_orders)):
        candidate_vehicles = vehicles.sample(n=CANDIDATES_PER_ORDER, replace=True)
        pre_text = f"{order['brand']} {order['model']} {order['year']}"
        for _, veh in candidate_vehicles.iterrows():
            veh_text = f"{veh['brand']} {veh['model']} {veh['year']}"
            
            # Label
            matched = (
                (veh['brand'] == order['brand']) and
                (veh['model'] == order['model']) and
                (order['price_min'] <= veh['price'] <= order['price_max'])
            )
            if not matched and np.random.rand() < MATCH_PROB:
                matched = True
            
            # Features
            brand_match = int(order['brand'] == veh['brand'])
            model_match = int(order['model'] == veh['model'])
            within_budget = int(order['price_min'] <= veh['price'] <= order['price_max'])
            price_diff_abs = abs(((order['price_min'] + order['price_max']) / 2) - veh['price'])
            year_diff = abs(order['year'] - veh['year'])
            same_location = int(order['location'] == veh['location'])
            condition_match = int(order['condition'] == veh['condition'])
            transmission_match = int(order['transmission'] == veh['transmission'])
            
            # text_similarity using SBERT
            text_similarity = compute_text_similarity([pre_text], [veh_text])[0]
            
            # profile_similarity as average of location/condition/transmission
            profile_similarity = np.mean([same_location, condition_match, transmission_match])
            
            rows.append({
                "pre_order_id": order["pre_order_id"],
                "user_id": order["user_id"],
                "pre_brand": order["brand"],
                "pre_model": order["model"],
                "pre_year": order["year"],
                "price_min": order["price_min"],
                "price_max": order["price_max"],
                "veh_brand": veh["brand"],
                "veh_model": veh["model"],
                "veh_year": veh["year"],
                "price": veh["price"],
                "brand_match": brand_match,
                "model_match": model_match,
                "within_budget": within_budget,
                "price_diff_abs": price_diff_abs,
                "year_diff": year_diff,
                "same_location": same_location,
                "condition_match": condition_match,
                "transmission_match": transmission_match,
                "text_similarity": text_similarity,
                "profile_similarity": profile_similarity,
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

# Normalize numeric columns (optional)
scaler = MinMaxScaler()
for col in ["price_diff_abs", "year_diff", "text_similarity", "profile_similarity"]:
    train_df[col] = scaler.fit_transform(train_df[[col]])

# Save parquet
train_df.to_parquet("data/train_ver_2.parquet", index=False)
print("Saved high-quality synthetic training data to data/train_ver_2.parquet")
