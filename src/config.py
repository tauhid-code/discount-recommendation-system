import os

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

DATA_DIR = os.path.join(BASE_DIR, "data")
RAW_DATA_PATH = os.path.join(DATA_DIR, "customer_data.csv")
PROCESSED_DATA_PATH = os.path.join(DATA_DIR, "processed_data.csv")

MODEL_DIR = os.path.join(BASE_DIR, "models")
MODEL_PATH = os.path.join(MODEL_DIR, "random_forest_model.pkl")

TARGET_COLUMN = "target_discount_growth"