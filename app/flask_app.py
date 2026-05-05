import os
import sys
import csv
from flask import Flask, request, jsonify

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(BASE_DIR, ".."))
sys.path.append(PROJECT_ROOT)

from src.predict import DiscountPredictor

app = Flask(__name__)

predictor = DiscountPredictor(
    model_path=os.path.join(PROJECT_ROOT, "models", "random_forest_model.pkl"),
    feature_columns_path=os.path.join(PROJECT_ROOT, "models", "feature_columns.pkl")
)

def load_global_shap():
    shap_path = os.path.join(PROJECT_ROOT, "models", "global_shap.csv")
    data = []
    with open(shap_path, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            data.append({
                "feature": row["feature"],
                "importance": float(row["importance"])
            })
    data.sort(key=lambda x: x["importance"], reverse=True)
    return data

GLOBAL_SHAP = load_global_shap()


@app.route("/predict", methods=["POST"])
def predict():
    try:
        user_input = request.json
        result = predictor.predict(user_input)
        return jsonify(result)
    except Exception as e:
        return jsonify({"error": str(e)}), 500


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True)
