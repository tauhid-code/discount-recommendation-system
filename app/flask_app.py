import os
import sys
from flask import Flask, request, jsonify


BASE_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(BASE_DIR, ".."))
sys.path.append(PROJECT_ROOT)

from src.predict import DiscountPredictor

app = Flask(__name__)

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(BASE_DIR, ".."))

predictor = DiscountPredictor(
    model_path=os.path.join(PROJECT_ROOT, "models", "random_forest_model.pkl"),
    feature_columns_path=os.path.join(PROJECT_ROOT, "models", "feature_columns.pkl")
)


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