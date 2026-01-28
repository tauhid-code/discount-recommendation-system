from flask import Flask, request, jsonify
import pandas as pd
import joblib

FEATURE_COLUMNS = joblib.load("models/feature_columns.pkl")

app = Flask(__name__)

MODEL_PATH = "models/random_forest_model.pkl"
PROCESSED_DATA_PATH = "data/processed_data.csv"

model = joblib.load(MODEL_PATH)
reference_df = pd.read_csv(PROCESSED_DATA_PATH)


NUMERIC_MEDIANS = reference_df.median()
CATEGORICAL_MODES = reference_df.mode().iloc[0]

HIGH_ORDER_THRESHOLD = reference_df["No. of orders placed"].quantile(0.7)

def build_feature_vector(user_input: dict) -> pd.DataFrame:
    features = {}

    
    for col in FEATURE_COLUMNS:
        if col in NUMERIC_MEDIANS:
            features[col] = NUMERIC_MEDIANS[col]
        else:
            features[col] = CATEGORICAL_MODES.get(col, 0)


    if "No. of orders placed" in features:
        features["No. of orders placed"] = user_input["orders"]

    if "More Offers and Discount" in features:
        features["More Offers and Discount"] = user_input["discount"]

    if "Order Value" in features:
        features["Order Value"] = user_input["order_value"]

    if "Delivery Rating" in features:
        features["Delivery Rating"] = user_input["delivery_exp"]


    return pd.DataFrame([features])[FEATURE_COLUMNS]

@app.route("/predict", methods=["POST"])
def predict():
    payload = request.json

    input_df = build_feature_vector(payload)
    probability = model.predict_proba(input_df)[0][1]

    return jsonify({
        "recommend_discount": int(probability >= 0.5),
        "confidence": round(probability, 2),
        "message": (
            "Offer discount to increase engagement"
            if probability >= 0.5
            else "Discount not required"
        )
    })

if __name__ == "__main__":
    app.run(debug=True)