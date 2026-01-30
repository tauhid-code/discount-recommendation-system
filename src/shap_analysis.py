import joblib
import shap
import pandas as pd
import numpy as np
import os


MODEL_PATH = "models/random_forest_model.pkl"
FEATURES_PATH = "models/feature_columns.pkl"
PROCESSED_DATA_PATH = "data/processed_data.csv"
OUTPUT_PATH = "models/global_shap.csv"


model = joblib.load(MODEL_PATH)
feature_columns = joblib.load(FEATURES_PATH)

df = pd.read_csv(PROCESSED_DATA_PATH)
X = df[feature_columns]

explainer = shap.TreeExplainer(model)
shap_values = explainer.shap_values(X)


if isinstance(shap_values, list):
   
    if len(shap_values) > 1:
        values = shap_values[1]
    else:
        values = shap_values[0]
else:
    values = shap_values


values = np.asarray(values)

if values.ndim == 3:
    values = values[:, :, 1] 

if values.ndim != 2:
    raise ValueError(f"Unexpected SHAP shape: {values.shape}")


importance = np.mean(np.abs(values), axis=0)

assert importance.ndim == 1, "Importance is not 1D!"

global_shap = pd.DataFrame({
    "feature": feature_columns,
    "importance": importance
}).sort_values(by="importance", ascending=False)


os.makedirs("models", exist_ok=True)
global_shap.to_csv(OUTPUT_PATH, index=False)

print(" Global SHAP saved to models/global_shap.csv")
print(global_shap.head(10))