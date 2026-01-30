import sys
import joblib
import shap
import pandas as pd
from src.logger import get_logger
from src.exception import CustomException

logger = get_logger()


class DiscountPredictor:
    def __init__(self, model_path, feature_columns_path):
        try:
            self.model = joblib.load(model_path)
            self.feature_columns = joblib.load(feature_columns_path)

            # Tree SHAP Explainer
            self.explainer = shap.TreeExplainer(self.model)

            logger.info(f"Model loaded | Classes: {self.model.classes_}")

        except Exception as e:
            logger.error("Failed to initialize DiscountPredictor")
            raise CustomException(e, sys)

   
    def _build_feature_vector(self, user_input: dict) -> pd.DataFrame:
        features = {}

        for col in self.feature_columns:
            if col in ["Age", "Family size", "Maximum wait time"]:
                features[col] = 30
            elif col in [
                "Delivery Rating",
                "Restaurant Rating",
                "Ease and convenient",
                "Late Delivery",
                "Bad past experience"
            ]:
                features[col] = 3
            else:
                features[col] = 0

        # User inputs fields visible on the interface
        features["No. of orders placed"] = user_input["orders"]
        features["More Offers and Discount"] = user_input["discount"]
        features["Order Value"] = user_input["order_value"]
        features["Delivery Rating"] = user_input["delivery_exp"]

        df = pd.DataFrame([features])
        df = df.reindex(columns=self.feature_columns, fill_value=0)

        return df


    def _get_local_shap(self, input_df: pd.DataFrame):
        """
        Compute LOCAL SHAP values for ONE prediction
        """
        try:
            shap_values = self.explainer.shap_values(input_df)

            # NORMALIZE SHAP
            if isinstance(shap_values, list):
               
                values = shap_values[1][0] if len(shap_values) > 1 else shap_values[0][0]
            else:
              
                values = shap_values[0]

            shap_df = pd.DataFrame({
                "feature": input_df.columns,
                "impact": values
            })

            shap_df["abs_impact"] = shap_df["impact"].abs()
            shap_df = shap_df.sort_values("abs_impact", ascending=False)

            return shap_df.head(6)[["feature", "impact"]].to_dict("records")

        except Exception as e:
            logger.error(f"LOCAL SHAP FAILED: {e}", exc_info=True)
            return []

 
    def predict(self, user_input: dict):
        try:
            orders = user_input["orders"]
            discount_pref = user_input["discount"]

            
            input_df = self._build_feature_vector(user_input)

            proba = self.model.predict_proba(input_df)[0]

            if len(self.model.classes_) == 1:
                probability = 1.0 if self.model.classes_[0] == 1 else 0.0
            else:
                class_index = list(self.model.classes_).index(1)
                probability = proba[class_index]

            prediction = int(probability >= 0.5)

        
            shap_explanation = self._get_local_shap(input_df)

    
            if orders < 20 and discount_pref <= 2:
                return {
                    "recommend_discount": 1,
                    "confidence": 0.75,
                    "reason": "Low frequency and low loyalty",
                    "shap_explanation": shap_explanation
                }

            if orders >= 60:
                return {
                    "recommend_discount": 0,
                    "confidence": 0.80,
                    "reason": "High order frequency (loyal customer)",
                    "shap_explanation": shap_explanation
                }

            return {
                "recommend_discount": prediction,
                "confidence": round(probability, 2),
                "reason": "ML-based decision",
                "shap_explanation": shap_explanation
            }

        except Exception as e:
            logger.error("Prediction failed", exc_info=True)
            raise CustomException(e, sys)