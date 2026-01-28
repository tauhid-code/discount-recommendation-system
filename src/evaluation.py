import sys
import pandas as pd 
import joblib 
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score 
from src.config import MODEL_PATH, PROCESSED_DATA_PATH, TARGET_COLUMN 
from src.exception import CustomException 
from src.logger import get_logger 

logger = get_logger()

class ModelEvaluator: 
    def evaluate(self):
        try: 
            logger.info("Starting model evaluation")
            df = pd.read_csv(PROCESSED_DATA_PATH)

            model = joblib.load(MODEL_PATH)

            X = df.drop(columns=[
            TARGET_COLUMN,
            "is_discount_sensitive",
            "is_high_order"
            ])
            y = df[TARGET_COLUMN]

            y_pred = model.predict(X)
            y_proba = model.predict_proba(X)[:,1]

            report = classification_report(y, y_pred)
            auc = roc_auc_score(y, y_proba)

            logger.info("Model evaluation successfully")
            return {
            "classification_report": report,
            "roc_auc": auc
            }

        except Exception as e:
            logger.error("Error during model evaluation")
            raise CustomException(e, sys)