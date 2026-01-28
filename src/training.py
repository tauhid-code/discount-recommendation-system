import sys 
import pandas as pd 
import joblib 
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from src.config import PROCESSED_DATA_PATH, MODEL_PATH, TARGET_COLUMN
from src.exception import CustomException
from src.logger import get_logger

logger = get_logger()

class ModelTrainer:
    def train_model(self):
        try: 
            logger.info("Starting model training")
             
            df = pd.read_csv(PROCESSED_DATA_PATH)

            X = df.drop(columns=[
            TARGET_COLUMN,
            "is_discount_sensitive",
            "is_high_order"
            ])
            import joblib

            FEATURE_COLUMNS = X.columns.tolist()

            joblib.dump(FEATURE_COLUMNS, "models/feature_columns.pkl")

            y = df[TARGET_COLUMN]


            X_train, X_test, y_train, y_test = train_test_split(
                X,y,test_size=0.2,random_state=42,stratify=y
            )
             
            model = RandomForestClassifier(
                n_estimators=100,
                random_state=42,
                max_depth=8,
                min_samples_split=10,
                min_samples_leaf=5,
                class_weight="balanced"
             )
            
            model.fit(X_train,y_train)

            joblib.dump(model, MODEL_PATH)

            logger.info("Model trained successfully")
            return model
        

        except Exception as e:
            logger.error("Error during model training")
            raise CustomException(e, sys)