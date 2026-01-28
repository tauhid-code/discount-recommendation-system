import sys
import pandas as pd 
from src.logger import get_logger 
from src.exception import CustomException 

logger = get_logger()


class FeatureEngineering:
    def add_features(self, df: pd.DataFrame) -> pd.DataFrame:
        try:
            logger.info("Feature engineering started")

            df['is_discount_sensitive']= (
                df["More Offers and Discount"]>=4
            ).astype(int)

            logger.info("Added discount sensitivity feature")

            high_order_threshold = df['No. of orders placed'].quantile(0.70)
            df['is_high_order'] = (
                df["No. of orders placed"]>= high_order_threshold
            ).astype(int)

            logger.info("Added high order frequency feature")

            df["target_discount_growth"] = (
            (df["is_discount_sensitive"] == 1) &
            (df["is_high_order"] == 0)
            ).astype(int)


            logger.info("Created target_discount_growth")
            return df
            
        except Exception as e: 
            logger.error("Error during feature engineering")
            raise CustomException(e,sys)
