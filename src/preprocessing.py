import sys
from turtle import mode
import pandas as pd
from src.exception import CustomException 
from src.logger import get_logger
from src.config import RAW_DATA_PATH, PROCESSED_DATA_PATH

logger = get_logger()

class DataPreprocessor: 
    def load_data(self):
        try:
            logger.info("Loading data from raw data path")
            return pd.read_csv(RAW_DATA_PATH)
        except Excepion as e:
            logger.error("Failed to load data")
            raise CustomException(e, sys)
        
    def preprocess(self,df):
        try:
            logger.info("Starting preprocessing pipeline")
            final_cols = [
                'Age', 'Occupation', 'Family size', 'Frequently used Medium',
                'Frequently ordered Meal category ', 'Perference',
                'Restaurnat Rating', 'Delivery Rating', 'Order Value',
                'No. of orders placed', 'Maximum wait time',
                'More Offers and Discount', 'Influence of rating',
                'Ease and convenient', 'Late Delivery', 'Bad past experience'
            ]
            df = df[final_cols].copy()
            logger.info("Selected final columns for preprocessing")

            num_cols = df.select_dtypes(include=['int64', 'float64']).columns
            cat_cols = df.select_dtypes(include=['object']).columns

            df[num_cols] = df[num_cols].fillna(df[num_cols].median())

            for col in cat_cols:
                mode = df[col].mode()
                if not mode.empty:
                    df[col] = df[col].fillna(mode[0])
                else:
                    df[col] = df[col].fillna("Unknown") 
            logger.info("Handled missing values")



            wait_time_map = {
                "15 minutes": 15,
                "30 minutes": 30,
                "45 minutes": 45,
                "60 minutes": 60,
                "more than 60 minutes": 75
            }

            df["Maximum wait time"] = (
                df["Maximum wait time"]
                .astype(str)
                .str.lower()
                .str.strip()
                .map(wait_time_map)
            )

          
            df["Maximum wait time"] = df["Maximum wait time"].fillna(
                df["Maximum wait time"].median()
            )

            logger.info("Converted Maximum wait time to numeric")


            ordinal_cols = [
            'Ease and convenient',
            'Late Delivery',
            'Bad past experience',
            'More Offers and Discount'
            ]
            binary_cols = ['Influence of rating']

            for col in ordinal_cols + binary_cols:
                df[col] = (
                df[col]
                .astype(str)
                .str.strip()
                .str.lower()
                ) 
            logger.info("Clean text columns")

            ordinal_map = {
            'strongly disagree': 1,
            'disagree': 2,
            'neutral': 3,
            'agree': 4,
            'strongly agree': 5
            }

            binary_map = {'yes': 1, 'no': 0}


            for col in ordinal_cols:
              df[col] = df[col].map(ordinal_map)


            df['Influence of rating'] = df['Influence of rating'].map(binary_map)


            logger.info("Applied ordinal & binary encoding")

            for col in ordinal_cols:
              df[col] = df[col].fillna(df[col].median())


            df['Influence of rating'] = df['Influence of rating'].fillna(df['Influence of rating'].mode()[0])
            logger.info("Handled post-encoding missing values")

            nominal_cols = [
            'Occupation',
            'Frequently used Medium',
            'Frequently ordered Meal category ',
            'Perference'
            ]


            df = pd.get_dummies(
            df,
            columns=nominal_cols,
            drop_first=True
            )
            logger.info("Applied one-hot encoding")
            return df
        
        except Exception as e:
            logger.error("Error during preprocessing")
            raise CustomException(e, sys)
        


    def save_processed_data(self, df):
        try:
            logger.info("Saving processed dataset")
            df.to_csv(PROCESSED_DATA_PATH, index=False) 
        except Exception as e:
            logger.error("Failed to save processed dataset")
            raise CustomException(e, sys)

