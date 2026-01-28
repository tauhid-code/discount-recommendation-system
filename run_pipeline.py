from src.preprocessing import DataPreprocessor
from src.feature_engineering import FeatureEngineering
from src.training import ModelTrainer
from src.evaluation import ModelEvaluator
from src.logger import get_logger

logger = get_logger()

def main():
    logger.info("Pipeline started")

    preprocessor = DataPreprocessor()
    engineer = FeatureEngineering()
    trainer = ModelTrainer()
    evaluator = ModelEvaluator()

  
    df_raw = preprocessor.load_data()

    df_processed = preprocessor.preprocess(df_raw)

    df_featured = engineer.add_features(df_processed)
  
    preprocessor.save_processed_data(df_featured)

    trainer.train_model()

    metrics = evaluator.evaluate()

    logger.info(f"ROC-AUC Score: {metrics['roc_auc']}")

    logger.info("Pipeline completed successfully")

if __name__ == "__main__":
    main()