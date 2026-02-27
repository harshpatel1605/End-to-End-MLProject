import sys
from src.mlproject.logger import logging
from src.mlproject.exception import CustomException
from src.mlproject.components.data_ingestion import DataIngestion
from src.mlproject.components.data_transformation import DataTransformation
from src.mlproject.components.model_trainer import ModelTrainer


class TrainPipeline:
    def __init__(self):
        pass

    def run_pipeline(self):
        try:
            logging.info("Training Pipeline Started")

            # ==============================
            # Step 1: Data Ingestion
            # ==============================
            data_ingestion = DataIngestion()
            train_data_path, test_data_path = data_ingestion.initiate_data_ingestion()

            logging.info("Data Ingestion Completed")

            # ==============================
            # Step 2: Data Transformation
            # ==============================
            data_transformation = DataTransformation()
            train_arr, test_arr, _ = data_transformation.initiate_data_transformation(
                train_data_path,
                test_data_path
            )

            logging.info("Data Transformation Completed")

            # ==============================
            # Step 3: Model Training
            # ==============================
            model_trainer = ModelTrainer()
            model_score = model_trainer.initiate_model_trainer(
                train_arr,
                test_arr
            )

            logging.info("Model Training Completed")
            logging.info(f"Best Model Score: {model_score}")

            return model_score

        except Exception as e:
            raise CustomException(e, sys)


if __name__ == "__main__":
    obj = TrainPipeline()
    obj.run_pipeline()