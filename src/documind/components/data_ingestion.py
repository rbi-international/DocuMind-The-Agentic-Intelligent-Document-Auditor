import os
from documind import logger
from documind.entity import DataIngestionConfig
from datasets import load_dataset
import pandas as pd

class DataIngestion:
    def __init__(self, config: DataIngestionConfig):
        self.config = config

    def download_data(self):
        """
        Downloads data from Hugging Face and saves it locally as CSV for DVC tracking.
        """
        try:
            logger.info(f"Downloading Data from HuggingFace '{self.config.dataset_name}' subset '{self.config.subset_name}'")
            
            # Load dataset from HuggingFace (This caches it in ~/.cache/huggingface)
            dataset = load_dataset(self.config.dataset_name, self.config.subset_name)
            
            logger.info(f"Dataset downloaded. Converting to CSV for local artifact storage...")

            # Convert to pandas for easier local handling/inspection
            # We only take a small slice (e.g., 5000 rows) for this demo to keep it fast on your laptop
            # If you want full data, remove .select(range(5000))
            train_df = pd.DataFrame(dataset['train'].select(range(5000))) 
            test_df = pd.DataFrame(dataset['test'].select(range(1000)))
            validation_df = pd.DataFrame(dataset['validation'].select(range(1000)))

            # Define output paths
            train_path = os.path.join(self.config.root_dir, "train.csv")
            test_path = os.path.join(self.config.root_dir, "test.csv")
            val_path = os.path.join(self.config.root_dir, "validation.csv")

            # Save to disk
            train_df.to_csv(train_path, index=False)
            test_df.to_csv(test_path, index=False)
            validation_df.to_csv(val_path, index=False)

            logger.info(f"Data saved to {self.config.root_dir}")
            logger.info(f"Train size: {train_df.shape}, Test size: {test_df.shape}")

        except Exception as e:
            logger.error(f"Error in Data Ingestion: {e}")
            raise e