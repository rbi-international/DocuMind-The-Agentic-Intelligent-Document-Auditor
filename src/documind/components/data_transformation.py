import os
from documind import logger
from transformers import AutoTokenizer
from datasets import load_dataset
from documind.entity import DataTransformationConfig

class DataTransformation:
    def __init__(self, config: DataTransformationConfig):
        self.config = config
        self.tokenizer = AutoTokenizer.from_pretrained(config.tokenizer_name)

    def convert_examples_to_features(self, example_batch):
        """
        Tokenizes the input text.
        """
        return self.tokenizer(
            example_batch['text'],
            padding='max_length',
            truncation=True,
            max_length=512 # Standard BERT length
        )

    def convert(self):
        try:
            logger.info("Loading validated data for transformation...")
            
            # We load the CSVs we saved in Stage 01
            # Note: We look at the data_ingestion folder, not just the single file path in config
            data_dir = os.path.dirname(self.config.data_path)
            
            data_files = {
                "train": os.path.join(data_dir, "train.csv"),
                "test": os.path.join(data_dir, "test.csv"),
                "validation": os.path.join(data_dir, "validation.csv")
            }

            dataset = load_dataset("csv", data_files=data_files)
            
            logger.info(f"Data loaded. Rows: {dataset.num_rows}")
            logger.info("Starting Tokenization (This may take a moment)...")

            # Map the tokenization function over the dataset
            encoded_dataset = dataset.map(self.convert_examples_to_features, batched=True)

            # Save the processed dataset to disk (Arrow format)
            save_path = os.path.join(self.config.root_dir, "samsum_dataset")
            encoded_dataset.save_to_disk(save_path)
            
            logger.info(f"Transformation completed. Saved to {save_path}")

        except Exception as e:
            logger.exception(e)
            raise e