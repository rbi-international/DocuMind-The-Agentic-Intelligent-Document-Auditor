from documind.constants import *
from documind.utils.common import read_yaml, create_directories
from documind.entity import DataIngestionConfig, DataValidationConfig, DataTransformationConfig
from documind.entity import ModelTrainerConfig
from documind.entity import ModelEvaluationConfig
from pathlib import Path

class ConfigurationManager:
    def __init__(
        self,
        config_filepath = CONFIG_FILE_PATH,
        params_filepath = PARAMS_FILE_PATH,
        schema_filepath = SCHEMA_FILE_PATH):

        self.config = read_yaml(config_filepath)
        self.params = read_yaml(params_filepath)
        self.schema = read_yaml(schema_filepath)

        create_directories([self.config.artifacts_root])

    def get_data_ingestion_config(self) -> DataIngestionConfig:
        config = self.config.data_ingestion
        create_directories([config.root_dir])

        data_ingestion_config = DataIngestionConfig(
            root_dir=Path(config.root_dir),
            dataset_name=config.dataset_name,
            subset_name=config.subset_name,
            local_data_file=Path(config.local_data_file)
        )

        return data_ingestion_config

    def get_data_validation_config(self) -> DataValidationConfig:
        config = self.config.data_validation
        # Note: We access COLUMNS from self.schema, not self.config
        schema = self.schema.COLUMNS 

        create_directories([config.root_dir])

        data_validation_config = DataValidationConfig(
            root_dir=Path(config.root_dir),
            report_file=Path(config.report_file),
            required_files=config.required_files,
            all_schema=schema
        )

        return data_validation_config
    
    # Add this method inside ConfigurationManager class
    def get_data_transformation_config(self) -> DataTransformationConfig:
        config = self.config.data_transformation

        create_directories([config.root_dir])

        data_transformation_config = DataTransformationConfig(
            root_dir=Path(config.root_dir),
            data_path=Path(config.data_path),
            tokenizer_name=config.tokenizer_name
        )

        return data_transformation_config
    
    # Add this method
    def get_model_trainer_config(self) -> ModelTrainerConfig:
        config = self.config.model_trainer
        params = self.params.TrainingArguments

        create_directories([config.root_dir])

        model_trainer_config = ModelTrainerConfig(
            root_dir=Path(config.root_dir),
            data_path=Path(config.data_path),
            model_ckpt=config.model_ckpt,
            num_train_epochs=int(params.epochs),
            per_device_train_batch_size=int(params.batch_size),
            weight_decay=float(params.weight_decay),
            learning_rate=float(params.learning_rate)
        )

        return model_trainer_config
    
    def get_model_evaluation_config(self) -> ModelEvaluationConfig:
        config = self.config.model_evaluation
        params = self.params.TrainingArguments

        create_directories([config.root_dir])

        model_evaluation_config = ModelEvaluationConfig(
            root_dir=Path(config.root_dir),
            data_path=Path(config.data_path),
            model_path=Path(config.model_path),
            tokenizer_path=Path(config.model_path), # Tokenizer is saved with model
            metric_file_name=Path(config.metric_file_name),
            eval_batch_size=params.batch_size
        )

        return model_evaluation_config