from dataclasses import dataclass
from pathlib import Path

@dataclass(frozen=True)
class DataIngestionConfig:
    root_dir: Path
    dataset_name: str
    subset_name: str
    local_data_file: Path
    
@dataclass(frozen=True)
class DataValidationConfig:
    root_dir: Path
    report_file: Path
    required_files: list
    all_schema: dict
    
@dataclass(frozen=True)
class DataTransformationConfig:
    root_dir: Path
    data_path: Path
    tokenizer_name: str
    
@dataclass(frozen=True)
class ModelTrainerConfig:
    root_dir: Path
    data_path: Path
    model_ckpt: str
    num_train_epochs: int
    per_device_train_batch_size: int
    weight_decay: float
    learning_rate: float
    
@dataclass(frozen=True)
class ModelEvaluationConfig:
    root_dir: Path
    data_path: Path
    model_path: Path
    tokenizer_path: Path
    metric_file_name: Path
    eval_batch_size: int