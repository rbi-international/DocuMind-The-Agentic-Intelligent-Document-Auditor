import torch
import pandas as pd
import mlflow
import mlflow.pytorch
from transformers import AutoModelForSequenceClassification, AutoTokenizer
from datasets import load_from_disk
from documind.entity import ModelEvaluationConfig
from documind.utils.common import save_json
from pathlib import Path
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
from documind import logger

class ModelEvaluation:
    def __init__(self, config: ModelEvaluationConfig):
        self.config = config

    def eval_metrics(self, actual, pred):
        accuracy = accuracy_score(actual, pred)
        precision, recall, f1, _ = precision_recall_fscore_support(actual, pred, average='weighted')
        return accuracy, precision, recall, f1

    def evaluation(self):
        device = "cuda" if torch.cuda.is_available() else "cpu"
        
        # 1. Load Model & Tokenizer
        model = AutoModelForSequenceClassification.from_pretrained(self.config.model_path).to(device)
        tokenizer = AutoTokenizer.from_pretrained(self.config.tokenizer_path)
        
        # 2. Load Data
        dataset = load_from_disk(self.config.data_path)
        eval_dataset = dataset["test"] # Use test set for final evaluation

        # 3. Batch Prediction
        logger.info("Starting Batch Evaluation...")
        predictions = []
        labels = []

        # We process in batches to avoid OOM on 6GB VRAM
        batch_size = self.config.eval_batch_size
        
        # Simple loop for inference
        for i in range(0, len(eval_dataset), batch_size):
            batch = eval_dataset[i : i + batch_size]
            
            # Tokenize on the fly or use existing features (we have features)
            input_ids = torch.tensor(batch["input_ids"]).to(device)
            attention_mask = torch.tensor(batch["attention_mask"]).to(device)
            
            with torch.no_grad():
                outputs = model(input_ids=input_ids, attention_mask=attention_mask)
            
            logits = outputs.logits
            preds = torch.argmax(logits, dim=1).cpu().numpy()
            
            predictions.extend(preds)
            labels.extend(batch["label"])

        # 4. Calculate Metrics
        accuracy, precision, recall, f1 = self.eval_metrics(labels, predictions)
        
        # 5. Save locally
        scores = {"accuracy": accuracy, "f1": f1, "precision": precision, "recall": recall}
        save_json(path=Path(self.config.metric_file_name), data=scores)
        
        # 6. Log to MLflow
        mlflow.set_registry_uri("file://" + str(Path("mlruns").absolute())) # Local MLflow
        mlflow.set_experiment("DocuMind-Classification")
        
        with mlflow.start_run():
            mlflow.log_params(self.config.__dict__)
            mlflow.log_metrics(scores)
            # We don't log the full model to MLflow here to save space, but you can if needed
            
        logger.info(f"Evaluation completed. Metrics: {scores}")