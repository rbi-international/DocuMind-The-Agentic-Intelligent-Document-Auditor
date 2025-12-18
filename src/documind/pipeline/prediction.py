import os
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from datasets import load_dataset
from documind import logger

class PredictionPipeline:
    def __init__(self):
        self.model_path = os.path.join("artifacts", "model_trainer", "bert-classifier")
        self.device = "cpu" # Keep CPU for tool usage
        
        logger.info(f"Loading Classification Model from {self.model_path}...")
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_path)
        self.model = AutoModelForSequenceClassification.from_pretrained(self.model_path).to(self.device)

        # --- FIX: Load Label Mappings ---
        # We fetch the dataset metadata to get the list of 100 label names
        logger.info("Loading Label Mappings...")
        dataset = load_dataset("lex_glue", "ledgar", split="train", trust_remote_code=False)
        self.id2label = {i: label for i, label in enumerate(dataset.features["label"].names)}

    def predict(self, text: str):
        try:
            # 1. Tokenize
            inputs = self.tokenizer(
                text, 
                return_tensors="pt", 
                truncation=True, 
                padding="max_length", 
                max_length=512
            ).to(self.device)

            # 2. Inference
            with torch.no_grad():
                outputs = self.model(**inputs)
            
            # 3. Get Prediction ID
            logits = outputs.logits
            predicted_class_id = torch.argmax(logits, dim=1).item()
            
            # 4. Convert to Text
            predicted_label = self.id2label[predicted_class_id]

            return predicted_label

        except Exception as e:
            logger.error(f"Prediction Error: {e}")
            return "Error"