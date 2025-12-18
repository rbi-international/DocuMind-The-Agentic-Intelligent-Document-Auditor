
import os
from documind import logger
from documind.entity import ModelTrainerConfig
from transformers import AutoModelForSequenceClassification, AutoTokenizer, TrainingArguments, Trainer, DataCollatorWithPadding
from datasets import load_from_disk
import torch

class ModelTrainer:
    def __init__(self, config: ModelTrainerConfig):
        self.config = config

    def train(self):
        device = "cuda" if torch.cuda.is_available() else "cpu"
        logger.info(f"Training on Device: {device}")
        
        # 1. Load the tokenizer and model
        tokenizer = AutoTokenizer.from_pretrained(self.config.model_ckpt)
        model = AutoModelForSequenceClassification.from_pretrained(
            self.config.model_ckpt, 
            num_labels=100 # LEDGAR has ~100 classes
        ).to(device)

        # 2. Load the processed dataset
        dataset = load_from_disk(self.config.data_path)

        # 3. Define Data Collator (Handles dynamic padding for batches)
        data_collator = DataCollatorWithPadding(tokenizer=tokenizer)

        # 4. Define Training Arguments
        # We enforce FP16 to save memory on your RTX 3060
        args = TrainingArguments(
            output_dir=self.config.root_dir,
            num_train_epochs=self.config.num_train_epochs,
            per_device_train_batch_size=self.config.per_device_train_batch_size,
            learning_rate=self.config.learning_rate,
            weight_decay=self.config.weight_decay,
            logging_steps=10,
            eval_strategy="steps",
            eval_steps=500,
            save_steps=1000,
            fp16=True, # <--- CRITICAL for 6GB VRAM
            report_to="none" # We will add MLflow later
        )

        # 5. Initialize Trainer
        trainer = Trainer(
            model=model,
            args=args,
            train_dataset=dataset["train"],
            eval_dataset=dataset["test"], # We use test as eval for this demo
            data_collator=data_collator,
        )

        # 6. Start Training
        logger.info("Starting Training...")
        trainer.train()

        # 7. Save Model and Tokenizer
        logger.info(f"Saving model to {self.config.root_dir}")
        model.save_pretrained(os.path.join(self.config.root_dir, "bert-classifier"))
        tokenizer.save_pretrained(os.path.join(self.config.root_dir, "bert-classifier"))