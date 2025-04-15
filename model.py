import logging
from transformers import BartForConditionalGeneration, BartTokenizer, T5ForConditionalGeneration, T5Tokenizer
import torch
from typing import List, Dict
from datasets import Dataset

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class SummarizationModel:
    """Class to handle loading, running, and fine-tuning summarization models."""
    
    def __init__(self, model_name: str = "facebook/bart-large-cnn"):
        self.model_name = model_name
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.tokenizer = None
        self.model = None
        if model_name == "./fine_tuned_model":
            self.model_type = "bart"
        else:
            self.model_type = "bart" if "bart" in model_name.lower() else "t5"
    
    def load_model(self) -> None:
        """Load the model and tokenizer."""
        try:
            logger.info(f"Loading model {self.model_name}...")
            if self.model_type == "bart":
                self.tokenizer = BartTokenizer.from_pretrained(self.model_name)
                self.model = BartForConditionalGeneration.from_pretrained(self.model_name).to(self.device)
            else:
                self.tokenizer = T5Tokenizer.from_pretrained(self.model_name)
                self.model = T5ForConditionalGeneration.from_pretrained(self.model_name).to(self.device)
            logger.info(f"Model loaded successfully on {self.device}.")
        except Exception as e:
            logger.error(f"Failed to load model: {str(e)}")
            raise
    
    def summarize(self, texts: List[str], max_length: int = 128, min_length: int = 30, **generate_kwargs) -> List[str]:
        """Generate summaries for texts with customizable generation parameters."""
        if self.model is None or self.tokenizer is None:
            raise ValueError("Model not loaded. Call load_model() first.")
        
        logger.info(f"Summarizing {len(texts)} texts...")
        summaries = []
        
        for text in texts:
            inputs = self.tokenizer(
                text if self.model_type == "bart" else f"summarize: {text}",
                max_length=512,
                truncation=True,
                padding="max_length",
                return_tensors="pt"
            ).to(self.device)
            
            summary_ids = self.model.generate(
                inputs["input_ids"],
                max_length=max_length,
                min_length=min_length,
                **generate_kwargs
            )
            
            summary = self.tokenizer.decode(summary_ids[0], skip_special_tokens=True).strip()
            summaries.append(summary)
        
        logger.info("Summarization complete.")
        return summaries
    
    def fine_tune(self, train_data: List[Dict], epochs: int = 1, batch_size: int = 1):
        """Fine-tune the model with memory-efficient settings and validation."""
        from transformers import Trainer, TrainingArguments
        
        if self.model is None or self.tokenizer is None:
            self.load_model()
        
        logger.info(f"Fine-tuning {self.model_name}...")
        
        # Split data into train (80%) and validation (20%)
        train_size = int(0.8 * len(train_data))
        train_subset = train_data[:train_size]
        val_subset = train_data[train_size:]
        
        train_dataset = self._prepare_dataset(train_subset, max_length=256)
        val_dataset = self._prepare_dataset(val_subset, max_length=256) if val_subset else None
        
        training_args = TrainingArguments(
            output_dir="./fine_tuned_model",
            num_train_epochs=epochs,
            per_device_train_batch_size=batch_size,
            per_device_eval_batch_size=batch_size,
            gradient_accumulation_steps=4,
            learning_rate=2e-5,
            warmup_steps=20,
            weight_decay=0.01,
            fp16=torch.cuda.is_available(),
            evaluation_strategy="steps" if val_dataset else "no",
            eval_steps=50 if val_dataset else None,
            save_steps=500,
            save_total_limit=2,
            logging_dir='./logs',
            logging_steps=10,
            disable_tqdm=False,
        )
        
        trainer = Trainer(
            model=self.model,
            args=training_args,
            train_dataset=train_dataset,
            eval_dataset=val_dataset,
        )
        
        trainer.train()
        self.model.save_pretrained("./fine_tuned_model")
        self.tokenizer.save_pretrained("./fine_tuned_model")
        logger.info("Fine-tuning complete. Model saved to ./fine_tuned_model.")
        
        # Clear GPU memory
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    
    def _prepare_dataset(self, data: List[Dict], max_length: int = 256) -> Dataset:
        """Prepare dataset for fine-tuning with validation."""
        inputs = [d['dialogue'].strip() if self.model_type == "bart" else f"summarize: {d['dialogue'].strip()}" for d in data]
        targets = [d['summary'].strip() for d in data]
        
        logger.info("Sample data:")
        for i in range(min(3, len(data))):
            logger.info(f"Input {i+1}: {inputs[i][:100]}...")
            logger.info(f"Target {i+1}: {targets[i][:100]}...")
        
        tokenized = self.tokenizer(
            inputs,
            text_target=targets,
            max_length=max_length,
            truncation=True,
            padding="max_length",
            return_tensors="pt"
        )
        
        if tokenized.input_ids.shape[0] == 0:
            logger.error("Tokenized dataset is empty!")
            raise ValueError("Tokenized dataset is empty!")
        
        return Dataset.from_dict({
            "input_ids": tokenized.input_ids,
            "attention_mask": tokenized.attention_mask,
            "labels": tokenized.labels
        })