import logging
from datasets import load_dataset
from typing import Dict, List

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class DataLoader:
    """Class to load and preprocess datasets for summarization."""
    
    def __init__(self, dataset_name: str = "samsum", split: str = "train"):
        self.dataset_name = dataset_name
        self.split = split
        self.dataset = None
    
    def load_data(self) -> None:
        """Load the dataset from Hugging Face."""
        try:
            logger.info(f"Loading {self.dataset_name} dataset, split: {self.split}...")
            self.dataset = load_dataset(self.dataset_name, split=self.split)
            logger.info(f"Dataset loaded successfully. Size: {len(self.dataset)} samples.")
        except Exception as e:
            logger.error(f"Failed to load dataset: {str(e)}")
            raise
    
    def preprocess(self, max_dialogue_len: int = 512, max_summary_len: int = 128) -> List[Dict]:
        """Preprocess dataset by filtering and formatting."""
        if self.dataset is None:
            raise ValueError("Dataset not loaded. Call load_data() first.")
        
        logger.info("Preprocessing dataset...")
        processed_data = []
        
        for sample in self.dataset:
            dialogue = sample['dialogue'].strip()
            summary = sample['summary'].strip()
            
            if len(dialogue.split()) <= max_dialogue_len and len(summary.split()) <= max_summary_len:
                processed_data.append({
                    'dialogue': dialogue,
                    'summary': summary
                })
        
        logger.info(f"Preprocessing complete. Retained {len(processed_data)} samples.")
        return processed_data