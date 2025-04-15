import logging
from typing import List, Dict
from data_loader import DataLoader
from model import SummarizationModel

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class SummarizationPipeline:
    """Class to orchestrate the summarization process."""
    
    def __init__(self, dataset_split: str = "test", model_name: str = "facebook/bart-large-cnn"):
        self.data_loader = DataLoader(split=dataset_split)
        self.model = SummarizationModel(model_name=model_name)
    
    def run(self, num_samples: int = 10, **generate_kwargs) -> List[Dict]:
        """Run the summarization pipeline with customizable generation parameters."""
        logger.info("Starting summarization pipeline...")
        
        self.data_loader.load_data()
        data = self.data_loader.preprocess()
        
        data = data[:min(num_samples, len(data))]
        
        self.model.load_model()
        
        dialogues = [sample['dialogue'] for sample in data]
        generated_summaries = self.model.summarize(dialogues, **generate_kwargs)
        
        results = []
        for i, sample in enumerate(data):
            results.append({
                'dialogue': sample['dialogue'],
                'reference_summary': sample['summary'],
                'generated_summary': generated_summaries[i]
            })
        
        logger.info("Pipeline completed.")
        return results