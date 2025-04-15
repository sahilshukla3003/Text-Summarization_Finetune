import logging
from data_loader import DataLoader
from pipeline import SummarizationPipeline

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def main():
    """Re-run fine-tuning for BART model with improved settings."""
    try:
        logger.info("Starting fine-tuning...")
        train_loader = DataLoader(split="train")
        train_loader.load_data()
        train_data = train_loader.preprocess()[:200]  # Increased for better learning
        bart_model = SummarizationPipeline(model_name="facebook/bart-large-cnn")
        bart_model.model.load_model()
        bart_model.model.fine_tune(train_data, epochs=2)  # More epochs
        logger.info("Fine-tuning completed. Model saved to ./fine_tuned_model.")
    
    except Exception as e:
        logger.error(f"Error in fine-tuning: {str(e)}")
        raise

if __name__ == "__main__":
    main()