import logging
import json
import os
import torch
from pipeline import SummarizationPipeline
from evaluator import Evaluator
from data_loader import DataLoader

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def main():
    """Test the fine-tuned BART model on SAMSum test data and debug outputs."""
    try:
        # Ensure output directories exist
        os.makedirs("outputs", exist_ok=True)
        os.makedirs("visualizations", exist_ok=True)
        
        # Best parameters from prior tuning
        best_params = {
            'num_beams': 4,
            'length_penalty': 2.0,
            'min_length': 50,  # Increase to avoid empty outputs
            'max_length': 150,  # Explicitly set for consistency
        }
        
        # Test fine-tuned model
        logger.info("Testing fine-tuned BART model...")
        fine_tuned_pipeline = SummarizationPipeline(dataset_split="test", model_name="./fine_tuned_model")
        fine_tuned_results = fine_tuned_pipeline.run(num_samples=10, **best_params)
        
        # Log raw summaries for debugging
        logger.info("Fine-Tuned BART Summaries:")
        for i, result in enumerate(fine_tuned_results):
            logger.info(f"Sample {i+1}:")
            logger.info(f"Dialogue: {result['dialogue'][:200]}...")  # Truncate for readability
            logger.info(f"Reference: {result['reference_summary']}")
            logger.info(f"Generated: {result['generated_summary']}")
            logger.info("-" * 50)
        
        # Evaluate fine-tuned model
        evaluator = Evaluator()
        fine_tuned_scores, fine_tuned_rouge_scores = evaluator.evaluate(fine_tuned_results)
        logger.info(f"Fine-Tuned BART Scores: {fine_tuned_scores}")
        
        # Save fine-tuned results
        with open("outputs/summarization_results_fine_tuned_bart.json", "w") as f:
            json.dump(fine_tuned_results, f, indent=2)
        with open("outputs/rouge_scores_fine_tuned_bart.json", "w") as f:
            json.dump(fine_tuned_scores, f, indent=2)
        
        # Qualitative analysis
        examples = evaluator.qualitative_analysis(fine_tuned_results, num_examples=3)
        with open("outputs/qualitative_examples_fine_tuned_bart.json", "w") as f:
            json.dump(examples, f, indent=2)
        
        # Visualize fine-tuned scores
        evaluator.visualize_scores(fine_tuned_rouge_scores, title="ROUGE Scores Fine-Tuned BART")
        
        # Compare with pre-fine-tuned BART
        logger.info("Testing pre-fine-tuned BART for comparison...")
        pre_tuned_pipeline = SummarizationPipeline(dataset_split="test", model_name="facebook/bart-large-cnn")
        pre_tuned_results = pre_tuned_pipeline.run(num_samples=10, **best_params)
        
        logger.info("Pre-Fine-Tuned BART Summaries:")
        for i, result in enumerate(pre_tuned_results):
            logger.info(f"Sample {i+1}:")
            logger.info(f"Dialogue: {result['dialogue'][:200]}...")
            logger.info(f"Reference: {result['reference_summary']}")
            logger.info(f"Generated: {result['generated_summary']}")
            logger.info("-" * 50)
        
        pre_tuned_scores, pre_tuned_rouge_scores = evaluator.evaluate(pre_tuned_results)
        logger.info(f"Pre-Fine-Tuned BART Scores: {pre_tuned_scores}")
        
        # Save pre-fine-tuned results
        with open("outputs/summarization_results_pre_tuned_bart.json", "w") as f:
            json.dump(pre_tuned_results, f, indent=2)
        with open("outputs/rouge_scores_pre_tuned_bart.json", "w") as f:
            json.dump(pre_tuned_scores, f, indent=2)
        
        evaluator.visualize_scores(pre_tuned_rouge_scores, title="ROUGE Scores Pre-Fine-Tuned BART")
        
        # Clear GPU memory
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    
    except Exception as e:
        logger.error(f"Error in testing: {str(e)}")
        raise

if __name__ == "__main__":
    main()