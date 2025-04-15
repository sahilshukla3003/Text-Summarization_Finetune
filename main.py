import logging
import json
from pipeline import SummarizationPipeline
from evaluator import Evaluator
from data_loader import DataLoader
import itertools

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def hyperparameter_tuning(pipeline: SummarizationPipeline, num_samples: int = 5):
    """Perform simple hyperparameter tuning."""
    param_grid = {
        'num_beams': [4, 6],
        'length_penalty': [1.0, 2.0]
    }
    
    best_scores = None
    best_params = None
    evaluator = Evaluator()
    
    for params in [dict(zip(param_grid.keys(), v)) for v in itertools.product(*param_grid.values())]:
        logger.info(f"Testing parameters: {params}")
        results = pipeline.run(num_samples=num_samples, **params)
        scores, _ = evaluator.evaluate(results)
        
        if best_scores is None or scores['rouge1'] > best_scores['rouge1']:
            best_scores = scores
            best_params = params
    
    logger.info(f"Best parameters: {best_params}, Best ROUGE scores: {best_scores}")
    return best_params

def main():
    """Run the enhanced summarization pipeline with tuning, visualization, and fine-tuning."""
    try:
        # Compare BART and T5
        models = [
            "facebook/bart-large-cnn",
            "t5-small"  # Smaller for faster experimentation
        ]
        all_results = {}
        all_scores = {}
        
        for model_name in models:
            logger.info(f"Processing model: {model_name}")
            pipeline = SummarizationPipeline(dataset_split="test", model_name=model_name)
            
            # Hyperparameter tuning
            best_params = hyperparameter_tuning(pipeline, num_samples=5)
            
            # Run with best parameters
            results = pipeline.run(num_samples=10, **best_params)
            all_results[model_name] = results
            
            # Evaluate
            evaluator = Evaluator()
            avg_scores, rouge_scores = evaluator.evaluate(results)
            all_scores[model_name] = avg_scores
            
            # Visualize
            evaluator.visualize_scores(rouge_scores, title=f"ROUGE Scores ({model_name})")
            
            # Save results
            with open(f"summarization_results_{model_name.split('/')[-1]}.json", "w") as f:
                json.dump(results, f, indent=2)
            with open(f"rouge_scores_{model_name.split('/')[-1]}.json", "w") as f:
                json.dump(avg_scores, f, indent=2)
            
            # Qualitative analysis
            examples = evaluator.qualitative_analysis(results)
            with open(f"qualitative_examples_{model_name.split('/')[-1]}.json", "w") as f:
                json.dump(examples, f, indent=2)
        
        # # Fine-tune BART on a small subset
        # logger.info("Fine-tuning BART...")
        # train_loader = DataLoader(split="train")
        # train_loader.load_data()
        # train_data = train_loader.preprocess()[:50]  # Small subset for demo
        # bart_model = SummarizationPipeline(model_name="facebook/bart-large-cnn")
        # bart_model.model.load_model()
        # bart_model.model.fine_tune(train_data, epochs=1)
        
        # # Re-run with fine-tuned model
        # fine_tuned_pipeline = SummarizationPipeline(model_name="./fine_tuned_model")
        # fine_tuned_results = fine_tuned_pipeline.run(num_samples=10, **best_params)
        # fine_tuned_scores, fine_tuned_rouge_scores = evaluator.evaluate(fine_tuned_results)
        
        # # Save fine-tuned results
        # with open("summarization_results_fine_tuned_bart.json", "w") as f:
        #     json.dump(fine_tuned_results, f, indent=2)
        # with open("rouge_scores_fine_tuned_bart.json", "w") as f:
        #     json.dump(fine_tuned_scores, f, indent=2)
        # evaluator.visualize_scores(fine_tuned_rouge_scores, title="ROUGE Scores (Fine-Tuned BART)")
        
        # # Compare models
        # logger.info("Model Comparison:")
        # for model_name, scores in all_scores.items():
        #     logger.info(f"{model_name}: {scores}")
        # logger.info(f"Fine-Tuned BART: {fine_tuned_scores}")
    
    except Exception as e:
        logger.error(f"Error in main execution: {str(e)}")
        raise

if __name__ == "__main__":
    main()