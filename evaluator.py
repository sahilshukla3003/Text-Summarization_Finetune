import logging
from typing import List, Dict
from rouge_score import rouge_scorer
import matplotlib.pyplot as plt
import os
import re
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class Evaluator:
    """Class to evaluate and visualize summarization results."""
    
    def __init__(self):
        self.scorer = rouge_scorer.RougeScorer(['rouge1', 'rouge2', 'rougeL'], use_stemmer=True)
    
    def evaluate(self, results: List[Dict]) -> Dict:
        """Evaluate summaries using ROUGE scores."""
        logger.info("Evaluating summaries...")
        
        rouge_scores = {'rouge1': [], 'rouge2': [], 'rougeL': []}
        
        for result in results:
            reference = result['reference_summary']
            generated = result['generated_summary']
            scores = self.scorer.score(reference, generated)
            
            for metric in rouge_scores:
                rouge_scores[metric].append(scores[metric].fmeasure)
        
        avg_scores = {
            metric: sum(scores) / len(scores) for metric, scores in rouge_scores.items()
        }
        
        logger.info(f"Average ROUGE scores: {avg_scores}")
        return avg_scores, rouge_scores
    
    def qualitative_analysis(self, results: List[Dict], num_examples: int = 3) -> List[Dict]:
        """Select examples for qualitative analysis."""
        logger.info("Performing qualitative analysis...")
        return results[:min(num_examples, len(results))]
    

    def visualize_scores(self, rouge_scores: Dict, title: str = "ROUGE Scores"):
        """Visualize ROUGE score distributions."""
        # Sanitize filename
        safe_title = re.sub(r'[^\w\s-]', '', title.lower().replace(' ', '_'))
        output_dir = "visualizations"
        os.makedirs(output_dir, exist_ok=True)
        output_path = os.path.join(output_dir, f"{safe_title}.png")
        
        plt.figure(figsize=(10, 6))
        for metric, scores in rouge_scores.items():
            plt.hist(scores, bins=20, alpha=0.5, label=metric)
        plt.title(title)
        plt.xlabel("Score")
        plt.ylabel("Frequency")
        plt.legend()
        plt.savefig(output_path)
        plt.close()
        logger.info(f"Saved visualization to {output_path}")
    # def visualize_scores(self, rouge_scores: Dict, title: str = "ROUGE Scores"):
    #     """Visualize ROUGE score distributions."""
    #     plt.figure(figsize=(10, 6))
    #     for metric, scores in rouge_scores.items():
    #         plt.hist(scores, bins=20, alpha=0.5, label=metric)
    #     plt.title(title)
    #     plt.xlabel("Score")
    #     plt.ylabel("Frequency")
    #     plt.legend()
    #     plt.savefig(f"{title.lower().replace(' ', '_')}.png")
    #     plt.close()