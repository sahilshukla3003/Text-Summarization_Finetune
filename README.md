# Enhanced Text Summarization Application

This project extends a modular text summarization application using the SAMSum dataset. It includes hyperparameter tuning, visualization, fine-tuning, and a comparison between BART and T5 models.

## Setup Instructions

1. **Clone the repository**:
   ```bash
   git clone https://github.com/sahilshukla3003/Text-Summarization_Finetune.git
   cd text_summarization
Create a virtual environment:

bash

Copy
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
Install dependencies:

bash

Copy
pip install -r requirements.txt
Run the application:

bash

Copy
python main.py
This will:

Load the SAMSum test dataset.
Perform hyperparameter tuning for BART and T5.
Generate summaries for 10 samples per model.
Fine-tune BART on 50 training samples.
Evaluate summaries using ROUGE scores.
Visualize score distributions.
Save results, scores, and qualitative examples.

Project Structure
data_loader.py: Loads and preprocesses the SAMSum dataset.
model.py: Handles loading, running, and fine-tuning BART/T5 models.
pipeline.py: Orchestrates data flow and summarization.
evaluator.py: Evaluates summaries and visualizes metrics.
main.py: Runs the pipeline with tuning, visualization.
fineTune.py : finetuning model.
test_fineTune.py : Testing the funetuning model and comparing with pre-finetune model.
requirements.txt: Lists dependencies.
insights.md: Documents findings, limitations, and enhancements.

Design Choices
Models: Compared facebook/bart-large-cnn (strong abstractive performance) and t5-small (lightweight for experimentation).
Tuning: Simple grid search over num_beams and length_penalty to optimize ROUGE-1.
Fine-Tuning: Trained BART on 50 samples to demo dialogue-specific improvements.
Visualization: Histogram of ROUGE scores to analyze distribution.
Modularity: Class-based structure for reusability.

Outputs
summarization_results_*.json: Results for each model.
rouge_scores_*.json: Average ROUGE scores.
qualitative_examples_*.json: Example outputs.
*.png: ROUGE score visualizations.
./fine_tuned_model/: Fine-tuned BART model.

