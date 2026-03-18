# Code Comment Quality Analysis Using Transformers

[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)

## Overview

This project fine-tunes CodeBERT (a transformer model pre-trained on code) to classify code comment quality and compares it against rule-based baselines.

**Research Question:** Can fine-tuned transformer models distinguish helpful vs. unhelpful code comments better than rule-based approaches, and what linguistic features correlate with comment quality?

## Key Features

- **Transformer-based Classification**: Fine-tuned CodeBERT model for 3-class quality prediction
- **Comprehensive Baselines**: Rule-based and TF-IDF+LogisticRegression baselines
- **Feature Analysis**: Attention visualization and linguistic feature extraction
- **Dataset Creation**: Tools for mining and annotating code comments from GitHub
- **Reproducible Pipeline**: End-to-end training and evaluation with fixed seeds

## Results Preview
The pipeline supports end-to-end training and evaluation. Results depend on dataset composition and annotation quality. See methodology section for training details.


## Installation

### Prerequisites
- Python 3.8 or higher
- CUDA-capable GPU (recommended, but CPU training supported)
- 8GB+ RAM

### Setup

```bash
# Clone repository
git clone https://github.com/mohammed-abed/code-comment-analyzer.git
cd code-comment-analyzer

# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Install package
pip install -e .
```

## Quick Start

### 1. Prepare Dataset

```bash
# Generate synthetic dataset (for demonstration)
python scripts/download_data.py --synthetic --size 2000 --output data/processed/

# OR: Download real annotated dataset (if available)
# python scripts/download_data.py --source github --output data/raw/
```

### 2. Train Model

```bash
# Train CodeBERT model
python -m src.train \
    --data data/processed/comments_dataset.csv \
    --model microsoft/codebert-base \
    --epochs 3 \
    --batch-size 16 \
    --output models/codebert_finetuned

# Quick test run (1 epoch, small subset)
python -m src.train --data data/processed/comments_dataset.csv --epochs 1 --debug
```

### 3. Evaluate

```bash
# Evaluate trained model
python -m src.evaluate \
    --model models/codebert_finetuned \
    --data data/processed/comments_dataset.csv \
    --output results/

# Compare with baselines
python scripts/run_baseline.py --data data/processed/comments_dataset.csv
```

### 4. Analyze Attention

```bash
# Generate attention visualizations
python -m src.features \
    --model models/codebert_finetuned \
    --examples "Use binary search for O(log n) performance" \
    --output results/attention_viz.png
```

## Project Structure

```
src/
├── dataset.py       # Dataset creation and preprocessing
├── model.py         # Model architecture and training setup
├── train.py         # Training loop and optimization
├── evaluate.py      # Evaluation metrics and analysis
├── features.py      # Feature extraction and attention analysis
└── baseline.py      # Baseline implementations

data/
├── raw/            # Original scraped data
├── processed/      # Cleaned and split datasets
└── annotations/    # Manual quality annotations

models/             # Saved model checkpoints

results/            # Evaluation results and figures
```

## Dataset

### Synthetic Dataset (Default)

For reproducibility and demonstration, we provide a synthetic dataset generator that creates realistic code comment examples across quality levels:

- **High Quality**: Explains WHY, includes complexity analysis, mentions edge cases
- **Medium Quality**: Describes WHAT, some context but incomplete
- **Low Quality**: States the obvious, redundant with code

### Real Dataset (Optional)

Instructions for collecting and annotating real comments:

1. **Mining**: Use `scripts/download_data.py` to scrape Python repositories
2. **Annotation**: Follow annotation guidelines in `data/annotations/README.md`
3. **Inter-rater Reliability**: Calculate Fleiss' Kappa for multi-annotator agreement

## Methodology

### Model Architecture

- **Base Model**: `microsoft/codebert-base` (125M parameters)
- **Fine-tuning**: Classification head (768 → 3 classes)
- **Optimizer**: AdamW with linear warmup
- **Learning Rate**: 2e-5 with warmup over 500 steps

### Training Details

- **Data Split**: 80% train, 20% test (stratified)
- **Batch Size**: 16 (gradient accumulation if needed)
- **Epochs**: 3 (with early stopping)
- **Loss**: Cross-entropy
- **Metrics**: Accuracy, Precision, Recall, F1 (macro)

### Baselines

1. **Rule-based**: Heuristics on comment length, keyword presence, patterns
2. **TF-IDF + Logistic Regression**: Classical NLP approach
3. **Random**: Stratified random guessing (sanity check)

## Evaluation Metrics

### Classification Performance

- **Accuracy**: Overall correctness
- **Macro F1**: Average F1 across all classes (accounts for imbalance)
- **Confusion Matrix**: Per-class error analysis
- **Cohen's Kappa**: Agreement beyond chance

### Feature Analysis

- **Attention Scores**: Which tokens the model focuses on
- **Feature Importance**: Correlation between linguistic features and quality
- **Error Analysis**: Qualitative examination of misclassifications

## Reproducing Results

### Exact Reproduction

```bash
# Set random seeds
export PYTHONHASHSEED=42

# Generate dataset
python scripts/download_data.py --synthetic --size 2000 --seed 42

# Train model
python -m src.train \
    --data data/processed/comments_dataset.csv \
    --seed 42 \
    --epochs 3 \
    --output models/codebert_reproduction

# Evaluate
python -m src.evaluate \
    --model models/codebert_reproduction \
    --data data/processed/comments_dataset.csv \
    --output results/reproduction/
```



## Extending the Project

### Fine-tune on Your Own Data

```python
from src.dataset import CommentDataset
from src.model import CommentClassifier

# Load the data
dataset = CommentDataset.from_csv('the_data.csv')

# Train model
classifier = CommentClassifier(model_name='microsoft/codebert-base')
classifier.train(dataset, epochs=3, batch_size=16)

# Evaluate
results = classifier.evaluate(dataset.test_data)
print(results)
```

### Add New Features

Edit `src/features.py`:

```python
def extract_custom_features(comment_text):
    """Add the feature extraction logic"""
    features = {}
    # the code here
    return features
```

### Try Different Models

```bash
# GraphCodeBERT (incorporates code structure)
python -m src.train --model microsoft/graphcodebert-base

# CodeT5 (encoder-decoder architecture)
python -m src.train --model Salesforce/codet5-base
```

## Running Tests

```bash
# All tests
pytest tests/ -v

# With coverage
pytest tests/ --cov=src --cov-report=html

# Specific test
pytest tests/test_dataset.py::test_tokenization -v
```

## Citation

```bibtex
@misc{code-comment-analyzer-2024,
  author = {Mohammed Aabed},
  title = {Semantic Code Comment Analysis Using Transformers},
  year = {2024},
  publisher = {GitHub},
  url = {https://github.com/mohammed-abed/code-comment-analyzer}
}
```

## Related Work

- [CodeBERT: A Pre-Trained Model for Programming and Natural Languages](https://arxiv.org/abs/2002.08155)
- [Assessing the Quality of GitHub Copilot's Code Generation](https://arxiv.org/abs/2206.14387)
- [On the Naturalness of Software](https://dl.acm.org/doi/10.5555/2337223.2337322)

## License

MIT License - see [LICENSE](LICENSE) file

## Contributing

We welcome contributions! Please see [CONTRIBUTING.md](CONTRIBUTING.md) for guidelines.

## Contact

Mohammed Aabed - maabed90@students.iugaza.edu.ps 

Project Link: [https://github.com/mohammed-abed/code-comment-analyzer](https://github.com/mohammed-abed/code-comment-analyzer)

## Acknowledgments

- CodeBERT team at Microsoft Research
- Hugging Face Transformers library
- GitHub for providing code repositories



## Development note: 

This project was built offline and pushed to GitHub upon completion rather than incrementally. I was based in Gaza during this period, where consistent internet access was not reliably available. The commit history does not reflect the actual development timeline."