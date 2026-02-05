"""
Evaluation script for comment quality classifier
"""

import argparse
import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.metrics import (
    accuracy_score,
    precision_recall_fscore_support,
    confusion_matrix,
    classification_report,
    cohen_kappa_score
)
import matplotlib.pyplot as plt
import seaborn as sns

from .dataset import CommentDataset
from .model import CommentQualityModel


def evaluate_model(
    model_path: str,
    data_path: str,
    output_dir: str = 'results',
    seed: int = 42
):
    """
    Evaluate trained model.
    
    Args:
        model_path: Path to saved model
        data_path: Path to dataset CSV
        output_dir: Directory to save results
        seed: Random seed
    """
    print("="*60)
    print("EVALUATING COMMENT QUALITY CLASSIFIER")
    print("="*60)
    
    # Create output directory
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    # Load model
    print(f"\n1. Loading model from {model_path}...")
    model = CommentQualityModel.load(model_path)
    
    # Load dataset
    print(f"\n2. Loading dataset from {data_path}...")
    dataset = CommentDataset.from_csv(data_path, seed=seed)
    _, test_dataset = dataset.split(test_size=0.2, stratify=True)
    
    print(f"Test set size: {len(test_dataset)}")
    
    # Get predictions
    print("\n3. Generating predictions...")
    test_comments = test_dataset.data['comment'].tolist()
    test_labels = test_dataset.data['label'].values
    
    predictions = model.predict(test_comments)
    
    # Calculate metrics
    print("\n4. Calculating metrics...")
    
    accuracy = accuracy_score(test_labels, predictions)
    precision, recall, f1, _ = precision_recall_fscore_support(
        test_labels, predictions, average='macro'
    )
    kappa = cohen_kappa_score(test_labels, predictions)
    
    # Per-class metrics
    precision_per_class, recall_per_class, f1_per_class, support = \
        precision_recall_fscore_support(test_labels, predictions, average=None)
    
    # Confusion matrix
    cm = confusion_matrix(test_labels, predictions)
    
    # Print results
    print("\n" + "="*60)
    print("RESULTS")
    print("="*60)
    
    print(f"\nOverall Metrics:")
    print(f"  Accuracy:  {accuracy:.4f}")
    print(f"  Precision: {precision:.4f}")
    print(f"  Recall:    {recall:.4f}")
    print(f"  F1 Score:  {f1:.4f}")
    print(f"  Cohen's κ: {kappa:.4f}")
    
    print(f"\nPer-Class Metrics:")
    class_names = ['Low', 'Medium', 'High']
    for i, name in enumerate(class_names):
        print(f"  {name:8s} - P: {precision_per_class[i]:.3f}, "
              f"R: {recall_per_class[i]:.3f}, "
              f"F1: {f1_per_class[i]:.3f}, "
              f"Support: {support[i]}")
    
    print(f"\nClassification Report:")
    print(classification_report(test_labels, predictions, target_names=class_names))
    
    # Save results
    results = {
        'accuracy': accuracy,
        'precision_macro': precision,
        'recall_macro': recall,
        'f1_macro': f1,
        'cohen_kappa': kappa,
        'precision_per_class': precision_per_class.tolist(),
        'recall_per_class': recall_per_class.tolist(),
        'f1_per_class': f1_per_class.tolist(),
        'support': support.tolist(),
        'confusion_matrix': cm.tolist()
    }
    
    import json
    with open(output_path / 'metrics.json', 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"\nMetrics saved to {output_path / 'metrics.json'}")
    
    # Plot confusion matrix
    plot_confusion_matrix(cm, class_names, output_path / 'confusion_matrix.png')
    
    # Error analysis
    print("\n5. Performing error analysis...")
    error_analysis(test_dataset.data, predictions, test_labels, output_path)
    
    return results


def plot_confusion_matrix(cm, class_names, save_path):
    """Plot and save confusion matrix"""
    plt.figure(figsize=(8, 6))
    sns.heatmap(
        cm,
        annot=True,
        fmt='d',
        cmap='Blues',
        xticklabels=class_names,
        yticklabels=class_names,
        cbar_kws={'label': 'Count'}
    )
    plt.xlabel('Predicted Label', fontsize=12)
    plt.ylabel('True Label', fontsize=12)
    plt.title('Confusion Matrix', fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"Confusion matrix saved to {save_path}")
    plt.close()


def error_analysis(data, predictions, true_labels, output_dir):
    """Analyze misclassifications"""
    errors = data.copy()
    errors['prediction'] = predictions
    errors['true_label'] = true_labels
    errors['correct'] = predictions == true_labels
    
    misclassified = errors[~errors['correct']]
    
    print(f"\nMisclassifications: {len(misclassified)} / {len(errors)} "
          f"({len(misclassified)/len(errors)*100:.1f}%)")
    
    # Sample errors
    print("\nSample Misclassifications:")
    for idx, row in misclassified.head(5).iterrows():
        print(f"\nComment: {row['comment'][:100]}...")
        print(f"  True: {row['true_label']}, Predicted: {row['prediction']}")
    
    # Save all errors
    misclassified.to_csv(output_dir / 'misclassifications.csv', index=False)
    print(f"\nAll misclassifications saved to {output_dir / 'misclassifications.csv'}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Evaluate comment quality classifier')
    parser.add_argument('--model', type=str, required=True,
                       help='Path to trained model')
    parser.add_argument('--data', type=str, required=True,
                       help='Path to dataset CSV')
    parser.add_argument('--output', type=str, default='results',
                       help='Output directory')
    parser.add_argument('--seed', type=int, default=42,
                       help='Random seed')
    
    args = parser.parse_args()
    
    evaluate_model(
        model_path=args.model,
        data_path=args.data,
        output_dir=args.output,
        seed=args.seed
    )

