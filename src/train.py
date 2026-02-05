"""
Training script for comment quality classifier
"""

import argparse
from pathlib import Path
import torch
from transformers import TrainingArguments, Trainer
import numpy as np

from .dataset import CommentDataset
from .model import CommentQualityModel, compute_metrics


def train_model(
    data_path: str,
    model_name: str = 'microsoft/codebert-base',
    output_dir: str = 'models/codebert_finetuned',
    epochs: int = 3,
    batch_size: int = 16,
    learning_rate: float = 2e-5,
    seed: int = 42,
    debug: bool = False
):
    """
    Train comment quality classification model.
    
    Args:
        data_path: Path to dataset CSV
        model_name: Hugging Face model identifier
        output_dir: Directory to save trained model
        epochs: Number of training epochs
        batch_size: Training batch size
        learning_rate: Learning rate
        seed: Random seed
        debug: If True, use small subset for quick testing
    """
    # Set seeds
    torch.manual_seed(seed)
    np.random.seed(seed)
    
    print("="*60)
    print("TRAINING COMMENT QUALITY CLASSIFIER")
    print("="*60)
    
    # Load dataset
    print(f"\n1. Loading dataset from {data_path}...")
    dataset = CommentDataset.from_csv(data_path, seed=seed)
    
    if debug:
        print("DEBUG MODE: Using subset of data")
        dataset.data = dataset.data.head(200)
    
    stats = dataset.get_statistics()
    print(f"Dataset statistics:")
    for key, value in stats.items():
        print(f"  {key}: {value}")
    
    # Split data
    print("\n2. Splitting data...")
    train_dataset, test_dataset = dataset.split(test_size=0.2, stratify=True)
    print(f"Train size: {len(train_dataset)}")
    print(f"Test size: {len(test_dataset)}")
    
    # Initialize model
    print(f"\n3. Initializing model: {model_name}...")
    model_wrapper = CommentQualityModel(model_name=model_name, num_labels=3)
    
    # Tokenize datasets
    print("\n4. Tokenizing datasets...")
    train_tokenized = model_wrapper.tokenize_dataset(train_dataset)
    test_tokenized = model_wrapper.tokenize_dataset(test_dataset)
    
    # Training arguments
    training_args = TrainingArguments(
        output_dir=output_dir,
        num_train_epochs=epochs,
        per_device_train_batch_size=batch_size,
        per_device_eval_batch_size=batch_size * 2,
        learning_rate=learning_rate,
        weight_decay=0.01,
        warmup_steps=500,
        logging_dir=f'{output_dir}/logs',
        logging_steps=50,
        evaluation_strategy="epoch",
        save_strategy="epoch",
        load_best_model_at_end=True,
        metric_for_best_model="f1",
        seed=seed,
        report_to="none",  # Disable wandb/tensorboard
    )
    
    # Initialize trainer
    trainer = Trainer(
        model=model_wrapper.model,
        args=training_args,
        train_dataset=train_tokenized,
        eval_dataset=test_tokenized,
        compute_metrics=compute_metrics,
    )
    
    # Train
    print("\n5. Training model...")
    print(f"Epochs: {epochs}")
    print(f"Batch size: {batch_size}")
    print(f"Learning rate: {learning_rate}")
    print(f"Device: {model_wrapper.device}")
    print()
    
    train_result = trainer.train()
    
    # Save model
    print(f"\n6. Saving model to {output_dir}...")
    trainer.save_model(output_dir)
    model_wrapper.tokenizer.save_pretrained(output_dir)
    
    # Final evaluation
    print("\n7. Final evaluation...")
    eval_results = trainer.evaluate()
    
    print("\nTraining complete!")
    print(f"Final results:")
    for key, value in eval_results.items():
        print(f"  {key}: {value:.4f}")
    
    return model_wrapper, eval_results


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Train comment quality classifier')
    parser.add_argument('--data', type=str, required=True,
                       help='Path to dataset CSV')
    parser.add_argument('--model', type=str, default='microsoft/codebert-base',
                       help='Model name or path')
    parser.add_argument('--output', type=str, default='models/codebert_finetuned',
                       help='Output directory')
    parser.add_argument('--epochs', type=int, default=3,
                       help='Number of epochs')
    parser.add_argument('--batch-size', type=int, default=16,
                       help='Batch size')
    parser.add_argument('--lr', type=float, default=2e-5,
                       help='Learning rate')
    parser.add_argument('--seed', type=int, default=42,
                       help='Random seed')
    parser.add_argument('--debug', action='store_true',
                       help='Debug mode (small subset)')
    
    args = parser.parse_args()
    
    train_model(
        data_path=args.data,
        model_name=args.model,
        output_dir=args.output,
        epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.lr,
        seed=args.seed,
        debug=args.debug
    )

