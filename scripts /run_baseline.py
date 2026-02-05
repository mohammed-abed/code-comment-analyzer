"""
Run baseline model comparisons
"""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.dataset import CommentDataset
from src.baseline import compare_baselines


def main():
    parser = argparse.ArgumentParser(description='Run baseline comparisons')
    parser.add_argument('--data', type=str, required=True,
                       help='Path to dataset CSV')
    parser.add_argument('--seed', type=int, default=42,
                       help='Random seed')
    
    args = parser.parse_args()
    
    print("Loading dataset...")
    dataset = CommentDataset.from_csv(args.data, seed=args.seed)
    train_dataset, test_dataset = dataset.split(test_size=0.2, stratify=True)
    
    print(f"Train size: {len(train_dataset)}")
    print(f"Test size: {len(test_dataset)}")
    
    results = compare_baselines(train_dataset, test_dataset)
    
    print("\n" + "="*60)
    print("SUMMARY")
    print("="*60)
    
    for model_name, metrics in results.items():
        print(f"\n{model_name}:")
        print(f"  Accuracy: {metrics['accuracy']:.4f}")
        print(f"  F1:       {metrics['f1']:.4f}")
    
    return 0


if __name__ == "__main__":
    sys.exit(main())

