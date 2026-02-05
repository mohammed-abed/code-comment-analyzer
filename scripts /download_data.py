"""
Download or generate dataset for comment quality analysis
"""

import argparse
import sys
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.dataset import CommentDataset


def main():
    parser = argparse.ArgumentParser(description='Download/generate dataset')
    parser.add_argument('--synthetic', action='store_true',
                       help='Generate synthetic dataset')
    parser.add_argument('--size', type=int, default=2000,
                       help='Dataset size (for synthetic)')
    parser.add_argument('--seed', type=int, default=42,
                       help='Random seed')
    parser.add_argument('--output', type=str, default='data/processed/',
                       help='Output directory')
    
    args = parser.parse_args()
    
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    if args.synthetic:
        print(f"Generating synthetic dataset with {args.size} samples...")
        dataset = CommentDataset.create_synthetic(
            n_samples=args.size,
            seed=args.seed
        )
        
        output_path = output_dir / 'comments_dataset.csv'
        dataset.save(str(output_path))
        
        # Print statistics
        print("\nDataset Statistics:")
        stats = dataset.get_statistics()
        for key, value in stats.items():
            print(f"  {key}: {value}")
        
        print(f"\nDataset saved to {output_path}")
    
    else:
        print("Real dataset download not implemented yet.")
        print("Use --synthetic flag to generate synthetic data.")
        return 1
    
    return 0


if __name__ == "__main__":
    sys.exit(main())
