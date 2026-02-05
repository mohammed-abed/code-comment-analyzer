"""
Feature extraction and attention visualization
"""

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from typing import List, Dict
from pathlib import Path


def visualize_attention(
    model,
    text: str,
    save_path: str = None,
    top_k: int = 10
):
    """
    Visualize attention weights for a comment.
    
    Args:
        model: CommentQualityModel instance
        text: Comment text
        save_path: Path to save figure
        top_k: Number of top attended tokens to highlight
    """
    attention_data = model.get_attention_weights(text)
    
    tokens = attention_data['tokens']
    scores = attention_data['attention_scores']
    
    # Remove special tokens
    valid_indices = [
        i for i, token in enumerate(tokens)
        if token not in ['<s>', '</s>', '<pad>']
    ]
    
    tokens_clean = [tokens[i] for i in valid_indices]
    scores_clean = scores[valid_indices]
    
    # Normalize scores
    scores_normalized = scores_clean / scores_clean.sum()
    
    # Create figure
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8))
    
    # Bar plot of attention scores
    colors = ['red' if i < top_k else 'blue' 
             for i in np.argsort(scores_normalized)[::-1]]
    
    ax1.barh(range(len(tokens_clean)), scores_normalized, color=colors, alpha=0.6)
    ax1.set_yticks(range(len(tokens_clean)))
    ax1.set_yticklabels(tokens_clean, fontsize=8)
    ax1.set_xlabel('Attention Weight', fontsize=11)
    ax1.set_title(f'Token Attention Weights\n(Red = Top {top_k})', 
                  fontsize=12, fontweight='bold')
    ax1.grid(axis='x', alpha=0.3)
    
    # Heatmap
    scores_matrix = scores_normalized.reshape(1, -1)
    sns.heatmap(
        scores_matrix,
        cmap='YlOrRd',
        xticklabels=tokens_clean,
        yticklabels=['Attention'],
        cbar_kws={'label': 'Weight'},
        ax=ax2
    )
    ax2.set_title('Attention Heatmap', fontsize=12, fontweight='bold')
    plt.setp(ax2.get_xticklabels(), rotation=45, ha='right', fontsize=8)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Attention visualization saved to {save_path}")
    else:
        plt.show()
    
    plt.close()
    
    # Print top attended tokens
    print(f"\nTop {top_k} attended tokens:")
    top_indices = np.argsort(scores_normalized)[::-1][:top_k]
    for rank, idx in enumerate(top_indices, 1):
        print(f"  {rank}. '{tokens_clean[idx]}': {scores_normalized[idx]:.4f}")


def extract_linguistic_features(comments: List[str]) -> Dict[str, np.ndarray]:
    """
    Extract linguistic features from comments.
    
    Args:
        comments: List of comment strings
        
    Returns:
        Dictionary of feature arrays
    """
    features = {
        'length_chars': [],
        'length_words': [],
        'avg_word_length': [],
        'num_sentences': [],
        'has_code_terms': [],
        'has_why_keywords': [],
        'has_numbers': [],
        'has_examples': [],
        'complexity_score': []
    }
    
    why_keywords = ['because', 'to avoid', 'to prevent', 'for', 'since']
    code_terms = ['o(', 'complexity', 'algorithm', 'recursive', 'iterative']
    
    for comment in comments:
        comment_lower = comment.lower()
        words = comment.split()
        
        features['length_chars'].append(len(comment))
        features['length_words'].append(len(words))
        features['avg_word_length'].append(
            np.mean([len(w) for w in words]) if words else 0
        )
        features['num_sentences'].append(comment.count('.') + comment.count('!') + 1)
        features['has_code_terms'].append(
            int(any(term in comment_lower for term in code_terms))
        )
        features['has_why_keywords'].append(
            int(any(kw in comment_lower for kw in why_keywords))
        )
        features['has_numbers'].append(int(bool(re.search(r'\d', comment))))
        features['has_examples'].append(
            int('example' in comment_lower or 'e.g.' in comment_lower)
        )
        
        # Simple complexity score
        complexity = (
            len(words) / 20 +  # Length factor
            features['has_code_terms'][-1] * 0.3 +
            features['has_why_keywords'][-1] * 0.3 +
            features['has_numbers'][-1] * 0.2
        )
        features['complexity_score'].append(min(complexity, 1.0))
    
    # Convert to arrays
    return {k: np.array(v) for k, v in features.items()}


def analyze_feature_importance(dataset, labels):
    """
    Analyze which features correlate with quality.
    
    Args:
        dataset: CommentDataset
        labels: Quality labels
    """
    from scipy.stats import f_oneway
    
    comments = dataset.data['comment'].tolist()
    features = extract_linguistic_features(comments)
    
    print("\n" + "="*60)
    print("FEATURE IMPORTANCE ANALYSIS")
    print("="*60)
    
    for feature_name, feature_values in features.items():
        # Group by label
        groups = [
            feature_values[labels == label]
            for label in [0, 1, 2]
        ]
        
        # ANOVA F-test
        f_stat, p_value = f_oneway(*groups)
        
        # Means per class
        means = [group.mean() for group in groups]
        
        print(f"\n{feature_name}:")
        print(f"  F-statistic: {f_stat:.4f}")
        print(f"  p-value: {p_value:.4f}")
        print(f"  Means by class: Low={means[0]:.3f}, "
              f"Medium={means[1]:.3f}, High={means[2]:.3f}")
        print(f"  Significant: {'Yes' if p_value < 0.05 else 'No'}")


import re
