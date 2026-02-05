"""
Baseline models for comparison
"""

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report
from typing import List, Tuple
import re


class RuleBasedBaseline:
    """Simple rule-based baseline using heuristics"""
    
    def __init__(self):
        self.name = "Rule-Based"
    
    def predict(self, comments: List[str]) -> np.ndarray:
        """
        Predict quality based on simple rules.
        
        Rules:
        - Low: Very short, obvious patterns
        - Medium: Moderate length, some context
        - High: Long, explanatory keywords, mentions complexity/edge cases
        """
        predictions = []
        
        for comment in comments:
            score = 50  # Start neutral
            words = comment.lower().split()
            
            # Length features
            if len(words) < 3:
                score -= 30
            elif len(words) > 15:
                score += 20
            
            # Explanatory keywords (WHY, not WHAT)
            why_keywords = ['because', 'to avoid', 'to prevent', 'for performance', 
                          'optimization', 'handles', 'edge case', 'prevents']
            if any(kw in comment.lower() for kw in why_keywords):
                score += 25
            
            # Technical depth
            technical_terms = ['complexity', 'o(', 'algorithm', 'thread', 
                             'async', 'cache', 'memory', 'recursive']
            if any(term in comment.lower() for term in technical_terms):
                score += 15
            
            # Examples/specifics
            if 'example' in comment.lower() or 'e.g.' in comment.lower():
                score += 15
            
            # Obvious/redundant patterns
            obvious = ['loop through', 'set to', 'return', 'call', 'get', 'print']
            if any(pattern in comment.lower() for pattern in obvious) and len(words) < 6:
                score -= 20
            
            # Number mentions (specific examples)
            if re.search(r'\d+', comment):
                score += 10
            
            # Convert score to label
            if score >= 65:
                predictions.append(2)  # High
            elif score >= 45:
                predictions.append(1)  # Medium
            else:
                predictions.append(0)  # Low
        
        return np.array(predictions)
    
    def fit(self, X, y):
        """No training needed for rule-based"""
        pass
    
    def score(self, X, y):
        """Calculate accuracy"""
        predictions = self.predict(X)
        return accuracy_score(y, predictions)


class TfidfBaseline:
    """TF-IDF + Logistic Regression baseline"""
    
    def __init__(self, max_features: int = 1000):
        self.name = "TF-IDF + LogReg"
        self.vectorizer = TfidfVectorizer(
            max_features=max_features,
            stop_words='english',
            ngram_range=(1, 2)
        )
        self.classifier = LogisticRegression(
            max_iter=1000,
            random_state=42,
            class_weight='balanced'
        )
    
    def fit(self, comments: List[str], labels: np.ndarray):
        """Train the model"""
        X = self.vectorizer.fit_transform(comments)
        self.classifier.fit(X, labels)
    
    def predict(self, comments: List[str]) -> np.ndarray:
        """Predict quality labels"""
        X = self.vectorizer.transform(comments)
        return self.classifier.predict(X)
    
    def score(self, comments: List[str], labels: np.ndarray) -> float:
        """Calculate accuracy"""
        predictions = self.predict(comments)
        return accuracy_score(labels, predictions)


class RandomBaseline:
    """Stratified random baseline (sanity check)"""
    
    def __init__(self):
        self.name = "Random"
        self.class_probs = None
    
    def fit(self, X, y):
        """Learn class distribution"""
        unique, counts = np.unique(y, return_counts=True)
        self.class_probs = counts / counts.sum()
    
    def predict(self, comments: List[str]) -> np.ndarray:
        """Random predictions following class distribution"""
        return np.random.choice(
            len(self.class_probs),
            size=len(comments),
            p=self.class_probs
        )
    
    def score(self, X, y):
        """Calculate accuracy"""
        predictions = self.predict(X)
        return accuracy_score(y, predictions)


def compare_baselines(dataset, test_dataset):
    """
    Compare all baseline models.
    
    Args:
        dataset: Training CommentDataset
        test_dataset: Test CommentDataset
        
    Returns:
        Dictionary of results
    """
    print("\n" + "="*60)
    print("BASELINE COMPARISON")
    print("="*60)
    
    train_comments = dataset.data['comment'].tolist()
    train_labels = dataset.data['label'].values
    test_comments = test_dataset.data['comment'].tolist()
    test_labels = test_dataset.data['label'].values
    
    baselines = [
        RandomBaseline(),
        RuleBasedBaseline(),
        TfidfBaseline(max_features=1000)
    ]
    
    results = {}
    
    for baseline in baselines:
        print(f"\n{baseline.name}:")
        print("-" * 40)
        
        # Train if applicable
        if hasattr(baseline, 'fit'):
            print("Training...")
            baseline.fit(train_comments, train_labels)
        
        # Predict
        print("Evaluating...")
        predictions = baseline.predict(test_comments)
        
        # Metrics
        from sklearn.metrics import precision_recall_fscore_support
        
        accuracy = accuracy_score(test_labels, predictions)
        precision, recall, f1, _ = precision_recall_fscore_support(
            test_labels, predictions, average='macro'
        )
        
        results[baseline.name] = {
            'accuracy': accuracy,
            'precision': precision,
            'recall': recall,
            'f1': f1,
            'predictions': predictions
        }
        
        print(f"  Accuracy:  {accuracy:.4f}")
        print(f"  Precision: {precision:.4f}")
        print(f"  Recall:    {recall:.4f}")
        print(f"  F1 Score:  {f1:.4f}")
        
        print(f"\nClassification Report:")
        print(classification_report(
            test_labels, 
            predictions,
            target_names=['Low', 'Medium', 'High']
        ))
    
    return results
