"""Unit tests for feature extraction"""

import pytest
import numpy as np
from src.features import extract_linguistic_features


def test_feature_extraction():
    """Test linguistic feature extraction"""
    comments = [
        "Loop through items",  # Low quality
        "Calculate average using dynamic programming for O(n) complexity",  # High quality
        "Process the input data"  # Medium quality
    ]
    
    features = extract_linguistic_features(comments)
    
    assert 'length_chars' in features
    assert 'length_words' in features
    assert 'has_code_terms' in features
    assert len(features['length_chars']) == 3


def test_complexity_score():
    """Test complexity score calculation"""
    comments = [
        "x = 0",  # Very simple
        "Use binary search for O(log n) performance because linear scan is too slow"  # Complex
    ]
    
    features = extract_linguistic_features(comments)
    
    assert features['complexity_score'][0] < features['complexity_score'][1]


def test_keyword_detection():
    """Test keyword detection"""
    comments = [
        "This is to avoid race conditions",
        "Simple loop"
    ]
    
    features = extract_linguistic_features(comments)
    
    assert features['has_why_keywords'][0] == 1
    assert features['has_why_keywords'][1] == 0


