"""Unit tests for dataset module"""

import pytest
import pandas as pd
from src.dataset import CommentDataset


def test_synthetic_dataset_creation():
    """Test synthetic dataset generation"""
    dataset = CommentDataset.create_synthetic(n_samples=100, seed=42)
    
    assert len(dataset) == 100
    assert 'comment' in dataset.data.columns
    assert 'label' in dataset.data.columns
    assert set(dataset.data['label'].unique()) == {0, 1, 2}


def test_dataset_split():
    """Test train/test split"""
    dataset = CommentDataset.create_synthetic(n_samples=100, seed=42)
    train, test = dataset.split(test_size=0.2, stratify=True)
    
    assert len(train) == 80
    assert len(test) == 20
    assert len(set(train.data.index) & set(test.data.index)) == 0


def test_dataset_statistics():
    """Test statistics calculation"""
    dataset = CommentDataset.create_synthetic(n_samples=100, seed=42)
    stats = dataset.get_statistics()
    
    assert 'total_samples' in stats
    assert stats['total_samples'] == 100
    assert 'label_distribution' in stats


def test_dataset_save_load(tmp_path):
    """Test saving and loading"""
    dataset = CommentDataset.create_synthetic(n_samples=50, seed=42)
    
    filepath = tmp_path / "test_dataset.csv"
    dataset.save(str(filepath))
    
    loaded = CommentDataset.from_csv(str(filepath))
    
    assert len(loaded) == len(dataset)
    pd.testing.assert_frame_equal(loaded.data, dataset.data)


def test_dataset_getitem():
    """Test indexing"""
    dataset = CommentDataset.create_synthetic(n_samples=10, seed=42)
    
    item = dataset[0]
    
    assert 'comment' in item
    assert 'label' in item
    assert isinstance(item['comment'], str)
    assert item['label'] in [0, 1, 2]

