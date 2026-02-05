"""
Transformer model for comment quality classification
"""

import torch
import torch.nn as nn
from transformers import (
    RobertaTokenizer,
    RobertaForSequenceClassification,
    AutoTokenizer,
    AutoModelForSequenceClassification,
    TrainingArguments,
    Trainer
)
from typing import Dict, List, Optional
import numpy as np
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, confusion_matrix


class CommentQualityModel:
    """Wrapper for transformer-based comment quality classifier"""
    
    def __init__(
        self,
        model_name: str = 'microsoft/codebert-base',
        num_labels: int = 3,
        device: str = None
    ):
        """
        Initialize model.
        
        Args:
            model_name: Hugging Face model identifier
            num_labels: Number of quality classes
            device: Device to use ('cuda' or 'cpu')
        """
        self.model_name = model_name
        self.num_labels = num_labels
        
        # Set device
        if device is None:
            self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        else:
            self.device = device
        
        # Load tokenizer and model
        print(f"Loading {model_name}...")
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForSequenceClassification.from_pretrained(
            model_name,
            num_labels=num_labels
        )
        self.model.to(self.device)
        
        print(f"Model loaded on {self.device}")
        print(f"Parameters: {sum(p.numel() for p in self.model.parameters()):,}")
    
    def tokenize_dataset(self, dataset, max_length: int = 128):
        """
        Tokenize dataset for training.
        
        Args:
            dataset: CommentDataset instance
            max_length: Maximum sequence length
            
        Returns:
            Hugging Face Dataset
        """
        from datasets import Dataset as HFDataset
        
        def tokenize_function(examples):
            return self.tokenizer(
                examples['comment'],
                padding='max_length',
                truncation=True,
                max_length=max_length
            )
        
        # Convert to Hugging Face dataset
        hf_dataset = HFDataset.from_pandas(dataset.data)
        
        # Tokenize
        tokenized = hf_dataset.map(
            tokenize_function,
            batched=True,
            remove_columns=['comment', 'code_context', 'quality_score']
        )
        
        return tokenized
    
    def predict(self, texts: List[str], batch_size: int = 32) -> np.ndarray:
        """
        Predict quality labels for comments.
        
        Args:
            texts: List of comment strings
            batch_size: Batch size for inference
            
        Returns:
            Array of predicted labels
        """
        self.model.eval()
        predictions = []
        
        with torch.no_grad():
            for i in range(0, len(texts), batch_size):
                batch_texts = texts[i:i + batch_size]
                
                inputs = self.tokenizer(
                    batch_texts,
                    padding=True,
                    truncation=True,
                    max_length=128,
                    return_tensors='pt'
                ).to(self.device)
                
                outputs = self.model(**inputs)
                logits = outputs.logits
                batch_predictions = torch.argmax(logits, dim=-1).cpu().numpy()
                predictions.extend(batch_predictions)
        
        return np.array(predictions)
    
    def get_attention_weights(self, text: str) -> Dict:
        """
        Get attention weights for a single comment.
        
        Args:
            text: Comment string
            
        Returns:
            Dictionary with tokens and attention weights
        """
        self.model.eval()
        
        inputs = self.tokenizer(
            text,
            return_tensors='pt',
            truncation=True,
            max_length=128
        ).to(self.device)
        
        with torch.no_grad():
            outputs = self.model(**inputs, output_attentions=True)
        
        # Average attention across all heads and layers
        attentions = outputs.attentions  # Tuple of attention matrices
        avg_attention = torch.stack(attentions).mean(dim=(0, 1, 2))  # Average over layers, heads, queries
        
        tokens = self.tokenizer.convert_ids_to_tokens(inputs['input_ids'][0])
        attention_scores = avg_attention.cpu().numpy()
        
        return {
            'tokens': tokens,
            'attention_scores': attention_scores,
            'text': text
        }
    
    def save(self, output_dir: str):
        """Save model and tokenizer"""
        from pathlib import Path
        Path(output_dir).mkdir(parents=True, exist_ok=True)
        
        self.model.save_pretrained(output_dir)
        self.tokenizer.save_pretrained(output_dir)
        print(f"Model saved to {output_dir}")
    
    @classmethod
    def load(cls, model_dir: str, device: str = None) -> 'CommentQualityModel':
        """Load saved model"""
        tokenizer = AutoTokenizer.from_pretrained(model_dir)
        model = AutoModelForSequenceClassification.from_pretrained(model_dir)
        
        instance = cls.__new__(cls)
        instance.tokenizer = tokenizer
        instance.model = model
        instance.device = device if device else ('cuda' if torch.cuda.is_available() else 'cpu')
        instance.model.to(instance.device)
        instance.num_labels = model.config.num_labels
        
        return instance


def compute_metrics(eval_pred):
    """Compute metrics for Hugging Face Trainer"""
    predictions, labels = eval_pred
    predictions = np.argmax(predictions, axis=1)
    
    precision, recall, f1, _ = precision_recall_fscore_support(
        labels, predictions, average='macro'
    )
    acc = accuracy_score(labels, predictions)
    
    return {
        'accuracy': acc,
        'f1': f1,
        'precision': precision,
        'recall': recall
    }
