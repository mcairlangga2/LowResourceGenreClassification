#!/usr/bin/env python3
"""Utility functions for training and evaluation."""

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from typing import List, Dict
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, confusion_matrix
from transformers.trainer_callback import TrainerCallback
import wandb

class LossLoggingCallback(TrainerCallback):
    """Callback for logging training and evaluation losses."""
    
    def __init__(self):
        super().__init__()
        self.training_loss = []
        self.eval_loss = []
        self.steps = []
        
    def on_log(self, args, state, control, logs=None, **kwargs):
        if logs:
            step = state.global_step
            if 'loss' in logs:
                self.training_loss.append(logs['loss'])
                self.steps.append(step)
                wandb.log({
                    'train/loss': logs['loss'],
                    'step': step
                }, step=step)
            if 'eval_loss' in logs:
                self.eval_loss.append(logs['eval_loss'])
                wandb.log({
                    'eval/loss': logs['eval_loss'],
                    'step': step
                }, step=step)

def plot_confusion_matrix(y_true: np.ndarray, y_pred: np.ndarray, 
                         labels: List[str], title: str = 'Confusion Matrix') -> plt.Figure:
    """
    Generate confusion matrix plot.
    
    Args:
        y_true: True labels
        y_pred: Predicted labels
        labels: Label names
        title: Plot title
        
    Returns:
        Matplotlib figure
    """
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(12, 10))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=labels, yticklabels=labels)
    plt.title(title)
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.xticks(rotation=45)
    plt.yticks(rotation=45)
    plt.tight_layout()
    return plt.gcf()

def compute_metrics(pred) -> Dict[str, float]:
    """
    Compute evaluation metrics.
    
    Args:
        pred: Prediction object from trainer
        
    Returns:
        Dictionary of metrics
    """
    labels = pred.label_ids
    preds = pred.predictions.argmax(-1)
    precision, recall, f1, _ = precision_recall_fscore_support(
        labels, preds, average='weighted'
    )
    acc = accuracy_score(labels, preds)
    
    return {
        'accuracy': acc,
        'f1': f1,
        'precision': precision,
        'recall': recall
    }
