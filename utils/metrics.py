import numpy as np
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score
import torch
import torch.nn.functional as F

def compute_metrics(outputs, targets, domain='medical'):
    """
    Computes metrics based on model outputs and targets.
    
    Args:
        outputs: Raw logits from the model (N, C) or (N,).
        targets: Ground truth labels (N,).
        domain: 'medical' or 'industrial'. 
                'medical' requires ROC-AUC.
                
    Returns:
        dict: Dictionary containing calculated metrics.
    """
    
    # Convert logits to probabilities
    probs = F.softmax(torch.tensor(outputs), dim=1).numpy()
    preds = np.argmax(probs, axis=1)
    targets = np.array(targets)
    
    metrics = {}
    
    # Accuracy
    metrics['accuracy'] = accuracy_score(targets, preds)
    
    # Macro F1
    metrics['f1_macro'] = f1_score(targets, preds, average='macro')
    
    # ROC-AUC for medical
    if domain == 'medical':
        try:
            # Handle multi-class case for ROC-AUC
            if probs.shape[1] > 2:
                 metrics['roc_auc'] = roc_auc_score(targets, probs, multi_class='ovr', average='macro')
            else:
                 # Binary case, use positive class prob
                 metrics['roc_auc'] = roc_auc_score(targets, probs[:, 1])
        except ValueError:
            # Fallback if AUC cannot be computed (e.g., only one class present in batch)
            metrics['roc_auc'] = float('nan')
            
    return metrics
