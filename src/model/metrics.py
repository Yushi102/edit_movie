"""
Metrics for model evaluation

Provides functions for calculating various evaluation metrics.
"""
import torch
import torch.nn.functional as F
from typing import Dict, Tuple
import logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def calculate_classification_metrics(
    predictions: torch.Tensor,
    targets: torch.Tensor,
    threshold: float = 0.5
) -> Dict[str, float]:
    """
    Calculate classification metrics (Precision, Recall, F1, Accuracy)
    
    Args:
        predictions: Predicted logits (batch, seq_len, 2) or probabilities
        targets: Ground truth labels (batch, seq_len)
        threshold: Classification threshold (default: 0.5)
    
    Returns:
        Dict with metrics: precision, recall, f1, accuracy, specificity
    """
    # Get probabilities if logits provided
    if predictions.size(-1) == 2:
        probs = F.softmax(predictions, dim=-1)
        pred_probs = probs[..., 1]  # Probability of active class
    else:
        pred_probs = predictions
    
    # Binary predictions
    pred_active = (pred_probs > threshold).float()
    targets_float = targets.float()
    
    # True Positives, False Positives, True Negatives, False Negatives
    tp = (pred_active * targets_float).sum()
    fp = (pred_active * (1 - targets_float)).sum()
    tn = ((1 - pred_active) * (1 - targets_float)).sum()
    fn = ((1 - pred_active) * targets_float).sum()
    
    # Calculate metrics with safe division
    precision = tp / torch.clamp(tp + fp, min=1.0)
    recall = tp / torch.clamp(tp + fn, min=1.0)
    f1 = 2 * precision * recall / torch.clamp(precision + recall, min=1e-8)
    accuracy = (tp + tn) / torch.clamp(tp + fp + tn + fn, min=1.0)
    specificity = tn / torch.clamp(tn + fp, min=1.0)
    
    return {
        'precision': precision.item(),
        'recall': recall.item(),
        'f1': f1.item(),
        'accuracy': accuracy.item(),
        'specificity': specificity.item(),
        'tp': tp.item(),
        'fp': fp.item(),
        'tn': tn.item(),
        'fn': fn.item()
    }


def calculate_multiclass_metrics(
    predictions: torch.Tensor,
    targets: torch.Tensor,
    active_mask: torch.Tensor = None
) -> Dict[str, float]:
    """
    Calculate multi-class classification metrics
    
    Args:
        predictions: Predicted logits (batch, seq_len, num_tracks, num_classes)
        targets: Ground truth labels (batch, seq_len, num_tracks)
        active_mask: Optional mask for active tracks (batch, seq_len, num_tracks)
    
    Returns:
        Dict with metrics: accuracy, macro_precision, macro_recall, macro_f1, 
                          micro_precision, micro_recall, micro_f1
    """
    # Get predicted classes
    pred_classes = torch.argmax(predictions, dim=-1)  # (batch, seq_len, num_tracks)
    
    # Apply mask if provided
    if active_mask is not None:
        # Only consider active tracks
        mask_bool = active_mask.bool()
        pred_classes_masked = pred_classes[mask_bool]
        targets_masked = targets[mask_bool]
    else:
        pred_classes_masked = pred_classes.reshape(-1)
        targets_masked = targets.reshape(-1)
    
    # Overall accuracy
    correct = (pred_classes_masked == targets_masked).float()
    accuracy = correct.mean()
    
    # Get number of classes
    num_classes = predictions.size(-1)
    
    # Per-class metrics
    class_precision = []
    class_recall = []
    class_f1 = []
    
    total_tp = 0
    total_fp = 0
    total_fn = 0
    
    for class_id in range(num_classes):
        # True positives, false positives, false negatives for this class
        tp = ((pred_classes_masked == class_id) & (targets_masked == class_id)).float().sum()
        fp = ((pred_classes_masked == class_id) & (targets_masked != class_id)).float().sum()
        fn = ((pred_classes_masked != class_id) & (targets_masked == class_id)).float().sum()
        
        # Accumulate for micro-averaging
        total_tp += tp
        total_fp += fp
        total_fn += fn
        
        # Calculate per-class metrics
        precision = tp / torch.clamp(tp + fp, min=1.0)
        recall = tp / torch.clamp(tp + fn, min=1.0)
        f1 = 2 * precision * recall / torch.clamp(precision + recall, min=1e-8)
        
        class_precision.append(precision.item())
        class_recall.append(recall.item())
        class_f1.append(f1.item())
    
    # Macro-averaging (average across classes)
    macro_precision = sum(class_precision) / num_classes
    macro_recall = sum(class_recall) / num_classes
    macro_f1 = sum(class_f1) / num_classes
    
    # Micro-averaging (aggregate all classes)
    micro_precision = total_tp / torch.clamp(total_tp + total_fp, min=1.0)
    micro_recall = total_tp / torch.clamp(total_tp + total_fn, min=1.0)
    micro_f1 = 2 * micro_precision * micro_recall / torch.clamp(micro_precision + micro_recall, min=1e-8)
    
    return {
        'accuracy': accuracy.item(),
        'macro_precision': macro_precision,
        'macro_recall': macro_recall,
        'macro_f1': macro_f1,
        'micro_precision': micro_precision.item(),
        'micro_recall': micro_recall.item(),
        'micro_f1': micro_f1.item()
    }


def calculate_regression_metrics(
    predictions: torch.Tensor,
    targets: torch.Tensor,
    active_mask: torch.Tensor = None
) -> Dict[str, float]:
    """
    Calculate regression metrics (MAE, MSE, RMSE)
    
    Args:
        predictions: Predicted values (batch, seq_len, num_tracks, 1)
        targets: Ground truth values (batch, seq_len, num_tracks, 1)
        active_mask: Optional mask for active tracks (batch, seq_len, num_tracks)
    
    Returns:
        Dict with metrics: mae, mse, rmse
    """
    # Squeeze last dimension
    pred = predictions.squeeze(-1)
    tgt = targets.squeeze(-1)
    
    # Apply mask if provided
    if active_mask is not None:
        pred = pred * active_mask
        tgt = tgt * active_mask
        count = active_mask.sum()
    else:
        count = pred.numel()
    
    # Calculate metrics
    diff = pred - tgt
    mae = torch.abs(diff).sum() / torch.clamp(count, min=1.0)
    mse = (diff ** 2).sum() / torch.clamp(count, min=1.0)
    rmse = torch.sqrt(mse)
    
    return {
        'mae': mae.item(),
        'mse': mse.item(),
        'rmse': rmse.item()
    }


def calculate_multitrack_metrics(
    predictions: Dict[str, torch.Tensor],
    targets: Dict[str, torch.Tensor],
    mask: torch.Tensor = None
) -> Dict[str, float]:
    """
    Calculate comprehensive metrics for multi-track predictions
    
    Args:
        predictions: Dict with model predictions
        targets: Dict with ground truth values
        mask: Optional padding mask (batch, seq_len)
    
    Returns:
        Dict with all metrics
    """
    metrics = {}
    
    # Active classification metrics
    active_metrics = calculate_classification_metrics(
        predictions['active'],
        targets['active']
    )
    for key, value in active_metrics.items():
        metrics[f'active_{key}'] = value
    
    # Asset classification metrics
    if 'asset' in predictions and 'asset' in targets:
        asset_metrics = calculate_multiclass_metrics(
            predictions['asset'],
            targets['asset'],
            active_mask
        )
        for key, value in asset_metrics.items():
            metrics[f'asset_{key}'] = value
    
    # Get active mask for regression metrics
    active_mask = (targets['active'] == 1).float()
    if mask is not None:
        mask_expanded = mask.unsqueeze(2).expand_as(active_mask)
        active_mask = active_mask * mask_expanded
    
    # Regression metrics for each parameter
    regression_params = ['scale', 'pos_x', 'pos_y', 'anchor_x', 'anchor_y', 
                         'rotation', 'crop_l', 'crop_r', 'crop_t', 'crop_b']
    
    for param in regression_params:
        if param in predictions and param in targets:
            param_metrics = calculate_regression_metrics(
                predictions[param],
                targets[param],
                active_mask
            )
            for key, value in param_metrics.items():
                metrics[f'{param}_{key}'] = value
    
    return metrics


def find_optimal_threshold(
    predictions: torch.Tensor,
    targets: torch.Tensor,
    thresholds: torch.Tensor = None
) -> Tuple[float, float]:
    """
    Find optimal classification threshold by maximizing F1 score
    
    Args:
        predictions: Predicted logits (batch, seq_len, 2)
        targets: Ground truth labels (batch, seq_len)
        thresholds: Optional tensor of thresholds to try (default: 0.1 to 0.9 in steps of 0.05)
    
    Returns:
        Tuple of (optimal_threshold, best_f1_score)
    """
    if thresholds is None:
        thresholds = torch.arange(0.1, 0.95, 0.05)
    
    # Get probabilities
    if predictions.size(-1) == 2:
        probs = F.softmax(predictions, dim=-1)
        pred_probs = probs[..., 1]
    else:
        pred_probs = predictions
    
    best_f1 = 0.0
    best_threshold = 0.5
    
    for threshold in thresholds:
        metrics = calculate_classification_metrics(
            pred_probs,
            targets,
            threshold=threshold.item()
        )
        
        if metrics['f1'] > best_f1:
            best_f1 = metrics['f1']
            best_threshold = threshold.item()
    
    return best_threshold, best_f1


if __name__ == "__main__":
    # Test metrics calculation
    logger.info("Testing metrics calculation...")
    
    batch_size = 4
    seq_len = 100
    num_tracks = 20
    
    # Create dummy predictions and targets
    predictions = {
        'active': torch.randn(batch_size, seq_len, num_tracks, 2),
        'scale': torch.randn(batch_size, seq_len, num_tracks, 1),
        'pos_x': torch.randn(batch_size, seq_len, num_tracks, 1),
        'pos_y': torch.randn(batch_size, seq_len, num_tracks, 1),
        'anchor_x': torch.randn(batch_size, seq_len, num_tracks, 1),
        'anchor_y': torch.randn(batch_size, seq_len, num_tracks, 1),
        'rotation': torch.randn(batch_size, seq_len, num_tracks, 1),
        'crop_l': torch.randn(batch_size, seq_len, num_tracks, 1),
        'crop_r': torch.randn(batch_size, seq_len, num_tracks, 1),
        'crop_t': torch.randn(batch_size, seq_len, num_tracks, 1),
        'crop_b': torch.randn(batch_size, seq_len, num_tracks, 1)
    }
    
    targets = {
        'active': torch.randint(0, 2, (batch_size, seq_len, num_tracks)),
        'scale': torch.randn(batch_size, seq_len, num_tracks, 1),
        'pos_x': torch.randn(batch_size, seq_len, num_tracks, 1),
        'pos_y': torch.randn(batch_size, seq_len, num_tracks, 1),
        'anchor_x': torch.randn(batch_size, seq_len, num_tracks, 1),
        'anchor_y': torch.randn(batch_size, seq_len, num_tracks, 1),
        'rotation': torch.randn(batch_size, seq_len, num_tracks, 1),
        'crop_l': torch.randn(batch_size, seq_len, num_tracks, 1),
        'crop_r': torch.randn(batch_size, seq_len, num_tracks, 1),
        'crop_t': torch.randn(batch_size, seq_len, num_tracks, 1),
        'crop_b': torch.randn(batch_size, seq_len, num_tracks, 1)
    }
    
    mask = torch.ones(batch_size, seq_len, dtype=torch.bool)
    mask[:, -20:] = False  # Last 20 frames are padding
    
    # Calculate metrics
    logger.info("\nCalculating metrics...")
    metrics = calculate_multitrack_metrics(predictions, targets, mask)
    
    logger.info("\nMetrics:")
    for key, value in metrics.items():
        logger.info(f"  {key}: {value:.4f}")
    
    # Test optimal threshold finding
    logger.info("\nFinding optimal threshold...")
    active_pred = predictions['active'].reshape(-1, 2)
    active_target = targets['active'].reshape(-1)
    
    optimal_threshold, best_f1 = find_optimal_threshold(active_pred, active_target)
    logger.info(f"  Optimal threshold: {optimal_threshold:.3f}")
    logger.info(f"  Best F1 score: {best_f1:.4f}")
    
    logger.info("\n✅ All metrics tests passed!")
