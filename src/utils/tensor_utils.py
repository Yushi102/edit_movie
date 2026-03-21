"""
Tensor utility functions for common operations

Provides reusable functions for tensor validation, NaN/Inf handling, and other common operations.
"""
import torch
import logging
from typing import Dict, Union, Optional

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def check_nan_inf(
    tensor: Union[torch.Tensor, Dict[str, torch.Tensor]],
    name: str = "tensor",
    raise_error: bool = False
) -> bool:
    """
    Check if tensor contains NaN or Inf values
    
    Args:
        tensor: Tensor or dict of tensors to check
        name: Name for logging purposes
        raise_error: Whether to raise an error if NaN/Inf found (default: False)
    
    Returns:
        True if NaN or Inf found, False otherwise
    
    Raises:
        ValueError: If raise_error=True and NaN/Inf found
    """
    has_issue = False
    
    if isinstance(tensor, dict):
        # Check each tensor in dict
        for key, value in tensor.items():
            if isinstance(value, torch.Tensor):
                has_nan = torch.isnan(value).any().item()
                has_inf = torch.isinf(value).any().item()
                
                if has_nan or has_inf:
                    has_issue = True
                    nan_count = torch.isnan(value).sum().item() if has_nan else 0
                    inf_count = torch.isinf(value).sum().item() if has_inf else 0
                    total = value.numel()
                    
                    msg = f"⚠️  {name}['{key}'] contains "
                    if has_nan:
                        msg += f"NaN ({nan_count}/{total}, {100*nan_count/total:.2f}%)"
                    if has_inf:
                        if has_nan:
                            msg += " and "
                        msg += f"Inf ({inf_count}/{total}, {100*inf_count/total:.2f}%)"
                    
                    if raise_error:
                        raise ValueError(msg)
                    else:
                        logger.warning(msg)
    
    elif isinstance(tensor, torch.Tensor):
        # Check single tensor
        has_nan = torch.isnan(tensor).any().item()
        has_inf = torch.isinf(tensor).any().item()
        
        if has_nan or has_inf:
            has_issue = True
            nan_count = torch.isnan(tensor).sum().item() if has_nan else 0
            inf_count = torch.isinf(tensor).sum().item() if has_inf else 0
            total = tensor.numel()
            
            msg = f"⚠️  {name} contains "
            if has_nan:
                msg += f"NaN ({nan_count}/{total}, {100*nan_count/total:.2f}%)"
            if has_inf:
                if has_nan:
                    msg += " and "
                msg += f"Inf ({inf_count}/{total}, {100*inf_count/total:.2f}%)"
            
            if raise_error:
                raise ValueError(msg)
            else:
                logger.warning(msg)
    
    return has_issue


def replace_nan_inf(
    tensor: torch.Tensor,
    nan_value: float = 0.0,
    inf_value: Optional[float] = None,
    name: str = "tensor"
) -> torch.Tensor:
    """
    Replace NaN and Inf values in tensor
    
    Args:
        tensor: Input tensor
        nan_value: Value to replace NaN with (default: 0.0)
        inf_value: Value to replace Inf with (default: None = use nan_value)
        name: Name for logging purposes
    
    Returns:
        Tensor with NaN/Inf replaced
    """
    if inf_value is None:
        inf_value = nan_value
    
    # Count issues before replacement
    nan_count = torch.isnan(tensor).sum().item()
    inf_count = torch.isinf(tensor).sum().item()
    
    if nan_count > 0 or inf_count > 0:
        logger.warning(f"Replacing {nan_count} NaN and {inf_count} Inf values in {name}")
        
        # Replace NaN
        tensor = torch.where(torch.isnan(tensor), torch.tensor(nan_value, device=tensor.device), tensor)
        
        # Replace Inf
        tensor = torch.where(torch.isinf(tensor), torch.tensor(inf_value, device=tensor.device), tensor)
    
    return tensor


def safe_divide(
    numerator: Union[torch.Tensor, float],
    denominator: Union[torch.Tensor, float],
    epsilon: float = 1e-8,
    default_value: float = 0.0
) -> Union[torch.Tensor, float]:
    """
    Safe division with epsilon to avoid division by zero
    
    Args:
        numerator: Numerator
        denominator: Denominator
        epsilon: Small value to add to denominator (default: 1e-8)
        default_value: Value to return if denominator is zero (default: 0.0)
    
    Returns:
        Result of division
    """
    if isinstance(denominator, torch.Tensor):
        # Clamp denominator to avoid division by zero
        safe_denom = torch.clamp(torch.abs(denominator), min=epsilon)
        # Preserve sign
        safe_denom = safe_denom * torch.sign(denominator + epsilon)
        return numerator / safe_denom
    else:
        # Scalar division
        if abs(denominator) < epsilon:
            return default_value
        return numerator / denominator


def calculate_modality_statistics(
    modality_mask: torch.Tensor,
    batch_size: int
) -> Dict[str, int]:
    """
    Calculate modality utilization statistics
    
    Args:
        modality_mask: Modality availability mask (batch, seq_len, num_modalities)
                      where num_modalities is typically 3 [audio, visual, track]
        batch_size: Batch size
    
    Returns:
        Dict with statistics:
            - total_samples: Total number of samples
            - audio_available: Number of samples with audio
            - visual_available: Number of samples with visual
            - both_available: Number of samples with both audio and visual
    """
    stats = {
        'total_samples': batch_size,
        'audio_available': 0,
        'visual_available': 0,
        'both_available': 0
    }
    
    if modality_mask is not None and modality_mask.size(-1) >= 2:
        # Count samples with audio/visual available
        audio_avail = modality_mask[:, :, 0].any(dim=1).sum().item()
        visual_avail = modality_mask[:, :, 1].any(dim=1).sum().item()
        both_avail = (modality_mask[:, :, 0].any(dim=1) & modality_mask[:, :, 1].any(dim=1)).sum().item()
        
        stats['audio_available'] = audio_avail
        stats['visual_available'] = visual_avail
        stats['both_available'] = both_avail
    
    return stats


def log_modality_statistics(
    stats: Dict[str, int],
    epoch: int,
    phase: str = "Train"
):
    """
    Log modality utilization statistics
    
    Args:
        stats: Statistics dict from calculate_modality_statistics()
        epoch: Current epoch number
        phase: Training phase ("Train" or "Val")
    """
    if stats['total_samples'] > 0:
        audio_pct = 100.0 * stats['audio_available'] / stats['total_samples']
        visual_pct = 100.0 * stats['visual_available'] / stats['total_samples']
        both_pct = 100.0 * stats['both_available'] / stats['total_samples']
        
        logger.info(f"\n📊 Modality Utilization (Epoch {epoch} {phase}):")
        logger.info(f"  Audio available: {audio_pct:.1f}% ({stats['audio_available']}/{stats['total_samples']})")
        logger.info(f"  Visual available: {visual_pct:.1f}% ({stats['visual_available']}/{stats['total_samples']})")
        logger.info(f"  Both available: {both_pct:.1f}% ({stats['both_available']}/{stats['total_samples']})")


if __name__ == "__main__":
    # Test tensor utilities
    logger.info("Testing tensor utilities...")
    
    # Test NaN/Inf detection
    logger.info("\n1. Testing NaN/Inf detection:")
    tensor_with_nan = torch.tensor([1.0, 2.0, float('nan'), 4.0])
    tensor_with_inf = torch.tensor([1.0, 2.0, float('inf'), 4.0])
    tensor_clean = torch.tensor([1.0, 2.0, 3.0, 4.0])
    
    assert check_nan_inf(tensor_with_nan, "tensor_with_nan") == True
    assert check_nan_inf(tensor_with_inf, "tensor_with_inf") == True
    assert check_nan_inf(tensor_clean, "tensor_clean") == False
    
    # Test dict of tensors
    tensor_dict = {
        'clean': tensor_clean,
        'nan': tensor_with_nan,
        'inf': tensor_with_inf
    }
    assert check_nan_inf(tensor_dict, "tensor_dict") == True
    
    # Test replacement
    logger.info("\n2. Testing NaN/Inf replacement:")
    replaced = replace_nan_inf(tensor_with_nan, nan_value=0.0)
    assert not torch.isnan(replaced).any()
    logger.info(f"  Original: {tensor_with_nan}")
    logger.info(f"  Replaced: {replaced}")
    
    # Test safe division
    logger.info("\n3. Testing safe division:")
    numerator = torch.tensor([1.0, 2.0, 3.0])
    denominator = torch.tensor([2.0, 0.0, 4.0])
    result = safe_divide(numerator, denominator)
    logger.info(f"  Numerator: {numerator}")
    logger.info(f"  Denominator: {denominator}")
    logger.info(f"  Result: {result}")
    assert not torch.isnan(result).any()
    assert not torch.isinf(result).any()
    
    # Test modality statistics
    logger.info("\n4. Testing modality statistics:")
    batch_size = 4
    seq_len = 10
    modality_mask = torch.zeros(batch_size, seq_len, 3)
    modality_mask[0, :, 0] = 1  # Sample 0: audio only
    modality_mask[1, :, 1] = 1  # Sample 1: visual only
    modality_mask[2, :, :2] = 1  # Sample 2: both
    modality_mask[3, :, :2] = 1  # Sample 3: both
    
    stats = calculate_modality_statistics(modality_mask, batch_size)
    logger.info(f"  Stats: {stats}")
    assert stats['total_samples'] == 4
    assert stats['audio_available'] == 3  # Samples 0, 2, 3
    assert stats['visual_available'] == 3  # Samples 1, 2, 3
    assert stats['both_available'] == 2  # Samples 2, 3
    
    log_modality_statistics(stats, epoch=1, phase="Test")
    
    logger.info("\n✅ All tensor utility tests passed!")
