"""
Clip extraction utilities for cut selection

Provides functions for extracting clips from predictions and calculating durations.
"""
import torch
import torch.nn.functional as F
from typing import List, Tuple
import logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def extract_clips_from_predictions(
    predictions: torch.Tensor,
    threshold: float = 0.5,
    min_clip_duration: float = 3.0,
    fps: float = 10.0
) -> List[List[Tuple[int, int]]]:
    """
    Extract clips from predictions with minimum duration filtering
    
    Args:
        predictions: Predicted logits (batch, seq_len, 2) or probabilities
        threshold: Classification threshold (default: 0.5)
        min_clip_duration: Minimum clip duration in seconds (default: 3.0)
        fps: Frames per second (default: 10.0)
    
    Returns:
        List of clips for each batch item, where each clip is (start_frame, end_frame)
    """
    # Get probabilities if logits provided
    if predictions.size(-1) == 2:
        probs = F.softmax(predictions, dim=-1)
        pred_probs = probs[..., 1]  # Probability of active class
    else:
        pred_probs = predictions
    
    # Binary predictions
    pred_active = (pred_probs > threshold).cpu().numpy()  # (batch, seq_len)
    
    batch_size, seq_len = pred_active.shape
    min_frames = int(min_clip_duration * fps)
    
    all_clips = []
    
    for batch_idx in range(batch_size):
        clips = []
        in_clip = False
        clip_start = 0
        
        for frame_idx in range(seq_len):
            if pred_active[batch_idx, frame_idx] == 1:
                if not in_clip:
                    # Start new clip
                    clip_start = frame_idx
                    in_clip = True
            else:
                if in_clip:
                    # End current clip
                    clip_end = frame_idx
                    clip_length = clip_end - clip_start
                    
                    # Only keep clips that meet minimum duration
                    if clip_length >= min_frames:
                        clips.append((clip_start, clip_end))
                    
                    in_clip = False
        
        # Handle clip that extends to end of sequence
        if in_clip:
            clip_end = seq_len
            clip_length = clip_end - clip_start
            if clip_length >= min_frames:
                clips.append((clip_start, clip_end))
        
        all_clips.append(clips)
    
    return all_clips


def calculate_total_duration(
    predictions: torch.Tensor,
    threshold: float = 0.5,
    min_clip_duration: float = 3.0,
    fps: float = 10.0,
    return_per_batch: bool = False
) -> torch.Tensor:
    """
    Calculate total duration of all valid clips
    
    Args:
        predictions: Predicted logits (batch, seq_len, 2) or probabilities
        threshold: Classification threshold (default: 0.5)
        min_clip_duration: Minimum clip duration in seconds (default: 3.0)
        fps: Frames per second (default: 10.0)
        return_per_batch: If True, return duration for each batch item (default: False)
    
    Returns:
        Total duration in seconds (scalar tensor or batch tensor)
    """
    # Extract clips
    all_clips = extract_clips_from_predictions(
        predictions, threshold, min_clip_duration, fps
    )
    
    # Calculate durations
    durations = []
    for clips in all_clips:
        total_frames = sum(end - start for start, end in clips)
        duration = total_frames / fps
        durations.append(duration)
    
    durations_tensor = torch.tensor(durations, device=predictions.device)
    
    if return_per_batch:
        return durations_tensor
    else:
        return durations_tensor.mean()


def calculate_total_duration_fast(
    predictions: torch.Tensor,
    threshold: float = 0.5,
    fps: float = 10.0
) -> torch.Tensor:
    """
    Fast approximation of total duration (ignores min_clip_duration)
    
    This is faster but less accurate than calculate_total_duration().
    Use for training when speed is critical.
    
    Args:
        predictions: Predicted logits (batch, seq_len, 2) or probabilities
        threshold: Classification threshold (default: 0.5)
        fps: Frames per second (default: 10.0)
    
    Returns:
        Approximate total duration in seconds (scalar tensor)
    """
    # Get probabilities if logits provided
    if predictions.size(-1) == 2:
        probs = F.softmax(predictions, dim=-1)
        pred_probs = probs[..., 1]
    else:
        pred_probs = predictions
    
    # Binary predictions
    pred_active = (pred_probs > threshold).float()  # (batch, seq_len)
    
    # Count active frames per batch
    active_frames = pred_active.sum(dim=1)  # (batch,)
    
    # Convert to duration (frames / fps)
    durations = active_frames / fps  # (batch,)
    
    # Average over batch
    avg_duration = durations.mean()
    
    return avg_duration


def calculate_clip_statistics(
    predictions: torch.Tensor,
    threshold: float = 0.5,
    min_clip_duration: float = 3.0,
    fps: float = 10.0
) -> dict:
    """
    Calculate detailed statistics about extracted clips
    
    Args:
        predictions: Predicted logits (batch, seq_len, 2) or probabilities
        threshold: Classification threshold (default: 0.5)
        min_clip_duration: Minimum clip duration in seconds (default: 3.0)
        fps: Frames per second (default: 10.0)
    
    Returns:
        Dict with statistics:
            - num_clips: Total number of clips
            - total_duration: Total duration in seconds
            - avg_clip_duration: Average clip duration in seconds
            - min_clip_duration_actual: Shortest clip duration
            - max_clip_duration: Longest clip duration
            - clips_per_batch: Average number of clips per batch item
    """
    # Extract clips
    all_clips = extract_clips_from_predictions(
        predictions, threshold, min_clip_duration, fps
    )
    
    # Calculate statistics
    all_clip_durations = []
    total_clips = 0
    
    for clips in all_clips:
        total_clips += len(clips)
        for start, end in clips:
            duration = (end - start) / fps
            all_clip_durations.append(duration)
    
    if len(all_clip_durations) == 0:
        return {
            'num_clips': 0,
            'total_duration': 0.0,
            'avg_clip_duration': 0.0,
            'min_clip_duration_actual': 0.0,
            'max_clip_duration': 0.0,
            'clips_per_batch': 0.0
        }
    
    return {
        'num_clips': total_clips,
        'total_duration': sum(all_clip_durations),
        'avg_clip_duration': sum(all_clip_durations) / len(all_clip_durations),
        'min_clip_duration_actual': min(all_clip_durations),
        'max_clip_duration': max(all_clip_durations),
        'clips_per_batch': total_clips / len(all_clips)
    }


if __name__ == "__main__":
    # Test clip extraction
    logger.info("Testing clip extraction...")
    
    batch_size = 4
    seq_len = 100
    fps = 10.0
    min_clip_duration = 3.0
    
    # Create dummy predictions with some clips
    # Pattern: [0, 0, 1, 1, 1, 1, 1, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, ...]
    predictions = torch.zeros(batch_size, seq_len, 2)
    
    # Batch 0: One long clip (50 frames = 5 seconds)
    predictions[0, 10:60, 1] = 5.0  # High logit for active
    
    # Batch 1: Two medium clips (30 frames = 3 seconds each)
    predictions[1, 10:40, 1] = 5.0
    predictions[1, 50:80, 1] = 5.0
    
    # Batch 2: Many short clips (10 frames = 1 second each, should be filtered)
    for i in range(0, 100, 20):
        predictions[2, i:i+10, 1] = 5.0
    
    # Batch 3: One clip at threshold (30 frames = 3 seconds, exactly min duration)
    predictions[3, 20:50, 1] = 5.0
    
    # Extract clips
    logger.info("\nExtracting clips...")
    clips = extract_clips_from_predictions(
        predictions, threshold=0.5, min_clip_duration=min_clip_duration, fps=fps
    )
    
    logger.info("\nExtracted clips:")
    for batch_idx, batch_clips in enumerate(clips):
        logger.info(f"  Batch {batch_idx}: {len(batch_clips)} clips")
        for start, end in batch_clips:
            duration = (end - start) / fps
            logger.info(f"    Clip: frames {start}-{end} ({duration:.1f}s)")
    
    # Calculate total duration (accurate)
    logger.info("\nCalculating total duration (accurate)...")
    total_duration = calculate_total_duration(
        predictions, threshold=0.5, min_clip_duration=min_clip_duration, fps=fps
    )
    logger.info(f"  Total duration: {total_duration:.2f}s")
    
    # Calculate total duration (fast approximation)
    logger.info("\nCalculating total duration (fast approximation)...")
    total_duration_fast = calculate_total_duration_fast(
        predictions, threshold=0.5, fps=fps
    )
    logger.info(f"  Total duration (fast): {total_duration_fast:.2f}s")
    
    # Calculate statistics
    logger.info("\nCalculating clip statistics...")
    stats = calculate_clip_statistics(
        predictions, threshold=0.5, min_clip_duration=min_clip_duration, fps=fps
    )
    
    logger.info("\nClip statistics:")
    for key, value in stats.items():
        logger.info(f"  {key}: {value:.2f}")
    
    # Verify filtering
    logger.info("\n✅ Verification:")
    logger.info(f"  Batch 0: Expected 1 clip (5s), got {len(clips[0])} clips")
    logger.info(f"  Batch 1: Expected 2 clips (3s each), got {len(clips[1])} clips")
    logger.info(f"  Batch 2: Expected 0 clips (all <3s), got {len(clips[2])} clips")
    logger.info(f"  Batch 3: Expected 1 clip (3s), got {len(clips[3])} clips")
    
    assert len(clips[0]) == 1, "Batch 0 should have 1 clip"
    assert len(clips[1]) == 2, "Batch 1 should have 2 clips"
    assert len(clips[2]) == 0, "Batch 2 should have 0 clips (filtered)"
    assert len(clips[3]) == 1, "Batch 3 should have 1 clip"
    
    logger.info("\n✅ All tests passed!")
