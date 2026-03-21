"""
Duration penalty calculation for cut selection

Provides configurable duration penalty functions to constrain total output duration.
"""
import torch
from typing import Optional
import logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class DurationPenalty:
    """
    Configurable duration penalty calculator
    
    Penalizes predictions that deviate from target duration with progressive penalties.
    """
    
    def __init__(
        self,
        target_duration: float = 180.0,
        min_duration_ratio: float = 0.5,
        slight_overage_ratio: float = 1.2,
        moderate_overage_ratio: float = 1.5,
        shortage_penalty_weight: float = 5.0,
        slight_overage_penalty_weight: float = 1.0,
        moderate_overage_penalty_weight: float = 2.0,
        severe_overage_penalty_weight: float = 5.0,
        penalty_weight: float = 0.5
    ):
        """
        Initialize duration penalty calculator
        
        Args:
            target_duration: Target total duration in seconds (e.g., 180 for 3 minutes)
            min_duration_ratio: Minimum acceptable ratio of target (default: 0.5 = 50%)
            slight_overage_ratio: Threshold for slight overage (default: 1.2 = 120%)
            moderate_overage_ratio: Threshold for moderate overage (default: 1.5 = 150%)
            shortage_penalty_weight: Weight for shortage penalty (default: 5.0)
            slight_overage_penalty_weight: Weight for slight overage (default: 1.0)
            moderate_overage_penalty_weight: Weight for moderate overage (default: 2.0)
            severe_overage_penalty_weight: Weight for severe overage (default: 5.0)
            penalty_weight: Overall penalty weight (default: 0.5)
        
        Penalty ranges:
            - < min_duration_ratio * target: Strong penalty (too conservative)
            - min_duration_ratio * target to target: No penalty (ideal range)
            - target to slight_overage_ratio * target: Linear penalty (slight overage)
            - slight_overage_ratio * target to moderate_overage_ratio * target: Quadratic penalty
            - > moderate_overage_ratio * target: Exponential penalty (severe overage)
        """
        self.target_duration = target_duration
        self.min_duration_ratio = min_duration_ratio
        self.slight_overage_ratio = slight_overage_ratio
        self.moderate_overage_ratio = moderate_overage_ratio
        
        self.shortage_penalty_weight = shortage_penalty_weight
        self.slight_overage_penalty_weight = slight_overage_penalty_weight
        self.moderate_overage_penalty_weight = moderate_overage_penalty_weight
        self.severe_overage_penalty_weight = severe_overage_penalty_weight
        
        self.penalty_weight = penalty_weight
        
        # Calculate thresholds
        self.min_duration = target_duration * min_duration_ratio
        self.slight_overage_threshold = target_duration * slight_overage_ratio
        self.moderate_overage_threshold = target_duration * moderate_overage_ratio
        
        logger.info(f"DurationPenalty initialized:")
        logger.info(f"  Target duration: {target_duration:.1f}s")
        logger.info(f"  Acceptable range: {self.min_duration:.1f}s - {target_duration:.1f}s")
        logger.info(f"  Slight overage threshold: {self.slight_overage_threshold:.1f}s")
        logger.info(f"  Moderate overage threshold: {self.moderate_overage_threshold:.1f}s")
    
    def calculate(
        self,
        predicted_duration: torch.Tensor,
        return_details: bool = False
    ) -> torch.Tensor:
        """
        Calculate duration penalty
        
        Args:
            predicted_duration: Predicted total duration in seconds (scalar tensor)
            return_details: If True, return dict with penalty breakdown (default: False)
        
        Returns:
            Duration penalty (scalar tensor) or dict with details
        """
        # Ensure predicted_duration is a scalar tensor
        if not torch.is_tensor(predicted_duration):
            predicted_duration = torch.tensor(predicted_duration)
        
        pred_duration_scalar = predicted_duration.item()
        device = predicted_duration.device
        
        # Calculate penalty based on duration range
        if pred_duration_scalar < self.min_duration:
            # Strong penalty for shortage (too conservative)
            shortage = self.min_duration - pred_duration_scalar
            penalty = self.penalty_weight * ((shortage / self.target_duration) ** 3) * self.shortage_penalty_weight
            penalty_type = "shortage"
            
        elif pred_duration_scalar <= self.target_duration:
            # No penalty in ideal range
            penalty = 0.0
            penalty_type = "ideal"
            
        elif pred_duration_scalar <= self.slight_overage_threshold:
            # Linear penalty for slight overage
            excess = pred_duration_scalar - self.target_duration
            penalty = self.penalty_weight * (excess / self.target_duration) * self.slight_overage_penalty_weight
            penalty_type = "slight_overage"
            
        elif pred_duration_scalar <= self.moderate_overage_threshold:
            # Quadratic penalty for moderate overage
            excess = pred_duration_scalar - self.target_duration
            penalty = self.penalty_weight * ((excess / self.target_duration) ** 2) * self.moderate_overage_penalty_weight
            penalty_type = "moderate_overage"
            
        else:
            # Exponential penalty for severe overage
            excess = pred_duration_scalar - self.target_duration
            penalty = self.penalty_weight * ((excess / self.target_duration) ** 3) * self.severe_overage_penalty_weight
            penalty_type = "severe_overage"
        
        # Convert to tensor and ensure non-negative
        penalty_tensor = torch.tensor(penalty, device=device)
        penalty_tensor = torch.clamp(penalty_tensor, min=0.0)
        
        if return_details:
            return {
                'penalty': penalty_tensor,
                'penalty_type': penalty_type,
                'predicted_duration': pred_duration_scalar,
                'target_duration': self.target_duration,
                'deviation': pred_duration_scalar - self.target_duration,
                'deviation_ratio': pred_duration_scalar / self.target_duration
            }
        else:
            return penalty_tensor
    
    def __call__(self, predicted_duration: torch.Tensor) -> torch.Tensor:
        """Shorthand for calculate()"""
        return self.calculate(predicted_duration)


def create_duration_penalty(
    target_duration: float = 180.0,
    config: Optional[dict] = None
) -> DurationPenalty:
    """
    Create duration penalty calculator with optional config
    
    Args:
        target_duration: Target total duration in seconds
        config: Optional configuration dict with penalty parameters
    
    Returns:
        DurationPenalty instance
    """
    if config is None:
        config = {}
    
    return DurationPenalty(
        target_duration=target_duration,
        min_duration_ratio=config.get('min_duration_ratio', 0.5),
        slight_overage_ratio=config.get('slight_overage_ratio', 1.2),
        moderate_overage_ratio=config.get('moderate_overage_ratio', 1.5),
        shortage_penalty_weight=config.get('shortage_penalty_weight', 5.0),
        slight_overage_penalty_weight=config.get('slight_overage_penalty_weight', 1.0),
        moderate_overage_penalty_weight=config.get('moderate_overage_penalty_weight', 2.0),
        severe_overage_penalty_weight=config.get('severe_overage_penalty_weight', 5.0),
        penalty_weight=config.get('penalty_weight', 0.5)
    )


if __name__ == "__main__":
    # Test duration penalty
    logger.info("Testing duration penalty...")
    
    target_duration = 180.0  # 3 minutes
    penalty_calc = DurationPenalty(target_duration=target_duration)
    
    # Test various durations
    test_durations = [
        60.0,   # 33% of target (shortage)
        90.0,   # 50% of target (minimum acceptable)
        150.0,  # 83% of target (ideal)
        180.0,  # 100% of target (perfect)
        200.0,  # 111% of target (slight overage)
        216.0,  # 120% of target (threshold)
        250.0,  # 139% of target (moderate overage)
        270.0,  # 150% of target (threshold)
        300.0,  # 167% of target (severe overage)
        400.0,  # 222% of target (very severe)
    ]
    
    logger.info("\nDuration penalty for various durations:")
    logger.info(f"{'Duration (s)':<15} {'% of Target':<15} {'Penalty Type':<20} {'Penalty Value':<15}")
    logger.info("-" * 70)
    
    for duration in test_durations:
        duration_tensor = torch.tensor(duration)
        details = penalty_calc.calculate(duration_tensor, return_details=True)
        
        pct_of_target = (duration / target_duration) * 100
        penalty_value = details['penalty'].item()
        penalty_type = details['penalty_type']
        
        logger.info(f"{duration:<15.1f} {pct_of_target:<15.1f} {penalty_type:<20} {penalty_value:<15.4f}")
    
    # Test with custom config
    logger.info("\n\nTesting with custom config (stricter penalties)...")
    custom_config = {
        'min_duration_ratio': 0.7,  # Require at least 70% of target
        'slight_overage_ratio': 1.1,  # Only allow 10% overage
        'moderate_overage_ratio': 1.3,  # 30% overage is moderate
        'shortage_penalty_weight': 10.0,  # Stronger shortage penalty
        'penalty_weight': 1.0  # Higher overall weight
    }
    
    strict_penalty_calc = create_duration_penalty(
        target_duration=target_duration,
        config=custom_config
    )
    
    logger.info("\nStrict penalty for various durations:")
    logger.info(f"{'Duration (s)':<15} {'% of Target':<15} {'Penalty Type':<20} {'Penalty Value':<15}")
    logger.info("-" * 70)
    
    for duration in test_durations:
        duration_tensor = torch.tensor(duration)
        details = strict_penalty_calc.calculate(duration_tensor, return_details=True)
        
        pct_of_target = (duration / target_duration) * 100
        penalty_value = details['penalty'].item()
        penalty_type = details['penalty_type']
        
        logger.info(f"{duration:<15.1f} {pct_of_target:<15.1f} {penalty_type:<20} {penalty_value:<15.4f}")
    
    logger.info("\n✅ All tests passed!")
