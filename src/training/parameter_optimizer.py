"""
Inference Parameter Optimizer

学習後に検証データを使って推論パラメータを最適化します。
最適化されるパラメータ:
- active_threshold: Active判定の閾値
- min_clip_duration: 最小クリップ継続時間
- max_gap_duration: ギャップ結合の最大長
- target_duration: 目標合計時間
- max_duration: 最大合計時間
"""
import numpy as np
import logging
from typing import Dict, Tuple
import scipy.special

logger = logging.getLogger(__name__)


class InferenceParameterOptimizer:
    """推論パラメータの最適化"""
    
    def __init__(self, fps: float = 10.0):
        """
        初期化
        
        Args:
            fps: フレームレート
        """
        self.fps = fps
    
    def optimize_parameters(
        self,
        val_predictions: Dict[str, np.ndarray],
        val_targets: Dict[str, np.ndarray]
    ) -> Dict[str, float]:
        """
        検証データを使ってパラメータを最適化
        
        Args:
            val_predictions: 検証データの予測結果
                - 'active': (num_samples, seq_len, num_tracks, 2)
            val_targets: 検証データの正解ラベル
                - 'active': (num_samples, seq_len, num_tracks)
        
        Returns:
            最適化されたパラメータの辞書
        """
        logger.info("\n" + "="*80)
        logger.info("Optimizing inference parameters on validation data")
        logger.info("="*80)
        
        # Active閾値を最適化
        optimal_threshold = self._optimize_active_threshold(
            val_predictions['active'],
            val_targets['active']
        )
        
        # クリップ統計を分析
        clip_stats = self._analyze_clip_statistics(val_targets['active'])
        
        # 最適なパラメータを決定
        optimal_params = {
            'active_threshold': optimal_threshold,
            'min_clip_duration': clip_stats['min_duration'],
            'max_gap_duration': clip_stats['max_gap'],
            'target_duration': clip_stats['target_duration'],
            'max_duration': clip_stats['max_duration']
        }
        
        logger.info("\nOptimized parameters:")
        for key, value in optimal_params.items():
            logger.info(f"  {key}: {value:.4f}")
        logger.info("="*80 + "\n")
        
        return optimal_params
    
    def _optimize_active_threshold(
        self,
        predictions: np.ndarray,
        targets: np.ndarray
    ) -> float:
        """
        Active閾値を最適化（F1スコアを最大化）
        
        Args:
            predictions: 予測結果 (num_samples, seq_len, num_tracks, 2)
            targets: 正解ラベル (num_samples, seq_len, num_tracks)
        
        Returns:
            最適な閾値
        """
        logger.info("\nOptimizing active threshold...")
        
        # Logitsを確率に変換
        probs = scipy.special.softmax(predictions, axis=-1)[:, :, :, 1]  # Active class
        
        # フラット化
        probs_flat = probs.reshape(-1)
        targets_flat = targets.reshape(-1)
        
        # 様々な閾値でF1スコアを計算
        thresholds = np.arange(0.1, 0.9, 0.05)
        best_threshold = 0.5
        best_f1 = 0.0
        
        for threshold in thresholds:
            pred_binary = (probs_flat > threshold).astype(int)
            
            # F1スコアを計算
            tp = np.sum((pred_binary == 1) & (targets_flat == 1))
            fp = np.sum((pred_binary == 1) & (targets_flat == 0))
            fn = np.sum((pred_binary == 0) & (targets_flat == 1))
            
            precision = tp / (tp + fp + 1e-8)
            recall = tp / (tp + fn + 1e-8)
            f1 = 2 * precision * recall / (precision + recall + 1e-8)
            
            logger.debug(f"  Threshold {threshold:.2f}: F1={f1:.4f}, Precision={precision:.4f}, Recall={recall:.4f}")
            
            if f1 > best_f1:
                best_f1 = f1
                best_threshold = threshold
        
        logger.info(f"  Best threshold: {best_threshold:.4f} (F1={best_f1:.4f})")
        
        return float(best_threshold)
    
    def _analyze_clip_statistics(
        self,
        targets: np.ndarray
    ) -> Dict[str, float]:
        """
        正解データからクリップ統計を分析
        
        Args:
            targets: 正解ラベル (num_samples, seq_len, num_tracks)
        
        Returns:
            統計情報の辞書
        """
        logger.info("\nAnalyzing clip statistics from ground truth...")
        
        num_samples, seq_len, num_tracks = targets.shape
        
        all_clip_durations = []
        all_gap_durations = []
        all_sequence_durations = []
        
        for sample_idx in range(num_samples):
            for track_idx in range(num_tracks):
                track_data = targets[sample_idx, :, track_idx]
                
                # アクティブな区間を検出
                active_mask = track_data == 1
                if not active_mask.any():
                    continue
                
                # 連続する区間をグループ化
                changes = np.diff(active_mask.astype(int), prepend=0, append=0)
                starts = np.where(changes == 1)[0]
                ends = np.where(changes == -1)[0]
                
                # クリップの継続時間を記録
                for start, end in zip(starts, ends):
                    duration_frames = end - start
                    duration_seconds = duration_frames / self.fps
                    all_clip_durations.append(duration_seconds)
                
                # ギャップの長さを記録
                for i in range(len(ends) - 1):
                    gap_frames = starts[i + 1] - ends[i]
                    gap_seconds = gap_frames / self.fps
                    all_gap_durations.append(gap_seconds)
            
            # シーケンス全体のアクティブ時間を計算
            sequence_active = targets[sample_idx].any(axis=1)  # (seq_len,)
            total_active_frames = sequence_active.sum()
            total_active_seconds = total_active_frames / self.fps
            all_sequence_durations.append(total_active_seconds)
        
        # 統計を計算
        if all_clip_durations:
            clip_durations = np.array(all_clip_durations)
            
            # 最小クリップ継続時間: 25パーセンタイル（短すぎるクリップを除外）
            min_duration = float(np.percentile(clip_durations, 25))
            min_duration = max(2.0, min(min_duration, 5.0))  # 2秒〜5秒の範囲に制限
            
            logger.info(f"  Clip duration stats:")
            logger.info(f"    Mean: {clip_durations.mean():.2f}s")
            logger.info(f"    Median: {np.median(clip_durations):.2f}s")
            logger.info(f"    25th percentile: {np.percentile(clip_durations, 25):.2f}s")
            logger.info(f"    75th percentile: {np.percentile(clip_durations, 75):.2f}s")
            logger.info(f"  → Recommended min_clip_duration: {min_duration:.2f}s")
        else:
            min_duration = 3.0
            logger.warning("  No clips found in validation data, using default min_duration=3.0s")
        
        if all_gap_durations:
            gap_durations = np.array(all_gap_durations)
            
            # 最大ギャップ: 75パーセンタイル（短いギャップを結合）
            max_gap = float(np.percentile(gap_durations, 75))
            max_gap = max(1.0, min(max_gap, 3.0))  # 1秒〜3秒の範囲に制限
            
            logger.info(f"  Gap duration stats:")
            logger.info(f"    Mean: {gap_durations.mean():.2f}s")
            logger.info(f"    Median: {np.median(gap_durations):.2f}s")
            logger.info(f"    75th percentile: {np.percentile(gap_durations, 75):.2f}s")
            logger.info(f"  → Recommended max_gap_duration: {max_gap:.2f}s")
        else:
            max_gap = 2.0
            logger.warning("  No gaps found in validation data, using default max_gap=2.0s")
        
        if all_sequence_durations:
            sequence_durations = np.array(all_sequence_durations)
            
            # 目標時間: 中央値
            target_duration = float(np.median(sequence_durations))
            
            # 最大時間: 90パーセンタイル
            max_duration = float(np.percentile(sequence_durations, 90))
            
            logger.info(f"  Sequence duration stats:")
            logger.info(f"    Mean: {sequence_durations.mean():.2f}s")
            logger.info(f"    Median: {np.median(sequence_durations):.2f}s")
            logger.info(f"    90th percentile: {np.percentile(sequence_durations, 90):.2f}s")
            logger.info(f"  → Recommended target_duration: {target_duration:.2f}s")
            logger.info(f"  → Recommended max_duration: {max_duration:.2f}s")
        else:
            target_duration = 90.0
            max_duration = 150.0
            logger.warning("  No sequences found in validation data, using defaults")
        
        return {
            'min_duration': min_duration,
            'max_gap': max_gap,
            'target_duration': target_duration,
            'max_duration': max_duration
        }


def optimize_and_save_parameters(
    model,
    val_loader,
    device: str,
    output_path: str,
    fps: float = 10.0
) -> Dict[str, float]:
    """
    検証データでパラメータを最適化してファイルに保存
    
    Args:
        model: 学習済みモデル
        val_loader: 検証データローダー
        device: デバイス
        output_path: 保存先パス
        fps: フレームレート
    
    Returns:
        最適化されたパラメータ
    """
    import torch
    
    logger.info("\nCollecting validation predictions for parameter optimization...")
    
    model.eval()
    
    all_predictions = []
    all_targets = []
    
    with torch.no_grad():
        for batch_idx, batch in enumerate(val_loader):
            # バッチデータを取得
            if isinstance(batch, dict):
                # Multimodal dataset
                audio = batch['audio'].to(device)
                visual = batch['visual'].to(device)
                track = batch['track'].to(device)
                targets = batch['targets']
                padding_mask = batch.get('padding_mask', None)
                modality_mask = batch.get('modality_mask', None)
                
                if padding_mask is not None:
                    padding_mask = padding_mask.to(device)
                if modality_mask is not None:
                    modality_mask = modality_mask.to(device)
                
                # 予測
                outputs = model(
                    audio=audio,
                    visual=visual,
                    track=track,
                    padding_mask=padding_mask,
                    modality_mask=modality_mask
                )
            else:
                # Track-only dataset
                inputs, targets, padding_mask = batch
                inputs = inputs.to(device)
                padding_mask = padding_mask.to(device)
                
                outputs = model(inputs, padding_mask)
            
            # Active予測と正解を保存
            all_predictions.append(outputs['active'].cpu().numpy())
            
            # targetsが辞書かタプルかを確認
            if isinstance(targets, dict):
                all_targets.append(targets['active'].cpu().numpy())
            else:
                # タプルの場合: (active, asset_id, scale, position, rotation, crop)
                all_targets.append(targets[0].cpu().numpy())
    
    # 結合
    val_predictions = {
        'active': np.concatenate(all_predictions, axis=0)  # (batch, seq, tracks, 2)
    }
    val_targets = {
        'active': np.concatenate(all_targets, axis=0)  # (batch*seq, tracks, 12) or (batch, seq, tracks)
    }
    
    logger.info(f"Predictions shape: {val_predictions['active'].shape}")
    logger.info(f"Targets shape: {val_targets['active'].shape}")
    
    # ターゲットの形状を調整
    if len(val_targets['active'].shape) == 3 and val_targets['active'].shape[-1] == 12:
        # (batch*seq, tracks, 12) -> (batch*seq, tracks) (activeのみ抽出)
        val_targets['active'] = val_targets['active'][:, :, 0]
        # 予測と同じバッチ数に調整
        batch_size = val_predictions['active'].shape[0]
        seq_len = val_predictions['active'].shape[1]
        num_tracks = val_predictions['active'].shape[2]
        expected_size = batch_size * seq_len * num_tracks
        actual_size = val_targets['active'].size
        
        if actual_size < expected_size:
            logger.warning(f"Target size mismatch: {actual_size} < {expected_size}. Truncating predictions.")
            # 予測を切り詰める
            num_samples = actual_size // (seq_len * num_tracks)
            val_predictions['active'] = val_predictions['active'][:num_samples]
            batch_size = num_samples
        
        # (batch*seq, tracks) -> (batch, seq, tracks)に再形成
        val_targets['active'] = val_targets['active'].reshape(batch_size, seq_len, num_tracks)
    
    logger.info(f"Adjusted predictions shape: {val_predictions['active'].shape}")
    logger.info(f"Adjusted targets shape: {val_targets['active'].shape}")
    
    # パラメータを最適化
    optimizer = InferenceParameterOptimizer(fps=fps)
    optimal_params = optimizer.optimize_parameters(val_predictions, val_targets)
    
    # ファイルに保存
    import yaml
    from pathlib import Path
    
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    with open(output_path, 'w', encoding='utf-8') as f:
        yaml.dump(optimal_params, f, default_flow_style=False)
    
    logger.info(f"\nOptimized parameters saved to: {output_path}")
    
    return optimal_params


if __name__ == "__main__":
    # テスト
    logging.basicConfig(level=logging.INFO)
    
    # ダミーデータでテスト
    num_samples = 5
    seq_len = 100
    num_tracks = 20
    
    # ダミーの予測と正解
    predictions = {
        'active': np.random.randn(num_samples, seq_len, num_tracks, 2)
    }
    targets = {
        'active': np.random.randint(0, 2, (num_samples, seq_len, num_tracks))
    }
    
    optimizer = InferenceParameterOptimizer(fps=10.0)
    params = optimizer.optimize_parameters(predictions, targets)
    
    print("\nOptimized parameters:")
    for key, value in params.items():
        print(f"  {key}: {value:.4f}")
