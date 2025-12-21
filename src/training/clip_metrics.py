"""
Clip-level Metrics for Evaluation

クリップレベルの評価指標を計算します。
これらの指標は、推論パラメータの最適化と学習の評価に使用されます。
"""
import numpy as np
import torch
import logging
from typing import Dict, Tuple
import scipy.special

logger = logging.getLogger(__name__)


class ClipMetricsCalculator:
    """クリップレベルの評価指標を計算"""
    
    def __init__(self, fps: float = 10.0):
        """
        初期化
        
        Args:
            fps: フレームレート
        """
        self.fps = fps
    
    def calculate_metrics(
        self,
        predictions: Dict[str, torch.Tensor],
        targets: Dict[str, torch.Tensor],
        active_threshold: float = 0.5
    ) -> Dict[str, float]:
        """
        クリップレベルの評価指標を計算
        
        Args:
            predictions: モデルの予測
                - 'active': (batch, seq_len, num_tracks, 2) - logits
            targets: 正解ラベル
                - 'active': (batch, seq_len, num_tracks) - binary labels
            active_threshold: Active判定の閾値
        
        Returns:
            評価指標の辞書
        """
        # Tensorをnumpy配列に変換
        if isinstance(predictions['active'], torch.Tensor):
            pred_logits = predictions['active'].detach().cpu().numpy()
        else:
            pred_logits = predictions['active']
        
        if isinstance(targets['active'], torch.Tensor):
            target_labels = targets['active'].detach().cpu().numpy()
        else:
            target_labels = targets['active']
        
        # Logitsを確率に変換
        pred_probs = scipy.special.softmax(pred_logits, axis=-1)[:, :, :, 1]  # Active class
        
        # 閾値を適用してバイナリ予測に変換
        pred_binary = (pred_probs > active_threshold).astype(int)
        
        # 基本的な分類指標
        classification_metrics = self._calculate_classification_metrics(
            pred_binary, target_labels
        )
        
        # クリップレベルの指標
        clip_metrics = self._calculate_clip_level_metrics(
            pred_binary, target_labels
        )
        
        # 統合
        metrics = {**classification_metrics, **clip_metrics}
        
        return metrics
    
    def _calculate_classification_metrics(
        self,
        predictions: np.ndarray,
        targets: np.ndarray
    ) -> Dict[str, float]:
        """
        基本的な分類指標を計算
        
        Args:
            predictions: 予測 (batch, seq_len, num_tracks)
            targets: 正解 (batch, seq_len, num_tracks)
        
        Returns:
            指標の辞書
        """
        # フラット化
        pred_flat = predictions.reshape(-1)
        target_flat = targets.reshape(-1)
        
        # 混同行列の要素を計算
        tp = np.sum((pred_flat == 1) & (target_flat == 1))
        tn = np.sum((pred_flat == 0) & (target_flat == 0))
        fp = np.sum((pred_flat == 1) & (target_flat == 0))
        fn = np.sum((pred_flat == 0) & (target_flat == 1))
        
        # 指標を計算
        accuracy = (tp + tn) / (tp + tn + fp + fn + 1e-8)
        precision = tp / (tp + fp + 1e-8)
        recall = tp / (tp + fn + 1e-8)
        f1 = 2 * precision * recall / (precision + recall + 1e-8)
        
        return {
            'clip_accuracy': float(accuracy),
            'clip_precision': float(precision),
            'clip_recall': float(recall),
            'clip_f1': float(f1)
        }
    
    def _calculate_clip_level_metrics(
        self,
        predictions: np.ndarray,
        targets: np.ndarray
    ) -> Dict[str, float]:
        """
        クリップレベルの詳細な指標を計算
        
        Args:
            predictions: 予測 (batch, seq_len, num_tracks)
            targets: 正解 (batch, seq_len, num_tracks)
        
        Returns:
            指標の辞書
        """
        batch_size, seq_len, num_tracks = predictions.shape
        
        # 各サンプルのクリップ統計を収集
        pred_clip_counts = []
        target_clip_counts = []
        
        pred_clip_durations = []
        target_clip_durations = []
        
        pred_total_durations = []
        target_total_durations = []
        
        clip_duration_errors = []
        clip_count_errors = []
        
        for sample_idx in range(batch_size):
            # 予測のクリップを抽出
            pred_clips = self._extract_clips(predictions[sample_idx])
            target_clips = self._extract_clips(targets[sample_idx])
            
            # クリップ数
            pred_clip_counts.append(len(pred_clips))
            target_clip_counts.append(len(target_clips))
            
            # クリップ継続時間
            if pred_clips:
                pred_durations = [clip['duration'] for clip in pred_clips]
                pred_clip_durations.extend(pred_durations)
                pred_total_durations.append(sum(pred_durations))
            else:
                pred_total_durations.append(0.0)
            
            if target_clips:
                target_durations = [clip['duration'] for clip in target_clips]
                target_clip_durations.extend(target_durations)
                target_total_durations.append(sum(target_durations))
            else:
                target_total_durations.append(0.0)
            
            # エラーを計算
            clip_count_errors.append(abs(len(pred_clips) - len(target_clips)))
            
            # 継続時間の誤差（合計時間の差）
            duration_error = abs(pred_total_durations[-1] - target_total_durations[-1])
            clip_duration_errors.append(duration_error)
        
        # 統計を計算
        metrics = {}
        
        # クリップ数の指標
        if target_clip_counts:
            metrics['avg_pred_clip_count'] = float(np.mean(pred_clip_counts))
            metrics['avg_target_clip_count'] = float(np.mean(target_clip_counts))
            metrics['clip_count_mae'] = float(np.mean(clip_count_errors))
            
            # クリップ数の相対誤差
            relative_errors = [
                abs(p - t) / (t + 1e-8) 
                for p, t in zip(pred_clip_counts, target_clip_counts)
            ]
            metrics['clip_count_relative_error'] = float(np.mean(relative_errors))
        
        # クリップ継続時間の指標
        if pred_clip_durations and target_clip_durations:
            metrics['avg_pred_clip_duration'] = float(np.mean(pred_clip_durations))
            metrics['avg_target_clip_duration'] = float(np.mean(target_clip_durations))
            
            # 継続時間の分布の類似度（KLダイバージェンス）
            # ヒストグラムを作成
            bins = np.linspace(0, 20, 21)  # 0-20秒を1秒刻み
            pred_hist, _ = np.histogram(pred_clip_durations, bins=bins, density=True)
            target_hist, _ = np.histogram(target_clip_durations, bins=bins, density=True)
            
            # 正規化（確率分布に変換）
            pred_hist = pred_hist / (pred_hist.sum() + 1e-8)
            target_hist = target_hist / (target_hist.sum() + 1e-8)
            
            # KLダイバージェンス（小さいほど良い）
            kl_div = np.sum(
                target_hist * np.log((target_hist + 1e-8) / (pred_hist + 1e-8))
            )
            metrics['clip_duration_kl_divergence'] = float(kl_div)
        
        # 合計時間の指標
        if pred_total_durations and target_total_durations:
            metrics['avg_pred_total_duration'] = float(np.mean(pred_total_durations))
            metrics['avg_target_total_duration'] = float(np.mean(target_total_durations))
            metrics['total_duration_mae'] = float(np.mean(clip_duration_errors))
            
            # 合計時間の相対誤差
            relative_errors = [
                abs(p - t) / (t + 1e-8)
                for p, t in zip(pred_total_durations, target_total_durations)
            ]
            metrics['total_duration_relative_error'] = float(np.mean(relative_errors))
        
        return metrics
    
    def _extract_clips(self, track_data: np.ndarray) -> list:
        """
        トラックデータからクリップを抽出
        
        Args:
            track_data: (seq_len, num_tracks)
        
        Returns:
            クリップのリスト
        """
        seq_len, num_tracks = track_data.shape
        
        clips = []
        
        for track_idx in range(num_tracks):
            track = track_data[:, track_idx]
            
            # アクティブな区間を検出
            active_mask = track == 1
            if not active_mask.any():
                continue
            
            # 連続する区間をグループ化
            changes = np.diff(active_mask.astype(int), prepend=0, append=0)
            starts = np.where(changes == 1)[0]
            ends = np.where(changes == -1)[0]
            
            # クリップを記録
            for start, end in zip(starts, ends):
                duration_frames = end - start
                duration_seconds = duration_frames / self.fps
                
                clips.append({
                    'track_id': track_idx,
                    'start_frame': start,
                    'end_frame': end,
                    'duration': duration_seconds
                })
        
        return clips


def calculate_batch_clip_metrics(
    predictions: Dict[str, torch.Tensor],
    targets: Dict[str, torch.Tensor],
    fps: float = 10.0,
    active_threshold: float = 0.5
) -> Dict[str, float]:
    """
    バッチのクリップレベル指標を計算（便利関数）
    
    Args:
        predictions: モデルの予測
        targets: 正解ラベル
        fps: フレームレート
        active_threshold: Active判定の閾値
    
    Returns:
        評価指標の辞書
    """
    calculator = ClipMetricsCalculator(fps=fps)
    return calculator.calculate_metrics(predictions, targets, active_threshold)


if __name__ == "__main__":
    # テスト
    logging.basicConfig(level=logging.INFO)
    
    # ダミーデータでテスト
    batch_size = 2
    seq_len = 100
    num_tracks = 20
    
    # ダミーの予測と正解
    predictions = {
        'active': torch.randn(batch_size, seq_len, num_tracks, 2)
    }
    targets = {
        'active': torch.randint(0, 2, (batch_size, seq_len, num_tracks))
    }
    
    metrics = calculate_batch_clip_metrics(predictions, targets, fps=10.0, active_threshold=0.5)
    
    print("\nClip-level metrics:")
    for key, value in metrics.items():
        print(f"  {key}: {value:.4f}")
