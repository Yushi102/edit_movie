"""
Class Balance Analyzer

学習データのクラス不均衡を分析し、適切なLoss重み付けを計算します。
"""
import numpy as np
import torch
import logging
from typing import Dict, Tuple
from pathlib import Path

logger = logging.getLogger(__name__)


class ClassBalanceAnalyzer:
    """クラス不均衡の分析"""
    
    def __init__(self):
        """初期化"""
        pass
    
    def analyze_dataset(
        self,
        data_path: str,
        num_tracks: int = 20
    ) -> Dict[str, any]:
        """
        データセットのクラス不均衡を分析
        
        Args:
            data_path: データファイルのパス（.npz）
            num_tracks: トラック数
        
        Returns:
            分析結果の辞書
        """
        logger.info(f"\n{'='*80}")
        logger.info(f"Analyzing class balance: {data_path}")
        logger.info(f"{'='*80}")
        
        # データをロード
        data = np.load(data_path)
        sequences = data['sequences']  # (num_samples, seq_len, 917) - integrated data
        
        num_samples, seq_len, total_features = sequences.shape
        
        logger.info(f"Dataset shape: {sequences.shape}")
        logger.info(f"  Samples: {num_samples}")
        logger.info(f"  Sequence length: {seq_len}")
        logger.info(f"  Total features: {total_features}")
        
        # Extract track features only (last 180 dimensions)
        # 917 = audio(215) + visual(522) + track(180)
        audio_dim = 215
        visual_dim = 522
        track_dim = 180
        
        track_sequences = sequences[:, :, audio_dim + visual_dim:]  # (num_samples, seq_len, 180)
        logger.info(f"  Track features extracted: {track_sequences.shape}")
        
        # シーケンスをトラックパラメータに分解
        # (num_samples, seq_len, num_tracks, 9)
        reshaped = track_sequences.reshape(num_samples, seq_len, num_tracks, 9)
        
        # Active状態を抽出（0番目のパラメータ）
        active_labels = reshaped[:, :, :, 0]  # (num_samples, seq_len, num_tracks)
        
        # Asset IDを抽出（1番目のパラメータ）
        asset_labels = reshaped[:, :, :, 1]  # (num_samples, seq_len, num_tracks)
        
        # Active状態の分析
        active_stats = self._analyze_active_balance(active_labels)
        
        # Asset IDの分析
        asset_stats = self._analyze_asset_balance(asset_labels, active_labels)
        
        # 推奨される重み付けを計算
        recommended_weights = self._calculate_recommended_weights(
            active_stats, asset_stats
        )
        
        return {
            'active_stats': active_stats,
            'asset_stats': asset_stats,
            'recommended_weights': recommended_weights
        }
    
    def _analyze_active_balance(
        self,
        active_labels: np.ndarray
    ) -> Dict[str, float]:
        """
        Active状態の不均衡を分析
        
        Args:
            active_labels: (num_samples, seq_len, num_tracks)
        
        Returns:
            統計情報
        """
        logger.info("\n--- Active State Analysis ---")
        
        # フラット化
        flat_labels = active_labels.reshape(-1)
        
        # クラスごとのカウント
        inactive_count = np.sum(flat_labels == 0)
        active_count = np.sum(flat_labels == 1)
        total_count = len(flat_labels)
        
        # 比率
        inactive_ratio = inactive_count / total_count
        active_ratio = active_count / total_count
        
        # 不均衡比率
        imbalance_ratio = inactive_count / (active_count + 1e-8)
        
        logger.info(f"  Inactive (0): {inactive_count:,} ({inactive_ratio*100:.2f}%)")
        logger.info(f"  Active (1):   {active_count:,} ({active_ratio*100:.2f}%)")
        logger.info(f"  Imbalance ratio (inactive/active): {imbalance_ratio:.2f}")
        
        if imbalance_ratio > 2.0:
            logger.warning(f"  ⚠️  Significant class imbalance detected!")
            logger.warning(f"     Inactive frames are {imbalance_ratio:.1f}x more common than active frames")
        
        return {
            'inactive_count': int(inactive_count),
            'active_count': int(active_count),
            'inactive_ratio': float(inactive_ratio),
            'active_ratio': float(active_ratio),
            'imbalance_ratio': float(imbalance_ratio)
        }
    
    def _analyze_asset_balance(
        self,
        asset_labels: np.ndarray,
        active_labels: np.ndarray
    ) -> Dict[str, any]:
        """
        Asset IDの不均衡を分析
        
        Args:
            asset_labels: (num_samples, seq_len, num_tracks)
            active_labels: (num_samples, seq_len, num_tracks)
        
        Returns:
            統計情報
        """
        logger.info("\n--- Asset ID Analysis ---")
        
        # Activeなフレームのみを対象
        active_mask = active_labels == 1
        active_assets = asset_labels[active_mask]
        
        if len(active_assets) == 0:
            logger.warning("  No active frames found!")
            return {
                'asset_counts': {},
                'asset_ratios': {},
                'num_unique_assets': 0
            }
        
        # ユニークなAsset IDとそのカウント
        unique_assets, counts = np.unique(active_assets, return_counts=True)
        
        # 統計
        asset_counts = {int(asset_id): int(count) for asset_id, count in zip(unique_assets, counts)}
        asset_ratios = {int(asset_id): float(count / len(active_assets)) for asset_id, count in zip(unique_assets, counts)}
        
        logger.info(f"  Number of unique assets: {len(unique_assets)}")
        logger.info(f"  Asset distribution:")
        
        # ソートして表示
        sorted_assets = sorted(asset_counts.items(), key=lambda x: x[1], reverse=True)
        for asset_id, count in sorted_assets[:10]:  # 上位10個
            ratio = asset_ratios[asset_id]
            logger.info(f"    Asset {asset_id}: {count:,} ({ratio*100:.2f}%)")
        
        if len(sorted_assets) > 10:
            logger.info(f"    ... and {len(sorted_assets) - 10} more")
        
        # 不均衡の検出
        max_count = max(counts)
        min_count = min(counts)
        asset_imbalance = max_count / (min_count + 1e-8)
        
        if asset_imbalance > 5.0:
            logger.warning(f"  ⚠️  Asset imbalance detected!")
            logger.warning(f"     Most common asset is {asset_imbalance:.1f}x more frequent than least common")
        
        return {
            'asset_counts': asset_counts,
            'asset_ratios': asset_ratios,
            'num_unique_assets': int(len(unique_assets)),
            'asset_imbalance': float(asset_imbalance)
        }
    
    def _calculate_recommended_weights(
        self,
        active_stats: Dict,
        asset_stats: Dict
    ) -> Dict[str, float]:
        """
        推奨されるLoss重み付けを計算
        
        Args:
            active_stats: Active状態の統計
            asset_stats: Asset IDの統計
        
        Returns:
            推奨される重み付け
        """
        logger.info("\n--- Recommended Loss Weights ---")
        
        # Active重み: 不均衡比率に基づいて調整
        # 不均衡が大きいほど、Activeクラスの重みを増やす
        imbalance_ratio = active_stats['imbalance_ratio']
        
        if imbalance_ratio > 2.0:
            # 不均衡がある場合は、Activeクラスの重みを増やす
            # ただし、極端な値にならないように制限
            active_weight = min(imbalance_ratio / 2.0, 5.0)
        else:
            active_weight = 1.0
        
        # Asset重み: Activeと同程度
        asset_weight = 1.0
        
        # 回帰パラメータの重み: Activeなフレームのみで計算されるため、
        # Activeが少ない場合は相対的に重要度を上げる
        if active_stats['active_ratio'] < 0.3:
            # Activeが30%未満の場合は回帰の重みを上げる
            regression_weight = 1.5
        else:
            regression_weight = 1.0
        
        recommended = {
            'active_weight': float(active_weight),
            'asset_weight': float(asset_weight),
            'scale_weight': float(regression_weight),
            'position_weight': float(regression_weight),
            'rotation_weight': float(regression_weight),
            'crop_weight': float(regression_weight)
        }
        
        logger.info("  Recommended weights:")
        for key, value in recommended.items():
            logger.info(f"    {key}: {value:.2f}")
        
        # 説明
        if active_weight > 1.0:
            logger.info(f"\n  💡 Active weight increased to {active_weight:.2f} due to class imbalance")
            logger.info(f"     This will help the model learn to predict active frames better")
        
        if regression_weight > 1.0:
            logger.info(f"\n  💡 Regression weights increased to {regression_weight:.2f}")
            logger.info(f"     This compensates for fewer active frames to learn from")
        
        logger.info(f"{'='*80}\n")
        
        return recommended


def analyze_and_save_weights(
    train_data_path: str,
    output_path: str,
    num_tracks: int = 20
) -> Dict[str, float]:
    """
    データセットを分析して推奨重みをファイルに保存
    
    Args:
        train_data_path: 学習データのパス
        output_path: 出力ファイルのパス
        num_tracks: トラック数
    
    Returns:
        推奨される重み付け
    """
    analyzer = ClassBalanceAnalyzer()
    results = analyzer.analyze_dataset(train_data_path, num_tracks)
    
    # YAMLファイルに保存
    import yaml
    
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    with open(output_path, 'w', encoding='utf-8') as f:
        yaml.dump(results['recommended_weights'], f, default_flow_style=False)
    
    logger.info(f"Recommended weights saved to: {output_path}")
    
    return results['recommended_weights']


if __name__ == "__main__":
    # テスト
    logging.basicConfig(level=logging.INFO)
    
    # ダミーデータでテスト
    num_samples = 10
    seq_len = 100
    num_tracks = 20
    
    # 不均衡なデータを生成
    sequences = np.zeros((num_samples, seq_len, num_tracks * 12))
    
    # Active状態を設定（20%のみActive）
    for i in range(num_samples):
        for t in range(seq_len):
            for track in range(num_tracks):
                if np.random.rand() < 0.2:  # 20%の確率でActive
                    idx = track * 12
                    sequences[i, t, idx] = 1  # Active
                    sequences[i, t, idx + 1] = np.random.randint(0, 10)  # Asset ID
    
    # 一時ファイルに保存
    temp_path = "temp_test_data.npz"
    np.savez(temp_path, sequences=sequences)
    
    # 分析
    analyzer = ClassBalanceAnalyzer()
    results = analyzer.analyze_dataset(temp_path, num_tracks=20)
    
    print("\n✅ Analysis complete!")
    print(f"Recommended active_weight: {results['recommended_weights']['active_weight']:.2f}")
    
    # クリーンアップ
    import os
    os.remove(temp_path)
