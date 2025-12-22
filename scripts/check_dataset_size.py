"""
学習データのサイズを確認するスクリプト
"""
import numpy as np

# 学習データを読み込み
train_data = np.load('preprocessed_data/train_sequences.npz')
val_data = np.load('preprocessed_data/val_sequences.npz')

print("="*70)
print("学習データ統計")
print("="*70)
print()

# 学習データ
train_sequences = train_data['sequences']
train_masks = train_data['masks']

print("学習データ:")
print(f"  シーケンス数: {train_sequences.shape[0]}")
print(f"  シーケンス長: {train_sequences.shape[1]}")
print(f"  特徴量次元数: {train_sequences.shape[2]}")
print(f"  総タイムステップ: {train_sequences.shape[0] * train_sequences.shape[1]:,}")
print(f"  有効タイムステップ: {int(train_masks.sum()):,}")
print()

# 検証データ
val_sequences = val_data['sequences']
val_masks = val_data['masks']

print("検証データ:")
print(f"  シーケンス数: {val_sequences.shape[0]}")
print(f"  シーケンス長: {val_sequences.shape[1]}")
print(f"  特徴量次元数: {val_sequences.shape[2]}")
print(f"  総タイムステップ: {val_sequences.shape[0] * val_sequences.shape[1]:,}")
print(f"  有効タイムステップ: {int(val_masks.sum()):,}")
print()

# 合計
total_timesteps = int(train_masks.sum() + val_masks.sum())
total_hours = total_timesteps / 10.0 / 3600.0  # 10fps, 秒→時間

print("合計:")
print(f"  有効タイムステップ: {total_timesteps:,}")
print(f"  動画時間: {total_timesteps / 10.0:.1f}秒 ({total_hours:.2f}時間)")
print()
print("="*70)
