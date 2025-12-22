"""
データの問題を調査するスクリプト
"""
import pandas as pd
from pathlib import Path

# サンプル動画で確認
video_id = "bandicam 2025-06-03 21-46-49-820"

features_file = Path(f'data/processed/input_features/{video_id}_features.csv')
tracks_file = Path(f'data/processed/tracks/{video_id}_tracks.csv')

print("="*70)
print("データ問題の調査")
print("="*70)
print()

# 特徴量ファイル
print(f"特徴量ファイル: {features_file.name}")
df_feat = pd.read_csv(features_file)
print(f"  タイムステップ数: {len(df_feat)}")
print(f"  時間範囲: {df_feat['time'].min():.1f}s ~ {df_feat['time'].max():.1f}s")
print(f"  時間間隔: {df_feat['time'].iloc[1] - df_feat['time'].iloc[0]:.2f}s")

# トラックファイル
print(f"\nトラックファイル: {tracks_file.name}")
df_track = pd.read_csv(tracks_file)
print(f"  タイムステップ数: {len(df_track)}")
print(f"  時間範囲: {df_track['time'].min():.1f}s ~ {df_track['time'].max():.1f}s")
print(f"  時間間隔: {df_track['time'].iloc[1] - df_track['time'].iloc[0]:.2f}s")

# 時間軸の比較
print(f"\n時間軸の比較:")
print(f"  特徴量の最初の5タイムスタンプ: {df_feat['time'].head().tolist()}")
print(f"  トラックの最初の5タイムスタンプ: {df_track['time'].head().tolist()}")

# Inner joinで統合
print(f"\n統合（inner join on 'time'）:")
df_merged = pd.merge(df_feat, df_track, on='time', how='inner')
print(f"  タイムステップ数: {len(df_merged)}")
if len(df_merged) > 0:
    print(f"  時間範囲: {df_merged['time'].min():.1f}s ~ {df_merged['time'].max():.1f}s")

# 問題の診断
print(f"\n問題の診断:")
if len(df_merged) < len(df_feat) * 0.5:
    print(f"  ⚠️  警告: 統合後のデータが大幅に減少しています")
    print(f"     特徴量: {len(df_feat)}タイムステップ")
    print(f"     トラック: {len(df_track)}タイムステップ")
    print(f"     統合後: {len(df_merged)}タイムステップ ({len(df_merged)/len(df_feat)*100:.1f}%)")
    print()
    print(f"  原因: 時間軸の精度が異なる可能性があります")
    print(f"     特徴量の時間精度: {df_feat['time'].iloc[0]}")
    print(f"     トラックの時間精度: {df_track['time'].iloc[0]}")
    
    # 時間の差分を確認
    feat_times = set(df_feat['time'].round(2))
    track_times = set(df_track['time'].round(2))
    common_times = feat_times & track_times
    print(f"\n  時間を0.01秒単位で丸めた場合:")
    print(f"     共通のタイムスタンプ数: {len(common_times)}")

print()
print("="*70)
