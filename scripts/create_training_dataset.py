"""
学習用データセット生成スクリプト

トラックデータと特徴量データを統合して、学習用のシーケンスデータを生成します。
"""
import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.model_selection import train_test_split
import sys

def load_and_align_data(video_id, tracks_path, features_path):
    """トラックデータと特徴量データを読み込んで時間軸で整列"""
    try:
        # トラックデータを読み込み
        df_tracks = pd.read_csv(tracks_path)
        
        # 特徴量データを読み込み
        df_features = pd.read_csv(features_path)
        
        # 時間軸で結合
        df_merged = pd.merge(df_features, df_tracks, on='time', how='inner')
        
        return df_merged
    except Exception as e:
        print(f"  ⚠️  {video_id}: {e}")
        return None

def main():
    print("="*70)
    print("学習用データセット生成")
    print("="*70)
    print()
    
    # マスターデータを読み込み
    master_csv = Path('data/processed/master_training_data.csv')
    if not master_csv.exists():
        print(f"❌ エラー: {master_csv} が見つかりません")
        return False
    
    df_master = pd.read_csv(master_csv)
    print(f"✓ {len(df_master)}個の動画データを読み込み")
    print()
    
    # 各動画のデータを統合
    print("データ統合中...")
    all_sequences = []
    
    for idx, row in df_master.iterrows():
        video_id = row['video_id']
        print(f"  処理中 ({idx+1}/{len(df_master)}): {video_id}")
        
        df_merged = load_and_align_data(
            video_id,
            row['tracks_path'],
            row['features_path']
        )
        
        if df_merged is not None and len(df_merged) > 0:
            df_merged['video_id'] = video_id
            all_sequences.append(df_merged)
    
    if not all_sequences:
        print("\n❌ エラー: 統合可能なデータがありません")
        return False
    
    # 全データを結合
    df_all = pd.concat(all_sequences, ignore_index=True)
    print()
    print(f"✓ 統合完了: {len(df_all)}タイムステップ")
    print(f"  特徴量次元数: {len(df_all.columns)}カラム")
    print()
    
    # 動画IDでtrain/val分割
    video_ids = df_all['video_id'].unique()
    train_ids, val_ids = train_test_split(
        video_ids,
        test_size=0.2,
        random_state=42
    )
    
    df_train = df_all[df_all['video_id'].isin(train_ids)]
    df_val = df_all[df_all['video_id'].isin(val_ids)]
    
    print(f"データ分割:")
    print(f"  学習データ: {len(train_ids)}動画, {len(df_train)}タイムステップ")
    print(f"  検証データ: {len(val_ids)}動画, {len(df_val)}タイムステップ")
    print()
    
    # 保存
    output_dir = Path('preprocessed_data')
    output_dir.mkdir(parents=True, exist_ok=True)
    
    train_csv = output_dir / 'train_data.csv'
    val_csv = output_dir / 'val_data.csv'
    
    df_train.to_csv(train_csv, index=False)
    df_val.to_csv(val_csv, index=False)
    
    print("保存完了:")
    print(f"  学習データ: {train_csv}")
    print(f"  検証データ: {val_csv}")
    print()
    
    # 統計情報
    print("="*70)
    print("データセット統計")
    print("="*70)
    
    # 音声特徴量
    audio_cols = [c for c in df_all.columns if c.startswith(('audio_', 'speaker_', 'pitch_', 'spectral_', 'zcr', 'mfcc_', 'text_', 'telop_'))]
    print(f"音声特徴量: {len(audio_cols)}次元")
    
    # 視覚特徴量
    visual_cols = [c for c in df_all.columns if c.startswith(('scene_', 'visual_', 'saliency_', 'face_', 'clip_'))]
    print(f"視覚特徴量: {len(visual_cols)}次元")
    
    # トラック特徴量
    track_cols = [c for c in df_all.columns if c.startswith('track_')]
    print(f"トラック特徴量: {len(track_cols)}次元")
    
    print()
    print("="*70)
    print("データセット生成完了！")
    print("="*70)
    print()
    
    return True

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
