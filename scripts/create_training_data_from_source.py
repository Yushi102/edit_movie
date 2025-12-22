"""
元動画の特徴量とトラックデータを統合して学習データを作成
"""
import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import sys

# 設定
SEQUENCE_LENGTH = 1000
OVERLAP = 500

def pivot_tracks_to_wide(df_tracks):
    """トラックデータをワイド形式に変換"""
    feature_cols = ['active', 'asset_id', 'scale', 'pos_x', 'pos_y', 
                    'anchor_x', 'anchor_y', 'rotation',
                    'crop_l', 'crop_r', 'crop_t', 'crop_b']
    
    pivoted_dfs = []
    for track_num in sorted(df_tracks['track'].unique()):
        track_data = df_tracks[df_tracks['track'] == track_num].copy()
        track_data = track_data.sort_values('time')
        track_features = track_data[['time'] + feature_cols].copy()
        rename_dict = {col: f'track_{track_num}_{col}' for col in feature_cols}
        track_features = track_features.rename(columns=rename_dict)
        pivoted_dfs.append(track_features)
    
    df_wide = pivoted_dfs[0]
    for df in pivoted_dfs[1:]:
        df_wide = pd.merge(df_wide, df, on='time', how='outer')
    
    df_wide = df_wide.sort_values('time').reset_index(drop=True)
    df_wide = df_wide.fillna(0)
    
    return df_wide


def extract_features_and_tracks(df_features, df_tracks):
    """特徴量とトラック情報を抽出して統合"""
    print(f"    トラックデータをピボット中...")
    df_tracks_wide = pivot_tracks_to_wide(df_tracks)
    print(f"      元: {len(df_tracks)}行 → ピボット後: {len(df_tracks_wide)}行")
    
    print(f"    特徴量とトラックを結合中...")
    df_merged = pd.merge(df_features, df_tracks_wide, on='time', how='inner')
    print(f"      特徴量: {len(df_features)}行, トラック: {len(df_tracks_wide)}行 → 結合後: {len(df_merged)}行")
    
    if len(df_merged) == 0:
        print(f"      ⚠️  警告: 結合後のデータが0行です")
        return None, None
    
    # 音声特徴量（215次元）
    audio_cols = [
        'audio_energy_rms', 'audio_is_speaking', 'silence_duration_ms', 'speaker_id',
        'text_is_active', 'telop_active',
        'pitch_f0', 'pitch_std', 'spectral_centroid', 'zcr'
    ]
    audio_cols += [f'speaker_emb_{i}' for i in range(192)]
    audio_cols += [f'mfcc_{i}' for i in range(13)]
    
    # 視覚特徴量（522次元）
    visual_cols = [
        'scene_change', 'visual_motion', 'saliency_x', 'saliency_y',
        'face_count', 'face_center_x', 'face_center_y', 
        'face_size', 'face_mouth_open', 'face_eyebrow_raise'
    ]
    visual_cols += [f'clip_{i}' for i in range(512)]
    
    # トラック特徴量（240次元）
    track_cols = []
    for i in range(20):
        track_cols += [
            f'track_{i}_active',
            f'track_{i}_asset_id',
            f'track_{i}_scale',
            f'track_{i}_pos_x',
            f'track_{i}_pos_y',
            f'track_{i}_anchor_x',
            f'track_{i}_anchor_y',
            f'track_{i}_rotation',
            f'track_{i}_crop_l',
            f'track_{i}_crop_r',
            f'track_{i}_crop_t',
            f'track_{i}_crop_b'
        ]
    
    # 存在するカラムのみ抽出
    audio_features = df_merged[[c for c in audio_cols if c in df_merged.columns]].values
    visual_features = df_merged[[c for c in visual_cols if c in df_merged.columns]].values
    track_features = df_merged[[c for c in track_cols if c in df_merged.columns]].values
    
    # 全特徴量を結合
    combined_data = np.concatenate([audio_features, visual_features, track_features], axis=1)
    timestamps = df_merged['time'].values
    
    # NaNを0で埋める
    nan_count = np.isnan(combined_data).sum()
    if nan_count > 0:
        print(f"      ⚠️  NaN検出: {nan_count}個 → 0で埋めます")
        combined_data = np.nan_to_num(combined_data, nan=0.0)
    
    print(f"      音声: {audio_features.shape[1]}次元, 視覚: {visual_features.shape[1]}次元, トラック: {track_features.shape[1]}次元")
    print(f"      合計: {combined_data.shape[1]}次元")
    
    return combined_data, timestamps


def create_sequences(data, sequence_length=1000, overlap=500):
    """データを固定長シーケンスに分割"""
    if len(data) < sequence_length:
        pad_length = sequence_length - len(data)
        padded_data = np.pad(data, ((0, pad_length), (0, 0)), mode='constant', constant_values=0)
        mask = np.concatenate([np.ones(len(data)), np.zeros(pad_length)])
        return padded_data[np.newaxis, :, :], mask[np.newaxis, :]
    
    step = sequence_length - overlap
    sequences = []
    masks = []
    
    for start in range(0, len(data) - sequence_length + 1, step):
        end = start + sequence_length
        sequences.append(data[start:end])
        masks.append(np.ones(sequence_length))
    
    if len(data) % step != 0:
        start = len(data) - sequence_length
        sequences.append(data[start:])
        masks.append(np.ones(sequence_length))
    
    return np.array(sequences), np.array(masks)


def normalize_features(train_features, val_features):
    """特徴量を正規化"""
    train_shape = train_features.shape
    val_shape = val_features.shape
    
    train_flat = train_features.reshape(-1, train_features.shape[-1])
    val_flat = val_features.reshape(-1, val_features.shape[-1])
    
    scaler = StandardScaler()
    train_normalized = scaler.fit_transform(train_flat)
    val_normalized = scaler.transform(val_flat)
    
    train_normalized = train_normalized.reshape(train_shape)
    val_normalized = val_normalized.reshape(val_shape)
    
    return train_normalized, val_normalized, scaler


def main():
    print("="*70)
    print("元動画特徴量 → NPZ 学習データ変換")
    print("="*70)
    print()
    
    # マッピングファイルを読み込み
    mapping_csv = Path("data/processed/source_video_mapping.csv")
    if not mapping_csv.exists():
        print(f"❌ エラー: {mapping_csv} が見つかりません")
        print("先に scripts/extract_source_videos.py を実行してください")
        return False
    
    df_mapping = pd.read_csv(mapping_csv)
    print(f"マッピングデータ読み込み: {mapping_csv}")
    print(f"  エントリ数: {len(df_mapping)}")
    print()
    
    # XMLごとにグループ化
    xml_groups = df_mapping.groupby('xml_id')
    print(f"  XML数: {len(xml_groups)}")
    print()
    
    # train/val分割
    xml_ids = list(xml_groups.groups.keys())
    train_ids, val_ids = train_test_split(xml_ids, test_size=0.2, random_state=42)
    
    print(f"  学習データ: {len(train_ids)}個のXML")
    print(f"  検証データ: {len(val_ids)}個のXML")
    print()
    
    # 出力ディレクトリ
    output_dir = Path("preprocessed_data")
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # データ処理
    train_sequences = []
    train_masks = []
    train_video_ids = []
    
    val_sequences = []
    val_masks = []
    val_video_ids = []
    
    print("学習データ処理:")
    for idx, xml_id in enumerate(train_ids, 1):
        print(f"  [{idx}/{len(train_ids)}] {xml_id}")
        
        xml_data = df_mapping[df_mapping['xml_id'] == xml_id]
        tracks_path = xml_data.iloc[0]['tracks_path']
        
        # トラックデータを読み込み
        if not Path(tracks_path).exists():
            print(f"    ⚠️  スキップ: トラックファイルなし")
            continue
        
        df_tracks = pd.read_csv(tracks_path)
        print(f"    トラック: {len(df_tracks)}行")
        
        # 各元動画の特徴量を処理
        for _, row in xml_data.iterrows():
            features_path = row['features_path']
            source_name = row['source_video_name']
            
            if not Path(features_path).exists():
                print(f"    ⚠️  スキップ: {source_name} - 特徴量ファイルなし")
                continue
            
            print(f"    元動画: {source_name}")
            
            try:
                df_features = pd.read_csv(features_path)
                print(f"      特徴量: {len(df_features)}行")
                
                # 統合
                combined_data, timestamps = extract_features_and_tracks(df_features, df_tracks)
                
                if combined_data is None:
                    print(f"      ⚠️  スキップ: データ統合失敗")
                    continue
                
                # シーケンス化
                seqs, masks = create_sequences(combined_data, SEQUENCE_LENGTH, OVERLAP)
                print(f"      シーケンス数: {len(seqs)}")
                
                train_sequences.append(seqs)
                train_masks.append(masks)
                train_video_ids.extend([f"{xml_id}_{source_name}"] * len(seqs))
                
            except Exception as e:
                print(f"      ⚠️  エラー: {e}")
                continue
    
    print()
    print("検証データ処理:")
    for idx, xml_id in enumerate(val_ids, 1):
        print(f"  [{idx}/{len(val_ids)}] {xml_id}")
        
        xml_data = df_mapping[df_mapping['xml_id'] == xml_id]
        tracks_path = xml_data.iloc[0]['tracks_path']
        
        if not Path(tracks_path).exists():
            print(f"    ⚠️  スキップ: トラックファイルなし")
            continue
        
        df_tracks = pd.read_csv(tracks_path)
        print(f"    トラック: {len(df_tracks)}行")
        
        for _, row in xml_data.iterrows():
            features_path = row['features_path']
            source_name = row['source_video_name']
            
            if not Path(features_path).exists():
                print(f"    ⚠️  スキップ: {source_name} - 特徴量ファイルなし")
                continue
            
            print(f"    元動画: {source_name}")
            
            try:
                df_features = pd.read_csv(features_path)
                print(f"      特徴量: {len(df_features)}行")
                
                combined_data, timestamps = extract_features_and_tracks(df_features, df_tracks)
                
                if combined_data is None:
                    print(f"      ⚠️  スキップ: データ統合失敗")
                    continue
                
                seqs, masks = create_sequences(combined_data, SEQUENCE_LENGTH, OVERLAP)
                print(f"      シーケンス数: {len(seqs)}")
                
                val_sequences.append(seqs)
                val_masks.append(masks)
                val_video_ids.extend([f"{xml_id}_{source_name}"] * len(seqs))
                
            except Exception as e:
                print(f"      ⚠️  エラー: {e}")
                continue
    
    if not train_sequences or not val_sequences:
        print("\n❌ エラー: 処理可能なデータがありません")
        return False
    
    # 配列に変換
    train_sequences = np.concatenate(train_sequences, axis=0)
    train_masks = np.concatenate(train_masks, axis=0)
    
    val_sequences = np.concatenate(val_sequences, axis=0)
    val_masks = np.concatenate(val_masks, axis=0)
    
    print()
    print(f"シーケンス化完了:")
    print(f"  学習シーケンス数: {len(train_sequences)}")
    print(f"  検証シーケンス数: {len(val_sequences)}")
    print(f"  シーケンス長: {SEQUENCE_LENGTH}")
    print(f"  特徴量次元数: {train_sequences.shape[-1]}")
    
    # 正規化
    print("\n特徴量を正規化中...")
    train_sequences, val_sequences, scaler = normalize_features(train_sequences, val_sequences)
    print("  ✓ 正規化完了")
    
    # NPZファイルに保存
    train_npz = output_dir / 'train_sequences.npz'
    val_npz = output_dir / 'val_sequences.npz'
    
    print(f"\nNPZファイルに保存中...")
    np.savez_compressed(
        train_npz,
        sequences=train_sequences.astype(np.float32),
        masks=train_masks.astype(np.float32),
        video_ids=np.array(train_video_ids),
        source_video_names=np.array(train_video_ids)
    )
    print(f"  ✓ {train_npz}")
    
    np.savez_compressed(
        val_npz,
        sequences=val_sequences.astype(np.float32),
        masks=val_masks.astype(np.float32),
        video_ids=np.array(val_video_ids),
        source_video_names=np.array(val_video_ids)
    )
    print(f"  ✓ {val_npz}")
    
    # Scalerを保存
    import pickle
    scaler_path = output_dir / 'feature_scaler.pkl'
    with open(scaler_path, 'wb') as f:
        pickle.dump(scaler, f)
    print(f"  ✓ {scaler_path}")
    
    # 統計情報を保存
    stats = {
        'train_sequences': len(train_sequences),
        'val_sequences': len(val_sequences),
        'sequence_length': SEQUENCE_LENGTH,
        'feature_dim': train_sequences.shape[-1],
        'overlap': OVERLAP,
        'train_xmls': len(train_ids),
        'val_xmls': len(val_ids)
    }
    
    stats_path = output_dir / 'dataset_stats.txt'
    with open(stats_path, 'w') as f:
        f.write("Dataset Statistics (Source Videos)\n")
        f.write("="*50 + "\n")
        for key, value in stats.items():
            f.write(f"{key}: {value}\n")
    print(f"  ✓ {stats_path}")
    
    print()
    print("="*70)
    print("変換完了！")
    print("="*70)
    print()
    print("次のステップ:")
    print("  python src/training/train.py --config configs/config_multimodal_experiment.yaml")
    print()
    
    return True


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
