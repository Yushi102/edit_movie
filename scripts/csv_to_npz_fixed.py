"""
CSV学習データをNPZ形式に変換するスクリプト（修正版）

トラックデータのワイド形式を正しく処理します。
"""
import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.preprocessing import StandardScaler
import sys

# 設定
SEQUENCE_LENGTH = 1000  # シーケンスの長さ（タイムステップ数）
OVERLAP = 500  # オーバーラップ（次のシーケンスとの重複）

def pivot_tracks_to_wide(df_tracks):
    """
    トラックデータをワイド形式に変換
    
    入力: (time, track, features) の縦長形式
    出力: (time, track_0_feature, track_1_feature, ...) の横長形式
    
    Args:
        df_tracks: トラックデータ (縦長形式)
    
    Returns:
        df_wide: ワイド形式のDataFrame
    """
    # 必要なカラム
    feature_cols = ['active', 'asset_id', 'scale', 'pos_x', 'pos_y', 
                    'crop_l', 'crop_r', 'crop_t', 'crop_b']
    
    # 各トラックごとにピボット
    pivoted_dfs = []
    
    for track_num in sorted(df_tracks['track'].unique()):
        track_data = df_tracks[df_tracks['track'] == track_num].copy()
        
        # timeでソート
        track_data = track_data.sort_values('time')
        
        # 必要なカラムを選択してリネーム
        track_features = track_data[['time'] + feature_cols].copy()
        
        # カラム名を変更 (例: active -> track_0_active)
        rename_dict = {col: f'track_{track_num}_{col}' for col in feature_cols}
        track_features = track_features.rename(columns=rename_dict)
        
        pivoted_dfs.append(track_features)
    
    # すべてのトラックをtimeで結合
    df_wide = pivoted_dfs[0]
    for df in pivoted_dfs[1:]:
        df_wide = pd.merge(df_wide, df, on='time', how='outer')
    
    # timeでソート
    df_wide = df_wide.sort_values('time').reset_index(drop=True)
    
    # NaNを0で埋める
    df_wide = df_wide.fillna(0)
    
    return df_wide


def extract_features_and_tracks(df_features, df_tracks):
    """
    特徴量とトラック情報を抽出して統合
    
    Args:
        df_features: 特徴量データ
        df_tracks: トラックデータ（縦長形式）
    
    Returns:
        combined_data: 統合されたデータ (timesteps, features)
        timestamps: タイムスタンプ
    """
    # トラックデータをワイド形式に変換
    print(f"    トラックデータをピボット中...")
    df_tracks_wide = pivot_tracks_to_wide(df_tracks)
    print(f"      元: {len(df_tracks)}行 → ピボット後: {len(df_tracks_wide)}行")
    
    # 特徴量とトラックを時間軸で結合
    print(f"    特徴量とトラックを結合中...")
    df_merged = pd.merge(df_features, df_tracks_wide, on='time', how='inner')
    print(f"      特徴量: {len(df_features)}行, トラック: {len(df_tracks_wide)}行 → 結合後: {len(df_merged)}行")
    
    if len(df_merged) == 0:
        print(f"      ⚠️  警告: 結合後のデータが0行です")
        return None, None
    
    # 音声特徴量（216次元）
    audio_cols = [
        'audio_energy_rms', 'audio_is_speaking', 'silence_duration_ms', 'speaker_id',
        'text_is_active', 'telop_active',
        'pitch_f0', 'pitch_std', 'spectral_centroid', 'zcr'
    ]
    # speaker_emb_0～191を追加
    audio_cols += [f'speaker_emb_{i}' for i in range(192)]
    # mfcc_0～12を追加
    audio_cols += [f'mfcc_{i}' for i in range(13)]
    
    # 視覚特徴量（522次元）
    visual_cols = [
        'scene_change', 'visual_motion', 'saliency_x', 'saliency_y',
        'face_count', 'face_center_x', 'face_center_y', 
        'face_size', 'face_mouth_open', 'face_eyebrow_raise'
    ]
    # clip_0～511を追加
    visual_cols += [f'clip_{i}' for i in range(512)]
    
    # トラック特徴量（各トラック9次元 × 20トラック = 180次元）
    track_cols = []
    for i in range(20):  # 20トラック
        track_cols += [
            f'track_{i}_active',
            f'track_{i}_asset_id',
            f'track_{i}_scale',
            f'track_{i}_pos_x',
            f'track_{i}_pos_y',
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
    
    # タイムスタンプ
    timestamps = df_merged['time'].values
    
    print(f"      音声: {audio_features.shape[1]}次元, 視覚: {visual_features.shape[1]}次元, トラック: {track_features.shape[1]}次元")
    print(f"      合計: {combined_data.shape[1]}次元")
    
    return combined_data, timestamps


def create_sequences(data, sequence_length=1000, overlap=500):
    """
    データを固定長シーケンスに分割
    
    Args:
        data: (timesteps, features)の配列
        sequence_length: シーケンスの長さ
        overlap: オーバーラップ
    
    Returns:
        sequences: (num_sequences, sequence_length, features)
        masks: (num_sequences, sequence_length) - 有効なタイムステップのマスク
    """
    if len(data) < sequence_length:
        # データが短い場合はパディング
        pad_length = sequence_length - len(data)
        padded_data = np.pad(data, ((0, pad_length), (0, 0)), mode='constant', constant_values=0)
        mask = np.concatenate([np.ones(len(data)), np.zeros(pad_length)])
        return padded_data[np.newaxis, :, :], mask[np.newaxis, :]
    
    # スライディングウィンドウでシーケンスを作成
    step = sequence_length - overlap
    sequences = []
    masks = []
    
    for start in range(0, len(data) - sequence_length + 1, step):
        end = start + sequence_length
        sequences.append(data[start:end])
        masks.append(np.ones(sequence_length))
    
    # 最後の部分が残っている場合
    if len(data) % step != 0:
        start = len(data) - sequence_length
        sequences.append(data[start:])
        masks.append(np.ones(sequence_length))
    
    return np.array(sequences), np.array(masks)


def normalize_features(train_features, val_features):
    """
    特徴量を正規化（StandardScaler）
    
    Args:
        train_features: 学習データの特徴量 (N, T, F)
        val_features: 検証データの特徴量 (M, T, F)
    
    Returns:
        normalized_train: 正規化された学習データ
        normalized_val: 正規化された検証データ
        scaler: 学習済みScaler
    """
    # 形状を保存
    train_shape = train_features.shape
    val_shape = val_features.shape
    
    # (N, T, F) -> (N*T, F)に変形
    train_flat = train_features.reshape(-1, train_features.shape[-1])
    val_flat = val_features.reshape(-1, val_features.shape[-1])
    
    # 正規化
    scaler = StandardScaler()
    train_normalized = scaler.fit_transform(train_flat)
    val_normalized = scaler.transform(val_flat)
    
    # 元の形状に戻す
    train_normalized = train_normalized.reshape(train_shape)
    val_normalized = val_normalized.reshape(val_shape)
    
    return train_normalized, val_normalized, scaler


def process_csv_to_npz(master_csv, output_dir):
    """
    マスターCSVからNPZ形式に変換
    
    Args:
        master_csv: マスターデータのCSVパス
        output_dir: 出力ディレクトリ
    """
    print("="*70)
    print("CSV → NPZ 変換（修正版）")
    print("="*70)
    print()
    
    # 出力ディレクトリを作成
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # マスターデータを読み込み
    print(f"マスターデータ読み込み: {master_csv}")
    df_master = pd.read_csv(master_csv)
    print(f"  動画数: {len(df_master)}")
    
    # train/val分割（video_idベース）
    from sklearn.model_selection import train_test_split
    train_ids, val_ids = train_test_split(
        df_master['video_id'].values,
        test_size=0.2,
        random_state=42
    )
    
    df_train = df_master[df_master['video_id'].isin(train_ids)]
    df_val = df_master[df_master['video_id'].isin(val_ids)]
    
    print(f"  学習データ: {len(df_train)}動画")
    print(f"  検証データ: {len(df_val)}動画")
    print()
    
    # 動画ごとにシーケンス化
    print("動画ごとにシーケンス化中...")
    
    train_sequences = []
    train_masks = []
    train_video_ids = []
    
    val_sequences = []
    val_masks = []
    val_video_ids = []
    
    # 学習データ処理
    print("\n学習データ処理:")
    for idx, row in df_train.iterrows():
        video_id = row['video_id']
        print(f"  [{idx+1}/{len(df_train)}] {video_id}")
        
        try:
            # 特徴量とトラックを読み込み
            df_features = pd.read_csv(row['features_path'])
            df_tracks = pd.read_csv(row['tracks_path'])
            
            print(f"    特徴量: {len(df_features)}行, トラック: {len(df_tracks)}行")
            
            # 統合
            combined_data, timestamps = extract_features_and_tracks(df_features, df_tracks)
            
            if combined_data is None:
                print(f"    ⚠️  スキップ: データ統合失敗")
                continue
            
            # シーケンス化
            seqs, masks = create_sequences(combined_data, SEQUENCE_LENGTH, OVERLAP)
            print(f"    シーケンス数: {len(seqs)}")
            
            train_sequences.append(seqs)
            train_masks.append(masks)
            train_video_ids.extend([video_id] * len(seqs))
            
        except Exception as e:
            print(f"    ⚠️  エラー: {e}")
            continue
    
    # 検証データ処理
    print("\n検証データ処理:")
    for idx, row in df_val.iterrows():
        video_id = row['video_id']
        print(f"  [{idx+1}/{len(df_val)}] {video_id}")
        
        try:
            # 特徴量とトラックを読み込み
            df_features = pd.read_csv(row['features_path'])
            df_tracks = pd.read_csv(row['tracks_path'])
            
            print(f"    特徴量: {len(df_features)}行, トラック: {len(df_tracks)}行")
            
            # 統合
            combined_data, timestamps = extract_features_and_tracks(df_features, df_tracks)
            
            if combined_data is None:
                print(f"    ⚠️  スキップ: データ統合失敗")
                continue
            
            # シーケンス化
            seqs, masks = create_sequences(combined_data, SEQUENCE_LENGTH, OVERLAP)
            print(f"    シーケンス数: {len(seqs)}")
            
            val_sequences.append(seqs)
            val_masks.append(masks)
            val_video_ids.extend([video_id] * len(seqs))
            
        except Exception as e:
            print(f"    ⚠️  エラー: {e}")
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
        'train_videos': len(set(train_video_ids)),
        'val_videos': len(set(val_video_ids))
    }
    
    stats_path = output_dir / 'dataset_stats.txt'
    with open(stats_path, 'w') as f:
        f.write("Dataset Statistics\n")
        f.write("="*50 + "\n")
        for key, value in stats.items():
            f.write(f"{key}: {value}\n")
    print(f"  ✓ {stats_path}")
    
    print()
    print("="*70)
    print("変換完了！")
    print("="*70)
    print()
    print("生成されたファイル:")
    print(f"  - {train_npz}")
    print(f"  - {val_npz}")
    print(f"  - {scaler_path}")
    print(f"  - {stats_path}")
    print()
    print("次のステップ:")
    print("  python src/training/train.py --config configs/config_multimodal_experiment.yaml")
    print()
    
    return True


def main():
    """メイン処理"""
    master_csv = 'data/processed/master_training_data.csv'
    output_dir = 'preprocessed_data'
    
    # ファイルの存在確認
    if not Path(master_csv).exists():
        print(f"❌ エラー: {master_csv} が見つかりません")
        return False
    
    # 変換実行
    success = process_csv_to_npz(master_csv, output_dir)
    
    return success


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
