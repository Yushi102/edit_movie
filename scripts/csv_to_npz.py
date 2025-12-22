"""
CSV学習データをNPZ形式に変換するスクリプト

CSVファイルから以下を生成:
1. シーケンス化されたデータ（固定長）
2. 正規化された特徴量
3. NPZ形式で保存
"""
import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.preprocessing import StandardScaler
import sys

# 設定
SEQUENCE_LENGTH = 1000  # シーケンスの長さ（タイムステップ数）
OVERLAP = 500  # オーバーラップ（次のシーケンスとの重複）

def extract_features_and_tracks(df):
    """
    DataFrameから特徴量とトラック情報を抽出
    
    Returns:
        audio_features: 音声特徴量
        visual_features: 視覚特徴量
        track_features: トラック特徴量
        timestamps: タイムスタンプ
    """
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
    
    # トラック特徴量（各トラック12次元 × 20トラック = 240次元）
    # active, asset_id, scale_x, scale_y, position_x, position_y, 
    # rotation, crop_left, crop_right, crop_top, crop_bottom, anchor
    track_cols = []
    for i in range(20):  # 20トラック
        track_cols += [
            f'track_{i}_active',
            f'track_{i}_asset_id',
            f'track_{i}_scale_x',
            f'track_{i}_scale_y',
            f'track_{i}_position_x',
            f'track_{i}_position_y',
            f'track_{i}_rotation',
            f'track_{i}_crop_left',
            f'track_{i}_crop_right',
            f'track_{i}_crop_top',
            f'track_{i}_crop_bottom',
            f'track_{i}_anchor'
        ]
    
    # 存在するカラムのみ抽出
    audio_features = df[[c for c in audio_cols if c in df.columns]].values
    visual_features = df[[c for c in visual_cols if c in df.columns]].values
    track_features = df[[c for c in track_cols if c in df.columns]].values
    
    # タイムスタンプ
    timestamps = df['time'].values if 'time' in df.columns else np.arange(len(df)) * 0.1
    
    return audio_features, visual_features, track_features, timestamps


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


def process_csv_to_npz(train_csv, val_csv, output_dir):
    """
    CSVファイルをNPZ形式に変換
    
    Args:
        train_csv: 学習データのCSVパス
        val_csv: 検証データのCSVパス
        output_dir: 出力ディレクトリ
    """
    print("="*70)
    print("CSV → NPZ 変換")
    print("="*70)
    print()
    
    # 出力ディレクトリを作成
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # 学習データを読み込み
    print(f"学習データ読み込み: {train_csv}")
    df_train = pd.read_csv(train_csv)
    print(f"  タイムステップ数: {len(df_train)}")
    print(f"  カラム数: {len(df_train.columns)}")
    
    # 検証データを読み込み
    print(f"\n検証データ読み込み: {val_csv}")
    df_val = pd.read_csv(val_csv)
    print(f"  タイムステップ数: {len(df_val)}")
    print(f"  カラム数: {len(df_val.columns)}")
    
    # 動画IDごとにグループ化
    print("\n動画ごとにシーケンス化中...")
    
    train_sequences = []
    train_masks = []
    train_video_ids = []
    
    val_sequences = []
    val_masks = []
    val_video_ids = []
    
    # 学習データ処理
    for video_id, group in df_train.groupby('video_id'):
        print(f"  処理中: {video_id} ({len(group)}タイムステップ)")
        
        # 特徴量とトラックを抽出
        audio_feat, visual_feat, track_feat, timestamps = extract_features_and_tracks(group)
        
        # 全特徴量を結合（audio + visual + track）
        combined_feat = np.concatenate([audio_feat, visual_feat, track_feat], axis=1)
        
        # シーケンス化
        seqs, masks = create_sequences(combined_feat, SEQUENCE_LENGTH, OVERLAP)
        
        train_sequences.append(seqs)
        train_masks.append(masks)
        train_video_ids.extend([video_id] * len(seqs))
    
    # 検証データ処理
    for video_id, group in df_val.groupby('video_id'):
        print(f"  処理中: {video_id} ({len(group)}タイムステップ)")
        
        # 特徴量とトラックを抽出
        audio_feat, visual_feat, track_feat, timestamps = extract_features_and_tracks(group)
        
        # 全特徴量を結合
        combined_feat = np.concatenate([audio_feat, visual_feat, track_feat], axis=1)
        
        # シーケンス化
        seqs, masks = create_sequences(combined_feat, SEQUENCE_LENGTH, OVERLAP)
        
        val_sequences.append(seqs)
        val_masks.append(masks)
        val_video_ids.extend([video_id] * len(seqs))
    
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
        source_video_names=np.array(train_video_ids)  # 互換性のため
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
    train_csv = 'preprocessed_data/train_data.csv'
    val_csv = 'preprocessed_data/val_data.csv'
    output_dir = 'preprocessed_data'
    
    # ファイルの存在確認
    if not Path(train_csv).exists():
        print(f"❌ エラー: {train_csv} が見つかりません")
        return False
    
    if not Path(val_csv).exists():
        print(f"❌ エラー: {val_csv} が見つかりません")
        return False
    
    # 変換実行
    success = process_csv_to_npz(train_csv, val_csv, output_dir)
    
    return success


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
