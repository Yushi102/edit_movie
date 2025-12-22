"""
話者埋め込みとAsset IDを対応付けるスクリプト

1. 全動画から話者埋め込みを収集してクラスタリング（動画を跨いで同じ人物を統一）
2. 各タイムステップで話者とAsset IDを対応付け
3. Asset IDを統一speaker_idに置き換え
"""
import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.cluster import AgglomerativeClustering
from collections import defaultdict, Counter
import sys

def collect_speaker_embeddings(mapping_csv):
    """全動画から話者埋め込みを収集"""
    print("話者埋め込みを収集中...")
    
    df_mapping = pd.read_csv(mapping_csv)
    
    all_embeddings = []
    embedding_info = []  # (video_id, local_speaker_id, embedding)
    
    for idx, row in df_mapping.iterrows():
        features_path = row['features_path']
        source_name = row['source_video_name']
        
        if not Path(features_path).exists():
            continue
        
        df_features = pd.read_csv(features_path)
        
        # 話者埋め込み列を抽出（speaker_emb_0 ~ speaker_emb_191）
        emb_cols = [f'speaker_emb_{i}' for i in range(192)]
        if not all(col in df_features.columns for col in emb_cols):
            print(f"  ⚠️  {source_name}: 話者埋め込みが見つかりません")
            continue
        
        # 各タイムステップの話者IDと埋め込みを取得
        for _, frame in df_features.iterrows():
            speaker_id = int(frame['speaker_id'])
            embedding = frame[emb_cols].values.astype(np.float32)
            
            # NaNチェック
            if np.isnan(embedding).any():
                continue
            
            # ゼロベクトルチェック（コサイン類似度で問題になる）
            if np.allclose(embedding, 0):
                continue
            
            all_embeddings.append(embedding)
            embedding_info.append({
                'video_id': row['xml_id'],
                'source_name': source_name,
                'local_speaker_id': speaker_id,
                'embedding': embedding
            })
    
    print(f"  収集した埋め込み数: {len(all_embeddings)}")
    return np.array(all_embeddings), embedding_info


def cluster_speakers(embeddings, embedding_info, n_clusters=None, distance_threshold=0.3, max_samples=10000):
    """話者埋め込みをクラスタリングして統一IDを割り当て"""
    print(f"\n話者をクラスタリング中（distance_threshold={distance_threshold}）...")
    
    # サンプリング（計算量削減のため）
    if len(embeddings) > max_samples:
        print(f"  サンプリング: {len(embeddings)} → {max_samples}")
        indices = np.random.choice(len(embeddings), max_samples, replace=False)
        sample_embeddings = embeddings[indices]
        sample_info = [embedding_info[i] for i in indices]
    else:
        sample_embeddings = embeddings
        sample_info = embedding_info
    
    # 階層的クラスタリング
    clustering = AgglomerativeClustering(
        n_clusters=n_clusters,
        distance_threshold=distance_threshold,
        metric='cosine',
        linkage='average'
    )
    
    sample_labels = clustering.fit_predict(sample_embeddings)
    
    # 全データに対してラベルを予測（最近傍）
    if len(embeddings) > max_samples:
        print(f"  全データにラベルを割り当て中...")
        from sklearn.metrics.pairwise import cosine_similarity
        
        # 各クラスタの中心を計算
        unique_labels = np.unique(sample_labels)
        centroids = []
        for label in unique_labels:
            mask = sample_labels == label
            centroid = sample_embeddings[mask].mean(axis=0)
            centroids.append(centroid)
        centroids = np.array(centroids)
        
        # 全埋め込みに対して最も近いクラスタを割り当て
        similarities = cosine_similarity(embeddings, centroids)
        labels = similarities.argmax(axis=1)
    else:
        labels = sample_labels
    
    n_speakers = len(np.unique(labels))
    print(f"  検出された話者数: {n_speakers}")
    
    return labels


def build_speaker_mapping(embedding_info, cluster_labels):
    """動画ごとのlocal_speaker_id → 統一speaker_idのマッピングを構築"""
    print("\n話者マッピングを構築中...")
    
    # (video_id, local_speaker_id) → [unified_speaker_ids]
    mapping_votes = defaultdict(list)
    
    for info, unified_id in zip(embedding_info, cluster_labels):
        key = (info['video_id'], info['source_name'], info['local_speaker_id'])
        mapping_votes[key].append(unified_id)
    
    # 各(video, local_id)に対して最頻値を取る
    speaker_mapping = {}
    for key, votes in mapping_votes.items():
        most_common = Counter(votes).most_common(1)[0][0]
        speaker_mapping[key] = most_common
    
    print(f"  マッピング数: {len(speaker_mapping)}")
    
    # マッピングを表示
    print("\n  動画ごとの話者マッピング:")
    current_video = None
    for (video_id, source_name, local_id), unified_id in sorted(speaker_mapping.items()):
        if video_id != current_video:
            print(f"\n  {video_id}:")
            current_video = video_id
        print(f"    {source_name}: local_speaker_{local_id} → unified_speaker_{unified_id}")
    
    return speaker_mapping


def align_speaker_to_asset(mapping_csv, speaker_mapping):
    """時間軸で話者とAsset IDを対応付け"""
    print("\n\n話者とAsset IDを対応付け中...")
    
    df_mapping = pd.read_csv(mapping_csv)
    
    # unified_speaker_id → asset_id の対応を記録
    speaker_asset_pairs = defaultdict(list)
    
    for idx, row in df_mapping.iterrows():
        xml_id = row['xml_id']
        features_path = row['features_path']
        tracks_path = row['tracks_path']
        source_name = row['source_video_name']
        
        if not Path(features_path).exists() or not Path(tracks_path).exists():
            continue
        
        print(f"  処理中: {xml_id} - {source_name}")
        
        df_features = pd.read_csv(features_path)
        df_tracks = pd.read_csv(tracks_path)
        
        # 時間軸でマージ
        df_merged = pd.merge(df_features, df_tracks, on='time', how='inner')
        
        for _, frame in df_merged.iterrows():
            local_speaker_id = int(frame['speaker_id'])
            key = (xml_id, source_name, local_speaker_id)
            
            # 統一speaker_idを取得
            if key not in speaker_mapping:
                continue
            
            unified_speaker_id = speaker_mapping[key]
            
            # このタイムステップでactiveなトラックのasset_idを取得
            if frame['active'] == 1:
                asset_id = int(frame['asset_id'])
                speaker_asset_pairs[unified_speaker_id].append(asset_id)
    
    # 各統一speaker_idに対して最頻のasset_idを決定
    speaker_to_asset = {}
    print("\n  統一話者 → Asset ID マッピング:")
    for unified_id in sorted(speaker_asset_pairs.keys()):
        asset_ids = speaker_asset_pairs[unified_id]
        if asset_ids:
            most_common_asset = Counter(asset_ids).most_common(1)[0][0]
            speaker_to_asset[unified_id] = most_common_asset
            count = len(asset_ids)
            print(f"    unified_speaker_{unified_id} → asset_id_{most_common_asset} ({count}回出現)")
    
    return speaker_to_asset


def replace_asset_ids(mapping_csv, speaker_mapping, speaker_to_asset):
    """Asset IDを統一speaker_idに置き換え"""
    print("\n\nAsset IDを置き換え中...")
    
    df_mapping = pd.read_csv(mapping_csv)
    
    for idx, row in df_mapping.iterrows():
        xml_id = row['xml_id']
        features_path = row['features_path']
        tracks_path = row['tracks_path']
        source_name = row['source_video_name']
        
        if not Path(features_path).exists() or not Path(tracks_path).exists():
            continue
        
        print(f"  処理中: {xml_id} - {source_name}")
        
        df_features = pd.read_csv(features_path)
        df_tracks = pd.read_csv(tracks_path)
        
        # 各タイムステップで話者に基づいてasset_idを更新
        updated_rows = []
        for _, track_row in df_tracks.iterrows():
            time = track_row['time']
            
            # この時間の話者を取得
            feature_row = df_features[df_features['time'] == time]
            if len(feature_row) == 0:
                updated_rows.append(track_row)
                continue
            
            local_speaker_id = int(feature_row.iloc[0]['speaker_id'])
            key = (xml_id, source_name, local_speaker_id)
            
            # 統一speaker_idを取得
            if key in speaker_mapping:
                unified_speaker_id = speaker_mapping[key]
                # asset_idを統一speaker_idに置き換え
                track_row = track_row.copy()
                track_row['asset_id'] = unified_speaker_id
            
            updated_rows.append(track_row)
        
        # 更新されたトラックデータを保存
        df_updated = pd.DataFrame(updated_rows)
        df_updated.to_csv(tracks_path, index=False)
        print(f"    更新完了: {tracks_path}")


def main():
    print("="*70)
    print("話者とAsset IDの対応付け")
    print("="*70)
    print()
    
    mapping_csv = Path("data/processed/source_video_mapping.csv")
    if not mapping_csv.exists():
        print(f"❌ エラー: {mapping_csv} が見つかりません")
        return False
    
    # 1. 話者埋め込みを収集
    embeddings, embedding_info = collect_speaker_embeddings(mapping_csv)
    
    if len(embeddings) == 0:
        print("❌ エラー: 話者埋め込みが見つかりません")
        return False
    
    # 2. クラスタリングして統一speaker_idを割り当て
    cluster_labels = cluster_speakers(embeddings, embedding_info, distance_threshold=0.3)
    
    # 3. 動画ごとのマッピングを構築
    speaker_mapping = build_speaker_mapping(embedding_info, cluster_labels)
    
    # 4. 話者とAsset IDを対応付け
    speaker_to_asset = align_speaker_to_asset(mapping_csv, speaker_mapping)
    
    # 5. Asset IDを置き換え
    replace_asset_ids(mapping_csv, speaker_mapping, speaker_to_asset)
    
    print("\n" + "="*70)
    print("完了！")
    print("="*70)
    print()
    print("次のステップ:")
    print("  python scripts/create_training_data_from_source.py")
    print()
    
    return True


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
