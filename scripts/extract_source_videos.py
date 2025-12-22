"""
XMLファイルから元動画（編集前）のパスを抽出し、特徴量を抽出するスクリプト
"""
import os
import sys
import xml.etree.ElementTree as ET
from pathlib import Path
from urllib.parse import unquote
from collections import defaultdict

# プロジェクトルートをPythonパスに追加
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

def extract_source_videos_from_xml(xml_path):
    """
    XMLファイルから元動画ファイルのパスを抽出
    
    Returns:
        list: [(video_path, clip_name), ...] のリスト
    """
    try:
        tree = ET.parse(xml_path)
        root = tree.getroot()
        
        source_videos = []
        seen_paths = set()
        
        # すべてのpathurlを探す
        for pathurl in root.iter('pathurl'):
            url = pathurl.text
            if url and url.startswith('file://localhost/'):
                path = url.replace('file://localhost/', '')
                path = unquote(path)
                path = path.replace('/', '\\')
                
                # 動画ファイルのみ（mp4, mov, avi など）
                if path.lower().endswith(('.mp4', '.mov', '.avi', '.mkv', '.m4v')):
                    if path not in seen_paths and os.path.exists(path):
                        seen_paths.add(path)
                        clip_name = Path(path).stem
                        source_videos.append((path, clip_name))
        
        return source_videos
    except Exception as e:
        print(f"  ⚠️  XMLパースエラー: {e}")
        return []


def main():
    print("="*70)
    print("元動画（編集前）の特徴量抽出")
    print("="*70)
    print()
    
    # ディレクトリ設定
    xml_dir = Path("data/raw/editxml")
    features_output = Path("data/processed/source_features")
    
    # 出力ディレクトリを作成
    features_output.mkdir(parents=True, exist_ok=True)
    
    # XMLファイルを検索
    xml_files = list(xml_dir.glob("*.xml"))
    if not xml_files:
        print(f"❌ エラー: XMLファイルが見つかりません: {xml_dir}")
        return False
    
    print(f"✓ {len(xml_files)}個のXMLファイルを発見")
    print()
    
    # ステップ1: XMLから元動画を抽出
    print("[ステップ 1/2] XMLファイルから元動画パスを抽出")
    print("-"*70)
    
    # XMLファイルごとに元動画を記録
    xml_to_videos = {}  # {xml_id: [(video_path, clip_name), ...]}
    all_source_videos = {}  # {video_path: clip_name}
    
    for xml_file in xml_files:
        xml_id = xml_file.stem
        print(f"処理中: {xml_file.name}")
        
        source_videos = extract_source_videos_from_xml(xml_file)
        
        if source_videos:
            xml_to_videos[xml_id] = source_videos
            print(f"  ✓ {len(source_videos)}個の元動画を発見:")
            for video_path, clip_name in source_videos:
                print(f"    - {clip_name}")
                all_source_videos[video_path] = clip_name
        else:
            print(f"  ⚠️  元動画が見つかりません")
    
    print()
    print(f"✓ 合計 {len(all_source_videos)}個のユニークな元動画を発見")
    print()
    
    # ステップ2: 元動画から特徴量を抽出
    print("[ステップ 2/2] 元動画から特徴量を抽出")
    print("-"*70)
    print("  音声特徴量: 215次元（基本4 + 話者192 + 感情16 + テキスト3）")
    print("  視覚特徴量: 522次元（シーン4 + 顔6 + CLIP512）")
    print("  合計: 737次元")
    print()
    print("⚠️  この処理には時間がかかります（動画1本あたり数分）")
    print()
    
    from src.data_preparation.extract_video_features_parallel import extract_features_worker
    
    success_count = 0
    for idx, (video_path, clip_name) in enumerate(all_source_videos.items(), 1):
        print(f"[{idx}/{len(all_source_videos)}] {clip_name}")
        
        # 既に抽出済みかチェック
        output_file = features_output / f"{clip_name}_features.csv"
        if output_file.exists():
            print(f"  ✓ 既に抽出済み（スキップ）")
            success_count += 1
            continue
        
        try:
            result = extract_features_worker(video_path, str(features_output))
            if result['status'] == 'Success':
                print(f"  ✓ 特徴量抽出完了 ({result['timesteps']}タイムステップ, {result['features']}次元)")
                success_count += 1
            else:
                print(f"  ⚠️  特徴量抽出失敗: {result.get('message', 'Unknown error')}")
        except Exception as e:
            print(f"  ⚠️  エラー: {e}")
    
    print()
    print(f"✓ {success_count}/{len(all_source_videos)}個の元動画の特徴量抽出完了")
    print()
    
    # ステップ3: マッピングファイルを作成
    print("[ステップ 3/3] マッピングファイルを作成")
    print("-"*70)
    
    import pandas as pd
    
    # XMLファイルごとのマッピングを作成
    mapping_data = []
    for xml_id, source_videos in xml_to_videos.items():
        tracks_path = f"data\\processed\\tracks\\{xml_id}_tracks.csv"
        
        # 各元動画の特徴量パスを記録
        for video_path, clip_name in source_videos:
            features_path = f"data\\processed\\source_features\\{clip_name}_features.csv"
            
            # 特徴量ファイルが存在する場合のみ追加
            if Path(features_path).exists():
                mapping_data.append({
                    'xml_id': xml_id,
                    'tracks_path': tracks_path,
                    'source_video_name': clip_name,
                    'source_video_path': video_path,
                    'features_path': features_path
                })
    
    # マッピングファイルを保存
    mapping_csv = Path("data/processed/source_video_mapping.csv")
    df = pd.DataFrame(mapping_data)
    df.to_csv(mapping_csv, index=False)
    
    print(f"✓ マッピングファイルを作成: {mapping_csv}")
    print(f"  エントリ数: {len(df)}")
    print()
    
    print("="*70)
    print("完了！")
    print("="*70)
    print()
    print("生成されたファイル:")
    print(f"  - 元動画特徴量: {features_output}/*_features.csv")
    print(f"  - マッピング: {mapping_csv}")
    print()
    print("次のステップ:")
    print("  python scripts/create_training_data_from_source.py")
    print()
    
    return True


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
