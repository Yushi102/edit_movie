"""
学習データ準備スクリプト

XMLファイルと動画ファイルから学習用データセットを生成します。

処理フロー:
1. XMLファイルからトラックデータを抽出
2. 動画ファイルから特徴量を抽出（音声215次元 + 視覚522次元）
3. トラックデータと特徴量を統合
4. 学習用データセット（train/val）を生成
"""
import os
import sys
import xml.etree.ElementTree as ET
from pathlib import Path
from urllib.parse import unquote
import subprocess
import pandas as pd

# プロジェクトルートをPythonパスに追加
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

def find_video_path_from_xml(xml_path):
    """XMLファイルから動画ファイルのパスを抽出"""
    try:
        tree = ET.parse(xml_path)
        root = tree.getroot()
        
        for pathurl in root.iter('pathurl'):
            url = pathurl.text
            if url and url.startswith('file://localhost/'):
                path = url.replace('file://localhost/', '')
                path = unquote(path)
                path = path.replace('/', '\\')
                if os.path.exists(path):
                    return path
    except Exception as e:
        print(f"  ⚠️  XMLパースエラー: {e}")
    return None


def main():
    print("="*70)
    print("動画編集AI - 学習データ準備パイプライン")
    print("="*70)
    print()
    
    # ディレクトリ設定
    xml_dir = Path("data/raw/editxml")
    tracks_output = Path("data/processed/tracks")
    features_output = Path("data/processed/input_features")
    
    # 出力ディレクトリを作成
    tracks_output.mkdir(parents=True, exist_ok=True)
    features_output.mkdir(parents=True, exist_ok=True)
    
    # XMLファイルを検索
    xml_files = list(xml_dir.glob("*.xml"))
    if not xml_files:
        print(f"❌ エラー: XMLファイルが見つかりません: {xml_dir}")
        return False
    
    print(f"✓ {len(xml_files)}個のXMLファイルを発見")
    print()
    
    # ステップ1: XMLからトラックデータを抽出
    print("[ステップ 1/4] XMLファイルからトラックデータを抽出")
    print("-"*70)
    
    video_paths = []
    for xml_file in xml_files:
        video_id = xml_file.stem
        print(f"処理中: {xml_file.name}")
        
        # 動画パスを取得
        video_path = find_video_path_from_xml(xml_file)
        if video_path:
            video_paths.append((video_id, video_path))
            print(f"  ✓ 動画発見: {Path(video_path).name}")
        else:
            print(f"  ⚠️  動画が見つかりません")
        
        # トラックデータを抽出
        try:
            cmd = [
                "python",
                "src/data_preparation/fcpxml_to_tracks.py",
                str(xml_file),
                "--output", str(tracks_output),
                "--format", "both"
            ]
            result = subprocess.run(cmd, capture_output=True, text=True)
            if result.returncode == 0:
                print(f"  ✓ トラックデータ抽出完了")
            else:
                print(f"  ⚠️  トラックデータ抽出失敗")
                print(f"     {result.stderr[:200]}")
        except Exception as e:
            print(f"  ⚠️  エラー: {e}")
    
    print()
    print(f"✓ {len(video_paths)}個の動画ファイルを発見")
    print()
    
    # ステップ2: 動画から特徴量を抽出
    print("[ステップ 2/4] 動画ファイルから特徴量を抽出")
    print("-"*70)
    print("  音声特徴量: 215次元（基本4 + 話者192 + 感情16 + テキスト3）")
    print("  視覚特徴量: 522次元（シーン4 + 顔6 + CLIP512）")
    print("  合計: 737次元")
    print()
    print("⚠️  この処理には時間がかかります（動画1本あたり数分）")
    print()
    
    from src.data_preparation.extract_video_features_parallel import extract_features_worker
    
    success_count = 0
    for video_id, video_path in video_paths:
        print(f"処理中: {Path(video_path).name}")
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
    print(f"✓ {success_count}/{len(video_paths)}個の動画の特徴量抽出完了")
    print()
    
    # ステップ3: データ統合
    print("[ステップ 3/4] トラックデータと特徴量を統合")
    print("-"*70)
    
    # トラックファイルと特徴量ファイルをマッチング
    all_data = []
    for track_file in tracks_output.glob("*_tracks.csv"):
        video_id = track_file.stem.replace("_tracks", "")
        features_file = features_output / f"{video_id}_features.csv"
        
        if features_file.exists():
            all_data.append({
                'video_id': video_id,
                'tracks_path': str(track_file),
                'features_path': str(features_file)
            })
            print(f"  ✓ {video_id}")
        else:
            print(f"  ⚠️  {video_id} - 特徴量ファイルなし")
    
    if not all_data:
        print("❌ エラー: 統合可能なデータがありません")
        return False
    
    # 統合データを保存
    master_csv = Path("data/processed/master_training_data.csv")
    master_csv.parent.mkdir(parents=True, exist_ok=True)
    df = pd.DataFrame(all_data)
    df.to_csv(master_csv, index=False)
    
    print()
    print(f"✓ {len(df)}個の動画データを統合")
    print(f"  保存先: {master_csv}")
    print()
    
    # ステップ4: 学習用データセット生成
    print("[ステップ 4/4] 学習用データセット生成")
    print("-"*70)
    
    try:
        cmd = [
            "python",
            "src/data_preparation/data_preprocessing.py",
            str(master_csv),
            "--output_dir", "preprocessed_data",
            "--val_ratio", "0.2"
        ]
        result = subprocess.run(cmd, capture_output=True, text=True)
        if result.returncode == 0:
            print("✓ 学習用データセット生成完了")
            print(result.stdout)
        else:
            print("⚠️  データセット生成に問題が発生しました")
            print(result.stderr)
    except Exception as e:
        print(f"❌ エラー: {e}")
        return False
    
    print()
    print("="*70)
    print("データ準備完了！")
    print("="*70)
    print()
    print("生成されたファイル:")
    print(f"  - トラックデータ: {tracks_output}/*_tracks.csv")
    print(f"  - 特徴量データ: {features_output}/*_features.csv")
    print(f"  - 統合データ: {master_csv}")
    print(f"  - 学習データ: preprocessed_data/train_sequences.npz")
    print(f"  - 検証データ: preprocessed_data/val_sequences.npz")
    print()
    print("次は学習を実行してください:")
    print("  python scripts/train_model.py")
    print()
    
    return True


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
