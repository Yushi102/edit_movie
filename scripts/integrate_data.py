"""
データ統合スクリプト

トラックデータと特徴量データを統合して、学習用のマスターデータを生成します。
"""
import pandas as pd
from pathlib import Path

def main():
    print("="*70)
    print("データ統合")
    print("="*70)
    print()
    
    tracks_dir = Path('data/processed/tracks')
    features_dir = Path('data/processed/input_features')
    output_csv = Path('data/processed/master_training_data.csv')
    
    # トラックファイルと特徴量ファイルをマッチング
    all_data = []
    for track_file in tracks_dir.glob('*_tracks.csv'):
        video_id = track_file.stem.replace('_tracks', '')
        features_file = features_dir / f'{video_id}_features.csv'
        
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
        print("\n❌ エラー: 統合可能なデータがありません")
        return False
    
    # 統合データを保存
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    df = pd.DataFrame(all_data)
    df.to_csv(output_csv, index=False)
    
    print()
    print(f"✓ {len(df)}個の動画データを統合")
    print(f"  保存先: {output_csv}")
    print()
    print("最初の5件:")
    print(df.head())
    print()
    
    return True

if __name__ == "__main__":
    success = main()
    exit(0 if success else 1)
