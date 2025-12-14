"""
Batch Video Feature Extraction Script

複数の動画ファイルから特徴量を一括抽出します。
"""
import os
import argparse
import glob
from pathlib import Path
from extract_video_features import FeatureExtractor
import torch

def batch_extract_features(
    input_dir,
    output_dir,
    video_extensions=None
):
    """
    ディレクトリ内の全動画から特徴量を抽出
    
    Args:
        input_dir: 動画ファイルが含まれるディレクトリ
        output_dir: 出力ディレクトリ
        video_extensions: 動画ファイルの拡張子リスト
    """
    if video_extensions is None:
        video_extensions = ['.mp4', '.mov', '.avi', '.mkv', '.m4v']
    
    # 出力ディレクトリを作成
    os.makedirs(output_dir, exist_ok=True)
    
    # 動画ファイルを検索
    video_files = []
    for ext in video_extensions:
        pattern = os.path.join(input_dir, f"**/*{ext}")
        video_files.extend(glob.glob(pattern, recursive=True))
    
    if not video_files:
        print(f"No video files found in {input_dir}")
        return
    
    print(f"\nFound {len(video_files)} video files")
    print("="*70)
    
    # 特徴量抽出器を初期化（1回だけ）
    extractor = FeatureExtractor()
    
    # 各動画を処理
    success_count = 0
    failed_files = []
    
    for i, video_path in enumerate(video_files, 1):
        print(f"\n[{i}/{len(video_files)}] Processing: {Path(video_path).name}")
        
        try:
            # 出力パスを決定
            video_stem = Path(video_path).stem
            output_path = os.path.join(output_dir, f"{video_stem}_features.csv")
            
            # すでに存在する場合はスキップ
            if os.path.exists(output_path):
                print(f"  ⚠ Skipping (already exists): {output_path}")
                continue
            
            # 特徴量を抽出
            extractor.extract_all_features(video_path, output_path)
            success_count += 1
            
        except Exception as e:
            print(f"  ✗ Error: {e}")
            failed_files.append((video_path, str(e)))
    
    # サマリーを表示
    print("\n" + "="*70)
    print("Batch Processing Complete!")
    print("="*70)
    print(f"Total files: {len(video_files)}")
    print(f"Successfully processed: {success_count}")
    print(f"Failed: {len(failed_files)}")
    
    if failed_files:
        print("\nFailed files:")
        for video_path, error in failed_files:
            print(f"  - {Path(video_path).name}: {error}")
    
    print("="*70 + "\n")


def main():
    parser = argparse.ArgumentParser(
        description="Batch extract features from multiple video files"
    )
    parser.add_argument(
        "input_dir",
        type=str,
        help="Directory containing video files"
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="./input_features",
        help="Output directory for feature CSV files (default: ./input_features)"
    )
    parser.add_argument(
        "--extensions",
        type=str,
        nargs="+",
        default=[".mp4", ".mov", ".avi", ".mkv", ".m4v"],
        help="Video file extensions to process (default: .mp4 .mov .avi .mkv .m4v)"
    )
    
    args = parser.parse_args()
    
    batch_extract_features(
        input_dir=args.input_dir,
        output_dir=args.output_dir,
        video_extensions=args.extensions
    )


if __name__ == "__main__":
    main()
