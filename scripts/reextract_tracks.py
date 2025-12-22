"""
トラックデータのみを再抽出するスクリプト

既存の動画特徴量はそのまま使用し、トラックデータだけを再抽出します。
"""
import subprocess
from pathlib import Path
import sys

def main():
    print("="*70)
    print("トラックデータ再抽出")
    print("="*70)
    print()
    
    # ディレクトリ設定
    xml_dir = Path("data/raw/editxml")
    tracks_output = Path("data/processed/tracks")
    
    # 出力ディレクトリを作成
    tracks_output.mkdir(parents=True, exist_ok=True)
    
    # XMLファイルを検索
    xml_files = list(xml_dir.glob("*.xml"))
    if not xml_files:
        print(f"❌ エラー: XMLファイルが見つかりません: {xml_dir}")
        return False
    
    print(f"✓ {len(xml_files)}個のXMLファイルを発見")
    print()
    
    # XMLからトラックデータを抽出
    print("XMLファイルからトラックデータを抽出中...")
    print("-"*70)
    
    success_count = 0
    for idx, xml_file in enumerate(xml_files, 1):
        video_id = xml_file.stem
        print(f"[{idx}/{len(xml_files)}] {xml_file.name}")
        
        try:
            cmd = [
                "python",
                "src/data_preparation/fcpxml_to_tracks.py",
                str(xml_file),
                "--output", str(tracks_output),
                "--format", "csv"
            ]
            result = subprocess.run(cmd, capture_output=True, text=True)
            if result.returncode == 0:
                # 抽出されたクリップ数を取得
                for line in result.stdout.split('\n'):
                    if 'Extracted' in line and 'clips' in line:
                        print(f"  ✓ {line.strip()}")
                        break
                success_count += 1
            else:
                print(f"  ⚠️  抽出失敗")
                if result.stderr:
                    print(f"     {result.stderr[:200]}")
        except Exception as e:
            print(f"  ⚠️  エラー: {e}")
    
    print()
    print("="*70)
    print(f"完了: {success_count}/{len(xml_files)}個のトラックデータを抽出")
    print("="*70)
    print()
    print("次のステップ:")
    print("  python scripts/csv_to_npz_fixed.py")
    print()
    
    return True


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
