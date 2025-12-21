"""
XMLファイルから採用/不採用（active）のみを抽出

各時刻で、どのクリップが採用されているか（active=1）を抽出します。
"""
import xml.etree.ElementTree as ET
from pathlib import Path
import pandas as pd
import sys
from datetime import datetime

def parse_time_xmeml(time_str, timebase):
    """XMEMLのフレーム数を秒に変換"""
    try:
        frames = int(time_str) if time_str else 0
        
        # タイムベースの補正：59は実際には59.94fps
        if timebase == 59:
            actual_fps = 59.94
        elif timebase == 29:
            actual_fps = 29.97
        elif timebase == 23:
            actual_fps = 23.976
        else:
            actual_fps = float(timebase)
        
        return frames / actual_fps if actual_fps > 0 else frames / 30.0
    except (ValueError, TypeError):
        return 0.0

def get_source_video_duration(video_name):
    """
    元動画の長さを特徴量ファイルから取得
    
    Args:
        video_name: 動画ファイル名（拡張子なし）
    
    Returns:
        動画の長さ（秒）、見つからない場合はNone
    """
    feature_path = Path('data/processed/source_features') / f'{video_name}_features.csv'
    if feature_path.exists():
        try:
            df = pd.read_csv(feature_path, usecols=['time'])
            return df['time'].max()
        except Exception as e:
            print(f"    警告: 特徴量ファイル読み込みエラー: {e}")
            return None
    return None

def extract_active_from_xml(xml_path, source_duration=None):
    """
    XMLファイルから採用/不採用情報を抽出
    
    元動画のどの部分が使われているか（in/out）を基に判定
    元動画以外のクリップ（グラフィック、タイトル等）は除外
    
    Args:
        xml_path: XMLファイルのパス
        source_duration: 元動画の長さ（秒）。Noneの場合はクリップから推定
    
    Returns:
        DataFrame with columns: time, active (0 or 1)
    """
    tree = ET.parse(xml_path)
    root = tree.getroot()
    
    # XMEMLフォーマットを想定
    sequence = root.find('.//sequence')
    if sequence is None:
        print(f"  警告: sequenceが見つかりません")
        return None
    
    # タイムベース（フレームレート）を取得
    rate_elem = sequence.find('.//rate/timebase')
    timebase = int(rate_elem.text) if rate_elem is not None and rate_elem.text else 30
    
    # XMLファイル名から元動画名を取得（拡張子なし）
    source_video_name = xml_path.stem
    
    # 全てのビデオトラックからクリップを収集
    # 元動画の in/out を使用（元動画のクリップのみ）
    used_ranges = []  # 元動画で使われている範囲
    
    for track in sequence.findall('.//video/track'):
        for clipitem in track.findall('clipitem'):
            # クリップ名を取得
            name_elem = clipitem.find('name')
            clip_name = name_elem.text if name_elem is not None else ""
            
            # 元動画のクリップかチェック
            # 1. クリップ名がXMLファイル名（元動画名）を含むかチェック
            # 2. クリップ名が.mp4で終わるかチェック（グラフィックを除外）
            if source_video_name not in clip_name:
                continue  # 元動画以外のクリップはスキップ
            
            if not clip_name.endswith('.mp4'):
                continue  # 動画ファイル以外（グラフィック等）はスキップ
            
            # in/out（元動画のどの部分を使ったか）を取得
            in_elem = clipitem.find('in')
            out_elem = clipitem.find('out')
            
            if in_elem is not None and out_elem is not None:
                in_frame = int(in_elem.text) if in_elem.text else 0
                out_frame = int(out_elem.text) if out_elem.text else 0
                
                if out_frame > in_frame:
                    in_time = parse_time_xmeml(str(in_frame), timebase)
                    out_time = parse_time_xmeml(str(out_frame), timebase)
                    
                    used_ranges.append({
                        'start_time': in_time,
                        'end_time': out_time
                    })
    
    if not used_ranges:
        print(f"  警告: 元動画のクリップが見つかりません")
        return None
    
    # 元動画の長さを決定
    if source_duration is None:
        # クリップから推定（最大のout時刻）
        source_duration = max(r['end_time'] for r in used_ranges)
    
    # 0.1秒ごとにサンプリング
    sample_interval = 0.1
    times = []
    
    current_time = 0.0
    while current_time <= source_duration:
        # この時刻が元動画の使用範囲に含まれるか確認
        is_active = any(
            r['start_time'] <= current_time < r['end_time']
            for r in used_ranges
        )
        
        times.append({
            'time': current_time,
            'active': 1 if is_active else 0
        })
        
        current_time += sample_interval
    
    df = pd.DataFrame(times)
    return df

def main():
    """メイン処理"""
    xml_dir = Path('data/raw/editxml')
    output_dir = Path('data/processed/active_labels')
    output_dir.mkdir(parents=True, exist_ok=True)
    
    xml_files = sorted(xml_dir.glob('*.xml'))
    
    print(f"XMLファイルから採用/不採用を抽出中...")
    print(f"入力: {xml_dir}")
    print(f"出力: {output_dir}")
    print(f"XMLファイル数: {len(xml_files)}\n")
    
    success_count = 0
    failed_files = []
    
    for i, xml_path in enumerate(xml_files, 1):
        print(f"[{i}/{len(xml_files)}] {xml_path.name}")
        
        try:
            # 元動画の長さを取得
            video_name = xml_path.stem
            source_duration = get_source_video_duration(video_name)
            
            if source_duration is not None:
                print(f"    元動画の長さ: {source_duration:.1f}秒")
            else:
                print(f"    警告: 元動画の特徴量が見つかりません。編集後の長さを使用します。")
            
            df = extract_active_from_xml(xml_path, source_duration)
            
            if df is not None and len(df) > 0:
                # 出力ファイル名
                output_name = xml_path.stem + '_active.csv'
                output_path = output_dir / output_name
                
                # CSVに保存
                df.to_csv(output_path, index=False)
                
                active_count = df['active'].sum()
                inactive_count = (df['active'] == 0).sum()
                
                print(f"  OK 抽出完了: {len(df)}行")
                print(f"    採用: {active_count}サンプル ({active_count/len(df)*100:.1f}%)")
                print(f"    不採用: {inactive_count}サンプル ({inactive_count/len(df)*100:.1f}%)")
                print(f"    時間範囲: {df['time'].min():.1f}秒 ~ {df['time'].max():.1f}秒")
                success_count += 1
            else:
                print(f"  NG データなし")
                failed_files.append(xml_path.name)
                
        except Exception as e:
            print(f"  NG エラー: {e}")
            failed_files.append(xml_path.name)
    
    print(f"\n{'='*60}")
    print(f"処理完了:")
    print(f"  成功: {success_count}/{len(xml_files)}")
    print(f"  失敗: {len(failed_files)}")
    
    if failed_files:
        print(f"\n失敗したファイル:")
        for fname in failed_files[:10]:
            print(f"  - {fname}")
        if len(failed_files) > 10:
            print(f"  ... 他 {len(failed_files)-10} ファイル")

if __name__ == '__main__':
    main()
