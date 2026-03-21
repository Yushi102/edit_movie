"""
XMLファイルからテロップ（字幕）情報を抽出

Premiere ProのXMLに含まれるBase64エンコードされたテロップ情報を解析し、
テキスト内容、表示時刻、位置情報を抽出します。
"""
import xml.etree.ElementTree as ET
from pathlib import Path
import pandas as pd
import base64
import re
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

def extract_japanese_text_from_base64(base64_str):
    """
    Base64エンコードされたデータから日本語テキストを抽出
    
    Args:
        base64_str: Base64エンコードされた文字列
    
    Returns:
        抽出されたテキスト（複数の場合は改行で結合）
    """
    try:
        # Base64デコード
        decoded = base64.b64decode(base64_str)
        
        # UTF-8でデコード（エラーは無視）
        text = decoded.decode('utf-8', errors='ignore')
        
        # 日本語文字（ひらがな、カタカナ、漢字、全角記号）を抽出
        # Unicode範囲:
        # \u3000-\u303f: 全角記号
        # \u3040-\u309f: ひらがな
        # \u30a0-\u30ff: カタカナ
        # \u4e00-\u9fff: 漢字
        # \uff00-\uffef: 全角英数字・記号
        japanese_chars = re.findall(r'[\u3000-\u9fff\uff00-\uffef]+', text)
        
        if japanese_chars:
            # 複数のテキスト断片を改行で結合
            return '\n'.join(japanese_chars)
        
        return None
    except Exception as e:
        return None

def extract_telop_from_xml(xml_path):
    """
    XMLファイルからテロップ情報を抽出
    
    Args:
        xml_path: XMLファイルのパス
    
    Returns:
        DataFrame with columns: start_time, end_time, duration, text
    """
    tree = ET.parse(xml_path)
    root = tree.getroot()
    
    # XMEMLフォーマットを想定
    sequence = root.find('.//sequence')
    if sequence is None:
        return None
    
    # タイムベース（フレームレート）を取得
    rate_elem = sequence.find('.//rate/timebase')
    timebase = int(rate_elem.text) if rate_elem is not None and rate_elem.text else 30
    
    telops = []
    
    # 全てのビデオトラックからテロップクリップを収集
    for track in sequence.findall('.//video/track'):
        for clipitem in track.findall('clipitem'):
            # クリップ名を取得
            name_elem = clipitem.find('name')
            clip_name = name_elem.text if name_elem is not None else ""
            
            # グラフィッククリップ（テロップ）のみを対象
            if 'グラフィック' not in clip_name:
                continue
            
            # start/end（タイムライン上の表示時刻）を取得
            start_elem = clipitem.find('start')
            end_elem = clipitem.find('end')
            
            if start_elem is None or end_elem is None:
                continue
            
            start_frame = int(start_elem.text) if start_elem.text else 0
            end_frame = int(end_elem.text) if end_elem.text else 0
            
            if end_frame <= start_frame:
                continue
            
            start_time = parse_time_xmeml(str(start_frame), timebase)
            end_time = parse_time_xmeml(str(end_frame), timebase)
            duration = end_time - start_time
            
            # テロップテキストを抽出
            # effectの中の「ソーステキスト」パラメータを探す
            text = None
            for effect in clipitem.findall('.//effect'):
                for param in effect.findall('parameter'):
                    name_elem = param.find('name')
                    if name_elem is not None and name_elem.text == 'ソーステキスト':
                        value_elem = param.find('value')
                        if value_elem is not None and value_elem.text:
                            # Base64データからテキストを抽出
                            text = extract_japanese_text_from_base64(value_elem.text)
                            break
                if text:
                    break
            
            # テキストが抽出できた場合のみ追加
            if text:
                telops.append({
                    'start_time': start_time,
                    'end_time': end_time,
                    'duration': duration,
                    'text': text
                })
    
    if not telops:
        return None
    
    df = pd.DataFrame(telops)
    # 時刻順にソート
    df = df.sort_values('start_time').reset_index(drop=True)
    
    return df

def main():
    """メイン処理"""
    xml_dir = Path('data/raw/editxml')
    output_dir = Path('data/processed/telop_labels')
    output_dir.mkdir(parents=True, exist_ok=True)
    
    xml_files = sorted(xml_dir.glob('*.xml'))
    
    print(f"XMLファイルからテロップ情報を抽出中...")
    print(f"入力: {xml_dir}")
    print(f"出力: {output_dir}")
    print(f"XMLファイル数: {len(xml_files)}\n")
    
    success_count = 0
    no_telop_count = 0
    failed_files = []
    
    total_telops = 0
    
    for i, xml_path in enumerate(xml_files, 1):
        print(f"[{i}/{len(xml_files)}] {xml_path.name}")
        
        try:
            df = extract_telop_from_xml(xml_path)
            
            if df is not None and len(df) > 0:
                # 出力ファイル名
                output_name = xml_path.stem + '_telop.csv'
                output_path = output_dir / output_name
                
                # CSVに保存
                df.to_csv(output_path, index=False, encoding='utf-8-sig')
                
                telop_count = len(df)
                total_telops += telop_count
                
                print(f"  OK テロップ抽出完了: {telop_count}個")
                print(f"    時間範囲: {df['start_time'].min():.1f}秒 ~ {df['end_time'].max():.1f}秒")
                print(f"    平均表示時間: {df['duration'].mean():.1f}秒")
                
                # 最初の3つのテロップを表示
                for idx, row in df.head(3).iterrows():
                    text_preview = row['text'].replace('\n', ' ')[:30]
                    print(f"      [{row['start_time']:.1f}s] {text_preview}...")
                
                if len(df) > 3:
                    print(f"      ... 他 {len(df)-3} 個")
                
                success_count += 1
            else:
                print(f"  - テロップなし")
                no_telop_count += 1
                
        except Exception as e:
            print(f"  NG エラー: {e}")
            failed_files.append(xml_path.name)
    
    print(f"\n{'='*60}")
    print(f"処理完了:")
    print(f"  テロップあり: {success_count}/{len(xml_files)}")
    print(f"  テロップなし: {no_telop_count}/{len(xml_files)}")
    print(f"  失敗: {len(failed_files)}")
    print(f"  総テロップ数: {total_telops}個")
    
    if success_count > 0:
        print(f"  平均テロップ数: {total_telops/success_count:.1f}個/動画")
    
    if failed_files:
        print(f"\n失敗したファイル:")
        for fname in failed_files[:10]:
            print(f"  - {fname}")
        if len(failed_files) > 10:
            print(f"  ... 他 {len(failed_files)-10} ファイル")

if __name__ == '__main__':
    main()
