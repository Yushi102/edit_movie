"""
シンプルなテロップ修正スクリプト
"""
import re
import sys

def fix_telops(input_xml, output_xml):
    with open(input_xml, 'r', encoding='utf-8') as f:
        content = f.read()
    
    # [Telop]マーカーを削除
    content = re.sub(r'\[Telop\]\s*', '', content)
    
    # 最初のテロップのfile要素を完全な形式に変換
    # パターン: テロップtrack内の最初のfile要素
    telop_track_pattern = r'(<track>.*?<name>はみがきさん.*?<file id=")(file-\d+)(">.*?</file>)'
    
    def replace_first_telop_file(match):
        prefix = match.group(1)
        file_id = match.group(2)
        
        # 新しいfile要素
        new_file = f'''{prefix}file-2">
                                    <name>グラフィック</name>
                                    <mediaSource>GraphicAndType</mediaSource>
                                    <rate>
                                        <timebase>60</timebase>
                                        <ntsc>TRUE</ntsc>
                                    </rate>
                                    <timecode>
                                        <rate>
                                            <timebase>60</timebase>
                                            <ntsc>TRUE</ntsc>
                                        </rate>
                                        <string>00:00:00:00</string>
                                        <frame>0</frame>
                                        <displayformat>NDF</displayformat>
                                    </timecode>
                                    <media>
                                        <video>
                                            <samplecharacteristics>
                                                <rate>
                                                    <timebase>59</timebase>
                                                    <ntsc>TRUE</ntsc>
                                                </rate>
                                                <width>1080</width>
                                                <height>1920</height>
                                                <anamorphic>FALSE</anamorphic>
                                                <pixelaspectratio>square</pixelaspectratio>
                                                <fielddominance>none</fielddominance>
                                            </samplecharacteristics>
                                        </video>
                                    </media>
                                </file>'''
        
        return new_file
    
    content = re.sub(telop_track_pattern, replace_first_telop_file, content, count=1, flags=re.DOTALL)
    
    # 残りのテロップのfile要素をfile-2参照に変換
    # パターン: テロップtrack内のfile要素（2番目以降）
    remaining_telop_pattern = r'(<track>.*?<name>(?:まっててー|うん|うちめっちゃ|大丈夫！|…ｗ|うざっ).*?<file id=")(file-\d+)(">.*?</file>|"\s*/>)'
    
    def replace_remaining_telop_file(match):
        prefix = match.group(1)
        return f'{prefix}file-2"/>'
    
    content = re.sub(remaining_telop_pattern, replace_remaining_telop_file, content, flags=re.DOTALL)
    
    with open(output_xml, 'w', encoding='utf-8') as f:
        f.write(content)
    
    print(f"Done! Output: {output_xml}")

if __name__ == "__main__":
    if len(sys.argv) != 3:
        print("Usage: python fix_telop_simple.py <input> <output>")
        sys.exit(1)
    
    fix_telops(sys.argv[1], sys.argv[2])
