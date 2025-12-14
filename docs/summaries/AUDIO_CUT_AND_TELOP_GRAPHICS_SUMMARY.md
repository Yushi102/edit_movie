# 音声カット & テロップグラフィック変換機能 - 実装完了

## 概要
モデルが予測した映像のカット位置で音声も同じ位置でカットし、テロップをPremiere Pro互換のグラフィックとして出力する機能が完成しました。

## 実装されたコード

### 1. `fix_telop_simple.py` - テロップ変換スクリプト（最終解決策）
**役割**: OTIOが生成したXMLのテロップマーカーをPremiere Pro互換のグラフィックに変換

**処理内容**:
- `[Telop]`マーカーを削除
- 最初のテロップ: 完全なfile要素を生成（`<name>グラフィック</name>`, `<mediaSource>GraphicAndType</mediaSource>`, 完全な`<media><video><samplecharacteristics>`構造）
- 2番目以降のテロップ: `<file id="file-2"/>`で最初のテロップのfile要素を参照
- file IDは`file-2`を使用（`file-1`はビデオファイル用）

**参考にした構造**:
- `editxml/bandicam 2025-06-02 00-03-33-780.xml`
- `bandicam_2025-06-02_final_test.xml`

### 2. `otio_xml_generator.py` - OTIO XML生成
**役割**: OpenTimelineIOを使用してPremiere Pro互換のXMLを生成

**音声カット機能**:
- `create_premiere_xml_with_otio`関数で実装
- 映像クリップと音声クリップを同じ`source_range`で追加
- 映像と音声が同じ位置でカットされる

**テロップ処理**:
- テロップクリップに`[Telop]`マーカーを付けて生成
- 後処理関数は現在コメントアウト（`# if telops:`の部分）
- 別スクリプト`fix_telop_simple.py`で後処理する方式に変更

### 3. `inference_pipeline.py` - 推論パイプライン
**役割**: 動画から編集を予測してXMLを生成

**設定**:
- active閾値: 0.29（カット数を約500個程度に調整）
- FPS: 10fps（推論時）
- 元動画のFPSに自動変換

## 使用方法

```bash
# ステップ1: 推論を実行してOTIO XMLを生成（音声カット済み）
python inference_pipeline.py "D:\切り抜き\2025-6\2025-6-02\bandicam 2025-06-02 00-03-33-780.mp4" --model checkpoints_50epochs/best_model.pth --output temp.xml

# ステップ2: テロップをグラフィックに変換
python fix_telop_simple.py temp.xml final.xml

# ステップ3: 最終的なXMLをPremiere Proで開く
```

## 成功したXML出力例
- `bandicam_2025-06-02_COMPLETE.xml` - 正しく生成され、Premiere Proで読み込めるXML

## 技術的な詳細

### 音声カットの仕組み
OTIOの`source_range`を使用して、映像と音声を同じ時間範囲で追加:
```python
source_range = otio.opentime.TimeRange(
    start_time=otio.opentime.RationalTime(start_frame_video, rate),
    duration=otio.opentime.RationalTime(end_frame_video - start_frame_video, rate)
)

# 映像クリップ
video_clip = otio.schema.Clip(name=video_name, media_reference=media_reference, source_range=source_range)
video_track.append(video_clip)

# 音声クリップ（同じsource_rangeを使用）
audio_clip = otio.schema.Clip(name=f"{video_name} - Audio", media_reference=media_reference, source_range=source_range)
audio_track.append(audio_clip)
```

### テロップのグラフィック変換
正規表現を使用してXMLを後処理:
1. `[Telop]`マーカーを削除
2. 最初のテロップのfile要素を完全な形式に変換
3. 残りのテロップは最初のfile要素を参照

## 重要な注意事項
- **OTIOは必ず使用する**（直接XML生成は不可）
- **`_generate_premiere_xml_directly_with_audio`関数は信用しない**
- テロップ変換は別スクリプト`fix_telop_simple.py`で行う
- モデルのactive閾値は0.29に設定（カット数調整）

## ファイル一覧
- `fix_telop_simple.py` - **テロップ変換スクリプト（最終解決策）**
- `otio_xml_generator.py` - OTIO XML生成（音声カット対応）
- `inference_pipeline.py` - 推論パイプライン
- `bandicam_2025-06-02_COMPLETE.xml` - 成功したXML出力例
- `editxml/bandicam 2025-06-02 00-03-33-780.xml` - 参考XML
- `bandicam_2025-06-02_final_test.xml` - 読み込める参考XML
