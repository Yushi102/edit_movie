# テロップ抽出ガイド

Premiere ProのXMLファイルからテロップ（字幕）情報を抽出する機能のガイドです。

## 概要

この機能は、Premiere ProのXMLファイルに含まれるBase64エンコードされたテロップ情報を解析し、以下の情報を抽出します：

- テロップのテキスト内容（日本語）
- 表示開始時刻
- 表示終了時刻
- 表示時間（duration）

## 使用方法

### 基本的な使い方

```bash
python scripts/extract_telop_from_xml.py
```

このコマンドを実行すると：
1. `data/raw/editxml/` 内の全XMLファイルを処理
2. テロップ情報を抽出
3. `data/processed/telop_labels/` にCSVファイルを出力

### 出力ファイル

各XMLファイルに対して、以下の形式のCSVファイルが生成されます：

**ファイル名**: `{元のXMLファイル名}_telop.csv`

**カラム**:
- `start_time`: テロップの表示開始時刻（秒）
- `end_time`: テロップの表示終了時刻（秒）
- `duration`: テロップの表示時間（秒）
- `text`: テロップのテキスト内容

**例**:
```csv
start_time,end_time,duration,text
0.62,2.90,2.28,"百鬼着地で
斬空ださない？"
4.10,6.21,2.10,"着地から対空
できますよ"
6.21,6.97,0.76,はいはいはい
```

## 技術的な詳細

### Base64デコード

Premiere Proはテロップのテキスト情報をBase64エンコードされたバイナリ形式で保存しています。このスクリプトは以下の手順でテキストを抽出します：

1. XMLから`<parameter>`タグの`<name>ソーステキスト</name>`を探す
2. 対応する`<value>`タグのBase64データをデコード
3. UTF-8でデコードし、日本語文字（ひらがな、カタカナ、漢字、全角記号）を抽出

### 対応する文字範囲

以下のUnicode範囲の文字を抽出します：
- `\u3000-\u303f`: 全角記号
- `\u3040-\u309f`: ひらがな
- `\u30a0-\u30ff`: カタカナ
- `\u4e00-\u9fff`: 漢字
- `\uff00-\uffef`: 全角英数字・記号

### フレームレート変換

XMLのフレーム数を秒に変換する際、以下のフレームレートを考慮します：
- タイムベース59 → 59.94fps（実際のフレームレート）
- タイムベース29 → 29.97fps
- タイムベース23 → 23.976fps
- その他 → タイムベースの値をそのまま使用

## 統計情報

実際のデータセットでの抽出結果（110個のXMLファイル）：

- **テロップあり**: 109ファイル（99.1%）
- **テロップなし**: 1ファイル（0.9%）
- **総テロップ数**: 3,262個
- **平均テロップ数**: 29.9個/動画
- **平均表示時間**: 約1.4秒

## 活用例

### 1. テロップの有無による動画分類

テロップが多い動画は、実況者の発言を強調している可能性が高く、視聴者の注目を集めやすい編集スタイルと言えます。

```python
import pandas as pd
from pathlib import Path

telop_dir = Path('data/processed/telop_labels')
for csv_file in telop_dir.glob('*_telop.csv'):
    df = pd.read_csv(csv_file)
    print(f"{csv_file.stem}: {len(df)}個のテロップ")
```

### 2. テロップ密度の計算

単位時間あたりのテロップ数を計算することで、編集の「テンポ」を定量化できます。

```python
import pandas as pd

df = pd.read_csv('data/processed/telop_labels/直線_telop.csv')
video_duration = df['end_time'].max()
telop_density = len(df) / video_duration
print(f"テロップ密度: {telop_density:.2f}個/秒")
```

### 3. テロップテキストの分析

テロップのテキスト内容を分析することで、動画の内容や雰囲気を把握できます。

```python
import pandas as pd
from collections import Counter
import re

df = pd.read_csv('data/processed/telop_labels/直線_telop.csv')

# 全テキストを結合
all_text = ' '.join(df['text'].tolist())

# 頻出単語を抽出（簡易版）
words = re.findall(r'[\u3040-\u309f\u30a0-\u30ff\u4e00-\u9fff]+', all_text)
word_freq = Counter(words)

print("頻出単語トップ10:")
for word, count in word_freq.most_common(10):
    print(f"  {word}: {count}回")
```

## 特徴量抽出への統合（今後の予定）

テロップ情報は、動画の特徴量として以下のように活用できます：

1. **テロップ密度特徴量**: 単位時間あたりのテロップ数
2. **テロップ表示時間特徴量**: 平均表示時間、最大表示時間
3. **テロップテキスト特徴量**: 感情分析、キーワード抽出
4. **テロップタイミング特徴量**: 音声との同期度、シーン変化との相関

これらの特徴量を既存の動画特徴量と組み合わせることで、より高精度な採用/不採用予測が可能になります。

## トラブルシューティング

### テロップが抽出されない

**原因**: XMLファイルにテロップ情報が含まれていない可能性があります。

**確認方法**:
```bash
# XMLファイルに「ソーステキスト」が含まれているか確認
findstr /C:"ソーステキスト" data\raw\editxml\{ファイル名}.xml
```

### 文字化けが発生する

**原因**: Base64データのデコードに失敗している可能性があります。

**対処法**: スクリプトは自動的にエラーを無視してデコードを試みますが、それでも問題がある場合は、XMLファイルのエンコーディングを確認してください。

### 出力CSVが空

**原因**: テロップクリップが「グラフィック」という名前でない可能性があります。

**対処法**: `extract_telop_from_xml.py` の以下の行を確認し、必要に応じて修正してください：
```python
if 'グラフィック' not in clip_name:
    continue
```

## 関連ファイル

- **抽出スクリプト**: `scripts/extract_telop_from_xml.py`
- **テストファイル**: `tests/test_telop_extraction.py`
- **出力ディレクトリ**: `data/processed/telop_labels/`
- **入力ディレクトリ**: `data/raw/editxml/`

## 参考情報

- Premiere ProのXMLフォーマット: XMEML (XML Media Exchange Language)
- Base64エンコーディング: RFC 4648
- Unicode文字範囲: Unicode Standard

## 今後の拡張予定

1. **テロップ位置情報の抽出**: テロップの画面上の位置（x, y座標）
2. **テロップスタイル情報の抽出**: フォント、色、サイズ
3. **テロップアニメーション情報の抽出**: フェードイン/アウト、移動
4. **特徴量抽出パイプラインへの統合**: ワンボタンで全て実行
5. **テロップベースの動画検索**: テキスト内容で動画を検索
