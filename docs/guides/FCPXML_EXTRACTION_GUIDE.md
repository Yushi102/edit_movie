# FCPXML トラック情報抽出ガイド

## 概要

`fcpxml_to_tracks.py`は、Final Cut Pro X（FCPXML）ファイルから編集トラック情報を抽出し、Multi-Track Transformerモデルの入力形式に変換するスクリプトです。

---

## 機能

### 抽出される情報

#### モデル入力用パラメータ（各トラック9次元）

1. **active** (0/1): トラックが有効かどうか
2. **asset_id** (0-9): クリップのアセットID
3. **scale**: スケール値
4. **pos_x**: X座標
5. **pos_y**: Y座標
6. **crop_l**: 左クロップ値
7. **crop_r**: 右クロップ値
8. **crop_t**: 上クロップ値
9. **crop_b**: 下クロップ値

#### 追加の詳細情報（CSV出力のみ）

- **clip_name**: クリップの名前
- **clip_ref**: クリップの参照ID
- **enabled**: クリップが有効かどうか（無効化されたクリップも検出）
- **source_start**: ソースファイルの使用開始位置（秒）
- **source_duration**: ソースファイルの使用時間（秒）
- **graphics_text**: グラフィック/テキスト要素の内容

### 出力形式

- **NPZ形式**: `(num_timesteps, 180)` - 20トラック × 9パラメータ
- **CSV形式**: 人間が読みやすい形式（検証用）

---

## インストール

### 必要なパッケージ

```bash
pip install numpy pandas
```

---

## 使用方法

### クイックスタート（Windows）

#### 単一ファイルのテスト

```cmd
test_fcpxml_extraction.bat "path\to\your_file.fcpxml"
```

#### 複数ファイルの一括処理

```cmd
batch_test_fcpxml.bat "path\to\fcpxml_directory"
```

これらのバッチファイルは自動的に：
1. 出力ディレクトリを作成
2. FCPXMLファイルをパース
3. NPZとCSVファイルを生成
4. 結果のサマリーを表示

### 基本的な使い方（コマンドライン）

```bash
python fcpxml_to_tracks.py input.fcpxml
```

これにより、`./output/`ディレクトリに以下のファイルが生成されます：
- `{video_id}_tracks.npz` - モデル入力用
- `{video_id}_tracks.csv` - 検証用（詳細情報付き）

### オプション

```bash
python fcpxml_to_tracks.py input.fcpxml \
    --output ./preprocessed_data \
    --video-id video001 \
    --max-tracks 20 \
    --fps 10.0 \
    --format npz
```

#### パラメータ説明

| パラメータ | デフォルト | 説明 |
|-----------|----------|------|
| `input` | (必須) | FCPXMLファイルのパス |
| `--output` | `output` | 出力ディレクトリ |
| `--video-id` | ファイル名 | ビデオID（識別子） |
| `--max-tracks` | `20` | 最大トラック数 |
| `--fps` | `10.0` | サンプリングレート（FPS） |
| `--format` | `both` | 出力形式（`npz`, `csv`, `both`） |

---

## 出力ファイル

### NPZ形式（モデル入力用）

```python
import numpy as np

# 読み込み
data = np.load('video001_tracks.npz')
sequences = data['sequences']  # (num_timesteps, 180)
video_ids = data['video_ids']  # ['video001']
num_tracks = data['num_tracks']  # 20
fps = data['fps']  # 10.0
asset_mapping = data['asset_mapping'].item()  # {'clip_name': asset_id}

print(f"Shape: {sequences.shape}")
print(f"Duration: {sequences.shape[0] / fps:.2f}s")
```

### CSV形式（検証用）

CSVファイルには、モデル入力パラメータに加えて、詳細な情報が含まれます：

| video_id | time | track | active | asset_id | clip_name | clip_ref | enabled | source_start | source_duration | graphics_text | scale | pos_x | pos_y | crop_l | crop_r | crop_t | crop_b |
|----------|------|-------|--------|----------|-----------|----------|---------|--------------|-----------------|---------------|-------|-------|-------|--------|--------|--------|--------|
| video001 | 0.0 | 0 | 1 | 0 | intro_clip | r1 | True | 0.0 | 5.0 | Welcome! | 1.0 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 |
| video001 | 0.1 | 0 | 1 | 0 | intro_clip | r1 | True | 0.1 | 5.0 | Welcome! | 1.0 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 |
| ... | ... | ... | ... | ... | ... | ... | ... | ... | ... | ... | ... | ... | ... | ... | ... | ... | ... |

**追加カラムの説明**:
- **clip_name**: クリップの名前（どのクリップが使われているか）
- **enabled**: クリップが有効かどうか（無効化されたクリップも記録）
- **source_start/duration**: ソースファイルのどの部分が使われているか
- **graphics_text**: テキスト/グラフィック要素の内容（タイトル、字幕など）

---

## バッチ処理

複数のFCPXMLファイルを一括処理する例：

```bash
#!/bin/bash

# バッチ処理スクリプト
for fcpxml in ./fcpxml_files/*.fcpxml; do
    video_id=$(basename "$fcpxml" .fcpxml)
    echo "Processing: $video_id"
    
    python fcpxml_to_tracks.py "$fcpxml" \
        --output ./preprocessed_data \
        --video-id "$video_id" \
        --fps 10.0 \
        --format npz
done

echo "Batch processing complete!"
```

Windowsの場合：

```powershell
# バッチ処理スクリプト (PowerShell)
Get-ChildItem ./fcpxml_files/*.fcpxml | ForEach-Object {
    $video_id = $_.BaseName
    Write-Host "Processing: $video_id"
    
    python fcpxml_to_tracks.py $_.FullName `
        --output ./preprocessed_data `
        --video-id $video_id `
        --fps 10.0 `
        --format npz
}

Write-Host "Batch processing complete!"
```

---

## データ検証

### 抽出されたデータの確認

```python
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# NPZファイルを読み込み
data = np.load('video001_tracks.npz')
sequences = data['sequences']
fps = float(data['fps'])

# 形状を確認
print(f"Shape: {sequences.shape}")
print(f"Duration: {sequences.shape[0] / fps:.2f}s")

# 20トラック × 9パラメータに変形
num_timesteps = sequences.shape[0]
tracks = sequences.reshape(num_timesteps, 20, 9)

# トラック0のアクティブ状態を可視化
plt.figure(figsize=(12, 4))
plt.plot(tracks[:, 0, 0])  # トラック0のactive
plt.xlabel('Timestep')
plt.ylabel('Active (0/1)')
plt.title('Track 0 Activity')
plt.grid(True)
plt.savefig('track0_activity.png')
plt.close()

# アクティブなトラック数を時系列で表示
active_counts = tracks[:, :, 0].sum(axis=1)
plt.figure(figsize=(12, 4))
plt.plot(active_counts)
plt.xlabel('Timestep')
plt.ylabel('Number of Active Tracks')
plt.title('Active Track Count Over Time')
plt.grid(True)
plt.savefig('active_track_count.png')
plt.close()

print("Validation plots saved!")
```

### CSVファイルでの確認

```python
import pandas as pd

# CSVを読み込み
df = pd.read_csv('video001_tracks.csv')

# 基本統計
print("Basic Statistics:")
print(df.describe())

# アクティブなクリップの数
active_clips = df[df['active'] == 1]
print(f"\nActive clips: {len(active_clips)}")
print(f"Unique tracks used: {active_clips['track'].nunique()}")
print(f"Unique assets: {active_clips['asset_id'].nunique()}")

# 時間範囲
print(f"\nTime range: {df['time'].min():.2f}s - {df['time'].max():.2f}s")
```

---

## トラブルシューティング

### 問題1: トラック数が20を超える

**症状**: 警告メッセージ「Reached max tracks (20), skipping remaining clips」

**解決策**:
```bash
python fcpxml_to_tracks.py input.fcpxml --max-tracks 30
```

### 問題2: パラメータが抽出されない

**症状**: scale, position, cropがすべて0

**原因**: FCPXMLの構造が想定と異なる可能性

**解決策**:
1. FCPXMLファイルをテキストエディタで開いて構造を確認
2. `param`要素の`name`属性を確認
3. 必要に応じてスクリプトの`extract_clip_info`メソッドを調整

### 問題3: アセットIDが10を超える

**症状**: 10個以上のユニークなクリップがある

**動作**: 自動的に0-9の範囲に収まるようにラップアラウンド（`asset_id % 10`）

**注意**: モデルは10クラスを想定しているため、10個以上のアセットがある場合は、
事前にクリップを統合するか、モデルの`max_asset_classes`パラメータを増やす必要があります。

---

## 統合ワークフロー

### 完全なデータ準備パイプライン

```bash
# 1. FCPXMLからトラック情報を抽出
python fcpxml_to_tracks.py project.fcpxml \
    --output ./preprocessed_data \
    --video-id video001 \
    --fps 10.0

# 2. 音声特徴量を抽出（別スクリプト）
# python extract_audio_features.py video001.mp4 \
#     --output ./input_features

# 3. 視覚特徴量を抽出（別スクリプト）
# python extract_visual_features.py video001.mp4 \
#     --output ./input_features

# 4. マルチモーダルモデルで訓練
python train.py \
    --sequences ./preprocessed_data/video001_tracks.npz \
    --features-dir ./input_features \
    --enable-multimodal
```

---

## 高度な使用例

### カスタムアセットマッピング

特定のクリップ名を特定のアセットIDにマッピングしたい場合：

```python
from fcpxml_to_tracks import FCPXMLParser

# パーサーを作成
parser = FCPXMLParser(max_tracks=20, fps=10.0)

# カスタムマッピングを設定
parser.asset_mapping = {
    'intro_clip': 0,
    'main_content': 1,
    'transition': 2,
    'outro_clip': 3
}
parser.next_asset_id = 4

# 通常通り処理
clips, duration = parser.parse_fcpxml('input.fcpxml')
sequence = parser.clips_to_track_sequence(clips, duration)
parser.save_sequence(sequence, 'output.npz', 'video001')
```

### 複数のFCPXMLを1つのNPZに統合

```python
import numpy as np
from fcpxml_to_tracks import FCPXMLParser

parser = FCPXMLParser(max_tracks=20, fps=10.0)

all_sequences = []
all_video_ids = []

for fcpxml_file in ['video001.fcpxml', 'video002.fcpxml', 'video003.fcpxml']:
    video_id = fcpxml_file.replace('.fcpxml', '')
    
    clips, duration = parser.parse_fcpxml(fcpxml_file)
    sequence = parser.clips_to_track_sequence(clips, duration)
    
    # Flatten to (num_timesteps, 180)
    flattened = sequence.reshape(sequence.shape[0], -1)
    
    all_sequences.append(flattened)
    all_video_ids.append(video_id)

# 統合して保存
np.savez_compressed(
    'combined_sequences.npz',
    sequences=all_sequences,
    video_ids=all_video_ids,
    num_tracks=20,
    fps=10.0
)

print(f"Combined {len(all_sequences)} sequences")
```

---

## 参考情報

### FCPXMLの構造

FCPXMLファイルは以下のような階層構造を持ちます：

```xml
<fcpxml version="1.9">
  <resources>
    <!-- アセット定義 -->
  </resources>
  <library>
    <event>
      <project>
        <sequence>
          <spine>
            <clip name="clip1" offset="0s" duration="5s">
              <video>
                <param name="scale" value="1.5"/>
                <param name="position_x" value="100"/>
              </video>
            </clip>
            <clip name="clip2" offset="5s" duration="3s">
              <!-- ... -->
            </clip>
          </spine>
        </sequence>
      </project>
    </event>
  </library>
</fcpxml>
```

### 時間形式

FCPXMLでは時間を分数形式で表現：
- `"1234/2400s"` = 1234フレーム / 2400fps = 0.514秒
- `"0s"` = 0秒

---

## まとめ

このスクリプトを使用することで、Final Cut Pro Xのプロジェクトから編集トラック情報を自動的に抽出し、
Multi-Track Transformerモデルの訓練データとして使用できます。

**次のステップ**:
1. FCPXMLファイルを準備
2. `fcpxml_to_tracks.py`でトラック情報を抽出
3. 音声・視覚特徴量を抽出（別途）
4. マルチモーダルモデルで訓練

質問や問題がある場合は、ログ出力を確認してください。詳細なデバッグ情報が記録されています。
