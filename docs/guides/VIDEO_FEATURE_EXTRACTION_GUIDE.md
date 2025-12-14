# 動画特徴量抽出ガイド

## 概要

`extract_video_features.py`は、動画ファイルから音声・視覚特徴量を抽出し、マルチモーダルモデルの入力データを生成するスクリプトです。

---

## 抽出される特徴量

### 音声特徴量 (7次元)

| 特徴量 | 説明 | 型 |
|--------|------|-----|
| `audio_energy_rms` | RMS energy (音量) | float |
| `audio_is_speaking` | 発話検出 (0/1) | int |
| `silence_duration_ms` | 無音時間 (ミリ秒) | int |
| `speaker_id` | 話者ID (プレースホルダ) | float (NaN) |
| `text_is_active` | テキスト検出 (0/1) | int |
| `text_word` | 単語数 | int/str |

### 視覚特徴量 (522次元)

#### スカラー特徴量 (10次元)

| 特徴量 | 説明 | 型 |
|--------|------|-----|
| `scene_change` | シーン転換スコア (0-1) | float |
| `visual_motion` | 動き量 (0-1) | float |
| `saliency_x` | 注目点X座標 (0-1) | float |
| `saliency_y` | 注目点Y座標 (0-1) | float |
| `face_count` | 顔の数 | int |
| `face_center_x` | 顔の中心X座標 (0-1) | float |
| `face_center_y` | 顔の中心Y座標 (0-1) | float |
| `face_size` | 顔のサイズ (0-1) | float |
| `face_mouth_open` | 口の開き具合 | float |
| `face_eyebrow_raise` | 眉の上がり具合 | float |

#### CLIP Embeddings (512次元)

| 特徴量 | 説明 |
|--------|------|
| `clip_0` ~ `clip_511` | CLIP視覚的意味表現 (512次元) |

### 合計: 529次元
- 音声: 7次元
- 視覚スカラー: 10次元
- CLIP: 512次元

---

## 必要なパッケージ

```bash
pip install numpy pandas opencv-python opencv-contrib-python
pip install librosa soundfile pydub
pip install openai-whisper
pip install torch torchvision
pip install transformers
pip install mediapipe
pip install tqdm
```

---

## 使用方法

### 単一ファイルの処理

```bash
python extract_video_features.py video.mp4
```

出力: `video_features.csv`

### 出力パスを指定

```bash
python extract_video_features.py video.mp4 --output ./output/video_features.csv
```

### バッチ処理

```bash
python batch_extract_features.py ./videos --output-dir ./input_features
```

---

## 出力形式

### CSV構造

```csv
time,audio_energy_rms,audio_is_speaking,silence_duration_ms,speaker_id,text_is_active,text_word,scene_change,visual_motion,saliency_x,saliency_y,face_count,face_center_x,face_center_y,face_size,face_mouth_open,face_eyebrow_raise,clip_0,clip_1,...,clip_511
0.0,0.023,1,0,,1,Hello,0.0,0.15,0.52,0.48,1,0.51,0.45,0.12,0.35,0.22,-0.123,0.456,...,0.789
0.1,0.031,1,0,,1,world,0.02,0.18,0.53,0.47,1,0.51,0.45,0.12,0.38,0.21,-0.125,0.458,...,0.791
...
```

### サンプリングレート

- **10 FPS** (0.1秒刻み)
- CLIP embeddingsは1秒ごとに更新（高速化のため）

---

## 設定のカスタマイズ

`extract_video_features.py`の冒頭で設定を変更できます：

```python
TIME_STEP = 0.1              # サンプリング間隔 (秒)
CLIP_STEP = 1.0              # CLIP解析間隔 (秒)
ANALYSIS_WIDTH = 640         # 解析用画像の横幅 (px)
WHISPER_MODEL_SIZE = "small" # Whisperモデルサイズ
```

### Whisperモデルサイズ

| サイズ | 精度 | 速度 | メモリ |
|--------|------|------|--------|
| tiny | 低 | 最速 | 最小 |
| base | 中 | 速い | 小 |
| small | 高 | 普通 | 中 |
| medium | 最高 | 遅い | 大 |
| large | 最高 | 最遅 | 最大 |

推奨: **small** (精度と速度のバランスが良い)

---

## パフォーマンス

### GPU使用時

- 10分の動画: 約5-10分
- CLIP処理が高速化

### CPU使用時

- 10分の動画: 約20-30分
- Whisper文字起こしが時間がかかる

### 高速化のヒント

1. **ANALYSIS_WIDTH を小さくする** (640 → 480)
2. **CLIP_STEP を大きくする** (1.0 → 2.0秒)
3. **Whisperモデルを小さくする** (small → base)

---

## トラブルシューティング

### 問題1: GPU out of memory

**解決策**:
```python
ANALYSIS_WIDTH = 480  # 640から縮小
CLIP_STEP = 2.0       # 1.0から増加
```

### 問題2: Whisperが遅い

**解決策**:
```python
WHISPER_MODEL_SIZE = "base"  # smallから変更
```

### 問題3: MediaPipeエラー

**症状**: `AttributeError: module 'mediapipe' has no attribute 'solutions'`

**解決策**:
```bash
pip install --upgrade mediapipe
```

### 問題4: CLIP safetensorsエラー

**症状**: `Error loading model with safetensors`

**解決策**: スクリプトが自動的にフォールバックします。問題なし。

---

## データ検証

### 抽出されたデータの確認

```python
import pandas as pd

# CSVを読み込み
df = pd.read_csv('video_features.csv')

# 基本情報
print(f"Timesteps: {len(df)}")
print(f"Duration: {df['time'].max():.2f}s")
print(f"Features: {len(df.columns)}")

# 音声特徴量の統計
print("\nAudio Features:")
print(df[['audio_energy_rms', 'audio_is_speaking', 'silence_duration_ms']].describe())

# 視覚特徴量の統計
print("\nVisual Features:")
print(df[['scene_change', 'visual_motion', 'face_count']].describe())

# 発話時間の割合
speaking_ratio = df['audio_is_speaking'].mean()
print(f"\nSpeaking ratio: {speaking_ratio:.2%}")

# 顔検出率
face_detection_ratio = (df['face_count'] > 0).mean()
print(f"Face detection ratio: {face_detection_ratio:.2%}")
```

---

## 統合ワークフロー

### 完全なデータ準備パイプライン

```bash
# 1. Premiere XMLからトラック情報を抽出
python premiere_xml_parser.py editxml/video.xml --output ./preprocessed_data --format npz

# 2. 動画から特徴量を抽出
python extract_video_features.py video.mp4 --output ./input_features/video_features.csv

# 3. マルチモーダルモデルで訓練
python train.py \
    --sequences ./preprocessed_data/video_tracks.npz \
    --features-dir ./input_features \
    --enable-multimodal
```

### バッチ処理の例

```bash
# 複数のXMLファイルを処理
for xml in editxml/*.xml; do
    python premiere_xml_parser.py "$xml" --output ./preprocessed_data --format npz
done

# 複数の動画ファイルを処理
python batch_extract_features.py ./videos --output-dir ./input_features

# 訓練
python train.py \
    --sequences ./preprocessed_data/*.npz \
    --features-dir ./input_features \
    --enable-multimodal
```

---

## 技術的な詳細

### 音声処理

1. **PyDub**: 動画から音声を抽出 (16kHz, mono)
2. **Librosa**: RMS energy計算、VAD
3. **Whisper**: 文字起こし（単語レベルのタイムスタンプ）

### 視覚処理

1. **OpenCV**: フレーム読み込み、シーン転換、動き検出
2. **MediaPipe**: 顔検出、ランドマーク、表情分析
3. **CLIP**: 視覚的意味表現（512次元）

### 時間同期

- すべての特徴量は同じタイムスタンプで整列
- 10 FPS (0.1秒刻み) で統一
- Premiere XMLのトラック情報と±0.05秒の精度でマッチング可能

---

## まとめ

✅ **音声特徴量**: RMS, 発話検出, 無音時間, 文字起こし

✅ **視覚特徴量**: シーン転換, 動き, 顔, CLIP embeddings

✅ **出力形式**: CSV (10 FPS, 529次元)

✅ **GPU対応**: CUDA自動検出

✅ **バッチ処理**: 複数ファイル一括処理

これで、マルチモーダルモデルの訓練に必要なすべての特徴量を抽出できます！
