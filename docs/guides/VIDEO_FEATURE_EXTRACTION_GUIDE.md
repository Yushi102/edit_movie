# 動画特徴量抽出ガイド

## 概要

`extract_video_features_parallel.py`は、動画ファイルから音声・視覚・時系列特徴量を抽出し、カット選択モデルの入力データを生成するスクリプトです。

**並列処理対応**: 複数の動画を同時に処理して高速化

---

## 抽出される特徴量

### 音声特徴量 (235次元)

#### 基本音声特徴量 (4次元)
| 特徴量 | 説明 | 型 |
|--------|------|-----|
| `audio_energy_rms` | RMS energy (音量) | float |
| `audio_is_speaking` | 発話検出 (VAD) | int (0/1) |
| `silence_duration_ms` | 無音時間 (ミリ秒) | int |
| `speaker_id` | 話者ID | float |

#### 話者埋め込み (192次元)
| 特徴量 | 説明 |
|--------|------|
| `speaker_emb_0` ~ `speaker_emb_191` | pyannote.audioによる話者埋め込み (192次元) |

#### 音響特徴量 (16次元)
| 特徴量 | 説明 |
|--------|------|
| `pitch` | ピッチ（音の高さ） |
| `spectral_centroid` | スペクトル重心 |
| `spectral_bandwidth` | スペクトル帯域幅 |
| `mfcc_0` ~ `mfcc_12` | MFCC（メル周波数ケプストラム係数、13次元） |

#### テキスト・テロップ (3次元)
| 特徴量 | 説明 |
|--------|------|
| `text_is_active` | Whisperによる音声認識フラグ | int (0/1) |
| `text_word` | 単語数 | int |
| `telop_is_active` | テロップ検出フラグ（未実装） | int (0/1) |

#### 時系列音声特徴量 (20次元)
- 移動平均: MA5, MA10, MA30（RMS energy）
- 変化率: DIFF1, DIFF2（RMS energy）
- 音声変化: audio_change_score, silence_to_speech, speech_to_silence, speaker_change, pitch_change
- その他: 累積統計等

**音声特徴量合計: 235次元**

---

### 視覚特徴量 (543次元)

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

#### 時系列視覚特徴量 (21次元)
- 移動平均: MA5, MA10, MA30（visual_motion）
- 変化率: DIFF1, DIFF2（visual_motion）
- CLIP類似度: clip_sim_prev, clip_sim_next, clip_sim_mean5
- 映像変化: visual_motion_change, face_count_change, saliency_movement
- その他: 累積統計等

**視覚特徴量合計: 543次元**

---

### 時系列特徴量 (6次元)

#### カットタイミング特徴量
| 特徴量 | 説明 |
|--------|------|
| `time_since_prev` | 前のカットからの時間（秒） |
| `time_to_next` | 次のカットまでの時間（秒） |
| `cut_duration` | カット長（秒） |
| `position_in_video` | 動画内位置（0-1） |
| `cut_density_10s` | 10秒間のカット密度 |
| `cumulative_adoption_rate` | 累積採用率 |

**時系列特徴量合計: 6次元**

---

### 合計: 784次元
- **音声**: 235次元
- **視覚**: 543次元
- **時系列**: 6次元

---

## 必要なパッケージ

```bash
pip install numpy pandas opencv-python opencv-contrib-python
pip install librosa soundfile pydub
pip install openai-whisper
pip install torch torchvision
pip install transformers
pip install mediapipe
pip install pyannote.audio
pip install tqdm
```

---

## 使用方法

### 並列処理（推奨）

```bash
python -m src.data_preparation.extract_video_features_parallel \
    --video_dir videos \
    --output_dir data/processed/source_features \
    --n_jobs 4
```

**オプション**:
- `--video_dir`: 動画ファイルのディレクトリ
- `--output_dir`: 出力先ディレクトリ
- `--n_jobs`: 並列処理数（デフォルト: 4）

**処理時間**: 5-10分/動画（10分の動画、GPU使用時）

### 単一ファイルの処理

```bash
python -m src.data_preparation.extract_video_features video.mp4
```

出力: `data/processed/source_features/video_features.csv`

---

## 時系列特徴量の追加

特徴量抽出後、時系列特徴量を追加します：

```bash
python scripts/add_temporal_features.py
```

**追加される特徴量（83個）**:
1. **移動統計量**: MA5, MA10, MA30, MA60, MA120, STD5, STD30, STD120
2. **変化率**: DIFF1, DIFF2, DIFF30
3. **カットタイミング**: time_since_prev, time_to_next, cut_duration, position_in_video, cut_density_10s
4. **CLIP類似度**: clip_sim_prev, clip_sim_next, clip_sim_mean5
5. **音声変化**: audio_change_score, silence_to_speech, speech_to_silence, speaker_change, pitch_change
6. **映像変化**: visual_motion_change, face_count_change, saliency_movement
7. **累積統計**: cumulative_position, cumulative_adoption_rate

**出力**: `data/processed/source_features/video_features_enhanced.csv`

---

## 出力形式

### CSV構造

```csv
time,audio_energy_rms,audio_is_speaking,silence_duration_ms,speaker_id,speaker_emb_0,...,speaker_emb_191,pitch,spectral_centroid,...,mfcc_12,text_is_active,text_word,telop_is_active,scene_change,visual_motion,saliency_x,saliency_y,face_count,face_center_x,face_center_y,face_size,face_mouth_open,face_eyebrow_raise,clip_0,...,clip_511,time_since_prev,time_to_next,cut_duration,position_in_video,cut_density_10s,cumulative_adoption_rate
0.0,0.023,1,0,0.0,-0.123,...,0.456,220.5,1500.2,...,0.35,1,5,0,0.0,0.15,0.52,0.48,1,0.51,0.45,0.12,0.35,0.22,-0.123,...,0.789,0.0,5.2,5.2,0.0,0.5,0.0
0.1,0.031,1,0,0.0,-0.125,...,0.458,225.3,1520.5,...,0.38,1,5,0,0.02,0.18,0.53,0.47,1,0.51,0.45,0.12,0.38,0.21,-0.125,...,0.791,0.1,5.1,5.2,0.001,0.5,0.0
...
```

### サンプリングレート

- **10 FPS** (0.1秒刻み)
- CLIP embeddingsは1秒ごとに更新（高速化のため）
- 話者埋め込みは0.5秒ごとに更新

---

## 設定のカスタマイズ

`src/data_preparation/extract_video_features.py`の冒頭で設定を変更できます：

```python
TIME_STEP = 0.1              # サンプリング間隔 (秒)
CLIP_STEP = 1.0              # CLIP解析間隔 (秒)
SPEAKER_STEP = 0.5           # 話者埋め込み解析間隔 (秒)
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

### GPU使用時（推奨）

- 10分の動画: 約5-10分
- CLIP処理が高速化
- 話者埋め込み処理が高速化

### CPU使用時

- 10分の動画: 約20-30分
- Whisper文字起こしが時間がかかる
- 話者埋め込みが時間がかかる

### 並列処理の効果

- 4並列: 約4倍高速化
- 30本の動画: 約2.5-5時間（4並列、GPU使用時）

### 高速化のヒント

1. **並列処理数を増やす** (`--n_jobs 8`)
2. **ANALYSIS_WIDTH を小さくする** (640 → 480)
3. **CLIP_STEP を大きくする** (1.0 → 2.0秒)
4. **Whisperモデルを小さくする** (small → base)
5. **GPU使用を確認** (`nvidia-smi`)

---

## トラブルシューティング

### 問題1: GPU out of memory

**解決策**:
```python
ANALYSIS_WIDTH = 480  # 640から縮小
CLIP_STEP = 2.0       # 1.0から増加
SPEAKER_STEP = 1.0    # 0.5から増加
```

または並列処理数を減らす:
```bash
python extract_video_features_parallel.py --n_jobs 2
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

**プロジェクトパスに日本語が含まれている場合**:
```
プロジェクトをASCII文字のみのパスに移動
例: D:\切り抜き\xmlai → C:\projects\xmlai
```

### 問題4: pyannote.audioのエラー

**症状**: `ImportError: cannot import name 'Pipeline' from 'pyannote.audio'`

**解決策**:
```bash
pip install --upgrade pyannote.audio
```

### 問題5: CLIP safetensorsエラー

**症状**: `Error loading model with safetensors`

**解決策**: スクリプトが自動的にフォールバックします。問題なし。

---

## データ検証

### 抽出されたデータの確認

```python
import pandas as pd
import numpy as np

# CSVを読み込み
df = pd.read_csv('data/processed/source_features/video_features_enhanced.csv')

# 基本情報
print(f"Timesteps: {len(df)}")
print(f"Duration: {df['time'].max():.2f}s")
print(f"Features: {len(df.columns)}")

# 特徴量の次元数を確認
audio_cols = [c for c in df.columns if c.startswith(('audio_', 'speaker_', 'pitch', 'spectral', 'mfcc', 'text_', 'telop_'))]
visual_cols = [c for c in df.columns if c.startswith(('scene_', 'visual_', 'saliency_', 'face_', 'clip_'))]
temporal_cols = [c for c in df.columns if c.startswith(('time_', 'cut_', 'position_', 'cumulative_'))]

print(f"\nAudio features: {len(audio_cols)}")
print(f"Visual features: {len(visual_cols)}")
print(f"Temporal features: {len(temporal_cols)}")
print(f"Total: {len(audio_cols) + len(visual_cols) + len(temporal_cols)}")

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

# 欠損値の確認
print(f"\nMissing values: {df.isnull().sum().sum()}")
```

---

## 統合ワークフロー

### 完全なデータ準備パイプライン

```bash
# 1. 動画から特徴量を抽出（並列処理）
python -m src.data_preparation.extract_video_features_parallel \
    --video_dir videos \
    --output_dir data/processed/source_features \
    --n_jobs 4

# 2. Premiere Pro XMLからアクティブラベルを抽出
python -m src.data_preparation.extract_active_labels \
    --xml_dir editxml \
    --feature_dir data/processed/source_features \
    --output_dir data/processed/active_labels

# 3. 時系列特徴量を追加
python scripts/add_temporal_features.py

# 4. K-Fold用データを作成
python scripts/create_combined_data_for_kfold.py

# 5. K-Fold Cross Validationで学習
train_cut_selection_kfold_enhanced.bat
```

---

## 技術的な詳細

### 音声処理

1. **PyDub**: 動画から音声を抽出 (16kHz, mono)
2. **Librosa**: RMS energy計算、VAD、MFCC、ピッチ、スペクトル特徴量
3. **Whisper**: 文字起こし（単語レベルのタイムスタンプ）
4. **pyannote.audio**: 話者埋め込み（192次元）

### 視覚処理

1. **OpenCV**: フレーム読み込み、シーン転換、動き検出
2. **MediaPipe**: 顔検出、ランドマーク、表情分析
3. **CLIP**: 視覚的意味表現（512次元）

### 時系列処理

1. **移動平均**: 短期・中期・長期のトレンド
2. **変化率**: 急激な変化の検出
3. **カットタイミング**: 編集のリズム
4. **CLIP類似度**: 視覚的な連続性
5. **音声・映像変化**: 転換点の検出

### 時間同期

- すべての特徴量は同じタイムスタンプで整列
- 10 FPS (0.1秒刻み) で統一
- Premiere XMLのトラック情報と±0.05秒の精度でマッチング可能

---

## まとめ

✅ **音声特徴量**: 235次元（RMS, VAD, 話者埋め込み, MFCC, ピッチ, Whisper等）

✅ **視覚特徴量**: 543次元（シーン転換, 動き, 顔, CLIP embeddings等）

✅ **時系列特徴量**: 6次元（カットタイミング, 位置, 密度等）

✅ **合計**: 784次元

✅ **出力形式**: CSV (10 FPS)

✅ **GPU対応**: CUDA自動検出

✅ **並列処理**: 複数ファイル同時処理

✅ **時系列特徴量**: 83個の追加特徴量

これで、カット選択モデルの訓練に必要なすべての特徴量を抽出できます！

---

**最終更新**: 2025-12-26  
**バージョン**: 3.0.0（時系列特徴量対応版）
