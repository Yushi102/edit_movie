# 各工程で必要なファイル一覧

## 📋 フェーズ別必要ファイル

---

## 1️⃣ データ準備フェーズ

### 📥 入力として必要なもの
```
editxml/
├── video1.mp4          # 編集済み動画
├── video1.xml          # Premiere ProからエクスポートしたXML
├── video2.mp4
├── video2.xml
└── ...
```

### 🔧 実行に必要なスクリプト

#### 1-1. XMLからラベル抽出
```bash
python premiere_xml_parser.py
```
**必要なファイル**:
- ✅ `premiere_xml_parser.py` - XMLパーサー

**出力**:
- `output_labels/video1_labels.csv`
- `output_labels/video2_labels.csv`

---

#### 1-2. 動画から特徴量抽出
```bash
python extract_video_features_parallel.py
```
**必要なファイル**:
- ✅ `extract_video_features_parallel.py` - 特徴量抽出スクリプト
- ✅ `telop_extractor.py` - テロップ抽出（OCR）
- ✅ `text_embedding.py` - テキスト埋め込み

**出力**:
- `input_features/video1_features.csv`
- `input_features/video2_features.csv`

---

#### 1-3. データ統合
```bash
python data_preprocessing.py
```
**必要なファイル**:
- ✅ `data_preprocessing.py` - データ統合スクリプト
- ✅ `feature_alignment.py` - 特徴量アライメント

**出力**:
- `master_training_data.csv` - 学習用データセット

---

### 📦 データ準備フェーズで必要なファイルまとめ

#### スクリプト (6個)
1. `premiere_xml_parser.py`
2. `extract_video_features_parallel.py`
3. `telop_extractor.py`
4. `text_embedding.py`
5. `data_preprocessing.py`
6. `feature_alignment.py`

#### 入力データ
- `editxml/*.mp4` - 編集済み動画
- `editxml/*.xml` - Premiere Pro XML

#### 出力データ
- `output_labels/*.csv` - 編集パラメータ
- `input_features/*.csv` - 特徴量
- `master_training_data.csv` - 統合データ

---

## 2️⃣ 学習フェーズ

### 🔧 実行に必要なスクリプト

```bash
python training.py --config config_multimodal.yaml
```

### 📦 学習フェーズで必要なファイルまとめ

#### スクリプト (8個)
1. ✅ `training.py` - 学習メインスクリプト
2. ✅ `model.py` - モデル定義
3. ✅ `multimodal_modules.py` - マルチモーダルエンコーダー
4. ✅ `multimodal_dataset.py` - データセット
5. ✅ `multimodal_preprocessing.py` - 前処理（正規化）
6. ✅ `feature_alignment.py` - アライメント
7. ✅ `loss.py` - 損失関数
8. ✅ `model_persistence.py` - モデル保存/読み込み

#### 設定ファイル (1個)
9. ✅ `config_multimodal.yaml` - 学習設定

#### 入力データ
- `master_training_data.csv` - 学習用データセット

#### 出力データ
- `checkpoints_50epochs/best_model.pth` - 学習済みモデル
- `checkpoints_50epochs/audio_preprocessor.pkl` - 音声前処理器
- `checkpoints_50epochs/visual_preprocessor.pkl` - 映像前処理器

---

## 3️⃣ 推論フェーズ（新しい動画の自動編集）

### 🔧 実行に必要なスクリプト

#### 3-1. 推論実行
```bash
python inference_pipeline.py "new_video.mp4" \
    --model checkpoints_50epochs/best_model.pth \
    --output temp.xml
```

#### 3-2. テロップ変換
```bash
python fix_telop_simple.py temp.xml final.xml
```

### 📦 推論フェーズで必要なファイルまとめ

#### スクリプト (11個)
1. ✅ `inference_pipeline.py` - 推論メインスクリプト
2. ✅ `model.py` - モデル定義
3. ✅ `multimodal_modules.py` - マルチモーダルエンコーダー
4. ✅ `multimodal_preprocessing.py` - 前処理（正規化）
5. ✅ `feature_alignment.py` - アライメント
6. ✅ `model_persistence.py` - モデル読み込み
7. ✅ `extract_video_features_parallel.py` - 特徴量抽出
8. ✅ `telop_extractor.py` - テロップ抽出
9. ✅ `text_embedding.py` - テキスト埋め込み
10. ✅ `otio_xml_generator.py` - OTIO XML生成
11. ✅ `fix_telop_simple.py` - テロップ変換

#### 学習済みモデル (3個)
12. ✅ `checkpoints_50epochs/best_model.pth` - 学習済みモデル
13. ✅ `checkpoints_50epochs/audio_preprocessor.pkl` - 音声前処理器
14. ✅ `checkpoints_50epochs/visual_preprocessor.pkl` - 映像前処理器

#### 入力データ
- `new_video.mp4` - 新しい動画（編集したい動画）

#### 出力データ
- `temp.xml` - OTIO生成XML（音声カット済み）
- `final.xml` - Premiere Pro互換XML（完成版）

---

## 📊 全体で必要なファイル一覧

### 🐍 Pythonスクリプト (15個)

#### データ準備用 (6個)
1. `premiere_xml_parser.py`
2. `extract_video_features_parallel.py`
3. `telop_extractor.py`
4. `text_embedding.py`
5. `data_preprocessing.py`
6. `feature_alignment.py`

#### モデル関連 (5個)
7. `model.py`
8. `multimodal_modules.py`
9. `multimodal_dataset.py`
10. `multimodal_preprocessing.py`
11. `model_persistence.py`

#### 学習用 (2個)
12. `training.py`
13. `loss.py`

#### 推論用 (3個)
14. `inference_pipeline.py`
15. `otio_xml_generator.py`
16. `fix_telop_simple.py`

### ⚙️ 設定ファイル (1個)
17. `config_multimodal.yaml`

---

## 🎯 フェーズ別クイックチェックリスト

### ✅ データ準備を始める前に
- [ ] `premiere_xml_parser.py`
- [ ] `extract_video_features_parallel.py`
- [ ] `telop_extractor.py`
- [ ] `text_embedding.py`
- [ ] `data_preprocessing.py`
- [ ] `feature_alignment.py`
- [ ] 編集済み動画とXML（`editxml/`フォルダ内）

### ✅ 学習を始める前に
- [ ] `training.py`
- [ ] `model.py`
- [ ] `multimodal_modules.py`
- [ ] `multimodal_dataset.py`
- [ ] `multimodal_preprocessing.py`
- [ ] `feature_alignment.py`
- [ ] `loss.py`
- [ ] `model_persistence.py`
- [ ] `config_multimodal.yaml`
- [ ] `master_training_data.csv`（データ準備で生成）

### ✅ 推論を始める前に
- [ ] `inference_pipeline.py`
- [ ] `model.py`
- [ ] `multimodal_modules.py`
- [ ] `multimodal_preprocessing.py`
- [ ] `feature_alignment.py`
- [ ] `model_persistence.py`
- [ ] `extract_video_features_parallel.py`
- [ ] `telop_extractor.py`
- [ ] `text_embedding.py`
- [ ] `otio_xml_generator.py`
- [ ] `fix_telop_simple.py`
- [ ] `checkpoints_50epochs/best_model.pth`（学習で生成）
- [ ] `checkpoints_50epochs/audio_preprocessor.pkl`（学習で生成）
- [ ] `checkpoints_50epochs/visual_preprocessor.pkl`（学習で生成）

---

## 🔍 依存関係の確認

### 各スクリプトが依存しているファイル

#### `inference_pipeline.py` の依存関係
```python
from model import create_model
from model_persistence import load_model
from multimodal_preprocessing import AudioFeaturePreprocessor, VisualFeaturePreprocessor
from feature_alignment import FeatureAligner
from extract_video_features_parallel import extract_features_worker
from text_embedding import SimpleTextEmbedder
from otio_xml_generator import create_premiere_xml_with_otio
```
→ 7個のファイルに依存

#### `training.py` の依存関係
```python
from model import create_model
from multimodal_dataset import MultimodalEditDataset
from multimodal_preprocessing import AudioFeaturePreprocessor, VisualFeaturePreprocessor
from loss import EditLoss
from model_persistence import save_model
```
→ 5個のファイルに依存

#### `otio_xml_generator.py` の依存関係
```python
import opentimelineio as otio
import cv2
```
→ 外部ライブラリのみ（他のスクリプトに依存しない）

---

## 📦 必要な外部ライブラリ

```bash
pip install torch torchvision
pip install opencv-python
pip install pandas numpy
pip install opentimelineio
pip install easyocr
pip install transformers
pip install scipy
pip install pyyaml
```

---

## 💡 ファイルが足りない場合

### エラーメッセージから判断
```
ModuleNotFoundError: No module named 'model'
→ model.py が必要

ModuleNotFoundError: No module named 'multimodal_modules'
→ multimodal_modules.py が必要

FileNotFoundError: checkpoints_50epochs/best_model.pth
→ 学習を実行してモデルを生成する必要がある
```

### 最小限で推論だけ実行したい場合
以下の11個のファイルがあればOK:
1. `inference_pipeline.py`
2. `model.py`
3. `multimodal_modules.py`
4. `multimodal_preprocessing.py`
5. `feature_alignment.py`
6. `model_persistence.py`
7. `extract_video_features_parallel.py`
8. `telop_extractor.py`
9. `text_embedding.py`
10. `otio_xml_generator.py`
11. `fix_telop_simple.py`

+ 学習済みモデル3個:
- `checkpoints_50epochs/best_model.pth`
- `checkpoints_50epochs/audio_preprocessor.pkl`
- `checkpoints_50epochs/visual_preprocessor.pkl`
