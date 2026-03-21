# Excitement Feature Guide

音声内容の盛り上がり度検出機能のガイド

## 概要

この機能は、Whisperで文字起こしした音声内容を分析し、「盛り上がり度」を特徴量として抽出します。感情の高まり、トピックの変化、発話パターンなどから、動画内の盛り上がっているシーンを検出します。

## 特徴量の種類

### 1. Transformer埋め込み（768次元）

日本語BERTモデル（cl-tohoku/bert-base-japanese-v3）を使用して、音声内容を意味的に理解します。

- **カラム名**: `speech_embedding_0` ~ `speech_embedding_767`
- **値の範囲**: 実数値（正規化済み）
- **用途**: 音声内容の意味的な類似性を捉える

### 2. 基本統計特徴量（5次元）

- **speech_presence**: 発話の有無（0 or 1）
- **cumulative_speech_count**: 累積発話数
- **time_since_last_speech**: 最後の発話からの時間（秒）
- **speech_text_length**: テキストの文字数
- **speech_density_10s**: 10秒間の発話密度（文字/秒）

### 3. 同時発話カウント（1次元）

- **simultaneous_speech_count**: 同時に発話しているセグメント数

### 4. 感情ベース特徴量（5次元）

- **positive_emotion_intensity**: ポジティブ感情の強度（0-1）
- **excited_emotion_intensity**: 興奮感情の強度（0-1）
- **emotion_change_rate**: 感情変化率
- **laughter_density**: 笑い密度（0-1）
- **emotion_variance_10s**: 10秒間の感情分散

### 5. トピック変化特徴量（5次元）

- **topic_change_rate**: トピック変化率
- **climax_keyword_density**: クライマックスキーワード密度（0-1）
- **semantic_similarity**: 連続セグメント間の意味的類似度（0-1）
- **topic_shift_intensity**: トピックシフト強度（0-1）
- **climax_score**: クライマックススコア（0-1）

### 6. 発話パターン特徴量（5次元）

- **speech_burst_intensity**: 発話バースト強度（0-1）
- **speech_pause_frequency**: ポーズ頻度（0-1）
- **speech_rhythm_variance**: リズム分散
- **speech_acceleration**: 発話加速度（0-1）
- **burst_pattern_score**: バーストパターンスコア（0-1）

## 使用例

### 基本的な使用方法

```python
from src.data_preparation.excitement_detector import ExcitementDetector

# 初期化
detector = ExcitementDetector()

# Whisper文字起こしセグメント
segments = [
    {"start": 0.0, "end": 2.5, "text": "やばい！すごい！"},
    {"start": 3.0, "end": 5.0, "text": "マジで最高！笑笑笑"}
]

# 特徴量生成
features_df = detector.generate_features(
    transcription_segments=segments,
    video_duration=10.0,
    sampling_rate=0.1
)

print(features_df.shape)  # (101, 790) - time + 789 features
print(features_df.columns)  # ['time', 'speech_embedding_0', ...]
```

### カスタム設定

```python
config = {
    "model_name": "cl-tohoku/bert-base-japanese-v3",
    "sampling_rate": 0.1,
    "use_gpu": True
}

detector = ExcitementDetector(config=config)
```

## 特徴量の値の例

### 盛り上がっているシーン

```
time: 5.0
speech_presence: 1.0
excited_emotion_intensity: 0.85
climax_keyword_density: 0.75
laughter_density: 0.60
climax_score: 0.78
speech_burst_intensity: 0.70
```

### 通常のシーン

```
time: 10.0
speech_presence: 1.0
excited_emotion_intensity: 0.15
climax_keyword_density: 0.05
laughter_density: 0.00
climax_score: 0.12
speech_burst_intensity: 0.10
```

### 無音シーン

```
time: 15.0
speech_presence: 0.0
excited_emotion_intensity: 0.00
climax_keyword_density: 0.00
laughter_density: 0.00
climax_score: 0.00
speech_burst_intensity: 0.00
```

## Transformerモデル

### 使用モデル

- **モデル名**: cl-tohoku/bert-base-japanese-v3
- **埋め込み次元**: 768
- **言語**: 日本語
- **ライセンス**: Apache 2.0

### モデルの特徴

- 日本語に特化したBERTモデル
- 形態素解析器（MeCab + IPAdic）を使用
- 高品質な日本語テキスト理解

## トラブルシューティング

### 1. Transformerモデルのロードに失敗する

**症状**: `RuntimeError: Transformer model initialization failed`

**原因**: PyTorchのバージョンが古い、またはモデルのダウンロードに失敗

**解決策**:
```bash
# PyTorchを更新（推奨: 2.6以上）
pip install --upgrade torch

# または、モデルを手動でダウンロード
python -c "from transformers import AutoModel; AutoModel.from_pretrained('cl-tohoku/bert-base-japanese-v3')"
```

### 2. GPU Out of Memory

**症状**: `torch.cuda.OutOfMemoryError`

**原因**: GPUメモリ不足

**解決策**:
- バッチサイズを小さくする
- CPUを使用する（`use_gpu=False`）
- より小さいモデルを使用する

### 3. 文字起こしデータがない

**症状**: 警告メッセージ「No transcription segments provided」

**原因**: Whisper文字起こしが実行されていない

**解決策**:
```bash
# Whisper文字起こしを実行
python scripts/extract_text_features.py --video_path <video_path>
```

### 4. 特徴量の次元数が合わない

**症状**: `ValueError: Feature dimension mismatch`

**原因**: 特徴量生成のバグ

**解決策**:
- ログを確認して、どの特徴量が欠けているか確認
- 最新版のコードを使用
- Issueを報告

## パフォーマンス

### 処理速度

- **GPU使用時**: 10分動画あたり約5秒
- **CPU使用時**: 10分動画あたり約30秒

### メモリ使用量

- **GPU**: 約2GB（BERT モデル）
- **CPU**: 約1GB

### キャッシュ効果

- 同じテキストの再エンコードは約100倍高速
- キャッシュサイズ: 10,000エントリ（デフォルト）

## 設定ファイル

設定ファイル（`configs/config_excitement_features.yaml`）で動作をカスタマイズできます：

```yaml
# Transformer設定
model_name: "cl-tohoku/bert-base-japanese-v3"
embedding_dim: 768
batch_size: 32
cache_size: 10000

# 特徴量計算設定
sampling_rate: 0.1  # 秒
emotion_window: 10.0  # 秒
speech_burst_threshold: 20.0  # 文字/秒

# クライマックスキーワード
climax_keywords:
  - "やばい"
  - "すごい"
  - "マジ"
  - "キター"

# パフォーマンス設定
use_gpu: true
num_workers: 4
```

## 既存特徴量との統合

盛り上がり度特徴量は、既存の特徴量と統合されます：

- **音声特徴量**: 215次元
- **映像特徴量**: 522次元
- **テキスト特徴量**: 15次元
- **盛り上がり度特徴量**: 789次元

**合計**: 1541次元

## 参考文献

- [BERT: Pre-training of Deep Bidirectional Transformers](https://arxiv.org/abs/1810.04805)
- [cl-tohoku/bert-base-japanese-v3](https://huggingface.co/cl-tohoku/bert-base-japanese-v3)
- [Whisper: Robust Speech Recognition](https://arxiv.org/abs/2212.04356)
