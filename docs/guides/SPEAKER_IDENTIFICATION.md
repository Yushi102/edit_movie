# 話者識別機能

## 概要

このプロジェクトでは、pyannote.audioを使用した話者識別機能を実装しています。これにより、100人以上のキャラクター立ち絵を自動的にマッチングできます。

## 仕組み

### 1. 話者埋め込みの抽出

- **モデル**: speechbrain/spkrec-ecapa-voxceleb
- **次元数**: 192次元
- **処理**:
  1. 音声から発話区間を検出（VADベース、0.5秒以上）
  2. 各発話区間から192次元の話者埋め込みを抽出
  3. コサイン類似度でクラスタリング（閾値0.7）
  4. 各タイムステップに話者IDと埋め込みを割り当て

### 2. 特徴量の構成

**音声特徴量（199次元）**:
- `audio_energy_rms`: 音声エネルギー（1次元）
- `audio_is_speaking`: 発話フラグ（1次元）
- `silence_duration_ms`: 無音時間（1次元）
- `speaker_id`: 話者ID（1次元、0-based、-1=不明）
- `speaker_emb_0~191`: 話者埋め込み（192次元）
- `text_is_active`: テキストアクティブフラグ（1次元）
- `telop_active`: テロップアクティブフラグ（1次元）

### 3. モデルの学習

モデルは以下を学習します：
- 話者埋め込み（256次元） → キャラクター画像の選択
- 「この話者の声 → この立ち絵」という対応関係

### 4. 推論時の動作

1. 入力動画から音声を抽出
2. 話者埋め込みを抽出
3. モデルが埋め込みから適切なキャラクター画像を予測
4. XMLに反映

## インストール

```bash
pip install pyannote.audio
```

**注意**: 一部のモデルはHugging Faceのトークンが必要な場合があります。

## 設定

### config_multimodal_experiment.yaml

```yaml
# Feature dimensions
audio_features: 199  # 4 base + 1 speaker_id + 192 speaker_emb + 2 flags

# Character Selection Strategy
max_asset_classes: 1  # Asset ID disabled
```

### extract_video_features_parallel.py

```python
# Speaker Identification
SPEAKER_EMBEDDING_MODEL = "speechbrain/spkrec-ecapa-voxceleb"  # 192次元
SPEAKER_EMBEDDING_DIM = 192  # 埋め込みの次元数
ENABLE_SPEAKER_ID = True  # 話者識別を有効化
```

## 使用方法

### 1. 特徴量抽出

```bash
python -m src.data_preparation.extract_video_features_parallel data/videos --output-dir data/processed/input_features
```

出力される特徴量CSVには以下が含まれます：
- `speaker_id`: 話者ID（0, 1, 2, ...）
- `speaker_emb_0` ~ `speaker_emb_191`: 話者埋め込み

### 2. データ前処理

```bash
python -m src.data_preparation.data_preprocessing
```

### 3. 学習

```bash
python -m src.training.train --config configs/config_multimodal_experiment.yaml
```

### 4. 推論

```bash
python -m src.inference.inference_pipeline input_video.mp4 output.xml
```

## 動作確認

### 話者識別が正しく動作しているか確認

特徴量CSVを確認：

```python
import pandas as pd

df = pd.read_csv('data/processed/input_features/video_001_features.csv')

# 話者IDの分布を確認
print(df['speaker_id'].value_counts())

# 話者埋め込みの統計
speaker_emb_cols = [f'speaker_emb_{i}' for i in range(192)]
print(df[speaker_emb_cols].describe())
```

期待される結果：
- `speaker_id`: 0, 1, 2, ... （動画内の話者数に応じて）
- `-1`: 発話がない区間
- 話者埋め込み: 非ゼロの値（話者識別が有効な場合）

## トラブルシューティング

### pyannote.audioがインストールできない

```bash
pip install pyannote.audio --upgrade
```

### 話者埋め込みがすべてゼロ

**原因**:
- pyannote.audioが正しくインストールされていない
- モデルのダウンロードに失敗
- GPU/CPUの互換性問題
- **Windows: シンボリックリンクの権限エラー**

**対処法**:

#### 方法1: 管理者権限で実行（推奨）
1. VSCodeを管理者として実行
2. または、コマンドプロンプトを管理者として開く
3. 特徴量抽出を実行

#### 方法2: 開発者モードを有効化
1. Windows設定 → 更新とセキュリティ → 開発者向け
2. 「開発者モード」をオンにする
3. PCを再起動
4. 特徴量抽出を再実行

#### 方法3: 話者識別を無効化
`src/data_preparation/extract_video_features_parallel.py`で：
```python
ENABLE_SPEAKER_ID = False  # 話者識別を無効化
```

この場合、音声特徴量は7次元になります（話者埋め込みなし）

### 話者IDが-1ばかり

**原因**:
- 音声に発話がない
- VAD閾値が高すぎる
- 発話区間が短すぎる（0.5秒未満）

**対処法**:
1. 音声を確認
2. `vad_threshold`を調整（デフォルト: 0.01）
3. `min_duration`を調整（デフォルト: 0.5秒）

### 話者が正しく分離されない

**原因**:
- クラスタリング閾値が不適切
- 話者の声が似ている

**対処法**:
1. `threshold`を調整（デフォルト: 0.7）
   - 高くする（0.8-0.9）: より厳密に分離
   - 低くする（0.5-0.6）: より緩く分離
2. より高度な話者ダイアライゼーションを使用

## パフォーマンス

### 処理時間

- 10分の動画: 約2-3分（GPU使用時）
- 話者埋め込み抽出: 約30秒
- クラスタリング: 約5秒

### メモリ使用量

- GPU: 約2-3GB
- CPU: 約1-2GB

## 今後の改善

1. **より高度な話者ダイアライゼーション**
   - pyannote.audioの完全なパイプラインを使用
   - オーバーラップ発話の処理

2. **感情検出との統合**
   - 話者の感情を検出
   - 適切な表情の立ち絵を選択

3. **CLIP特徴量との統合**
   - 立ち絵の視覚的特徴も考慮
   - より正確なマッチング

4. **リアルタイム処理**
   - ストリーミング対応
   - オンライン話者識別

## 参考資料

- [pyannote.audio](https://github.com/pyannote/pyannote-audio)
- [SpeechBrain](https://speechbrain.github.io/)
- [ECAPA-TDNN](https://arxiv.org/abs/2005.07143)
