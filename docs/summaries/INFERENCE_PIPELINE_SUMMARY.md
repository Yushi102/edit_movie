# 推論パイプライン改善まとめ

## 完了した改善

### 1. 特徴量抽出の統合 ✅
- `extract_video_features.py`の`FeatureExtractor`クラスを直接インポート
- サブプロセス呼び出しから直接関数呼び出しに変更
- 音声・映像特徴量を自動的に分割

### 2. 正規化パラメータの自動読み込み ✅
- トレーニング時に保存した前処理器ファイルを自動検出
- モデルと同じディレクトリから`audio_preprocessor.pkl`と`visual_preprocessor.pkl`を読み込み
- マルチモーダルモデルの場合のみ前処理器を適用

### 3. 前処理の適用 ✅
- 音声特徴量: RMSとsilence_durationを正規化
- 映像特徴量: スカラー特徴を正規化、CLIP embeddingsをL2正規化
- face_countに基づいて顔特徴をゼロ埋め

### 4. XML生成の改善 ✅
- `csv2xml3.py`のパターンを参考に実装
- `xml.etree.ElementTree`を使用して構造化されたXML生成
- モーションエフェクト（スケール、ポジション）を追加
- クロップエフェクト（上下左右）を追加
- 縦長動画対応（1080x1920）

### 5. 古いモデルとの互換性 ✅
- `config`キーがない古いモデルに対応
- `state_dict`から入力次元を自動推測
- 180次元（古いモデル）と240次元（新しいモデル）の両方に対応

## 使用方法

### 基本的な使い方

```bash
python inference_pipeline.py <動画ファイル> --model checkpoints/best_model.pth --output output.xml
```

### オプション

- `--model`: 学習済みモデルのパス（デフォルト: `checkpoints_experiment/best_model.pth`）
- `--output`: 出力XMLファイルのパス（デフォルト: `{video_name}_edited.xml`）
- `--device`: 使用デバイス（`cpu` or `cuda`、デフォルト: `cpu`）
- `--fps`: フレームレート（デフォルト: 10.0）
- `--num_tracks`: トラック数（デフォルト: 20）

### 例

```bash
# 基本的な使用
python inference_pipeline.py my_video.mp4

# GPUを使用
python inference_pipeline.py my_video.mp4 --device cuda

# カスタム出力パス
python inference_pipeline.py my_video.mp4 --output custom_edit.xml

# 異なるFPS
python inference_pipeline.py my_video.mp4 --fps 30.0
```

## パイプラインの流れ

1. **特徴量抽出**
   - 動画から音声・映像特徴量を抽出
   - Whisper（音声認識）、CLIP（映像理解）、MediaPipe（顔検出）を使用

2. **前処理とアライメント**
   - 特徴量を正規化
   - タイムスタンプを揃える
   - モダリティマスクを生成

3. **モデル予測**
   - 学習済みモデルで編集パラメータを予測
   - active, asset_id, scale, position, cropを出力

4. **XML生成**
   - 予測結果をPremiere Pro XML形式に変換
   - モーションとクロップエフェクトを適用
   - 縦長動画（1080x1920）に対応

## 今後の改善点

### 必須
1. **実際のトレーニングデータから前処理器を保存**
   - 現在はダミーの前処理器を使用
   - `training.py`を修正して、トレーニング時に前処理器を保存する必要がある

2. **トラック特徴量の統合**
   - 現在はダミーのトラック特徴量（ゼロ）を使用
   - 実際の編集履歴があれば、それを入力として使用可能

### オプション
1. **バッチ処理対応**
   - 複数の動画を一度に処理

2. **GPU最適化**
   - 特徴量抽出とモデル推論の並列化

3. **キーフレーム対応**
   - 時間変化するパラメータをキーフレームとして出力

4. **プレビュー機能**
   - 編集結果をプレビュー表示

## 参考ファイル

- `csv2xml3.py`: XML生成のパターン
- `inference/2inference.py`: 推論パイプラインの基本構造
- `extract_video_features.py`: 特徴量抽出
- `multimodal_preprocessing.py`: 前処理
- `feature_alignment.py`: アライメント
- `model_persistence.py`: モデルの保存・読み込み

## テスト

```bash
# パイプラインの初期化テスト
python test_inference_pipeline.py

# 実際の動画で推論テスト（動画ファイルが必要）
python inference_pipeline.py path/to/video.mp4 --output test_output.xml
```

## 注意事項

1. **前処理器ファイル**
   - `checkpoints/audio_preprocessor.pkl`
   - `checkpoints/visual_preprocessor.pkl`
   - これらのファイルは実際のトレーニングデータから生成する必要があります

2. **モデルの互換性**
   - 古いモデル（180次元入力）と新しいモデル（240次元入力）の両方に対応
   - マルチモーダルモデルとトラックオンリーモデルの両方に対応

3. **特徴量抽出の時間**
   - Whisper、CLIP、MediaPipeを使用するため、処理に時間がかかります
   - GPUを使用すると高速化できます

4. **XML形式**
   - Premiere Pro XMEML version 5形式
   - 縦長動画（1080x1920）に最適化
   - 必要に応じてシーケンス設定を変更可能
