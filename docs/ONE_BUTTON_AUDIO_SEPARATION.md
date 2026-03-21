# ワンボタンパイプラインでの音声分離

ワンボタントレーニングパイプラインで音声分離を簡単に有効化できます。

## 概要

音声分離機能をワンボタンパイプラインに統合することで、特徴量抽出時に自動的にゲーム音と実況者の声を分離し、Whisper音声認識の精度を向上させます。

## 使用方法

### 基本的な使用（音声分離なし）

```bash
# 通常のパイプライン実行
python scripts/train_pipeline_onebutton.py

# または Windows バッチファイル
batch\train_pipeline_onebutton.bat
```

### 音声分離を有効化

```bash
# 音声分離を有効化（推奨）
python scripts/train_pipeline_onebutton.py --enable-audio-separation

# または Windows バッチファイル
batch\train_pipeline_onebutton_with_audio_separation.bat
```

### 品質プリセットの選択

```bash
# 高速モード（2-3分/10分動画）
python scripts/train_pipeline_onebutton.py --enable-audio-separation --audio-separation-quality fast

# バランスモード（3-5分/10分動画、推奨）
python scripts/train_pipeline_onebutton.py --enable-audio-separation --audio-separation-quality balanced

# 高品質モード（5-10分/10分動画）
python scripts/train_pipeline_onebutton.py --enable-audio-separation --audio-separation-quality high
```

## 効果

### WER（Word Error Rate）改善

音声分離により、Whisper音声認識の精度が向上します：

- **ゲーム実況**: 10-30%改善
- **チュートリアル**: 5-15%改善
- **インタビュー**: 5-10%改善

### 処理時間

音声分離を有効化すると、特徴量抽出の処理時間が増加します：

| 品質プリセット | 追加時間（10分動画） | 合計時間 |
|--------------|-------------------|---------|
| fast | +2-3分 | 7-13分 |
| balanced | +3-5分 | 8-15分 |
| high | +5-10分 | 10-20分 |

## オプション組み合わせ

### 音声分離 + レジューム

```bash
# 中断した場合、音声分離を有効化して再開
python scripts/train_pipeline_onebutton.py --enable-audio-separation --resume
```

### 音声分離 + 特徴量抽出のみ

```bash
# 特徴量抽出のみ実行（音声分離あり）
python scripts/train_pipeline_onebutton.py --enable-audio-separation --only-train
```

### 実行計画の確認

```bash
# 音声分離を有効化した場合の実行計画を確認
python scripts/train_pipeline_onebutton.py --enable-audio-separation --dry-run
```

## 環境変数

ワンボタンパイプラインは内部で以下の環境変数を設定します：

```bash
ENABLE_AUDIO_SEPARATION=true
AUDIO_SEPARATION_QUALITY=balanced  # または fast, high
```

これらは特徴量抽出スクリプト（`extract_video_features_parallel.py`）で自動的に読み込まれます。

## トラブルシューティング

### メモリ不足エラー

高品質モードでメモリ不足が発生する場合：

```bash
# 高速モードに変更
python scripts/train_pipeline_onebutton.py --enable-audio-separation --audio-separation-quality fast
```

### 処理が遅い

GPUが利用可能か確認：

```bash
python -c "import torch; print(torch.cuda.is_available())"
```

GPUが利用できない場合、高速モードを使用：

```bash
python scripts/train_pipeline_onebutton.py --enable-audio-separation --audio-separation-quality fast
```

### 音声分離が失敗する

音声分離が失敗した場合、自動的に元の音声にフォールバックします。ログを確認してください：

```
WARNING: Audio separation failed: ...
Falling back to original audio
```

## 推奨設定

### ゲーム実況動画

```bash
python scripts/train_pipeline_onebutton.py --enable-audio-separation --audio-separation-quality balanced
```

### チュートリアル動画

```bash
python scripts/train_pipeline_onebutton.py --enable-audio-separation --audio-separation-quality balanced
```

### インタビュー動画

```bash
python scripts/train_pipeline_onebutton.py --enable-audio-separation --audio-separation-quality high
```

## 詳細情報

音声分離機能の詳細については、以下のドキュメントを参照してください：

- [Audio Separation Guide](AUDIO_SEPARATION_GUIDE.md) - 音声分離機能の詳細
- [Whisper Enhancement Guide](WHISPER_ENHANCEMENT_GUIDE.md) - Whisper拡張機能
- [One Button Training Guide](ONE_BUTTON_TRAINING_GUIDE.md) - ワンボタンパイプラインの詳細

## まとめ

ワンボタンパイプラインで音声分離を有効化することで、簡単にWhisper音声認識の精度を向上させることができます。

**推奨コマンド**:
```bash
python scripts/train_pipeline_onebutton.py --enable-audio-separation
```

これにより、特徴量抽出時に自動的に音声分離が実行され、より正確な文字起こしが得られます。
