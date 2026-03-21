# Audio Separation for Whisper Enhancement

音声分離機能により、ゲーム音声と実況者の声を分離してWhisper音声認識の精度を向上させます。

## 概要

ゲーム実況動画では、ゲーム音（BGM、効果音）と実況者の声が混在しているため、Whisperの音声認識精度が低下することがあります。この機能は、Demucsを使用して音声を分離し、実況者の声のみをWhisperに入力することで、認識精度を向上させます。

## 主な機能

- **音声分離**: Demucsを使用してゲーム音と実況者の声を分離
- **キャッシング**: 分離済み音声をキャッシュして再処理を高速化
- **品質評価**: SDR（Signal-to-Distortion Ratio）で分離品質を評価
- **自動フォールバック**: 分離失敗時は元の音声を使用
- **環境変数制御**: 簡単なON/OFF切り替え

## セットアップ

### 1. 依存関係のインストール

```bash
pip install demucs museval
```

### 2. 設定ファイルの確認

`configs/config_audio_separation.yaml` で設定を確認・変更できます：

```yaml
# 音声分離を有効化
enabled: false  # true に変更して有効化

# モデル設定
model:
  type: "demucs"
  quality: "balanced"  # fast, balanced, high
  device: "auto"       # auto, cuda, cpu

# キャッシュ設定
cache:
  directory: "preprocessed_data/audio_cache"
  enabled: true
  max_age_days: 30

# 品質閾値
quality:
  min_sdr: 5.0  # 最小SDR（デシベル）
```

## 使用方法

### 環境変数で有効化

最も簡単な方法は環境変数を設定することです：

```bash
# Windows (PowerShell)
$env:ENABLE_AUDIO_SEPARATION="true"

# Windows (CMD)
set ENABLE_AUDIO_SEPARATION=true

# Linux/Mac
export ENABLE_AUDIO_SEPARATION=true
```

その後、通常通り特徴量抽出を実行：

```bash
python src/data_preparation/extract_video_features_parallel.py --video_dir data/videos --output_dir preprocessed_data
```

### 設定ファイルで有効化

`configs/config_audio_separation.yaml` の `enabled` を `true` に変更：

```yaml
enabled: true
```

### 品質プリセット

処理速度と品質のトレードオフを選択できます：

| プリセット | モデル | 処理時間（10分動画） | 品質 | 推奨用途 |
|----------|--------|-------------------|------|---------|
| fast | mdx | 2-3分 | 良 | テスト、プロトタイピング |
| balanced | htdemucs | 3-5分 | 優 | 通常使用（推奨） |
| high | htdemucs_ft | 5-10分 | 最高 | 最終処理、高品質要求 |

環境変数で変更：

```bash
$env:AUDIO_SEPARATION_QUALITY="high"
```

## パフォーマンス

### 処理時間

- **GPU使用時**: 2-5分/10分動画（balanced）
- **CPU使用時**: 10-20分/10分動画（balanced）

### メモリ使用量

- **GPU**: 2-4GB VRAM
- **CPU**: 2-4GB RAM

### キャッシュ効果

初回処理後、同じ動画の再処理は数秒で完了します。

## 期待される効果

### WER（Word Error Rate）改善

- **ゲーム実況**: 10-30%改善
- **チュートリアル**: 5-15%改善
- **インタビュー**: 5-10%改善

### SDR（Signal-to-Distortion Ratio）

- **良好**: SDR > 10dB
- **許容**: SDR 5-10dB
- **不良**: SDR < 5dB（元の音声にフォールバック推奨）

## トラブルシューティング

### 音声分離が遅い

1. GPUを使用していることを確認：
   ```bash
   python -c "import torch; print(torch.cuda.is_available())"
   ```

2. 品質プリセットを下げる：
   ```bash
   $env:AUDIO_SEPARATION_QUALITY="fast"
   ```

### 分離品質が低い

1. 品質プリセットを上げる：
   ```bash
   $env:AUDIO_SEPARATION_QUALITY="high"
   ```

2. SDRを確認してログを確認：
   ```
   Separation complete: SDR=8.5dB, time=3.2s
   ```

### メモリ不足エラー

1. 品質プリセットを下げる
2. CPUモードを使用：
   ```bash
   $env:AUDIO_SEPARATION_DEVICE="cpu"
   ```

### 分離が失敗する

自動的に元の音声にフォールバックします。ログを確認：
```
WARNING: Audio separation failed: ...
Falling back to original audio
```

## キャッシュ管理

### キャッシュの場所

デフォルト: `preprocessed_data/audio_cache/`

### キャッシュのクリア

古いキャッシュファイルを削除：

```python
from src.data_preparation.audio_separator import AudioSeparator

separator = AudioSeparator()
separator.clear_cache(max_age_days=30)  # 30日以上古いファイルを削除
```

### キャッシュの無効化

```bash
# 環境変数で無効化
$env:AUDIO_CACHE_ENABLED="false"
```

または設定ファイルで：

```yaml
cache:
  enabled: false
```

## 環境変数リファレンス

| 変数名 | デフォルト | 説明 |
|-------|----------|------|
| ENABLE_AUDIO_SEPARATION | false | 音声分離の有効化 |
| AUDIO_SEPARATION_MODEL | demucs | 分離モデル |
| AUDIO_SEPARATION_QUALITY | balanced | 品質プリセット（fast/balanced/high） |
| AUDIO_SEPARATION_DEVICE | auto | デバイス（auto/cuda/cpu） |
| AUDIO_CACHE_DIR | preprocessed_data/audio_cache | キャッシュディレクトリ |

## コンテンツタイプ別の推奨設定

### ゲーム実況

```yaml
presets:
  gaming:
    model: "demucs"
    quality: "balanced"
    min_sdr: 5.0
```

### チュートリアル

```yaml
presets:
  tutorial:
    model: "demucs"
    quality: "balanced"
    min_sdr: 6.0
```

### インタビュー

```yaml
presets:
  interview:
    model: "demucs"
    quality: "high"
    min_sdr: 8.0
```

## API使用例

### 基本的な使用

```python
from src.data_preparation.audio_separator import AudioSeparator

# 初期化
separator = AudioSeparator(
    model='demucs',
    quality='balanced',
    device='auto'
)

# 音声分離
clean_voice_path, metrics = separator.separate('path/to/audio.wav')

print(f"SDR: {metrics['sdr']:.2f}dB")
print(f"Processing time: {metrics['processing_time']:.1f}s")
print(f"Model used: {metrics['model_used']}")
print(f"Cache hit: {metrics['cache_hit']}")
```

### カスタム設定

```python
separator = AudioSeparator(
    model='demucs',
    quality='high',
    cache_dir='custom_cache',
    device='cuda',
    config_path='custom_config.yaml'
)
```

### バッチ処理

```python
import glob

separator = AudioSeparator()

for audio_file in glob.glob('data/audio/*.wav'):
    try:
        clean_path, metrics = separator.separate(audio_file)
        print(f"{audio_file}: SDR={metrics['sdr']:.2f}dB")
    except Exception as e:
        print(f"{audio_file}: Failed - {e}")
```

## 技術詳細

### Demucsモデル

- **mdx**: 軽量モデル、高速処理
- **htdemucs**: 標準モデル、バランス型
- **htdemucs_ft**: ファインチューニング済み、最高品質

### 分離プロセス

1. 音声ファイルをロード
2. ステレオに変換（Demucs要件）
3. Demucsで4トラックに分離（drums, bass, other, vocals）
4. vocalsトラックを抽出
5. モノラルに変換
6. 品質評価（SDR計算）
7. キャッシュに保存

### SDR計算

```
SDR = 10 * log10(signal_power / distortion_power)
```

- SDR > 10dB: 優秀な分離
- SDR 5-10dB: 良好な分離
- SDR < 5dB: 不十分な分離

## 制限事項

- **Python 3.11+**: Spleeterは非対応のため、Demucsのみ使用可能
- **GPU推奨**: CPU処理は5-10倍遅い
- **メモリ**: 長時間動画（>30分）は大量のメモリを消費
- **品質**: 音声が非常に小さい場合、分離効果が限定的

## 参考資料

- [Demucs GitHub](https://github.com/facebookresearch/demucs)
- [Whisper Enhancement Guide](WHISPER_ENHANCEMENT_GUIDE.md)
- [Text Analysis Features](TEXT_ANALYSIS_FEATURES.md)

## サポート

問題が発生した場合は、以下の情報を含めてIssueを作成してください：

1. エラーメッセージ
2. 使用した設定（環境変数、config.yaml）
3. 動画の長さと形式
4. システム情報（GPU/CPU、メモリ）
5. ログ出力
