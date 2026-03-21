# One-Button Training Pipeline - User Guide

## 概要

特徴量抽出からトレーニングまでの全プロセスを1コマンドで実行できる自動化パイプラインです。

## 基本的な使い方

### 1. 全プロセスを実行

```bash
# Pythonスクリプトで実行
python scripts/train_pipeline_onebutton.py

# またはバッチファイルで実行（Windows）
batch\train_pipeline_onebutton.bat
```

これにより、以下のステップが自動的に実行されます:
1. 特徴量抽出（5-10分/動画）
2. ラベル抽出（数秒）- XMLから採用/不採用を抽出
3. 時系列特徴量の追加（数分）
4. データセット作成（数分）
5. モデルトレーニング（1-2時間）

**ラベル抽出について:**
- XMLファイルから元動画の使用範囲（in/out）を抽出
- グラフィックやタイトルなどの非動画クリップは自動的に除外
- 0.1秒間隔でActive(1)/Inactive(0)のラベルを生成
- 元動画の長さは特徴量ファイルから自動取得

### 2. 実行内容を確認（Dry-run）

実際に実行する前に、何が実行されるかを確認できます:

```bash
python scripts/train_pipeline_onebutton.py --dry-run
```

## オプション

### ステップのスキップ

既に完了しているステップをスキップできます:

```bash
# 特徴量抽出をスキップ
python scripts/train_pipeline_onebutton.py --skip-extraction

# ラベル抽出をスキップ
python scripts/train_pipeline_onebutton.py --skip-labels

# 時系列特徴量追加をスキップ
python scripts/train_pipeline_onebutton.py --skip-temporal

# データセット作成をスキップ
python scripts/train_pipeline_onebutton.py --skip-dataset

# 複数のステップをスキップ
python scripts/train_pipeline_onebutton.py --skip-extraction --skip-labels
```

### トレーニングのみ実行

前処理をすべてスキップして、トレーニングのみを実行:

```bash
python scripts/train_pipeline_onebutton.py --only-train
```

### 中断と再開

#### 中断

実行中に `Ctrl+C` を押すと、パイプラインが中断されます。
現在の状態は自動的に保存されます。

#### 再開

前回の状態から再開するには:

```bash
python scripts/train_pipeline_onebutton.py --resume
```

完了済みのステップは自動的にスキップされます。

### ステートのリセット

最初からやり直したい場合:

```bash
python scripts/train_pipeline_onebutton.py --reset
```

### 詳細なログ

より詳細なログを表示:

```bash
python scripts/train_pipeline_onebutton.py --verbose
```

## 実行例

### 例1: 初回実行

```bash
# 全プロセスを実行
python scripts/train_pipeline_onebutton.py
```

### 例2: 特徴量は既に抽出済み

```bash
# 特徴量抽出をスキップして実行
python scripts/train_pipeline_onebutton.py --skip-extraction
```

### 例3: トレーニングのみ再実行

```bash
# トレーニングのみを実行
python scripts/train_pipeline_onebutton.py --only-train
```

### 例4: 途中で中断した場合

```bash
# 1. 実行中に Ctrl+C で中断
# 2. 再開
python scripts/train_pipeline_onebutton.py --resume
```

### 例5: 実行内容を確認してから実行

```bash
# 1. 実行内容を確認
python scripts/train_pipeline_onebutton.py --dry-run

# 2. 問題なければ実行
python scripts/train_pipeline_onebutton.py
```

## 出力ファイル

### ステート管理

- `outputs/pipeline_state.json` - パイプラインの実行状態
- `outputs/pipeline_log.txt` - 詳細なログ

### トレーニング結果

- `checkpoints_cut_selection_fullvideo/best_model.pth` - 最良モデル
- `checkpoints_cut_selection_fullvideo/training_history.csv` - トレーニング履歴
- `checkpoints_cut_selection_fullvideo/training_progress.png` - 進捗グラフ

## トラブルシューティング

### 問題1: スクリプトが見つからない

**症状:**
```
Script not found: ...
```

**解決策:**
プロジェクトルートから実行していることを確認してください。

```bash
# 正しい実行方法
cd /path/to/xmlai
python scripts/train_pipeline_onebutton.py
```

### 問題2: ステートファイルが壊れた

**症状:**
```
JSONDecodeError: ...
```

**解決策:**
ステートをリセットしてください。

```bash
python scripts/train_pipeline_onebutton.py --reset
```

### 問題3: 途中で失敗した

**症状:**
エラーメッセージが表示されてパイプラインが停止

**解決策:**
1. エラーメッセージを確認
2. 問題を修正
3. `--resume` で再開

```bash
python scripts/train_pipeline_onebutton.py --resume
```

### 問題4: メモリ不足

**症状:**
```
CUDA out of memory
```

**解決策:**
1. 並列処理数を減らす（既存スクリプトの設定）
2. バッチサイズを減らす（設定ファイル）
3. 一部のステップを個別に実行

## 従来の手動実行との比較

### 従来の方法（5つのステップを手動実行）

```bash
# 1. 特徴量抽出
python -m src.data_preparation.extract_video_features_parallel \
    --video_dir videos \
    --output_dir data/processed/source_features \
    --n_jobs 4

# 2. ラベル抽出
python scripts/extract_active_labels.py

# 3. 時系列特徴量追加
python scripts/add_temporal_features.py

# 4. データセット作成
python scripts/create_cut_selection_data_enhanced_fullvideo.py

# 5. トレーニング
batch/train_fullvideo.bat
```

### 新しい方法（1コマンド）

```bash
# 全て自動実行
python scripts/train_pipeline_onebutton.py
```

## 利点

1. **手動操作の削減**: 5つのステップを1コマンドで実行
2. **エラーハンドリング**: 失敗時に適切なエラーメッセージを表示
3. **中断と再開**: いつでも中断して、後から再開可能
4. **進捗の可視化**: 各ステップの進捗が明確に表示
5. **柔軟性**: 必要なステップだけを実行可能

## ヘルプ

全てのオプションを確認:

```bash
python scripts/train_pipeline_onebutton.py --help
```

## 関連ドキュメント

- [README.md](../README.md) - プロジェクト全体の説明
- [Spec Document](.kiro/specs/one-button-training-pipeline/README.md) - 技術仕様
- [Implementation Guide](.kiro/specs/one-button-training-pipeline/implementation-guide.md) - 実装ガイド
