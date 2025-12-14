# 動画編集AI - 自動編集パラメータ予測システム

動画から自動的に編集パラメータ（カット位置、ズーム、クロップ、テロップ）を予測し、Premiere Pro用のXMLを生成するAIシステムです。

## 🎯 機能

- **マルチモーダル学習**: 音声・映像・トラックの3つのモダリティを統合
- **自動カット検出**: AIが最適なカット位置を予測
- **音声同期カット**: 映像と音声を同じ位置で自動カット
- **テロップ自動配置**: OCRでテロップを検出し、Premiere Pro互換のグラフィックとして出力
- **Premiere Pro連携**: 生成されたXMLをそのままPremiere Proで開ける

## 📁 プロジェクト構造

```
xmlai/
├── src/                          # ソースコード
│   ├── data_preparation/         # データ準備
│   ├── model/                    # モデル定義
│   ├── training/                 # 学習
│   ├── inference/                # 推論
│   └── utils/                    # ユーティリティ
├── tests/                        # テストコード
├── configs/                      # 設定ファイル
├── docs/                         # ドキュメント
├── data/                         # データ
├── models/                       # 学習済みモデル
├── outputs/                      # 出力ファイル
├── scripts/                      # 補助スクリプト
└── archive/                      # アーカイブ
```

## 🚀 クイックスタート

### 必要な環境
- Python 3.8+
- CUDA対応GPU（推奨）
- Premiere Pro（XML読み込み用）

### インストール
```bash
pip install -r requirements.txt
```

### 新しい動画を自動編集（超簡単！）

**方法1: バッチファイルを使う（推奨）**
```bash
run_inference.bat "D:\videos\my_video.mp4"
```

**方法2: 手動で実行**
```bash
# Pythonパスを設定
set PYTHONPATH=%PYTHONPATH%;%CD%\src

# 1. 推論実行
python src/inference/inference_pipeline.py "your_video.mp4" ^
    --model models/checkpoints_50epochs/best_model.pth ^
    --output outputs/inference_results/temp.xml

# 2. テロップをグラフィックに変換
python src/inference/fix_telop_simple.py ^
    outputs/inference_results/temp.xml ^
    outputs/inference_results/final.xml

# 3. Premiere Proで final.xml を開く
```

詳しくは [QUICK_START.md](QUICK_START.md) を参照してください。

## 📚 ドキュメント

- [プロジェクト全体の流れ](docs/guides/PROJECT_WORKFLOW_GUIDE.md)
- [必要なファイル一覧](docs/guides/REQUIRED_FILES_BY_PHASE.md)
- [音声カット & テロップ変換](docs/summaries/AUDIO_CUT_AND_TELOP_GRAPHICS_SUMMARY.md)

## 🔧 開発

### データ準備
```bash
# XMLからラベル抽出
python src/data_preparation/premiere_xml_parser.py

# 動画から特徴量抽出
python src/data_preparation/extract_video_features_parallel.py

# データ統合
python src/data_preparation/data_preprocessing.py
```

### 学習
```bash
python src/training/training.py --config configs/config_multimodal.yaml
```

### テスト
```bash
# ユニットテスト
pytest tests/unit/

# 統合テスト
pytest tests/integration/
```

## 📊 性能

- **学習データ**: 約10本の編集済み動画
- **学習時間**: 50エポック（約2-3時間、GPU使用時）
- **推論時間**: 約30秒/動画（特徴量抽出含む）
- **カット数**: 約500個/動画（閾値0.29の場合）

## 🤝 貢献

プルリクエストを歓迎します！

## 📝 ライセンス

MIT License


