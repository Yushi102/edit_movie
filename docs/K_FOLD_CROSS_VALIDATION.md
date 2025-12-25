# K-Fold Cross Validation ガイド

## 概要

K-Fold Cross Validation（K分割交差検証）は、モデルの性能をより信頼性高く評価するための手法です。

### 通常の学習との違い

**通常の学習（Train/Val Split）:**
- データを1回だけ学習用と検証用に分割
- 1つのモデルを学習
- 検証データの選び方によって性能が変動する可能性

**K-Fold Cross Validation:**
- データをK個（通常5個）に分割
- K回学習を実行（各回で異なる分割を検証用に使用）
- K個のモデルの平均性能を計算
- より安定した性能評価が可能

### メリット

1. **信頼性の高い評価**: 複数の分割で評価するため、偶然の良い/悪い分割に左右されない
2. **データの有効活用**: 全データが学習と検証の両方に使われる
3. **性能のばらつき確認**: 標準偏差で性能の安定性を評価できる
4. **過学習の検出**: Foldごとの性能差が大きい場合、過学習の可能性

### デメリット

1. **学習時間**: K倍の時間がかかる（5-Foldなら5倍）
2. **計算リソース**: より多くのGPUメモリとストレージが必要

---

## 使い方

### 1. データ準備

K-Foldでは、train と val を結合した1つのデータセットを使用します。

```bash
python scripts/create_combined_data_for_kfold.py
```

これにより `preprocessed_data/combined_sequences_cut_selection.npz` が作成されます。

### 2. 設定ファイルの確認

`configs/config_cut_selection_kfold.yaml` を確認・編集します。

```yaml
# K-Fold設定
n_folds: 5  # 分割数（5が標準）
random_state: 42  # 再現性のためのシード値

# データパス
data_path: preprocessed_data/combined_sequences_cut_selection.npz

# その他の設定は通常の学習と同じ
```

### 3. K-Fold学習の実行

```bash
train_cut_selection_kfold.bat
```

または手動で：

```bash
python src/cut_selection/train_cut_selection_kfold.py --config configs/config_cut_selection_kfold.yaml
```

### 4. 結果の確認

学習が完了すると、以下のファイルが生成されます：

#### モデルファイル
```
checkpoints_cut_selection_kfold/
├── fold_1_best_model.pth  # Fold 1の最良モデル
├── fold_2_best_model.pth  # Fold 2の最良モデル
├── fold_3_best_model.pth  # Fold 3の最良モデル
├── fold_4_best_model.pth  # Fold 4の最良モデル
└── fold_5_best_model.pth  # Fold 5の最良モデル
```

#### 可視化ファイル
```
checkpoints_cut_selection_kfold/
├── kfold_comparison.png   # 全Foldの比較グラフ
├── kfold_summary.csv      # 統計サマリー（CSV）
└── inference_params.yaml  # 推論用パラメータ（平均閾値）
```

---

## 結果の解釈

### kfold_comparison.png

4つのグラフが表示されます：

1. **F1スコアの推移（全Fold）**
   - 各Foldの学習曲線（エポックごと）
   - 過学習の有無を確認

2. **各Foldの最良F1スコア**
   - 棒グラフで各Foldの最高性能を比較
   - 赤い破線が平均値 ± 標準偏差

3. **Precision vs Recall（各Foldの最良値）**
   - 各Foldの最良値を1点ずつプロット
   - 理想的には右上（高Precision & 高Recall）
   - 赤い星印が平均値

4. **最適閾値（各Fold）**
   - 棒グラフで各Foldの最適閾値を表示
   - 赤い破線が平均値 ± 標準偏差

### kfold_summary.csv

各Foldの詳細な統計情報：

```csv
fold,best_epoch,best_val_f1,best_val_accuracy,best_val_precision,best_val_recall,optimal_threshold
1,4,0.49423045131426685,0.7362586206896552,0.3693965304206,0.7465041949660407,-0.5581883788108826
2,1,0.4121643171525165,0.36444827586206896,0.27851293103448277,0.7924331616384597,-0.4739184081554413
3,20,0.4309829669806833,0.4845,0.3093546800721272,0.7102176503794769,-0.5725248456001282
4,9,0.4556681004578882,0.5941724137931035,0.3354329100132784,0.7102588133515969,-0.5092343688011169
5,3,0.32201033683835323,0.33259649122807017,0.19888164846777034,0.8454052030694367,-0.5503854155540466
Mean ± Std,7.4 ± 6.8,0.4230 ± 0.0575,0.5024 ± 0.1492,0.2983 ± 0.0580,0.7610 ± 0.0519,-0.533 ± 0.036
```

### コンソール出力

学習終了時に以下のようなサマリーが表示されます：

```
================================================================================
K-FOLD CROSS VALIDATION SUMMARY
================================================================================

Mean F1 Score: 0.4230 ± 0.0575
Mean Accuracy: 0.5024 ± 0.1492
Mean Precision: 0.2983 ± 0.0580
Mean Recall: 0.7610 ± 0.0519
Mean Optimal Threshold: -0.533 ± 0.036
================================================================================
```

---

## 性能の評価基準

### 良い結果の目安

1. **高い平均F1スコア**: 0.55以上
2. **小さい標準偏差**: ±0.01以下（安定している）
3. **Foldごとの性能差が小さい**: 過学習していない
4. **閾値のばらつきが小さい**: モデルが安定している

### 問題がある場合

1. **標準偏差が大きい（±0.05以上）**
   - データの偏りが大きい
   - モデルが不安定
   - 対策: データ拡張、正則化の強化

2. **特定のFoldだけ性能が悪い**
   - そのFoldに特殊なデータが含まれている
   - 対策: データの確認、前処理の見直し

3. **全Foldで性能が低い**
   - モデルの容量不足
   - 特徴量が不十分
   - 対策: モデルの拡大、特徴量の追加

---

## 推論での使用

K-Fold学習後、推論には以下の2つの選択肢があります：

### 選択肢1: 最良のFoldのモデルを使用（推奨）

最もF1スコアが高かったFoldのモデルを使用：

```bash
# 例: Fold 3が最良だった場合
copy checkpoints_cut_selection_kfold\fold_3_best_model.pth checkpoints_cut_selection\best_model.pth
copy checkpoints_cut_selection_kfold\inference_params.yaml checkpoints_cut_selection\inference_params.yaml
```

### 選択肢2: アンサンブル（高度）

全Foldのモデルで予測し、多数決または平均を取る：

```python
# 疑似コード
predictions = []
for fold in range(1, 6):
    model = load_model(f'fold_{fold}_best_model.pth')
    pred = model.predict(input)
    predictions.append(pred)

# 多数決
final_prediction = majority_vote(predictions)

# または平均
final_prediction = average(predictions)
```

---

## よくある質問

### Q1: K=5が標準なのはなぜ？

A: 経験的に、K=5が計算コストと性能評価のバランスが良いとされています。
- K=3: 計算は速いが評価が不安定
- K=5: バランスが良い（標準）
- K=10: より正確だが計算時間が長い

### Q2: 通常の学習とK-Foldはどちらを使うべき？

A: 
- **開発中・実験中**: 通常の学習（速い）
- **最終評価・論文**: K-Fold（信頼性が高い）
- **本番デプロイ**: 通常の学習で最良モデルを選択

### Q3: K-Foldの結果が通常の学習より悪い

A: 正常です。K-Foldは厳しい評価なので、通常より低くなることがあります。
- 通常の学習: 運が良い分割で高い性能
- K-Fold: 平均的な性能（より現実的）

### Q4: 学習時間を短縮したい

A: 以下の方法があります：
1. `n_folds` を3に減らす
2. `num_epochs` を減らす
3. `early_stopping_patience` を小さくする
4. より強力なGPUを使用

---

## トラブルシューティング

### エラー: FileNotFoundError: combined_sequences_cut_selection.npz

**原因**: データが結合されていない

**解決策**:
```bash
python scripts/create_combined_data_for_kfold.py
```

### エラー: CUDA out of memory

**原因**: GPUメモリ不足

**解決策**:
1. `batch_size` を小さくする（16 → 8 → 4）
2. `use_amp: true` を確認（メモリ使用量が半分）
3. 一部のFoldだけ実行

### 学習が遅い

**対策**:
1. `num_workers` を増やす（Windows以外）
2. `use_amp: true` を有効化
3. より強力なGPUを使用
4. `n_folds` を減らす

---

## まとめ

K-Fold Cross Validationは、モデルの性能をより正確に評価するための強力な手法です。

**使うべき場面**:
- 最終的な性能評価
- モデルの比較
- 論文やレポート作成
- 本番環境への導入前

**メリット**:
- 信頼性の高い評価
- データの有効活用
- 過学習の検出

**デメリット**:
- 学習時間が長い
- 計算リソースが必要

開発中は通常の学習で素早く実験し、最終評価ではK-Foldを使うのがおすすめです。
