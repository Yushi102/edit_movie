# Premiere Pro XML Parser - 機能拡張

## 更新日時
2025-12-13

## 概要
Premiere Pro XMLパーサーに、アンカーポイント、回転、キーフレーム情報の抽出機能を追加しました。

---

## 追加された機能

### 1. アンカーポイント（Anchor Point）
- **anchor_x**: アンカーポイントのX座標
- **anchor_y**: アンカーポイントのY座標
- クリップの回転や変形の中心点を表します

### 2. 回転（Rotation）
- **rotation**: 回転角度（度数法）
- 0-360度の範囲で、時計回りの回転を表します

### 3. キーフレーム情報
- **has_keyframes**: キーフレームアニメーションの有無（True/False）
- **keyframe_count**: キーフレームの数
- アニメーション効果が適用されているクリップを識別できます

---

## パラメータ数の変更

### 従来（v1.0）
- **9パラメータ/トラック**
- 合計: 20トラック × 9 = **180次元**
- パラメータ: [active, asset_id, scale, pos_x, pos_y, crop_l, crop_r, crop_t, crop_b]

### 拡張版（v2.0）⭐
- **12パラメータ/トラック**
- 合計: 20トラック × 12 = **240次元**
- パラメータ: [active, asset_id, scale, pos_x, pos_y, **anchor_x, anchor_y, rotation**, crop_l, crop_r, crop_t, crop_b]

---

## 出力形式

### NPZファイル
```python
data = np.load('video_tracks.npz')
sequences = data['sequences']  # Shape: (num_timesteps, 240)
```

### CSVファイル
新しいカラムが追加されました：

| カラム名 | 説明 | 型 |
|---------|------|-----|
| anchor_x | アンカーポイントX座標 | float |
| anchor_y | アンカーポイントY座標 | float |
| rotation | 回転角度（度） | float |
| has_keyframes | キーフレームの有無 | bool |
| keyframe_count | キーフレーム数 | int |

---

## テスト結果

### テスト1: `editxml/1.xml`
```
✅ 成功
- 出力形状: (1142, 240) ← 従来 (1142, 180)
- 新パラメータ: anchor_x, anchor_y, rotation すべて抽出
- キーフレーム情報: has_keyframes, keyframe_count 記録
```

### テスト2: `editxml/bandicam 2025-03-03 22-34-57-492.xml`
```
✅ 成功
- 出力形状: (551, 240)
- 26クリップ、6アセット
- すべての新パラメータが正常に抽出
```

---

## 使用例

### 基本的な使用方法
```bash
python premiere_xml_parser.py "editxml/your_file.xml" --output output_dir --format both
```

### 出力の確認
```python
import pandas as pd

# CSVを読み込み
df = pd.read_csv('output_dir/your_file_tracks.csv')

# 回転が適用されているクリップを確認
rotated_clips = df[df['rotation'] != 0.0]
print(f"回転クリップ数: {len(rotated_clips)}")

# キーフレームアニメーションがあるクリップを確認
animated_clips = df[df['has_keyframes'] == True]
print(f"アニメーションクリップ数: {len(animated_clips)}")

# アンカーポイントが設定されているクリップを確認
anchor_clips = df[(df['anchor_x'] != 0.0) | (df['anchor_y'] != 0.0)]
print(f"アンカーポイント設定クリップ数: {len(anchor_clips)}")
```

---

## 技術的な詳細

### キーフレーム検出
Premiere Pro XMLの`<keyframe>`要素を検出し、以下の情報を抽出：
- キーフレームの存在
- キーフレームの数
- 各キーフレームのタイミング（`when`属性）
- 各キーフレームの値（`value`要素）

現在の実装では、最初のキーフレームの値を使用していますが、将来的には：
- 時間に応じた補間
- すべてのキーフレーム値の記録
- ベジェ曲線の情報

などを追加できます。

### パラメータID検出
Premiere Pro XMLの`<parameter>`要素から、以下のIDを検出：
- `rotation`, `angle` → rotation
- `anchor`, `anchorpoint` → anchor_x, anchor_y
- `scale` → scale
- `position` → pos_x, pos_y
- `crop` → crop_l, crop_r, crop_t, crop_b

---

## 互換性

### 後方互換性
- 既存のコードは影響を受けません
- NPZファイルの形状が変更されているため、モデル側で対応が必要です

### モデル側の対応
```python
# 従来（180次元）
track_dim = 180  # 20 tracks × 9 params

# 拡張版（240次元）
track_dim = 240  # 20 tracks × 12 params
```

---

## 次のステップ

### 1. モデルの更新
Multi-Track Transformerモデルを240次元入力に対応させる

### 2. キーフレーム補間
時間に応じたキーフレーム値の補間を実装

### 3. 一括処理
全XMLファイル（100+）を新しいパーサーで処理

### 4. データ分析
- 回転が使われている頻度
- キーフレームアニメーションの使用パターン
- アンカーポイントの設定傾向

---

## まとめ

✅ **アンカーポイント抽出**: anchor_x, anchor_y を追加

✅ **回転抽出**: rotation（度数法）を追加

✅ **キーフレーム情報**: has_keyframes, keyframe_count を追加

✅ **出力次元拡張**: 180次元 → 240次元

✅ **テスト完了**: 実際のファイルで動作確認済み

これにより、より詳細な編集情報を機械学習モデルに提供できるようになりました！
