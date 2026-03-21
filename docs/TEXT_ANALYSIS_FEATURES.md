# テキスト分析機能 - 詳細ドキュメント

## 概要

Whisperで文字起こしした内容を分析し、動画の特徴量として追加する機能です。

## 追加された特徴量

### 1. 言語検出

**特徴量名:** `text_language`

**説明:** 音声の言語を自動検出

**値:**
- `ja`: 日本語
- `en`: 英語
- `mixed`: 混在
- `unknown`: 不明

**検出方法:**
- 日本語文字（ひらがな、カタカナ、漢字）の割合で判定
- 30%以上: 日本語
- 10%未満: 英語
- その他: 混在

### 2. 基本統計

#### 文字数
**特徴量名:** `text_char_count`

**説明:** セグメント内の文字数（空白を除く）

**用途:** 発話量の指標

#### 単語数
**特徴量名:** `text_word_count`

**説明:** セグメント内の単語数

**計算方法:**
- 日本語: 連続した日本語文字列の数
- 英語: 単語の数

#### 発話速度
**特徴量名:** `text_speech_rate`

**説明:** 単位時間あたりの文字数（文字/秒）

**用途:** 
- 興奮度の指標
- 重要な場面の検出

### 3. 感情分析

キーワードベースの感情分析を実行します。

#### ポジティブ感情
**特徴量名:** `text_emotion_positive`

**キーワード例:**
- 日本語: 嬉しい、楽しい、最高、すごい、良い、素晴らしい、ありがとう、感謝、幸せ、笑、www、草
- 英語: happy, great, awesome, wonderful, amazing, excellent, love, thank, lol, haha

**値:** 0.0 ~ 1.0（正規化済み）

#### ネガティブ感情
**特徴量名:** `text_emotion_negative`

**キーワード例:**
- 日本語: 悲しい、辛い、嫌、最悪、ダメ、困る、心配、不安、怖い
- 英語: sad, bad, terrible, awful, hate, worry, afraid, scared

**値:** 0.0 ~ 1.0（正規化済み）

#### 興奮
**特徴量名:** `text_emotion_excited`

**キーワード例:**
- 日本語: やばい、ヤバい、マジ、すげー、ヤッター、キター、！！、!!、！
- 英語: wow, omg, amazing, incredible, !!!, yay

**値:** 0.0 ~ 1.0（正規化済み）

**用途:** ハイライトシーンの検出

#### 疑問
**特徴量名:** `text_emotion_question`

**キーワード例:**
- 日本語: ？、?、なぜ、どう、何、誰、いつ、どこ
- 英語: ?, why, how, what, who, when, where

**値:** 0.0 ~ 1.0（正規化済み）

#### ニュートラル
**特徴量名:** `text_emotion_neutral`

**説明:** 他の感情が検出されない場合

**値:** 0.0 ~ 1.0（正規化済み）

### 4. トピック分類

内容のトピックを自動分類します。

#### ゲーム
**特徴量名:** `text_topic_game`

**キーワード例:**
- 日本語: ゲーム、プレイ、攻略、クリア、レベル、スキル、アイテム、ボス、敵
- 英語: game, play, level, skill, item, boss, enemy, quest

**値:** 0.0 ~ 1.0（正規化済み）

#### 技術
**特徴量名:** `text_topic_tech`

**キーワード例:**
- 日本語: 技術、プログラミング、コード、開発、AI、機械学習、アルゴリズム
- 英語: tech, programming, code, development, ai, machine learning, algorithm

**値:** 0.0 ~ 1.0（正規化済み）

#### エンタメ
**特徴量名:** `text_topic_entertainment`

**キーワード例:**
- 日本語: 映画、アニメ、音楽、ドラマ、芸能、エンタメ
- 英語: movie, anime, music, drama, entertainment, show

**値:** 0.0 ~ 1.0（正規化済み）

#### 日常
**特徴量名:** `text_topic_daily`

**キーワード例:**
- 日本語: 日常、生活、料理、食事、買い物、旅行
- 英語: daily, life, cooking, food, shopping, travel

**値:** 0.0 ~ 1.0（正規化済み）

#### チュートリアル
**特徴量名:** `text_topic_tutorial`

**キーワード例:**
- 日本語: 解説、説明、方法、やり方、手順、チュートリアル、講座
- 英語: tutorial, how to, guide, explanation, instruction, lesson

**値:** 0.0 ~ 1.0（正規化済み）

### 5. キーワード抽出

**特徴量名:** `text_keywords`

**説明:** セグメント内の重要キーワード（上位3個）

**形式:** カンマ区切りの文字列

**例:** "ゲーム,攻略,クリア"

**抽出方法:**
- 頻出単語を抽出
- ストップワードを除外
- 上位3個を選択

## 使用例

### 特徴量の確認

```python
import pandas as pd

# 特徴量ファイルを読み込み
df = pd.read_csv('data/processed/source_features/video1_features.csv')

# テキスト分析の特徴量を確認
text_cols = [col for col in df.columns if col.startswith('text_')]
print(df[text_cols].head())
```

### 感情の時系列変化を可視化

```python
import matplotlib.pyplot as plt

# 感情の時系列変化をプロット
plt.figure(figsize=(12, 6))
plt.plot(df['time'], df['text_emotion_positive'], label='Positive')
plt.plot(df['time'], df['text_emotion_negative'], label='Negative')
plt.plot(df['time'], df['text_emotion_excited'], label='Excited')
plt.xlabel('Time (s)')
plt.ylabel('Emotion Score')
plt.legend()
plt.title('Emotion Timeline')
plt.show()
```

### ハイライトシーンの検出

```python
# 興奮度が高いシーンを検出
threshold = 0.5
highlight_scenes = df[df['text_emotion_excited'] > threshold]

print(f"Found {len(highlight_scenes)} highlight scenes")
print(highlight_scenes[['time', 'text_emotion_excited', 'text_word']])
```

## 追加された特徴量の一覧

| カテゴリ | 特徴量名 | 型 | 説明 |
|---------|---------|-----|------|
| 言語 | text_language | str | 検出された言語 |
| 基本統計 | text_char_count | int | 文字数 |
| 基本統計 | text_word_count | int | 単語数 |
| 基本統計 | text_speech_rate | float | 発話速度 |
| 感情 | text_emotion_positive | float | ポジティブ感情 |
| 感情 | text_emotion_negative | float | ネガティブ感情 |
| 感情 | text_emotion_excited | float | 興奮 |
| 感情 | text_emotion_question | float | 疑問 |
| 感情 | text_emotion_neutral | float | ニュートラル |
| トピック | text_topic_game | float | ゲーム |
| トピック | text_topic_tech | float | 技術 |
| トピック | text_topic_entertainment | float | エンタメ |
| トピック | text_topic_daily | float | 日常 |
| トピック | text_topic_tutorial | float | チュートリアル |
| キーワード | text_keywords | str | 重要キーワード |

**合計:** 15個の新規特徴量

## トレーニングへの影響

### 特徴量の増加

**以前:**
- 音声: 215次元
- 視覚: 522次元
- 合計: 737次元

**現在:**
- 音声: 215次元
- 視覚: 522次元
- テキスト分析: 15次元
- 合計: 752次元

### 期待される効果

1. **カット選択の精度向上**
   - 感情の高まりを検出
   - 重要な発言を識別
   - トピックの変化を捉える

2. **ハイライトシーンの検出**
   - 興奮度の高いシーン
   - ポジティブな反応
   - 重要なキーワード

3. **コンテキストの理解**
   - 動画の内容を理解
   - トピックに応じた編集
   - 言語に応じた処理

## カスタマイズ

### キーワードの追加

`src/data_preparation/text_analysis.py` を編集：

```python
EMOTION_KEYWORDS = {
    "positive": {
        "ja": ["嬉しい", "楽しい", "最高", ...],  # ここに追加
        "en": ["happy", "great", "awesome", ...]  # ここに追加
    },
    # ...
}
```

### トピックの追加

```python
TOPIC_KEYWORDS = {
    "game": {...},
    "tech": {...},
    "your_topic": {  # 新しいトピックを追加
        "ja": ["キーワード1", "キーワード2", ...],
        "en": ["keyword1", "keyword2", ...]
    }
}
```

## トラブルシューティング

### 問題1: 言語検出が不正確

**原因:** 短いテキストや混在言語

**解決策:** 
- より長いセグメントで分析
- 言語を手動指定

### 問題2: 感情スコアが常に0

**原因:** キーワードが含まれていない

**解決策:**
- キーワードリストを拡張
- より多様なキーワードを追加

### 問題3: トピック分類が不正確

**原因:** ドメイン固有の用語が不足

**解決策:**
- 対象ドメインのキーワードを追加
- カスタムトピックを定義

## 今後の拡張

### Phase 2（予定）

- [ ] セマンティック埋め込み（BERT/Sentence-BERT）
- [ ] より高度な感情分析（深層学習モデル）
- [ ] エンティティ抽出（人名、地名など）
- [ ] 要約生成
- [ ] 多言語対応の拡張

### Phase 3（予定）

- [ ] リアルタイム分析
- [ ] カスタムモデルのトレーニング
- [ ] ドメイン適応

## 参考資料

- [Whisper Documentation](https://github.com/openai/whisper)
- [テキスト分析の基礎](https://ja.wikipedia.org/wiki/テキストマイニング)
- [感情分析](https://ja.wikipedia.org/wiki/感情分析)
