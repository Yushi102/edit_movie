"""
Text Analysis Module

Whisperで文字起こしした内容を分析し、特徴量として抽出

機能:
1. 言語検出
2. 感情分析
3. トピック分類
4. 重要キーワード抽出
5. セマンティック埋め込み
"""
import re
import numpy as np
from typing import List, Dict, Optional, Tuple
from collections import Counter
import logging

logger = logging.getLogger(__name__)

# 感情を表すキーワード（日本語・英語）
EMOTION_KEYWORDS = {
    "positive": {
        "ja": ["嬉しい", "楽しい", "最高", "すごい", "良い", "素晴らしい", "ありがとう", "感謝", "幸せ", "笑", "www", "草"],
        "en": ["happy", "great", "awesome", "wonderful", "amazing", "excellent", "love", "thank", "lol", "haha"]
    },
    "negative": {
        "ja": ["悲しい", "辛い", "嫌", "最悪", "ダメ", "困る", "心配", "不安", "怖い"],
        "en": ["sad", "bad", "terrible", "awful", "hate", "worry", "afraid", "scared"]
    },
    "excited": {
        "ja": ["やばい", "ヤバい", "マジ", "すげー", "ヤッター", "キター", "！！", "!!", "！"],
        "en": ["wow", "omg", "amazing", "incredible", "!!!", "yay"]
    },
    "question": {
        "ja": ["？", "?", "なぜ", "どう", "何", "誰", "いつ", "どこ"],
        "en": ["?", "why", "how", "what", "who", "when", "where"]
    }
}

# トピックキーワード
TOPIC_KEYWORDS = {
    "game": {
        "ja": ["ゲーム", "プレイ", "攻略", "クリア", "レベル", "スキル", "アイテム", "ボス", "敵"],
        "en": ["game", "play", "level", "skill", "item", "boss", "enemy", "quest"]
    },
    "tech": {
        "ja": ["技術", "プログラミング", "コード", "開発", "AI", "機械学習", "アルゴリズム"],
        "en": ["tech", "programming", "code", "development", "ai", "machine learning", "algorithm"]
    },
    "entertainment": {
        "ja": ["映画", "アニメ", "音楽", "ドラマ", "芸能", "エンタメ"],
        "en": ["movie", "anime", "music", "drama", "entertainment", "show"]
    },
    "daily": {
        "ja": ["日常", "生活", "料理", "食事", "買い物", "旅行"],
        "en": ["daily", "life", "cooking", "food", "shopping", "travel"]
    },
    "tutorial": {
        "ja": ["解説", "説明", "方法", "やり方", "手順", "チュートリアル", "講座"],
        "en": ["tutorial", "how to", "guide", "explanation", "instruction", "lesson"]
    }
}


class TextAnalyzer:
    """テキスト分析クラス"""
    
    def __init__(self):
        """初期化"""
        self.language = None
        
    def detect_language(self, text: str) -> str:
        """
        言語を検出
        
        Args:
            text: 分析するテキスト
        
        Returns:
            言語コード ('ja', 'en', 'unknown')
        """
        if not text or len(text.strip()) == 0:
            return "unknown"
        
        # 日本語文字（ひらがな、カタカナ、漢字）の割合をチェック
        japanese_chars = len(re.findall(r'[\u3040-\u309F\u30A0-\u30FF\u4E00-\u9FFF]', text))
        total_chars = len(re.sub(r'\s', '', text))
        
        if total_chars == 0:
            return "unknown"
        
        japanese_ratio = japanese_chars / total_chars
        
        if japanese_ratio > 0.3:
            return "ja"
        elif japanese_ratio < 0.1:
            return "en"
        else:
            return "mixed"
    
    def analyze_emotion(self, text: str, language: str = None) -> Dict[str, float]:
        """
        感情を分析
        
        Args:
            text: 分析するテキスト
            language: 言語コード（Noneの場合は自動検出）
        
        Returns:
            感情スコアの辞書
        """
        if not text:
            return {
                "positive": 0.0,
                "negative": 0.0,
                "excited": 0.0,
                "question": 0.0,
                "neutral": 1.0
            }
        
        if language is None:
            language = self.detect_language(text)
        
        text_lower = text.lower()
        scores = {}
        
        # 各感情のキーワードをカウント
        for emotion, keywords_dict in EMOTION_KEYWORDS.items():
            count = 0
            if language in keywords_dict:
                for keyword in keywords_dict[language]:
                    count += text_lower.count(keyword.lower())
            scores[emotion] = count
        
        # 正規化
        total = sum(scores.values())
        if total > 0:
            for key in scores:
                scores[key] = scores[key] / total
            scores["neutral"] = 0.0
        else:
            scores["neutral"] = 1.0
        
        return scores
    
    def analyze_topic(self, text: str, language: str = None) -> Dict[str, float]:
        """
        トピックを分析
        
        Args:
            text: 分析するテキスト
            language: 言語コード（Noneの場合は自動検出）
        
        Returns:
            トピックスコアの辞書
        """
        if not text:
            return {topic: 0.0 for topic in TOPIC_KEYWORDS.keys()}
        
        if language is None:
            language = self.detect_language(text)
        
        text_lower = text.lower()
        scores = {}
        
        # 各トピックのキーワードをカウント
        for topic, keywords_dict in TOPIC_KEYWORDS.items():
            count = 0
            if language in keywords_dict:
                for keyword in keywords_dict[language]:
                    count += text_lower.count(keyword.lower())
            scores[topic] = count
        
        # 正規化
        total = sum(scores.values())
        if total > 0:
            for key in scores:
                scores[key] = scores[key] / total
        
        return scores
    
    def extract_keywords(self, text: str, top_n: int = 5) -> List[str]:
        """
        重要キーワードを抽出
        
        Args:
            text: 分析するテキスト
            top_n: 抽出する上位N個
        
        Returns:
            キーワードのリスト
        """
        if not text:
            return []
        
        # 単語に分割（簡易版）
        # 日本語の場合は文字単位、英語の場合は単語単位
        language = self.detect_language(text)
        
        if language == "ja":
            # 日本語: 2文字以上の連続した文字列を抽出
            words = re.findall(r'[\u3040-\u309F\u30A0-\u30FF\u4E00-\u9FFF]{2,}', text)
        else:
            # 英語: 単語を抽出
            words = re.findall(r'\b[a-zA-Z]{3,}\b', text.lower())
        
        # ストップワードを除外（簡易版）
        stopwords_ja = ["これ", "それ", "あれ", "この", "その", "あの", "ここ", "そこ", "あそこ"]
        stopwords_en = ["the", "and", "for", "are", "but", "not", "you", "all", "can", "her", "was", "one", "our", "out", "day", "get", "has", "him", "his", "how", "man", "new", "now", "old", "see", "two", "way", "who", "boy", "did", "its", "let", "put", "say", "she", "too", "use"]
        
        stopwords = stopwords_ja if language == "ja" else stopwords_en
        words = [w for w in words if w not in stopwords]
        
        # 頻度をカウント
        word_counts = Counter(words)
        
        # 上位N個を返す
        return [word for word, count in word_counts.most_common(top_n)]
    
    def calculate_speech_density(self, text: str) -> float:
        """
        発話密度を計算（単位時間あたりの文字数）
        
        Args:
            text: 分析するテキスト
        
        Returns:
            発話密度
        """
        if not text:
            return 0.0
        
        # 空白を除いた文字数
        char_count = len(re.sub(r'\s', '', text))
        return float(char_count)
    
    def analyze_segment(self, text: str, duration: float = 1.0) -> Dict[str, any]:
        """
        テキストセグメントを総合的に分析
        
        Args:
            text: 分析するテキスト
            duration: セグメントの長さ（秒）
        
        Returns:
            分析結果の辞書
        """
        if not text or len(text.strip()) == 0:
            return {
                "language": "unknown",
                "char_count": 0,
                "word_count": 0,
                "speech_rate": 0.0,
                "emotion_positive": 0.0,
                "emotion_negative": 0.0,
                "emotion_excited": 0.0,
                "emotion_question": 0.0,
                "emotion_neutral": 1.0,
                "topic_game": 0.0,
                "topic_tech": 0.0,
                "topic_entertainment": 0.0,
                "topic_daily": 0.0,
                "topic_tutorial": 0.0,
                "keywords": ""
            }
        
        # 言語検出
        language = self.detect_language(text)
        
        # 文字数・単語数
        char_count = len(re.sub(r'\s', '', text))
        
        if language == "ja":
            word_count = len(re.findall(r'[\u3040-\u309F\u30A0-\u30FF\u4E00-\u9FFF]+', text))
        else:
            word_count = len(re.findall(r'\b[a-zA-Z]+\b', text))
        
        # 発話速度（文字数/秒）
        speech_rate = char_count / duration if duration > 0 else 0.0
        
        # 感情分析
        emotions = self.analyze_emotion(text, language)
        
        # トピック分析
        topics = self.analyze_topic(text, language)
        
        # キーワード抽出
        keywords = self.extract_keywords(text, top_n=3)
        keywords_str = ",".join(keywords) if keywords else ""
        
        return {
            "language": language,
            "char_count": char_count,
            "word_count": word_count,
            "speech_rate": speech_rate,
            "emotion_positive": emotions.get("positive", 0.0),
            "emotion_negative": emotions.get("negative", 0.0),
            "emotion_excited": emotions.get("excited", 0.0),
            "emotion_question": emotions.get("question", 0.0),
            "emotion_neutral": emotions.get("neutral", 1.0),
            "topic_game": topics.get("game", 0.0),
            "topic_tech": topics.get("tech", 0.0),
            "topic_entertainment": topics.get("entertainment", 0.0),
            "topic_daily": topics.get("daily", 0.0),
            "topic_tutorial": topics.get("tutorial", 0.0),
            "keywords": keywords_str
        }


def analyze_transcription_segments(
    whisper_results: List[Dict],
    time_step: float = 0.1
) -> List[Dict]:
    """
    Whisperの文字起こし結果を時系列で分析
    
    Args:
        whisper_results: Whisperの結果（segmentsを含む）
        time_step: 時間ステップ（秒）
    
    Returns:
        時系列の分析結果リスト
    """
    analyzer = TextAnalyzer()
    analyzed_segments = []
    
    for segment in whisper_results.get('segments', []):
        text = segment.get('text', '')
        start = segment.get('start', 0.0)
        end = segment.get('end', 0.0)
        duration = end - start
        
        # セグメントを分析
        analysis = analyzer.analyze_segment(text, duration)
        analysis['start'] = start
        analysis['end'] = end
        analysis['text'] = text
        
        analyzed_segments.append(analysis)
    
    return analyzed_segments
