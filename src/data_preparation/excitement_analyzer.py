"""
Excitement Analyzer Module

音声内容から盛り上がり度を検出する

機能:
1. 感情強度の計算（ポジティブ、興奮）
2. クライマックスキーワードの検出
3. 笑い検出
4. 総合的な盛り上がりスコアの計算
"""
import re
from typing import Dict, List
import logging

logger = logging.getLogger(__name__)


class ExcitementAnalyzer:
    """音声内容から盛り上がり度を分析するクラス"""
    
    # クライマックスキーワード
    CLIMAX_KEYWORDS = {
        "ja": [
            "やばい", "ヤバい", "ヤバイ", "やばっ",
            "すごい", "スゴい", "スゴイ", "すげー", "すげえ",
            "マジ", "まじ", "マジで", "まじで",
            "キター", "きたー", "キタ", "きた",
            "ヤッター", "やったー", "やった",
            "最高", "サイコー", "さいこう",
            "神", "かみ"
        ],
        "en": [
            "wow", "omg", "amazing", "awesome", "incredible",
            "yes", "yeah", "yay", "woohoo"
        ]
    }
    
    # 笑い指標
    LAUGHTER_INDICATORS = {
        "ja": ["笑", "ｗ", "w", "W", "ｗｗｗ", "www", "WWW", "草"],
        "en": ["lol", "LOL", "haha", "HAHA", "hehe", "HEHE", "lmao", "LMAO"]
    }
    
    # 感情キーワード（ポジティブ）
    POSITIVE_KEYWORDS = {
        "ja": [
            "嬉しい", "楽しい", "良い", "いい", "素晴らしい",
            "ありがとう", "感謝", "幸せ", "最高", "好き"
        ],
        "en": [
            "happy", "great", "good", "wonderful", "excellent",
            "thank", "love", "best", "nice", "perfect"
        ]
    }
    
    # 感情キーワード（興奮）
    EXCITED_KEYWORDS = {
        "ja": [
            "やばい", "すごい", "マジ", "キター", "ヤッター",
            "！", "!!", "！！", "？！", "!?"
        ],
        "en": [
            "wow", "omg", "amazing", "awesome", "incredible",
            "!", "!!", "?!", "!?"
        ]
    }
    
    def __init__(self):
        """初期化"""
        pass
    
    def analyze_emotion(self, text: str, language: str = "ja") -> Dict[str, float]:
        """
        感情強度を計算
        
        Args:
            text: 分析するテキスト
            language: 言語コード（"ja" or "en"）
        
        Returns:
            感情スコアの辞書 {positive: float, excited: float}
        """
        if not text or len(text.strip()) == 0:
            return {"positive": 0.0, "excited": 0.0}
        
        text_lower = text.lower()
        
        # ポジティブ感情のカウント
        positive_count = 0
        if language in self.POSITIVE_KEYWORDS:
            for keyword in self.POSITIVE_KEYWORDS[language]:
                positive_count += text_lower.count(keyword.lower())
        
        # 興奮感情のカウント
        excited_count = 0
        if language in self.EXCITED_KEYWORDS:
            for keyword in self.EXCITED_KEYWORDS[language]:
                excited_count += text_lower.count(keyword.lower())
        
        # 感嘆符のカウント（興奮の指標）
        excited_count += text.count("！") + text.count("!")
        
        # 正規化（0-1の範囲に）
        # テキスト長に対する割合として計算
        text_length = len(text)
        positive_intensity = min(1.0, positive_count / max(1, text_length / 10))
        excited_intensity = min(1.0, excited_count / max(1, text_length / 10))
        
        return {
            "positive": positive_intensity,
            "excited": excited_intensity
        }
    
    def detect_climax_keywords(self, text: str, language: str = "ja") -> float:
        """
        クライマックスキーワードを検出し、密度を計算
        
        Args:
            text: 分析するテキスト
            language: 言語コード
        
        Returns:
            クライマックスキーワード密度（0-1）
        """
        if not text or len(text.strip()) == 0:
            return 0.0
        
        text_lower = text.lower()
        count = 0
        
        if language in self.CLIMAX_KEYWORDS:
            for keyword in self.CLIMAX_KEYWORDS[language]:
                count += text_lower.count(keyword.lower())
        
        # 密度を計算（テキスト長に対する割合）
        density = min(1.0, count / max(1, len(text) / 20))
        
        return density
    
    def detect_laughter(self, text: str, language: str = "ja") -> float:
        """
        笑い指標を検出し、密度を計算
        
        Args:
            text: 分析するテキスト
            language: 言語コード
        
        Returns:
            笑い密度（0-1）
        """
        if not text or len(text.strip()) == 0:
            return 0.0
        
        text_lower = text.lower()
        count = 0
        
        if language in self.LAUGHTER_INDICATORS:
            for indicator in self.LAUGHTER_INDICATORS[language]:
                count += text_lower.count(indicator.lower())
        
        # 連続した'w'や'笑'のパターンを検出
        # 例: "wwwww", "笑笑笑"
        w_pattern = re.findall(r'[wｗW]{2,}', text)
        count += sum(len(match) for match in w_pattern)
        
        laugh_pattern = re.findall(r'笑{2,}', text)
        count += sum(len(match) for match in laugh_pattern)
        
        # 密度を計算
        density = min(1.0, count / max(1, len(text) / 10))
        
        return density
    
    def compute_excitement_score(
        self,
        text: str,
        speech_rate: float,
        emotion: Dict[str, float],
        language: str = "ja"
    ) -> float:
        """
        総合的な盛り上がりスコアを計算
        
        Args:
            text: 分析するテキスト
            speech_rate: 発話速度（文字/秒）
            emotion: 感情分析結果
            language: 言語コード
        
        Returns:
            盛り上がりスコア（0-1）
        """
        if not text or len(text.strip()) == 0:
            return 0.0
        
        # 各要素のスコアを計算
        climax_score = self.detect_climax_keywords(text, language)
        laughter_score = self.detect_laughter(text, language)
        positive_score = emotion.get("positive", 0.0)
        excited_score = emotion.get("excited", 0.0)
        
        # 発話速度のスコア（速いほど盛り上がっている）
        # 通常の発話速度を10文字/秒と仮定
        # 20文字/秒以上で最大スコア
        speech_score = min(1.0, max(0.0, (speech_rate - 10) / 10))
        
        # 重み付き平均で総合スコアを計算
        weights = {
            "climax": 0.25,      # クライマックスキーワード
            "laughter": 0.20,    # 笑い
            "positive": 0.15,    # ポジティブ感情
            "excited": 0.25,     # 興奮感情
            "speech": 0.15       # 発話速度
        }
        
        excitement_score = (
            climax_score * weights["climax"] +
            laughter_score * weights["laughter"] +
            positive_score * weights["positive"] +
            excited_score * weights["excited"] +
            speech_score * weights["speech"]
        )
        
        return min(1.0, excitement_score)
    
    def analyze_comprehensive(
        self,
        text: str,
        speech_rate: float = 10.0,
        language: str = "ja"
    ) -> Dict[str, float]:
        """
        包括的な分析を実行
        
        Args:
            text: 分析するテキスト
            speech_rate: 発話速度（文字/秒）
            language: 言語コード
        
        Returns:
            分析結果の辞書
        """
        if not text or len(text.strip()) == 0:
            return {
                "excitement_score": 0.0,
                "positive_intensity": 0.0,
                "excited_intensity": 0.0,
                "climax_density": 0.0,
                "laughter_density": 0.0,
                "speech_rate": speech_rate
            }
        
        # 感情分析
        emotion = self.analyze_emotion(text, language)
        
        # クライマックスキーワード検出
        climax_density = self.detect_climax_keywords(text, language)
        
        # 笑い検出
        laughter_density = self.detect_laughter(text, language)
        
        # 総合スコア計算
        excitement_score = self.compute_excitement_score(
            text, speech_rate, emotion, language
        )
        
        return {
            "excitement_score": excitement_score,
            "positive_intensity": emotion["positive"],
            "excited_intensity": emotion["excited"],
            "climax_density": climax_density,
            "laughter_density": laughter_density,
            "speech_rate": speech_rate
        }
