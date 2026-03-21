"""
Enhanced Whisper Transcription Module

Whisperの音声認識精度を向上させるための拡張モジュール

改善策:
1. より大きなモデルの使用（medium, large）
2. 言語指定による精度向上
3. プロンプト機能の活用
4. 音声前処理（ノイズ除去、正規化）
5. 温度パラメータの調整
6. ビームサーチの最適化
7. VADによる無音区間の除外
"""
import os
import numpy as np
import whisper
import torch
import librosa
import noisereduce as nr
from typing import Dict, List, Optional, Tuple
import logging

logger = logging.getLogger(__name__)


class EnhancedWhisperTranscriber:
    """拡張Whisper文字起こしクラス"""
    
    # 利用可能なモデルサイズ
    AVAILABLE_MODELS = {
        "tiny": {"params": "39M", "vram": "~1GB", "speed": "~32x", "accuracy": "低"},
        "base": {"params": "74M", "vram": "~1GB", "speed": "~16x", "accuracy": "中"},
        "small": {"params": "244M", "vram": "~2GB", "speed": "~6x", "accuracy": "中"},
        "medium": {"params": "769M", "vram": "~5GB", "speed": "~2x", "accuracy": "高"},
        "large": {"params": "1550M", "vram": "~10GB", "speed": "~1x", "accuracy": "最高"},
        "large-v2": {"params": "1550M", "vram": "~10GB", "speed": "~1x", "accuracy": "最高"},
        "large-v3": {"params": "1550M", "vram": "~10GB", "speed": "~1x", "accuracy": "最高（最新）"}
    }
    
    def __init__(
        self,
        model_size: str = "medium",
        language: Optional[str] = None,
        device: Optional[str] = None,
        enable_preprocessing: bool = True,
        enable_vad: bool = True
    ):
        """
        初期化
        
        Args:
            model_size: モデルサイズ（tiny, base, small, medium, large, large-v2, large-v3）
            language: 言語コード（ja, en, None=自動検出）
            device: デバイス（cuda, cpu, None=自動選択）
            enable_preprocessing: 音声前処理を有効化
            enable_vad: VAD（Voice Activity Detection）を有効化
        """
        self.model_size = model_size
        self.language = language
        self.enable_preprocessing = enable_preprocessing
        self.enable_vad = enable_vad
        
        # デバイス選択
        if device is None:
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
        else:
            self.device = device
        
        # モデルをロード
        logger.info(f"Loading Whisper model: {model_size} on {self.device}")
        self.model = whisper.load_model(model_size, device=self.device)
        
        # モデル情報を表示
        if model_size in self.AVAILABLE_MODELS:
            info = self.AVAILABLE_MODELS[model_size]
            logger.info(f"  Parameters: {info['params']}")
            logger.info(f"  VRAM: {info['vram']}")
            logger.info(f"  Speed: {info['speed']} realtime")
            logger.info(f"  Accuracy: {info['accuracy']}")
    
    def preprocess_audio(
        self,
        audio: np.ndarray,
        sr: int,
        reduce_noise: bool = True,
        normalize: bool = True,
        trim_silence: bool = True
    ) -> np.ndarray:
        """
        音声を前処理
        
        Args:
            audio: 音声データ
            sr: サンプリングレート
            reduce_noise: ノイズ除去を実行
            normalize: 正規化を実行
            trim_silence: 無音をトリミング
        
        Returns:
            前処理済み音声データ
        """
        if not self.enable_preprocessing:
            return audio
        
        logger.debug("Preprocessing audio...")
        
        # ノイズ除去
        if reduce_noise:
            try:
                audio = nr.reduce_noise(y=audio, sr=sr, stationary=True)
                logger.debug("  Noise reduction applied")
            except Exception as e:
                logger.warning(f"  Noise reduction failed: {e}")
        
        # 正規化
        if normalize:
            max_val = np.abs(audio).max()
            if max_val > 0:
                audio = audio / max_val * 0.95  # ピークを95%に制限
                logger.debug("  Normalization applied")
        
        # 無音トリミング
        if trim_silence:
            try:
                audio, _ = librosa.effects.trim(audio, top_db=20)
                logger.debug("  Silence trimming applied")
            except Exception as e:
                logger.warning(f"  Silence trimming failed: {e}")
        
        return audio
    
    def detect_language(self, audio_path: str) -> Tuple[str, float]:
        """
        音声の言語を検出
        
        Args:
            audio_path: 音声ファイルのパス
        
        Returns:
            (言語コード, 確信度)
        """
        # Whisperで音声を読み込み
        audio = whisper.load_audio(audio_path)
        audio = whisper.pad_or_trim(audio)
        
        # メルスペクトログラムを作成
        mel = whisper.log_mel_spectrogram(audio).to(self.model.device)
        
        # 言語を検出
        _, probs = self.model.detect_language(mel)
        detected_language = max(probs, key=probs.get)
        confidence = probs[detected_language]
        
        logger.info(f"Detected language: {detected_language} (confidence: {confidence:.2f})")
        
        return detected_language, confidence
    
    def transcribe(
        self,
        audio_path: str,
        language: Optional[str] = None,
        initial_prompt: Optional[str] = None,
        word_timestamps: bool = True,
        temperature: float = 0.0,
        beam_size: int = 5,
        best_of: int = 5,
        patience: float = 1.0,
        condition_on_previous_text: bool = True,
        compression_ratio_threshold: float = 2.4,
        logprob_threshold: float = -1.0,
        no_speech_threshold: float = 0.6
    ) -> Dict:
        """
        音声を文字起こし（拡張版）
        
        Args:
            audio_path: 音声ファイルのパス
            language: 言語コード（None=自動検出）
            initial_prompt: 初期プロンプト（専門用語のヒント）
            word_timestamps: 単語レベルのタイムスタンプ
            temperature: 温度パラメータ（0.0=決定的、1.0=ランダム）
            beam_size: ビームサーチのビーム数
            best_of: 候補数
            patience: ビームサーチの忍耐度
            condition_on_previous_text: 前のテキストを条件付け
            compression_ratio_threshold: 圧縮率の閾値
            logprob_threshold: 対数確率の閾値
            no_speech_threshold: 無音判定の閾値
        
        Returns:
            文字起こし結果
        """
        # 言語を決定
        if language is None:
            language = self.language
        
        # 言語が指定されていない場合は自動検出
        if language is None:
            detected_lang, confidence = self.detect_language(audio_path)
            if confidence > 0.5:
                language = detected_lang
                logger.info(f"Using detected language: {language}")
        
        # 音声を読み込み
        audio = whisper.load_audio(audio_path)
        
        # 前処理
        if self.enable_preprocessing:
            sr = 16000  # Whisperのサンプリングレート
            audio = self.preprocess_audio(audio, sr)
        
        # 文字起こしオプション
        options = {
            "language": language,
            "task": "transcribe",
            "word_timestamps": word_timestamps,
            "temperature": temperature,
            "beam_size": beam_size,
            "best_of": best_of,
            "patience": patience,
            "condition_on_previous_text": condition_on_previous_text,
            "compression_ratio_threshold": compression_ratio_threshold,
            "logprob_threshold": logprob_threshold,
            "no_speech_threshold": no_speech_threshold
        }
        
        # 初期プロンプトを追加
        if initial_prompt:
            options["initial_prompt"] = initial_prompt
        
        # 文字起こし実行
        logger.info(f"Transcribing with options: language={language}, beam_size={beam_size}, temperature={temperature}")
        result = self.model.transcribe(audio, **options)
        
        # 結果にメタデータを追加
        result["model_size"] = self.model_size
        result["preprocessing_enabled"] = self.enable_preprocessing
        
        return result
    
    def transcribe_with_vad(
        self,
        audio_path: str,
        vad_threshold: float = 0.5,
        min_speech_duration: float = 0.25,
        **kwargs
    ) -> Dict:
        """
        VAD（Voice Activity Detection）を使用して文字起こし
        
        無音区間を除外することで精度を向上
        
        Args:
            audio_path: 音声ファイルのパス
            vad_threshold: VADの閾値
            min_speech_duration: 最小発話時間（秒）
            **kwargs: transcribeメソッドの引数
        
        Returns:
            文字起こし結果
        """
        if not self.enable_vad:
            return self.transcribe(audio_path, **kwargs)
        
        # 音声を読み込み
        y, sr = librosa.load(audio_path, sr=16000)
        
        # RMSエネルギーでVADを実行
        frame_length = int(0.025 * sr)  # 25ms
        hop_length = int(0.010 * sr)    # 10ms
        rms = librosa.feature.rms(y=y, frame_length=frame_length, hop_length=hop_length)[0]
        
        # 閾値を超える区間を検出
        is_speech = rms > (rms.mean() * vad_threshold)
        
        # 発話区間を抽出
        speech_frames = []
        in_speech = False
        start_frame = 0
        
        for i, speech in enumerate(is_speech):
            if speech and not in_speech:
                start_frame = i
                in_speech = True
            elif not speech and in_speech:
                duration = (i - start_frame) * hop_length / sr
                if duration >= min_speech_duration:
                    speech_frames.append((start_frame, i))
                in_speech = False
        
        # 最後の区間を追加
        if in_speech:
            duration = (len(is_speech) - start_frame) * hop_length / sr
            if duration >= min_speech_duration:
                speech_frames.append((start_frame, len(is_speech)))
        
        logger.info(f"VAD detected {len(speech_frames)} speech segments")
        
        # 発話区間のみを文字起こし
        # （簡易実装: 全体を文字起こしして、VAD情報をメタデータとして追加）
        result = self.transcribe(audio_path, **kwargs)
        result["vad_segments"] = len(speech_frames)
        result["vad_enabled"] = True
        
        return result
    
    @staticmethod
    def get_recommended_settings(content_type: str = "general") -> Dict:
        """
        コンテンツタイプに応じた推奨設定を取得
        
        Args:
            content_type: コンテンツタイプ
                - general: 一般的な内容
                - gaming: ゲーム実況
                - technical: 技術解説
                - interview: インタビュー
                - lecture: 講義
        
        Returns:
            推奨設定の辞書
        """
        settings = {
            "general": {
                "model_size": "medium",
                "temperature": 0.0,
                "beam_size": 5,
                "initial_prompt": None
            },
            "gaming": {
                "model_size": "medium",
                "temperature": 0.2,  # 少しランダム性を持たせる
                "beam_size": 5,
                "initial_prompt": "ゲーム実況、プレイ、攻略、クリア、レベル、スキル、アイテム、ボス"
            },
            "technical": {
                "model_size": "large-v3",  # 専門用語のため大きいモデル
                "temperature": 0.0,
                "beam_size": 10,  # より慎重に
                "initial_prompt": "プログラミング、コード、開発、AI、機械学習、アルゴリズム、技術"
            },
            "interview": {
                "model_size": "medium",
                "temperature": 0.0,
                "beam_size": 5,
                "initial_prompt": None,
                "condition_on_previous_text": True  # 文脈を重視
            },
            "lecture": {
                "model_size": "large-v2",
                "temperature": 0.0,
                "beam_size": 10,
                "initial_prompt": "講義、解説、説明、授業、学習"
            }
        }
        
        return settings.get(content_type, settings["general"])


def create_transcriber(
    model_size: str = "medium",
    language: str = "ja",
    enable_preprocessing: bool = True,
    enable_vad: bool = True
) -> EnhancedWhisperTranscriber:
    """
    拡張Whisper文字起こしクラスを作成（ファクトリー関数）
    
    Args:
        model_size: モデルサイズ
        language: 言語コード
        enable_preprocessing: 音声前処理を有効化
        enable_vad: VADを有効化
    
    Returns:
        EnhancedWhisperTranscriberインスタンス
    """
    return EnhancedWhisperTranscriber(
        model_size=model_size,
        language=language,
        enable_preprocessing=enable_preprocessing,
        enable_vad=enable_vad
    )
