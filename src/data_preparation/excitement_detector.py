"""
Excitement Detector Module

音声内容から盛り上がり度特徴量を生成する

機能:
1. Whisper文字起こしデータから時系列特徴量を生成
2. Transformer埋め込み（768次元）
3. 基本統計特徴量（5次元）
4. 同時発話カウント（1次元）
5. 感情ベース特徴量（5次元）
6. トピック変化特徴量（5次元）
7. 発話パターン特徴量（5次元）

合計: 789次元
"""
import numpy as np
import pandas as pd
from typing import Dict, List, Optional
import logging

from src.data_preparation.transformer_encoder import TransformerEncoder
from src.data_preparation.excitement_analyzer import ExcitementAnalyzer

logger = logging.getLogger(__name__)


class ExcitementDetector:
    """音声内容から盛り上がり度特徴量を生成するクラス"""
    
    # Feature dimensions
    EMBEDDING_DIM = 768  # BERT embeddings
    BASIC_STATS_DIM = 5  # Basic statistics
    SIMULTANEOUS_DIM = 1  # Simultaneous speech count
    EMOTION_DIM = 5  # Emotion-based features
    TOPIC_CHANGE_DIM = 5  # Topic change features
    SPEECH_PATTERN_DIM = 5  # Speech pattern features
    TOTAL_DIM = 789  # Total feature dimensions
    
    def __init__(self, config: Optional[Dict] = None):
        """
        初期化
        
        Args:
            config: 設定辞書（オプション）
        """
        self.config = config or {}
        
        # Transformer encoder (lazy loading)
        self._encoder = None
        
        # Excitement analyzer
        self.analyzer = ExcitementAnalyzer()
        
        # Sampling rate (seconds)
        self.sampling_rate = self.config.get("sampling_rate", 0.1)
        
        # Feature column names
        self.feature_columns = self._define_feature_columns()
        
        logger.info(f"ExcitementDetector initialized with {self.TOTAL_DIM} feature dimensions")
    
    @property
    def encoder(self) -> TransformerEncoder:
        """Lazy load Transformer encoder"""
        if self._encoder is None:
            model_name = self.config.get("model_name", "cl-tohoku/bert-base-japanese-v3")
            self._encoder = TransformerEncoder(model_name=model_name)
            logger.info(f"Loaded Transformer encoder: {model_name}")
        return self._encoder
    
    def _define_feature_columns(self) -> List[str]:
        """
        特徴量カラム名を定義
        
        Returns:
            特徴量カラム名のリスト（789個）
        """
        columns = []
        
        # 1. Transformer embeddings (768 dims)
        columns.extend([f"speech_embedding_{i}" for i in range(self.EMBEDDING_DIM)])
        
        # 2. Basic statistics (5 dims)
        columns.extend([
            "speech_presence",
            "cumulative_speech_count",
            "time_since_last_speech",
            "speech_text_length",
            "speech_density_10s"
        ])
        
        # 3. Simultaneous speech count (1 dim)
        columns.append("simultaneous_speech_count")
        
        # 4. Emotion-based excitement (5 dims)
        columns.extend([
            "positive_emotion_intensity",
            "excited_emotion_intensity",
            "emotion_change_rate",
            "laughter_density",
            "emotion_variance_10s"
        ])
        
        # 5. Topic change detection (5 dims)
        columns.extend([
            "topic_change_rate",
            "climax_keyword_density",
            "semantic_similarity",
            "topic_shift_intensity",
            "climax_score"
        ])
        
        # 6. Speech pattern analysis (5 dims)
        columns.extend([
            "speech_burst_intensity",
            "speech_pause_frequency",
            "speech_rhythm_variance",
            "speech_acceleration",
            "burst_pattern_score"
        ])
        
        assert len(columns) == self.TOTAL_DIM, f"Expected {self.TOTAL_DIM} columns, got {len(columns)}"
        
        return columns
    
    def generate_features(
        self,
        transcription_segments: List[Dict],
        video_duration: float,
        sampling_rate: Optional[float] = None
    ) -> pd.DataFrame:
        """
        盛り上がり度特徴量を生成
        
        Args:
            transcription_segments: Whisper文字起こしセグメントのリスト
                各セグメント: {"start": float, "end": float, "text": str}
            video_duration: 動画の長さ（秒）
            sampling_rate: サンプリングレート（秒）、デフォルトは0.1
        
        Returns:
            特徴量DataFrame（カラム: time + 789特徴量）
        """
        if sampling_rate is None:
            sampling_rate = self.sampling_rate
        
        logger.info(f"Generating excitement features for {len(transcription_segments)} segments")
        logger.info(f"Video duration: {video_duration:.2f}s, Sampling rate: {sampling_rate}s")
        
        # Create time array
        timestamps = self._create_timestamps(video_duration, sampling_rate)
        n_frames = len(timestamps)
        
        # Initialize feature array
        features = np.zeros((n_frames, self.TOTAL_DIM))
        
        # If no transcription data, return zero-filled features
        if not transcription_segments:
            logger.warning("No transcription segments provided. Returning zero-filled features.")
            return self._create_feature_dataframe(timestamps, features)
        
        # Filter and validate segments
        valid_segments = self._filter_valid_segments(transcription_segments)
        if len(valid_segments) < len(transcription_segments):
            logger.warning(
                f"Filtered out {len(transcription_segments) - len(valid_segments)} invalid segments"
            )
        
        if not valid_segments:
            logger.warning("No valid segments after filtering. Returning zero-filled features.")
            return self._create_feature_dataframe(timestamps, features)
        
        # Compute basic statistics
        features = self._compute_basic_statistics(features, timestamps, valid_segments)
        
        # Compute Transformer embeddings
        features = self._compute_transformer_embeddings(features, timestamps, valid_segments)
        
        # Compute emotion-based features
        features = self._compute_emotion_features(features, timestamps, valid_segments)
        
        # Compute topic change features
        features = self._compute_topic_change_features(features, timestamps, valid_segments)
        
        # Compute speech pattern features
        features = self._compute_speech_pattern_features(features, timestamps, valid_segments)
        
        logger.info(f"Generated {n_frames} frames with {self.TOTAL_DIM} features")
        
        return self._create_feature_dataframe(timestamps, features)
    
    def _create_timestamps(self, video_duration: float, sampling_rate: float) -> np.ndarray:
        """
        タイムスタンプ配列を作成
        
        Args:
            video_duration: 動画の長さ（秒）
            sampling_rate: サンプリングレート（秒）
        
        Returns:
            タイムスタンプ配列
        """
        n_frames = int(np.ceil(video_duration / sampling_rate)) + 1
        timestamps = np.arange(n_frames) * sampling_rate
        return timestamps
    
    def _create_feature_dataframe(
        self,
        timestamps: np.ndarray,
        features: np.ndarray
    ) -> pd.DataFrame:
        """
        特徴量DataFrameを作成
        
        Args:
            timestamps: タイムスタンプ配列
            features: 特徴量配列 (n_frames, n_features)
        
        Returns:
            特徴量DataFrame
        """
        df = pd.DataFrame(features, columns=self.feature_columns)
        df.insert(0, "time", timestamps)
        return df

    
    def _filter_valid_segments(self, segments: List[Dict]) -> List[Dict]:
        """
        無効なセグメントをフィルタリング
        
        Args:
            segments: 文字起こしセグメントのリスト
        
        Returns:
            有効なセグメントのリスト
        """
        valid_segments = []
        
        for seg in segments:
            # Check for required fields
            if "start" not in seg or "end" not in seg or "text" not in seg:
                continue
            
            # Check for valid timestamps
            start = seg["start"]
            end = seg["end"]
            if start < 0 or end <= start:
                continue
            
            # Check for non-empty text
            text = seg["text"].strip()
            if not text:
                continue
            
            valid_segments.append({
                "start": start,
                "end": end,
                "text": text
            })
        
        return valid_segments
    
    def _compute_basic_statistics(
        self,
        features: np.ndarray,
        timestamps: np.ndarray,
        segments: List[Dict]
    ) -> np.ndarray:
        """
        基本統計特徴量を計算
        
        Args:
            features: 特徴量配列 (n_frames, n_features)
            timestamps: タイムスタンプ配列
            segments: 有効な文字起こしセグメント
        
        Returns:
            更新された特徴量配列
        """
        n_frames = len(timestamps)
        
        # Feature indices
        idx_presence = self.EMBEDDING_DIM + 0
        idx_cumulative = self.EMBEDDING_DIM + 1
        idx_time_since = self.EMBEDDING_DIM + 2
        idx_text_length = self.EMBEDDING_DIM + 3
        idx_density = self.EMBEDDING_DIM + 4
        idx_simultaneous = self.EMBEDDING_DIM + 5
        
        # Initialize tracking variables
        cumulative_count = 0
        last_speech_time = -1.0
        
        # Create segment mapping for each timestamp
        for i, t in enumerate(timestamps):
            # Find segments that contain this timestamp
            active_segments = []
            for seg in segments:
                if seg["start"] <= t < seg["end"]:
                    active_segments.append(seg)
            
            # Speech presence
            if active_segments:
                features[i, idx_presence] = 1.0
                cumulative_count += 1
                last_speech_time = t
                
                # Text length (use first segment if multiple)
                features[i, idx_text_length] = len(active_segments[0]["text"])
                
                # Simultaneous speech count
                features[i, idx_simultaneous] = len(active_segments)
            else:
                features[i, idx_presence] = 0.0
                features[i, idx_text_length] = 0.0
                features[i, idx_simultaneous] = 0.0
            
            # Cumulative speech count
            features[i, idx_cumulative] = cumulative_count
            
            # Time since last speech
            if last_speech_time >= 0:
                features[i, idx_time_since] = t - last_speech_time
            else:
                features[i, idx_time_since] = 0.0
        
        # Compute speech density (10-second moving average)
        window_size = int(10.0 / self.sampling_rate)  # 10 seconds
        for i in range(n_frames):
            start_idx = max(0, i - window_size)
            end_idx = i + 1
            
            # Count characters in window
            char_count = 0
            for j in range(start_idx, end_idx):
                char_count += features[j, idx_text_length]
            
            # Compute density (chars per second)
            window_duration = (end_idx - start_idx) * self.sampling_rate
            if window_duration > 0:
                features[i, idx_density] = char_count / window_duration
            else:
                features[i, idx_density] = 0.0
        
        return features

    
    def _compute_transformer_embeddings(
        self,
        features: np.ndarray,
        timestamps: np.ndarray,
        segments: List[Dict]
    ) -> np.ndarray:
        """
        Transformer埋め込みを計算
        
        Args:
            features: 特徴量配列 (n_frames, n_features)
            timestamps: タイムスタンプ配列
            segments: 有効な文字起こしセグメント
        
        Returns:
            更新された特徴量配列
        """
        n_frames = len(timestamps)
        
        # 各タイムスタンプに対して埋め込みを計算
        for i, t in enumerate(timestamps):
            # このタイムスタンプに含まれるセグメントを見つける
            active_segments = []
            for seg in segments:
                if seg["start"] <= t < seg["end"]:
                    active_segments.append(seg)
            
            if active_segments:
                # 複数セグメントがある場合は平均プーリング
                texts = [seg["text"] for seg in active_segments]
                embeddings = self.encoder.encode_batch(texts, batch_size=32)
                
                if len(embeddings) > 0:
                    # 平均プーリング
                    mean_embedding = np.mean(embeddings, axis=0)
                    features[i, 0:self.EMBEDDING_DIM] = mean_embedding
        
        return features
    
    def _compute_emotion_features(
        self,
        features: np.ndarray,
        timestamps: np.ndarray,
        segments: List[Dict]
    ) -> np.ndarray:
        """
        感情ベース特徴量を計算
        
        Args:
            features: 特徴量配列 (n_frames, n_features)
            timestamps: タイムスタンプ配列
            segments: 有効な文字起こしセグメント
        
        Returns:
            更新された特徴量配列
        """
        n_frames = len(timestamps)
        
        # Feature indices
        idx_positive = self.EMBEDDING_DIM + 6 + 0  # After basic stats + simultaneous
        idx_excited = self.EMBEDDING_DIM + 6 + 1
        idx_emotion_change = self.EMBEDDING_DIM + 6 + 2
        idx_laughter = self.EMBEDDING_DIM + 6 + 3
        idx_emotion_var = self.EMBEDDING_DIM + 6 + 4
        
        # 各セグメントの感情分析結果を保存
        segment_emotions = []
        for seg in segments:
            emotion = self.analyzer.analyze_emotion(seg["text"], language="ja")
            laughter = self.analyzer.detect_laughter(seg["text"], language="ja")
            segment_emotions.append({
                "start": seg["start"],
                "end": seg["end"],
                "positive": emotion["positive"],
                "excited": emotion["excited"],
                "laughter": laughter
            })
        
        # 各タイムスタンプに対して感情特徴量を計算
        for i, t in enumerate(timestamps):
            # このタイムスタンプに含まれるセグメントを見つける
            active_emotions = []
            for emo in segment_emotions:
                if emo["start"] <= t < emo["end"]:
                    active_emotions.append(emo)
            
            if active_emotions:
                # 平均を取る
                features[i, idx_positive] = np.mean([e["positive"] for e in active_emotions])
                features[i, idx_excited] = np.mean([e["excited"] for e in active_emotions])
                features[i, idx_laughter] = np.mean([e["laughter"] for e in active_emotions])
        
        # 感情変化率を計算
        for i in range(1, n_frames):
            if features[i, idx_positive] > 0 or features[i-1, idx_positive] > 0:
                change = abs(features[i, idx_positive] - features[i-1, idx_positive]) + \
                        abs(features[i, idx_excited] - features[i-1, idx_excited])
                features[i, idx_emotion_change] = change
        
        # 感情分散を計算（10秒ウィンドウ）
        window_size = int(10.0 / self.sampling_rate)
        for i in range(n_frames):
            start_idx = max(0, i - window_size)
            end_idx = i + 1
            
            window_positive = features[start_idx:end_idx, idx_positive]
            window_excited = features[start_idx:end_idx, idx_excited]
            
            if len(window_positive) > 1:
                var = np.var(window_positive) + np.var(window_excited)
                features[i, idx_emotion_var] = var
        
        return features
    
    def _compute_topic_change_features(
        self,
        features: np.ndarray,
        timestamps: np.ndarray,
        segments: List[Dict]
    ) -> np.ndarray:
        """
        トピック変化特徴量を計算
        
        Args:
            features: 特徴量配列 (n_frames, n_features)
            timestamps: タイムスタンプ配列
            segments: 有効な文字起こしセグメント
        
        Returns:
            更新された特徴量配列
        """
        n_frames = len(timestamps)
        
        # Feature indices
        idx_topic_change = self.EMBEDDING_DIM + 6 + 5 + 0
        idx_climax_density = self.EMBEDDING_DIM + 6 + 5 + 1
        idx_semantic_sim = self.EMBEDDING_DIM + 6 + 5 + 2
        idx_topic_shift = self.EMBEDDING_DIM + 6 + 5 + 3
        idx_climax_score = self.EMBEDDING_DIM + 6 + 5 + 4
        
        # 各セグメントのクライマックス密度を計算
        segment_climax = []
        for seg in segments:
            climax = self.analyzer.detect_climax_keywords(seg["text"], language="ja")
            segment_climax.append({
                "start": seg["start"],
                "end": seg["end"],
                "climax": climax,
                "text": seg["text"]
            })
        
        # 各タイムスタンプに対してトピック変化特徴量を計算
        for i, t in enumerate(timestamps):
            # このタイムスタンプに含まれるセグメントを見つける
            active_segments = []
            for seg in segment_climax:
                if seg["start"] <= t < seg["end"]:
                    active_segments.append(seg)
            
            if active_segments:
                # クライマックス密度
                features[i, idx_climax_density] = np.mean([s["climax"] for s in active_segments])
                
                # クライマックススコア（感情と組み合わせ）
                idx_excited = self.EMBEDDING_DIM + 6 + 1
                idx_text_length = self.EMBEDDING_DIM + 3
                
                speech_rate = features[i, idx_text_length] / max(0.1, self.sampling_rate)
                emotion_score = features[i, idx_excited]
                climax_score = (features[i, idx_climax_density] * 0.4 + 
                               emotion_score * 0.4 + 
                               min(1.0, speech_rate / 20.0) * 0.2)
                features[i, idx_climax_score] = climax_score
        
        # セマンティック類似度を計算（連続セグメント間）
        if len(segments) > 1:
            for i in range(len(segments) - 1):
                seg1 = segments[i]
                seg2 = segments[i + 1]
                
                # 埋め込みを取得
                emb1 = self.encoder.encode(seg1["text"])
                emb2 = self.encoder.encode(seg2["text"])
                
                # コサイン類似度
                if np.linalg.norm(emb1) > 0 and np.linalg.norm(emb2) > 0:
                    similarity = np.dot(emb1, emb2) / (np.linalg.norm(emb1) * np.linalg.norm(emb2))
                else:
                    similarity = 0.0
                
                # seg2の時間範囲に類似度を割り当て
                for j, t in enumerate(timestamps):
                    if seg2["start"] <= t < seg2["end"]:
                        features[j, idx_semantic_sim] = similarity
                        
                        # トピックシフト強度（低い類似度 = 高いシフト）
                        features[j, idx_topic_shift] = 1.0 - similarity
        
        # トピック変化率を計算
        window_size = int(10.0 / self.sampling_rate)
        for i in range(n_frames):
            start_idx = max(0, i - window_size)
            end_idx = i + 1
            
            window_shift = features[start_idx:end_idx, idx_topic_shift]
            if len(window_shift) > 0:
                # 高いシフト値の頻度
                high_shift_count = np.sum(window_shift > 0.5)
                features[i, idx_topic_change] = high_shift_count / len(window_shift)
        
        return features
    
    def _compute_speech_pattern_features(
        self,
        features: np.ndarray,
        timestamps: np.ndarray,
        segments: List[Dict]
    ) -> np.ndarray:
        """
        発話パターン特徴量を計算
        
        Args:
            features: 特徴量配列 (n_frames, n_features)
            timestamps: タイムスタンプ配列
            segments: 有効な文字起こしセグメント
        
        Returns:
            更新された特徴量配列
        """
        n_frames = len(timestamps)
        
        # Feature indices
        idx_burst = self.EMBEDDING_DIM + 6 + 5 + 5 + 0
        idx_pause_freq = self.EMBEDDING_DIM + 6 + 5 + 5 + 1
        idx_rhythm_var = self.EMBEDDING_DIM + 6 + 5 + 5 + 2
        idx_acceleration = self.EMBEDDING_DIM + 6 + 5 + 5 + 3
        idx_burst_pattern = self.EMBEDDING_DIM + 6 + 5 + 5 + 4
        
        # 各セグメントの発話速度を計算
        segment_rates = []
        for seg in segments:
            duration = seg["end"] - seg["start"]
            if duration > 0:
                rate = len(seg["text"]) / duration
                segment_rates.append({
                    "start": seg["start"],
                    "end": seg["end"],
                    "rate": rate
                })
        
        # 各タイムスタンプに対して発話パターン特徴量を計算
        for i, t in enumerate(timestamps):
            # このタイムスタンプに含まれるセグメントを見つける
            active_rates = []
            for sr in segment_rates:
                if sr["start"] <= t < sr["end"]:
                    active_rates.append(sr["rate"])
            
            if active_rates:
                avg_rate = np.mean(active_rates)
                
                # バースト強度（高速発話）
                burst_threshold = 20.0  # 文字/秒
                if avg_rate > burst_threshold:
                    features[i, idx_burst] = min(1.0, (avg_rate - burst_threshold) / burst_threshold)
        
        # ポーズ頻度を計算
        if len(segments) > 1:
            for i in range(len(segments) - 1):
                pause_start = segments[i]["end"]
                pause_end = segments[i + 1]["start"]
                pause_duration = pause_end - pause_start
                
                # ポーズ期間中のタイムスタンプにポーズ頻度を記録
                for j, t in enumerate(timestamps):
                    if pause_start <= t < pause_end:
                        features[j, idx_pause_freq] = 1.0
        
        # リズム分散を計算（10秒ウィンドウ）
        window_size = int(10.0 / self.sampling_rate)
        for i in range(n_frames):
            start_idx = max(0, i - window_size)
            end_idx = i + 1
            
            # ウィンドウ内の発話速度の分散
            window_rates = []
            for j in range(start_idx, end_idx):
                t = timestamps[j]
                for sr in segment_rates:
                    if sr["start"] <= t < sr["end"]:
                        window_rates.append(sr["rate"])
                        break
            
            if len(window_rates) > 1:
                features[i, idx_rhythm_var] = np.var(window_rates) / 100.0  # 正規化
        
        # 発話加速度を計算
        for i in range(1, n_frames):
            # 現在と前のタイムスタンプの発話速度を比較
            curr_rates = []
            prev_rates = []
            
            t_curr = timestamps[i]
            t_prev = timestamps[i-1]
            
            for sr in segment_rates:
                if sr["start"] <= t_curr < sr["end"]:
                    curr_rates.append(sr["rate"])
                if sr["start"] <= t_prev < sr["end"]:
                    prev_rates.append(sr["rate"])
            
            if curr_rates and prev_rates:
                curr_avg = np.mean(curr_rates)
                prev_avg = np.mean(prev_rates)
                
                if curr_avg > prev_avg:
                    features[i, idx_acceleration] = min(1.0, (curr_avg - prev_avg) / 10.0)
        
        # バーストパターンスコア（リズム分散と感情を組み合わせ）
        idx_excited = self.EMBEDDING_DIM + 6 + 1
        for i in range(n_frames):
            burst_score = features[i, idx_burst]
            rhythm_score = features[i, idx_rhythm_var]
            emotion_score = features[i, idx_excited]
            
            pattern_score = (burst_score * 0.4 + rhythm_score * 0.3 + emotion_score * 0.3)
            features[i, idx_burst_pattern] = pattern_score
        
        return features
