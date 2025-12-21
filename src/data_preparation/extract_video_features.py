"""
Video Feature Extraction Script (統合版)

動画から以下の特徴量を抽出します：
【音声特徴量】(7次元)
- audio_energy_rms: RMS energy (音量)
- audio_is_speaking: 発話検出 (0/1)
- silence_duration_ms: 無音時間 (ミリ秒)
- speaker_id: 話者ID (プレースホルダ)
- text_is_active: テキスト検出 (0/1)
- text_word: 単語数

【視覚特徴量】(522次元)
- scene_change: シーン転換スコア
- visual_motion: 動き量
- saliency_x, saliency_y: 注目点座標
- face_count: 顔の数
- face_center_x, face_center_y: 顔の中心座標
- face_size: 顔のサイズ
- face_mouth_open: 口の開き具合
- face_eyebrow_raise: 眉の上がり具合
- clip_0 ~ clip_511: CLIP embeddings (512次元)

出力形式: CSV (10 FPS サンプリング)
合計: 529次元 (audio: 7, visual: 522)
"""
import os
import argparse
import math
import numpy as np
import pandas as pd
import cv2
import librosa
import soundfile as sf
import whisper
import torch
import mediapipe as mp
from PIL import Image
from pydub import AudioSegment
from transformers import CLIPProcessor, CLIPModel
from pathlib import Path
from tqdm import tqdm
from typing import List, Dict, Any
import warnings
warnings.filterwarnings('ignore')

# ==========================================
# 設定
# ==========================================
TIME_STEP = 0.1              # 基本サンプリング (0.1秒 = 10 FPS)
CLIP_STEP = 1.0              # CLIP解析間隔 (1.0秒)
ANALYSIS_WIDTH = 640         # 解析用画像の横幅 (高速化)
WHISPER_MODEL_SIZE = "small" # Whisperモデル ("tiny", "base", "small", "medium", "large")
CLIP_MODEL_NAME = "openai/clip-vit-base-patch32"
USE_GPU = torch.cuda.is_available()

print("="*70)
print("Video Feature Extraction Script")
print("="*70)
print(f"Device: {'GPU' if USE_GPU else 'CPU'}")
if USE_GPU:
    print(f"GPU: {torch.cuda.get_device_name(0)}")
print(f"Sampling Rate: {1.0/TIME_STEP:.1f} FPS")
print(f"CLIP Interval: {CLIP_STEP}s")
print(f"Analysis Width: {ANALYSIS_WIDTH}px")
print(f"Whisper Model: {WHISPER_MODEL_SIZE}")
print("="*70 + "\n")

# ==========================================
# モデル初期化
# ==========================================
class FeatureExtractor:
    """動画から音声・視覚特徴量を抽出"""
    
    def __init__(self):
        print("Initializing models...")
        
        # 1. Whisper (音声認識)
        print(f"[1/3] Loading Whisper model ({WHISPER_MODEL_SIZE})...")
        self.whisper_model = whisper.load_model(WHISPER_MODEL_SIZE)
        print("  ✓ Whisper loaded")
        
        # 2. CLIP (視覚的意味表現)
        print("[2/3] Loading CLIP model...")
        try:
            self.clip_model = CLIPModel.from_pretrained(CLIP_MODEL_NAME, use_safetensors=True)
            self.clip_processor = CLIPProcessor.from_pretrained(CLIP_MODEL_NAME)
        except:
            self.clip_model = CLIPModel.from_pretrained(CLIP_MODEL_NAME)
            self.clip_processor = CLIPProcessor.from_pretrained(CLIP_MODEL_NAME)
        
        if USE_GPU:
            self.clip_model = self.clip_model.to("cuda")
            torch.backends.cudnn.benchmark = True
        print("  ✓ CLIP loaded")
        
        # 3. MediaPipe Face Mesh (顔検出)
        print("[3/3] Loading MediaPipe FaceMesh...")
        try:
            self.mp_face_mesh = mp.solutions.face_mesh
            self.face_mesh = self.mp_face_mesh.FaceMesh(
                static_image_mode=False,
                max_num_faces=1,
                refine_landmarks=True,
                min_detection_confidence=0.5
            )
            print("  ✓ MediaPipe loaded")
        except Exception as e:
            print(f"  ⚠ MediaPipe initialization failed: {e}")
            print("  Continuing without face detection...")
            print("  Face features will be set to default values (zeros)")
            print("  Common causes:")
            print("    - Non-ASCII characters in project path (日本語など)")
            print("    - Missing dependencies: pip install mediapipe opencv-contrib-python")
            print("  To fix path issue: Move project to ASCII-only path")
            self.face_mesh = None
        print("All models loaded!\n")
    
    # ==========================================
    # 音声特徴量抽出
    # ==========================================
    def extract_audio_features(self, video_path: str) -> pd.DataFrame:
        """音声特徴量を抽出"""
        print("Extracting Audio Features...")
        
        temp_wav = "temp_audio.wav"
        
        try:
            # 1. 音声を抽出 (16kHz, mono)
            print("  -> Extracting audio...")
            audio = AudioSegment.from_file(video_path)
            audio = audio.set_frame_rate(16000).set_channels(1)
            audio.export(temp_wav, format="wav")
            
            # 2. Librosaで読み込み
            y, sr = librosa.load(temp_wav, sr=16000)
            total_duration = librosa.get_duration(y=y, sr=sr)
            
            # 3. RMS Energy (音量)
            frame_length = int(TIME_STEP * sr)
            hop_length = frame_length
            rms = librosa.feature.rms(y=y, frame_length=frame_length, hop_length=hop_length)[0]
            
            # 4. VAD (発話検出)
            vad_threshold = 0.01
            is_speaking = (rms > vad_threshold).astype(int)
            
            # 5. 無音時間
            silence_duration_ms = []
            current_silence = 0
            for speak_flag in is_speaking:
                if speak_flag == 0:
                    current_silence += int(TIME_STEP * 1000)
                else:
                    current_silence = 0
                silence_duration_ms.append(current_silence)
            
            # 6. Whisper文字起こし
            print("  -> Running Whisper transcription...")
            whisper_results = self._get_whisper_features(temp_wav)
            df_text = self._align_text_features(whisper_results, total_duration)
            
            # 7. DataFrameに統合
            min_len = min(len(rms), len(df_text))
            
            df_audio = pd.DataFrame({
                'time': df_text['time'][:min_len],
                'audio_energy_rms': rms[:min_len],
                'audio_is_speaking': is_speaking[:min_len],
                'silence_duration_ms': silence_duration_ms[:min_len],
                'speaker_id': np.nan,  # プレースホルダ
                'text_is_active': df_text['text_is_active'][:min_len],
                'text_word': df_text['text_word'][:min_len]
            })
            
            print(f"  ✓ Audio features extracted: {len(df_audio)} timesteps")
            return df_audio
            
        finally:
            if os.path.exists(temp_wav):
                os.remove(temp_wav)
    
    def _get_whisper_features(self, audio_path: str) -> List[Dict[str, Any]]:
        """Whisperで文字起こし"""
        try:
            result = self.whisper_model.transcribe(audio_path, word_timestamps=True)
            word_list = []
            for segment in result.get('segments', []):
                for word_info in segment.get('words', []):
                    word_list.append({
                        "word": word_info['word'],
                        "start": word_info['start'],
                        "end": word_info['end']
                    })
            return word_list
        except Exception as e:
            print(f"  ⚠ Whisper error: {e}")
            return []
    
    def _align_text_features(self, whisper_results: List[Dict], total_duration: float) -> pd.DataFrame:
        """Whisper結果を時系列データに変換"""
        num_steps = int(math.ceil(total_duration / TIME_STEP))
        time_points = [round(i * TIME_STEP, 6) for i in range(num_steps + 1)]
        
        text_records = []
        for t in time_points:
            current_word = np.nan
            is_active = 0
            
            for w in whisper_results:
                if w['start'] <= t < w['end']:
                    current_word = w['word'].strip()
                    is_active = 1
                    break
            
            text_records.append({
                'time': t,
                'text_is_active': is_active,
                'text_word': current_word
            })
        
        return pd.DataFrame(text_records)
    
    # ==========================================
    # 視覚特徴量抽出
    # ==========================================
    def extract_visual_features(self, video_path: str) -> pd.DataFrame:
        """視覚特徴量を抽出"""
        print("Extracting Visual Features...")
        
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            raise ValueError(f"Cannot open video: {video_path}")
        
        fps = cap.get(cv2.CAP_PROP_FPS)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        duration = total_frames / fps
        
        print(f"  Video: {fps:.2f} FPS, {total_frames} frames, {duration:.2f}s")
        
        frame_step = int(fps * TIME_STEP)
        if frame_step < 1:
            frame_step = 1
        
        records = []
        current_frame_idx = 0
        
        # 状態保持
        prev_gray = None
        prev_hist = None
        last_clip_emb = [0.0] * 512
        last_clip_time = -999.0
        
        print(f"  Processing frames (step={frame_step})...")
        pbar = tqdm(total=total_frames, desc="  Frames")
        
        while True:
            ret, raw_frame = cap.read()
            if not ret:
                break
            
            pbar.update(1)
            
            if current_frame_idx % frame_step == 0:
                timestamp = current_frame_idx / fps
                
                # リサイズして高速化
                h_raw, w_raw = raw_frame.shape[:2]
                scale = ANALYSIS_WIDTH / w_raw
                h_new = int(h_raw * scale)
                frame = cv2.resize(raw_frame, (ANALYSIS_WIDTH, h_new))
                
                # 1. シーン転換
                scene_score, prev_hist = self._calculate_scene_change(prev_hist, frame)
                
                # 2. 動き & 注目点
                motion, sal_x, sal_y, prev_gray = self._calculate_motion_and_saliency(prev_gray, frame)
                
                # 3. 顔・表情
                frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                face_data = self._extract_face_features(frame_rgb, h_new, ANALYSIS_WIDTH)
                
                # 4. CLIP (1秒ごと)
                if (timestamp - last_clip_time) >= CLIP_STEP:
                    last_clip_emb = self._extract_clip_features(frame_rgb)
                    last_clip_time = timestamp
                
                # レコード作成
                row = {
                    'time': round(timestamp, 3),
                    'scene_change': scene_score,
                    'visual_motion': motion,
                    'saliency_x': sal_x,
                    'saliency_y': sal_y,
                    **face_data
                }
                row.update({f'clip_{i}': v for i, v in enumerate(last_clip_emb)})
                records.append(row)
            
            current_frame_idx += 1
        
        pbar.close()
        cap.release()
        
        df = pd.DataFrame(records)
        print(f"  ✓ Visual features extracted: {len(df)} timesteps")
        return df
    
    def _calculate_scene_change(self, prev_hist, curr_frame):
        """シーン転換スコア"""
        curr_hsv = cv2.cvtColor(curr_frame, cv2.COLOR_BGR2HSV)
        curr_hist = cv2.calcHist([curr_hsv], [0, 1], None, [32, 32], [0, 180, 0, 256])
        cv2.normalize(curr_hist, curr_hist, 0, 1, cv2.NORM_MINMAX)
        
        if prev_hist is None:
            return 0.0, curr_hist
        
        similarity = cv2.compareHist(prev_hist, curr_hist, cv2.HISTCMP_CORREL)
        return max(0.0, 1.0 - similarity), curr_hist
    
    def _calculate_motion_and_saliency(self, prev_gray, curr_frame):
        """動きと注目点"""
        curr_gray = cv2.cvtColor(curr_frame, cv2.COLOR_BGR2GRAY)
        
        if prev_gray is None:
            return 0.0, np.nan, np.nan, curr_gray
        
        diff = cv2.absdiff(prev_gray, curr_gray)
        _, thresh = cv2.threshold(diff, 25, 255, cv2.THRESH_BINARY)
        
        non_zero = cv2.countNonZero(thresh)
        h, w = diff.shape
        motion = non_zero / (h * w)
        
        M = cv2.moments(thresh)
        if M["m00"] > 0:
            cX = (M["m10"] / M["m00"]) / w
            cY = (M["m01"] / M["m00"]) / h
        else:
            cX, cY = np.nan, np.nan
        
        return motion, cX, cY, curr_gray
    
    def _extract_face_features(self, frame_rgb, h, w):
        """顔特徴量"""
        if self.face_mesh is None:
            return {
                'face_count': 0,
                'face_center_x': np.nan,
                'face_center_y': np.nan,
                'face_size': 0.0,
                'face_mouth_open': 0.0,
                'face_eyebrow_raise': 0.0
            }
        
        results = self.face_mesh.process(frame_rgb)
        
        face_data = {
            'face_count': 0,
            'face_center_x': np.nan,
            'face_center_y': np.nan,
            'face_size': 0.0,
            'face_mouth_open': 0.0,
            'face_eyebrow_raise': 0.0
        }
        
        if results.multi_face_landmarks:
            face_data['face_count'] = len(results.multi_face_landmarks)
            lm = results.multi_face_landmarks[0].landmark
            
            # 座標
            xs = [l.x for l in lm]
            ys = [l.y for l in lm]
            face_data['face_center_x'] = sum(xs) / len(xs)
            face_data['face_center_y'] = sum(ys) / len(ys)
            face_data['face_size'] = (max(xs) - min(xs)) * (max(ys) - min(ys))
            
            # 口の開き
            upper = np.array([lm[13].x * w, lm[13].y * h])
            lower = np.array([lm[14].x * w, lm[14].y * h])
            left = np.array([lm[61].x * w, lm[61].y * h])
            right = np.array([lm[291].x * w, lm[291].y * h])
            face_data['face_mouth_open'] = np.linalg.norm(upper - lower) / (np.linalg.norm(left - right) + 1e-6)
            
            # 眉の上がり
            left_eye = np.array([lm[159].x * w, lm[159].y * h])
            left_brow = np.array([lm[65].x * w, lm[65].y * h])
            face_h = abs(lm[152].y * h - lm[10].y * h)
            face_data['face_eyebrow_raise'] = np.linalg.norm(left_eye - left_brow) / (face_h + 1e-6)
        
        return face_data
    
    def _extract_clip_features(self, frame_rgb):
        """CLIP embeddings"""
        pil_img = Image.fromarray(frame_rgb)
        inputs = self.clip_processor(images=pil_img, return_tensors="pt")
        
        if USE_GPU:
            inputs = {k: v.to("cuda") for k, v in inputs.items()}
        
        with torch.no_grad():
            img_feats = self.clip_model.get_image_features(**inputs)
        
        return img_feats.cpu().numpy().flatten().tolist()
    
    # ==========================================
    # 統合処理
    # ==========================================
    def extract_all_features(self, video_path: str, output_path: str = None):
        """全特徴量を抽出してCSVに保存"""
        if not os.path.exists(video_path):
            raise FileNotFoundError(f"Video not found: {video_path}")
        
        print("\n" + "="*70)
        print(f"Processing: {Path(video_path).name}")
        print("="*70 + "\n")
        
        # 音声特徴量
        df_audio = self.extract_audio_features(video_path)
        
        # 視覚特徴量
        df_visual = self.extract_visual_features(video_path)
        
        # 統合
        print("\nMerging features...")
        min_len = min(len(df_audio), len(df_visual))
        df_audio = df_audio.iloc[:min_len].reset_index(drop=True)
        df_visual = df_visual.iloc[:min_len].reset_index(drop=True)
        
        # timeカラムを削除してから結合
        df_visual = df_visual.drop(columns=['time'])
        df_final = pd.concat([df_audio, df_visual], axis=1)
        
        # 出力
        if output_path is None:
            output_path = f"{Path(video_path).stem}_features.csv"
        
        df_final.to_csv(output_path, index=False, float_format='%.6f')
        
        print("\n" + "="*70)
        print("Extraction Complete!")
        print("="*70)
        print(f"Output: {output_path}")
        print(f"Timesteps: {len(df_final)}")
        print(f"Total Features: {len(df_final.columns)}")
        print(f"  - Audio: 7 (rms, speaking, silence, speaker_id, text_active, text_word)")
        print(f"  - Visual Scalar: 10 (scene, motion, saliency, face)")
        print(f"  - CLIP: 512")
        print(f"  - Total: {len(df_final.columns)}")
        print("="*70 + "\n")
        
        return df_final


# ==========================================
# メイン処理
# ==========================================
def main():
    parser = argparse.ArgumentParser(description="Extract audio and visual features from video")
    parser.add_argument("video", type=str, help="Path to video file")
    parser.add_argument("--output", type=str, default=None, help="Output CSV path")
    
    args = parser.parse_args()
    
    extractor = FeatureExtractor()
    extractor.extract_all_features(args.video, args.output)


if __name__ == "__main__":
    main()
