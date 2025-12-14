"""
推論パイプラインのテストスクリプト
"""
import sys
from pathlib import Path

# 前処理器ファイルを作成（トレーニングデータから）
print("=" * 70)
print("前処理器ファイルの作成")
print("=" * 70)

from multimodal_preprocessing import AudioFeaturePreprocessor, VisualFeaturePreprocessor
import numpy as np

# ダミーの前処理器を作成（実際にはトレーニング時に保存すべき）
print("\n⚠️  注意: 実際のトレーニングデータから前処理器を作成する必要があります")
print("今回はダミーの前処理器を作成します...\n")

# 音声前処理器
audio_prep = AudioFeaturePreprocessor()
dummy_audio = np.random.randn(1000, 4)
audio_prep.fit(dummy_audio)
audio_prep.save('checkpoints/audio_preprocessor.pkl')
print("✅ 音声前処理器を保存: checkpoints/audio_preprocessor.pkl")

# 映像前処理器
visual_prep = VisualFeaturePreprocessor()
dummy_visual = np.random.randn(1000, 522)
visual_prep.fit(dummy_visual)
visual_prep.save('checkpoints/visual_preprocessor.pkl')
print("✅ 映像前処理器を保存: checkpoints/visual_preprocessor.pkl")

print("\n" + "=" * 70)
print("推論パイプラインのテスト")
print("=" * 70)

# 推論パイプラインをインポート
from inference_pipeline import InferencePipeline

# パイプラインを初期化
print("\nパイプラインを初期化中...")
try:
    pipeline = InferencePipeline(
        model_path='checkpoints/best_model.pth',
        device='cpu',
        fps=10.0,
        num_tracks=20
    )
    print("✅ パイプライン初期化成功！")
    print(f"   モデルタイプ: {pipeline.config.get('model_type', 'unknown')}")
    print(f"   デバイス: {pipeline.device}")
    print(f"   FPS: {pipeline.fps}")
    
except Exception as e:
    print(f"❌ エラー: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

print("\n" + "=" * 70)
print("テスト完了")
print("=" * 70)
print("\n使用方法:")
print("  python inference_pipeline.py <動画ファイル> --model checkpoints/best_model.pth")
print("\n例:")
print("  python inference_pipeline.py test_video.mp4 --output test_output.xml")
