"""
Test inference with real extracted features to debug NaN issue
"""
import torch
import numpy as np
from model import MultimodalTransformer
from inference_pipeline import InferencePipeline

# Initialize pipeline
pipeline = InferencePipeline(
    model_path='checkpoints/best_model.pth',
    device='cpu',
    fps=10.0
)

# Extract features from video
video_path = r"D:\切り抜き\2025-6\2025-6-02\bandicam 2025-06-02 00-03-33-780.mp4"

print("Step 1: Extracting features...")
audio_df, visual_df = pipeline._extract_features(video_path)

print(f"\nAudio features shape: {audio_df.shape}")
print(f"Visual features shape: {visual_df.shape}")

print(f"\nAudio columns: {audio_df.columns.tolist()}")
print(f"\nAudio data sample:")
print(audio_df.head())

print(f"\nAudio NaN check:")
for col in audio_df.columns:
    nan_count = audio_df[col].isna().sum()
    if nan_count > 0:
        print(f"  {col}: {nan_count} NaN values")

print(f"\nVisual NaN check:")
for col in visual_df.columns:
    nan_count = visual_df[col].isna().sum()
    if nan_count > 0:
        print(f"  {col}: {nan_count} NaN values")

print("\nStep 2: Preprocessing and aligning...")
features = pipeline._preprocess_and_align(audio_df, visual_df)

print(f"\nFeature shapes:")
for key, value in features.items():
    print(f"  {key}: {value.shape}")

print(f"\nFeature NaN check:")
for key, value in features.items():
    if torch.is_tensor(value) and torch.is_floating_point(value):
        has_nan = torch.isnan(value).any()
        has_inf = torch.isinf(value).any()
        print(f"  {key}: NaN={has_nan}, Inf={has_inf}")
        if has_nan:
            nan_count = torch.isnan(value).sum().item()
            total = value.numel()
            print(f"    NaN count: {nan_count}/{total} ({100*nan_count/total:.2f}%)")

print("\nStep 3: Running model prediction...")
predictions = pipeline._predict_with_model(features)

print(f"\nPrediction shapes:")
for key, value in predictions.items():
    print(f"  {key}: {value.shape}")

print(f"\nPrediction NaN check:")
for key, value in predictions.items():
    has_nan = np.isnan(value).any()
    has_inf = np.isinf(value).any()
    print(f"  {key}: NaN={has_nan}, Inf={has_inf}")
    if not has_nan and not has_inf:
        print(f"    Min: {value.min():.4f}, Max: {value.max():.4f}, Mean: {value.mean():.4f}")
