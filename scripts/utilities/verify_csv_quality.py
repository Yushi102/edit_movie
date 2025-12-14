import pandas as pd
import numpy as np

# Load CSV
df = pd.read_csv('master_training_data.csv')

print('=' * 60)
print('Data Quality Report')
print('=' * 60)
print(f'Total rows: {len(df):,}')
print(f'Total columns: {len(df.columns)}')
print(f'Unique videos: {df["video_id"].nunique()}')
print(f'Time range: {df["time"].min():.1f}s - {df["time"].max():.1f}s')
print(f'\nMissing values: {df.isnull().sum().sum()}')

print(f'\n{"=" * 60}')
print('Active Tracks Distribution')
print('=' * 60)
for i in range(1, 6):
    col = f'target_v{i}_active'
    active_count = (df[col] == 1).sum()
    percentage = active_count / len(df) * 100
    print(f'V{i}: {active_count:,} frames ({percentage:.1f}%)')

print(f'\n{"=" * 60}')
print('AssetID Distribution (V1-V5)')
print('=' * 60)
for i in range(1, 6):
    active_col = f'target_v{i}_active'
    asset_col = f'target_v{i}_asset'
    active_df = df[df[active_col] == 1]
    if len(active_df) > 0:
        asset_dist = active_df[asset_col].value_counts().to_dict()
        print(f'V{i}: {asset_dist}')
    else:
        print(f'V{i}: No active frames')

print(f'\n{"=" * 60}')
print('Parameter Ranges (V1)')
print('=' * 60)
v1_active = df[df['target_v1_active'] == 1]
if len(v1_active) > 0:
    print(f'Scale: {v1_active["target_v1_scale"].min():.1f} - {v1_active["target_v1_scale"].max():.1f}')
    print(f'Position X: {v1_active["target_v1_x"].min():.1f} - {v1_active["target_v1_x"].max():.1f}')
    print(f'Position Y: {v1_active["target_v1_y"].min():.1f} - {v1_active["target_v1_y"].max():.1f}')
    print(f'Crop Left: {v1_active["target_v1_crop_l"].min():.1f} - {v1_active["target_v1_crop_l"].max():.1f}')

print(f'\n{"=" * 60}')
print('Sample Data (First 3 rows)')
print('=' * 60)
print(df[['video_id', 'source_video_name', 'time', 'target_v1_active', 'target_v1_asset', 'target_v1_scale']].head(3))

print(f'\n✅ CSV data quality verification complete!')
