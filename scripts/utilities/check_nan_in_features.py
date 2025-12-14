"""
Check for NaN/Inf in feature CSV files
"""
import pandas as pd
import numpy as np
import glob
from pathlib import Path

files = glob.glob('input_features/*_features.csv')
files = [f for f in files if 'visual' not in f]

print(f"Checking {len(files)} files for NaN/Inf...")

problems = []

for f in files[:20]:  # Check first 20
    df = pd.read_csv(f, low_memory=False)
    
    # Check numeric columns only
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    
    # Check for NaN
    nan_counts = df[numeric_cols].isna().sum()
    has_nan = nan_counts.sum() > 0
    
    # Check for Inf
    inf_counts = np.isinf(df[numeric_cols]).sum()
    has_inf = inf_counts.sum() > 0
    
    if has_nan or has_inf:
        problems.append({
            'file': Path(f).name,
            'nan_cols': nan_counts[nan_counts > 0].to_dict() if has_nan else {},
            'inf_cols': inf_counts[inf_counts > 0].to_dict() if has_inf else {}
        })

print(f"\nFound {len(problems)} files with NaN/Inf:")

for p in problems[:10]:
    print(f"\n{p['file']}:")
    if p['nan_cols']:
        print(f"  NaN columns: {list(p['nan_cols'].keys())[:5]}")
    if p['inf_cols']:
        print(f"  Inf columns: {list(p['inf_cols'].keys())[:5]}")
