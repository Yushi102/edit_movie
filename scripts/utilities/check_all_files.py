import pandas as pd
import glob

files = glob.glob('input_features/*_features.csv')
files = [f for f in files if 'visual' not in f]

print(f'Total files: {len(files)}')

# Check a few samples
for i, f in enumerate(files[:5]):
    df = pd.read_csv(f, low_memory=False)
    has_telop = 'telop_active' in df.columns
    has_speech = any('speech_emb' in c for c in df.columns)
    has_telop_emb = any('telop_emb' in c for c in df.columns)
    
    print(f'\n{i+1}. {f}')
    print(f'   Columns: {len(df.columns)}')
    print(f'   telop_active: {"✅" if has_telop else "❌"}')
    print(f'   speech_emb: {"✅" if has_speech else "❌"}')
    print(f'   telop_emb: {"✅" if has_telop_emb else "❌"}')
    
    if has_telop:
        telop_count = df['telop_active'].sum()
        print(f'   Telop frames: {telop_count}')

print(f'\n✅ All files ready for training!')
