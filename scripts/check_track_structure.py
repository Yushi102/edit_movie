import pandas as pd

df = pd.read_csv('data/processed/tracks/bandicam 2025-03-03 22-34-57-492_tracks.csv')

print(f'総行数: {len(df)}')
print(f'ユニークなtime: {df["time"].nunique()}')
print(f'ユニークなtrack: {df["track"].nunique()}')
print(f'\ntimeの値（最初の20個）:')
print(df["time"].unique()[:20])
print(f'\ntrackの値:')
print(df["track"].unique())
