import pandas as pd

df = pd.read_csv('input_features/bandicam 2025-03-03 22-34-57-492_features.csv')

print('全カラム:')
print(df.columns.tolist())

print('\ntext_is_activeが1の行:')
active = df[df['text_is_active'] == 1]
print(f'件数: {len(active)}')

if len(active) > 0:
    print('\n最初の10行:')
    print(active[['time', 'text_is_active', 'text_word']].head(10))
else:
    print('text_is_activeが1の行はありません')

print('\ntext_wordの統計:')
print(f'非NaN件数: {df["text_word"].notna().sum()}')
if df['text_word'].notna().sum() > 0:
    print('\n非NaNの例:')
    print(df[df['text_word'].notna()][['time', 'text_is_active', 'text_word']].head(10))
