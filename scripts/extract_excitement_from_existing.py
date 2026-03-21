"""
Extract excitement features from existing video features

既存の動画特徴量ファイルから盛り上がり度特徴量を抽出
"""
import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import pandas as pd
import numpy as np
from pathlib import Path
from src.data_preparation.excitement_analyzer import ExcitementAnalyzer

def extract_excitement_from_csv():
    """既存のCSVファイルから盛り上がり度特徴量を抽出"""
    
    print("=" * 70)
    print("既存動画からの盛り上がり度特徴量抽出")
    print("=" * 70)
    print()
    
    # 既存の特徴量ファイルを探す
    feature_dir = Path("data/processed/source_features")
    csv_files = list(feature_dir.glob("*.csv"))
    
    if not csv_files:
        print("✗ 特徴量ファイルが見つかりません")
        return
    
    # 最初のファイルを使用
    csv_file = csv_files[0]
    print(f"対象ファイル: {csv_file.name}")
    print()
    
    # CSVを読み込む
    print("CSVファイルを読み込み中...")
    try:
        df = pd.read_csv(csv_file)
        print(f"✓ 読み込み完了")
        print(f"  行数: {len(df)}")
        print(f"  列数: {len(df.columns)}")
        print()
    except Exception as e:
        print(f"✗ 読み込み失敗: {e}")
        return
    
    # テキスト関連のカラムを確認
    print("=" * 70)
    print("テキスト関連カラムの確認")
    print("=" * 70)
    print()
    
    text_cols = [col for col in df.columns if 'text' in col.lower() or 'speech' in col.lower() or 'whisper' in col.lower()]
    
    if text_cols:
        print(f"✓ テキスト関連カラム: {len(text_cols)}個")
        for col in text_cols[:10]:  # 最初の10個を表示
            print(f"  - {col}")
        if len(text_cols) > 10:
            print(f"  ... 他 {len(text_cols) - 10}個")
        print()
    else:
        print("✗ テキスト関連カラムが見つかりません")
        print()
    
    # 全カラム名を表示
    print("=" * 70)
    print("全カラム名（最初の30個）")
    print("=" * 70)
    print()
    
    for i, col in enumerate(df.columns[:30], 1):
        print(f"{i:2d}. {col}")
    
    if len(df.columns) > 30:
        print(f"... 他 {len(df.columns) - 30}個")
    print()
    
    # テキストデータがあるか確認
    text_col = None
    if 'text' in df.columns:
        text_col = 'text'
    elif 'text_word' in df.columns:
        text_col = 'text_word'
    elif 'telop_text' in df.columns:
        text_col = 'telop_text'
    
    if text_col:
        print("=" * 70)
        print(f"テキストデータのサンプル（カラム: {text_col}）")
        print("=" * 70)
        print()
        
        # 空でないテキストを探す
        non_empty_text = df[df[text_col].notna() & (df[text_col] != '')]
        
        if len(non_empty_text) > 0:
            print(f"✓ テキストデータあり: {len(non_empty_text)}/{len(df)} 行")
            print()
            print("サンプル（最初の10行）:")
            for idx, row in non_empty_text.head(10).iterrows():
                time = row.get('time', idx * 0.1)
                text = row[text_col]
                print(f"  {time:.1f}秒: {text[:50]}{'...' if len(str(text)) > 50 else ''}")
            print()
            
            # ExcitementAnalyzerで分析
            print("=" * 70)
            print("盛り上がり度分析")
            print("=" * 70)
            print()
            
            analyzer = ExcitementAnalyzer()
            
            # 各行を分析
            excitement_scores = []
            positive_scores = []
            excited_scores = []
            climax_scores = []
            laughter_scores = []
            
            print("分析中...")
            for idx, row in df.iterrows():
                text = row.get(text_col, '')
                
                if pd.notna(text) and str(text).strip() != '':
                    # 発話速度を推定（仮に10文字/秒）
                    speech_rate = 10.0
                    
                    analysis = analyzer.analyze_comprehensive(
                        text=str(text),
                        speech_rate=speech_rate,
                        language="ja"
                    )
                    
                    excitement_scores.append(analysis['excitement_score'])
                    positive_scores.append(analysis['positive_intensity'])
                    excited_scores.append(analysis['excited_intensity'])
                    climax_scores.append(analysis['climax_density'])
                    laughter_scores.append(analysis['laughter_density'])
                else:
                    excitement_scores.append(0.0)
                    positive_scores.append(0.0)
                    excited_scores.append(0.0)
                    climax_scores.append(0.0)
                    laughter_scores.append(0.0)
            
            # DataFrameに追加
            df['excitement_score'] = excitement_scores
            df['positive_intensity'] = positive_scores
            df['excited_intensity'] = excited_scores
            df['climax_density'] = climax_scores
            df['laughter_density'] = laughter_scores
            
            print("✓ 分析完了")
            print()
            
            # 統計情報
            print("=" * 70)
            print("統計情報")
            print("=" * 70)
            print()
            
            print(f"平均盛り上がりスコア: {np.mean(excitement_scores):.3f}")
            print(f"最大盛り上がりスコア: {np.max(excitement_scores):.3f}")
            print(f"盛り上がりシーン数 (>0.5): {sum(1 for s in excitement_scores if s > 0.5)}")
            print()
            
            print(f"平均ポジティブ強度: {np.mean(positive_scores):.3f}")
            print(f"平均興奮強度: {np.mean(excited_scores):.3f}")
            print(f"平均クライマックス密度: {np.mean(climax_scores):.3f}")
            print(f"平均笑い密度: {np.mean(laughter_scores):.3f}")
            print()
            
            # 最も盛り上がっているシーンを表示
            print("=" * 70)
            print("最も盛り上がっているシーン TOP 10")
            print("=" * 70)
            print()
            
            df_sorted = df.sort_values('excitement_score', ascending=False)
            top10 = df_sorted[df_sorted['excitement_score'] > 0].head(10)
            
            if len(top10) > 0:
                for i, (idx, row) in enumerate(top10.iterrows(), 1):
                    time = row.get('time', idx * 0.1)
                    text = row.get(text_col, '')
                    score = row['excitement_score']
                    
                    print(f"{i}. {time:.1f}秒 - スコア: {score:.3f}")
                    print(f"   テキスト: {str(text)[:60]}{'...' if len(str(text)) > 60 else ''}")
                    print(f"   ポジティブ: {row['positive_intensity']:.2f}, "
                          f"興奮: {row['excited_intensity']:.2f}, "
                          f"クライマックス: {row['climax_density']:.2f}, "
                          f"笑い: {row['laughter_density']:.2f}")
                    print()
            else:
                print("  盛り上がりシーンが検出されませんでした")
                print()
            
            # 結果を保存
            output_file = csv_file.parent / f"{csv_file.stem}_with_excitement.csv"
            df.to_csv(output_file, index=False)
            print(f"✓ 結果を保存: {output_file.name}")
            print()
            
        else:
            print("✗ テキストデータが空です")
            print()
    else:
        print("✗ テキストカラムが見つかりません")
        print()
    
    print("=" * 70)
    print("処理完了")
    print("=" * 70)

if __name__ == "__main__":
    extract_excitement_from_csv()
