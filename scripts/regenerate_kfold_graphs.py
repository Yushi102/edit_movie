"""
K-Fold結果グラフの再生成スクリプト

既存のkfold_summary.csvと各Foldの履歴から、修正されたグラフを再生成します。
"""
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from pathlib import Path
import pandas as pd
import logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# 日本語フォント対応
try:
    import japanize_matplotlib
    logger.info("✅ japanize_matplotlib loaded")
except ImportError:
    logger.warning("⚠️  japanize_matplotlib not installed")


def regenerate_comparison_graph(checkpoint_dir: Path):
    """K-Fold比較グラフを再生成"""
    
    # CSVからサマリーを読み込み
    csv_path = checkpoint_dir / 'kfold_summary.csv'
    if not csv_path.exists():
        logger.error(f"❌ {csv_path} not found")
        return
    
    df = pd.read_csv(csv_path)
    
    # 統計行を除外
    df = df[df['fold'] != 'Mean ± Std']
    
    # 数値に変換
    folds = df['fold'].astype(int).tolist()
    f1_scores = df['best_val_f1'].astype(float).tolist()
    accuracies = df['best_val_accuracy'].astype(float).tolist()
    precisions = df['best_val_precision'].astype(float).tolist()
    recalls = df['best_val_recall'].astype(float).tolist()
    thresholds = df['optimal_threshold'].astype(float).tolist()
    
    # 各Foldの履歴を読み込み
    fold_histories = []
    for fold in folds:
        fold_dir = checkpoint_dir / f"fold_{fold}"
        history_path = fold_dir / 'training_history.csv'
        if history_path.exists():
            history_df = pd.read_csv(history_path)
            fold_histories.append({
                'fold': fold,
                'history': history_df
            })
    
    # グラフ作成
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    fig.suptitle(f'K-Fold Cross Validation Results (K={len(folds)})', 
                fontsize=16, fontweight='bold')
    
    # 1. F1スコアの推移（全Fold）
    ax = axes[0, 0]
    for fold_data in fold_histories:
        fold = fold_data['fold']
        history = fold_data['history']
        ax.plot(history['epoch'], history['val_f1'], 
               label=f'Fold {fold}', linewidth=2, marker='o', markersize=3)
    ax.set_title('F1スコアの推移（全Fold）')
    ax.set_xlabel('エポック')
    ax.set_ylabel('F1 Score')
    ax.set_ylim([0, 1])
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # 2. 各Foldの最良F1スコア
    ax = axes[0, 1]
    colors = plt.cm.viridis(np.linspace(0, 1, len(folds)))
    bars = ax.bar(folds, f1_scores, color=colors, alpha=0.7, edgecolor='black')
    
    # 平均値と標準偏差を表示
    mean_f1 = np.mean(f1_scores)
    std_f1 = np.std(f1_scores)
    ax.axhline(y=mean_f1, color='red', linestyle='--', linewidth=2, 
              label=f'平均: {mean_f1:.4f} ± {std_f1:.4f}')
    
    ax.set_title('各Foldの最良F1スコア')
    ax.set_xlabel('Fold')
    ax.set_ylabel('Best F1 Score')
    ax.set_ylim([0, 1])
    ax.set_xticks(folds)
    ax.legend()
    ax.grid(True, alpha=0.3, axis='y')
    
    # 各バーに値を表示
    for i, (fold, f1) in enumerate(zip(folds, f1_scores)):
        ax.text(fold, f1 + 0.02, f'{f1:.4f}', 
               ha='center', va='bottom', fontsize=9)
    
    # 3. Precision vs Recall（各Foldの最良値）
    ax = axes[1, 0]
    colors = plt.cm.viridis(np.linspace(0, 1, len(folds)))
    
    for i, (fold, prec, rec, color) in enumerate(zip(folds, precisions, recalls, colors)):
        ax.scatter(rec, prec, s=200, color=color, alpha=0.7, 
                  edgecolor='black', linewidth=2, label=f'Fold {fold}', zorder=3)
        ax.text(rec, prec, f'{fold}', ha='center', va='center', 
               fontsize=10, fontweight='bold', color='white')
    
    # 平均値をプロット
    mean_prec = np.mean(precisions)
    mean_rec = np.mean(recalls)
    ax.scatter(mean_rec, mean_prec, s=300, color='red', alpha=0.8, 
              edgecolor='black', linewidth=3, marker='*', label='平均', zorder=4)
    
    ax.set_title('Precision vs Recall（各Foldの最良値）')
    ax.set_xlabel('Recall')
    ax.set_ylabel('Precision')
    ax.set_xlim([0, 1])
    ax.set_ylim([0, 1])
    ax.legend(loc='best', fontsize=9)
    ax.grid(True, alpha=0.3)
    
    # 4. 最適閾値（各Fold）
    ax = axes[1, 1]
    colors = plt.cm.viridis(np.linspace(0, 1, len(folds)))
    bars = ax.bar(folds, thresholds, color=colors, alpha=0.7, edgecolor='black')
    
    # 平均値と標準偏差を表示
    mean_threshold = np.mean(thresholds)
    std_threshold = np.std(thresholds)
    ax.axhline(y=mean_threshold, color='red', linestyle='--', linewidth=2,
              label=f'平均: {mean_threshold:.3f} ± {std_threshold:.3f}')
    
    ax.set_title('最適閾値（各Fold）')
    ax.set_xlabel('Fold')
    ax.set_ylabel('Confidence Threshold')
    ax.set_xticks(folds)
    ax.legend()
    ax.grid(True, alpha=0.3, axis='y')
    
    # 各バーに値を表示
    for i, (fold, th) in enumerate(zip(folds, thresholds)):
        ax.text(fold, th + 0.02, f'{th:.3f}', 
               ha='center', va='bottom', fontsize=9)
    
    plt.tight_layout()
    
    # 保存
    save_path = checkpoint_dir / 'kfold_comparison.png'
    fig.savefig(save_path, dpi=300, bbox_inches='tight')
    logger.info(f"✅ K-Fold comparison graph regenerated: {save_path}")
    plt.close(fig)


if __name__ == '__main__':
    checkpoint_dir = Path('checkpoints_cut_selection_kfold_enhanced')
    
    if not checkpoint_dir.exists():
        logger.error(f"❌ Checkpoint directory not found: {checkpoint_dir}")
        exit(1)
    
    logger.info(f"📊 Regenerating K-Fold comparison graph...")
    regenerate_comparison_graph(checkpoint_dir)
    logger.info(f"✅ Done!")
