#!/usr/bin/env python
"""
Auto Training Loop - Recall-First Strategy

Early stopping後に自動で分析・調整・再起動するループ。

優先順位:
  1. recall >= TARGET_RECALL (本番運用に必要)
  2. precision >= MIN_PRECISION (あまりに低いと使えない)
  3. F1最大化

調整ロジック:
  - recall低い  → inactive_weight_multiplier を下げる（activeを取りやすくする）
  - recall OK & precision低すぎ → inactive_weight_multiplier を少し上げる
  - 過学習      → dropout / weight_decay / label_smoothing を増やす
  - 未学習      → dropout を下げる / LR を上げる
  - 変化なし    → LRをリセットして局所最適脱出

Usage:
    python scripts/auto_train_loop.py
    python scripts/auto_train_loop.py --max-rounds 8
    python scripts/auto_train_loop.py --target-recall 0.75 --min-precision 0.35
"""
import argparse
import subprocess
import sys
import os
import time
import shutil
import logging
from pathlib import Path
import yaml
import pandas as pd

Path('outputs').mkdir(exist_ok=True)

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler('outputs/auto_train_loop.log', encoding='utf-8')
    ]
)
logger = logging.getLogger(__name__)

# ─────────────────────────────────────────────
# 目標値（コマンドライン引数で上書き可能）
# ─────────────────────────────────────────────
DEFAULT_TARGET_RECALL = 0.75   # これを下回ったら積極的に調整
DEFAULT_MIN_PRECISION = 0.35   # これを下回ったら少しprecision改善
DEFAULT_MAX_ROUNDS = 8


# ─────────────────────────────────────────────
# Config I/O
# ─────────────────────────────────────────────

def load_config(config_path: Path) -> dict:
    with open(config_path, 'r', encoding='utf-8') as f:
        return yaml.safe_load(f)


def save_config(config_path: Path, cfg: dict):
    with open(config_path, 'w', encoding='utf-8') as f:
        yaml.dump(cfg, f, allow_unicode=True, default_flow_style=False, sort_keys=False)


# ─────────────────────────────────────────────
# Metrics
# ─────────────────────────────────────────────

def load_metrics(metrics_csv: Path):
    if not metrics_csv.exists():
        return None
    df = pd.read_csv(metrics_csv)
    return df if len(df) >= 3 else None


def diagnose(df: pd.DataFrame, target_recall: float, min_precision: float,
             max_active_ratio: float = 0.60) -> dict:
    """
    メトリクスを分析して問題を診断する。

    best_epoch は「recall >= target_recall かつ active率 <= max_active_ratio」の中でF1最大を優先。
    """
    cond = (df['val_recall'] >= target_recall) & (df['pred_active_ratio'] <= max_active_ratio)
    recall_ok_rows = df[cond]
    if len(recall_ok_rows) > 0:
        best_idx = recall_ok_rows['val_f1'].idxmax()
        recall_satisfied = True
        active_ok = True
    else:
        recall_only = df[df['val_recall'] >= target_recall]
        if len(recall_only) > 0:
            best_idx = recall_only['val_f1'].idxmax()
            recall_satisfied = True
            active_ok = float(df.loc[best_idx, 'pred_active_ratio']) <= max_active_ratio
        else:
            best_idx = df['val_recall'].idxmax()
            recall_satisfied = False
            active_ok = float(df.loc[best_idx, 'pred_active_ratio']) <= max_active_ratio

    best = df.loc[best_idx]
    last_n = df.tail(10)
    gap = last_n['val_loss'].mean() - last_n['train_loss'].mean()

    best_f1           = float(best['val_f1'])
    best_precision    = float(best['val_precision'])
    best_recall       = float(best['val_recall'])
    pred_active_ratio = float(best['pred_active_ratio'])
    max_recall_ever   = float(df['val_recall'].max())

    overfitting       = gap > 0.08
    underfitting      = best_f1 < 0.35 and gap < 0.03
    recall_low        = best_recall < target_recall
    precision_too_low = best_precision < min_precision
    active_too_high   = pred_active_ratio > max_active_ratio

    issues = []
    if recall_low:
        issues.append(f"recall低({best_recall:.3f} < {target_recall})")
    if active_too_high:
        issues.append(f"active取りすぎ({pred_active_ratio:.2%} > {max_active_ratio:.0%})")
    if precision_too_low:
        issues.append(f"precision極低({best_precision:.3f} < {min_precision})")
    if overfitting:
        issues.append(f"過学習(gap={gap:.3f})")
    if underfitting:
        issues.append(f"未学習(F1={best_f1:.3f})")

    return dict(
        best_f1=best_f1,
        best_precision=best_precision,
        best_recall=best_recall,
        max_recall_ever=max_recall_ever,
        pred_active_ratio=pred_active_ratio,
        max_active_ratio=max_active_ratio,
        recall_satisfied=recall_satisfied,
        active_ok=active_ok,
        overfitting=overfitting,
        underfitting=underfitting,
        recall_low=recall_low,
        precision_too_low=precision_too_low,
        active_too_high=active_too_high,
        gap=gap,
        diagnosis=", ".join(issues) if issues else "OK",
        total_epochs=len(df),
    )


# ─────────────────────────────────────────────
# Config 調整ロジック
# ─────────────────────────────────────────────

def adjust_config(cfg: dict, diag: dict, round_num: int) -> tuple:
    """
    診断結果に基づいてconfigを調整する。

    優先順位:
      1. active取りすぎ → inactive_weight_multiplierを上げる（recallより優先）
      2. recall低い     → inactive_weight_multiplierを下げる
      3. recall OK & precision極低 → 少しだけ上げる
      4. recall OK & active OK → F1最大化（precision/recallバランス微調整）
      5. 過学習 / 未学習 → dropout / LR調整
    """
    changes = []
    mult = cfg.get('inactive_weight_multiplier', 1.2)

    # ── 1. recall低い → inactive_weight_multiplierを下げる ──
    if diag['recall_low']:
        recall = diag['best_recall']
        target = diag.get('target_recall', 0.75)
        gap_r = target - recall
        if gap_r > 0.25:
            delta = -1.0
        elif gap_r > 0.15:
            delta = -0.5
        else:
            delta = -0.3
        old = mult
        mult = max(0.3, round(mult + delta, 1))
        if mult != old:
            changes.append(
                f"inactive_weight_multiplier {old} → {mult} "
                f"(recall改善: {recall:.3f} → 目標{target})"
            )

    # ── 2. recall OK & precision極低 → 少しだけ上げる ──
    elif diag['precision_too_low']:
        old = mult
        mult = min(5.0, round(mult + 0.3, 1))
        if mult != old:
            changes.append(
                f"inactive_weight_multiplier {old} → {mult} "
                f"(precision改善: {diag['best_precision']:.3f})"
            )

    # ── 3. recall OK → F1最大化フェーズ ──
    elif diag['recall_satisfied']:
        precision = diag['best_precision']
        recall    = diag['best_recall']
        # precision が recall より大幅に低い → multiplierを少し下げてrecallを伸ばす
        if precision < recall * 0.7:
            old = mult
            mult = max(0.3, round(mult - 0.2, 1))
            if mult != old:
                changes.append(
                    f"inactive_weight_multiplier {old} → {mult} "
                    f"(F1改善: precision={precision:.3f} << recall={recall:.3f})"
                )
        # precision が recall より大幅に高い → multiplierを少し上げてprecisionを伸ばす
        elif precision > recall * 1.4:
            old = mult
            mult = min(5.0, round(mult + 0.2, 1))
            if mult != old:
                changes.append(
                    f"inactive_weight_multiplier {old} → {mult} "
                    f"(F1改善: precision={precision:.3f} >> recall={recall:.3f})"
                )

    cfg['inactive_weight_multiplier'] = mult

    # ── 4. 過学習対策 ──
    if diag['overfitting']:
        old = cfg['dropout']
        cfg['dropout'] = min(0.6, round(cfg['dropout'] + 0.05, 2))
        if cfg['dropout'] != old:
            changes.append(f"dropout {old} → {cfg['dropout']} (過学習対策)")

        old = cfg['weight_decay']
        cfg['weight_decay'] = min(0.3, round(cfg['weight_decay'] * 1.5, 4))
        if cfg['weight_decay'] != old:
            changes.append(f"weight_decay {old} → {cfg['weight_decay']} (過学習対策)")

        old = cfg.get('label_smoothing', 0.0)
        cfg['label_smoothing'] = min(0.2, round(old + 0.05, 2))
        if cfg['label_smoothing'] != old:
            changes.append(f"label_smoothing {old} → {cfg['label_smoothing']} (過学習対策)")

    # ── 5. 未学習対策 ──
    if diag['underfitting']:
        old = cfg['dropout']
        cfg['dropout'] = max(0.1, round(cfg['dropout'] - 0.1, 2))
        if cfg['dropout'] != old:
            changes.append(f"dropout {old} → {cfg['dropout']} (未学習対策)")

        old = cfg['learning_rate']
        cfg['learning_rate'] = min(0.001, round(old * 3.0, 6))
        if cfg['learning_rate'] != old:
            changes.append(f"learning_rate {old} → {cfg['learning_rate']} (未学習対策)")

    # ── 6. Cosine LR常に有効 ──
    cfg['use_cosine_lr'] = True

    # ── 7. early_stopping_patience: ラウンドが進むほど長く ──
    old_patience = cfg.get('early_stopping_patience', 30)
    new_patience = min(50, 30 + round_num * 5)
    if new_patience != old_patience:
        cfg['early_stopping_patience'] = new_patience
        changes.append(f"early_stopping_patience {old_patience} → {new_patience}")

    # ── 8. 変化がない場合はLRをリセット（局所最適脱出） ──
    if not changes:
        old_lr = cfg['learning_rate']
        cfg['learning_rate'] = min(0.001, round(old_lr * 2.0, 6))
        changes.append(f"learning_rate {old_lr} → {cfg['learning_rate']} (局所最適脱出)")

    return cfg, changes


# ─────────────────────────────────────────────
# Checkpoint / Training
# ─────────────────────────────────────────────

def clean_checkpoint(checkpoint_dir: Path):
    for fname in ['best_model.pth', 'training_metrics.csv', 'training_progress.png']:
        p = checkpoint_dir / fname
        if p.exists():
            p.unlink()
    logger.info("古いcheckpointを削除しました")


def backup_round(checkpoint_dir: Path, round_num: int):
    for fname in ['best_model.pth', 'training_metrics.csv']:
        src = checkpoint_dir / fname
        if src.exists():
            dst = checkpoint_dir / f'{src.stem}_round{round_num}{src.suffix}'
            shutil.copy2(src, dst)
    logger.info(f"ラウンド{round_num}のファイルをバックアップしました")


def run_training() -> int:
    cmd = [sys.executable, 'scripts/train_pipeline_onebutton.py', '--only-train']
    logger.info(f"実行: {' '.join(cmd)}")
    result = subprocess.run(cmd, cwd=str(Path(__file__).parent.parent))
    # Windows環境でsys.exit(0)が120として返ることがある
    # best_model.pthが存在すれば成功とみなす
    return result.returncode


# ─────────────────────────────────────────────
# Main Loop
# ─────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description='Recall-first auto training loop')
    parser.add_argument('--max-rounds',              type=int,   default=DEFAULT_MAX_ROUNDS)
    parser.add_argument('--config',                  type=str,   default='configs/config_cut_selection_fullvideo.yaml')
    parser.add_argument('--target-recall',           type=float, default=DEFAULT_TARGET_RECALL,
                        help=f'目標recall (default: {DEFAULT_TARGET_RECALL})')
    parser.add_argument('--min-precision',           type=float, default=DEFAULT_MIN_PRECISION,
                        help=f'最低precision (default: {DEFAULT_MIN_PRECISION})')
    parser.add_argument('--max-active-ratio',        type=float, default=0.50,
                        help='pred_active_ratioの上限 (default: 0.50 = 50%%)')
    args = parser.parse_args()

    # ── 多重起動防止ロック ──
    lock_file = Path('outputs/auto_train_loop.lock')
    if lock_file.exists():
        try:
            existing_pid = int(lock_file.read_text().strip())
            import psutil
            if psutil.pid_exists(existing_pid):
                print(f"[ERROR] auto_train_loop はすでに実行中です (PID: {existing_pid})")
                print(f"  停止するには: taskkill /PID {existing_pid} /F")
                sys.exit(1)
        except Exception:
            pass  # lockファイルが壊れていれば無視して続行
    lock_file.write_text(str(os.getpid()))

    import atexit
    atexit.register(lambda: lock_file.unlink(missing_ok=True))

    config_path = Path(args.config)
    cfg = load_config(config_path)
    checkpoint_dir = Path(cfg['checkpoint_dir'])
    metrics_csv = checkpoint_dir / 'training_metrics.csv'

    logger.info("=" * 70)
    logger.info("Auto Training Loop (Recall >= 80%%, Active <= 60%%) 開始")
    logger.info(f"  最大ラウンド数:   {args.max_rounds}")
    logger.info(f"  目標recall:      {args.target_recall:.0%}")
    logger.info(f"  active率上限:    {args.max_active_ratio:.0%}")
    logger.info(f"  最低precision:   {args.min_precision:.0%}")
    logger.info("=" * 70)

    best_recall_overall = 0.0
    best_f1_overall = 0.0
    best_round = 0

    for round_num in range(1, args.max_rounds + 1):
        logger.info(f"\n{'='*60}")
        logger.info(f"  Round {round_num} / {args.max_rounds}")
        logger.info(f"{'='*60}")

        # ラウンド2以降: 前回の結果を分析してconfig調整
        if round_num > 1:
            df = load_metrics(metrics_csv)
            if df is not None:
                diag = diagnose(df, args.target_recall, args.min_precision, args.max_active_ratio)
                diag['target_recall'] = args.target_recall

                logger.info(f"\n[METRICS] 分析結果 (ラウンド{round_num-1}):")
                logger.info(f"  Best Recall:    {diag['best_recall']:.4f}  (max ever: {diag['max_recall_ever']:.4f}  目標: {args.target_recall:.0%})")
                logger.info(f"  Active Ratio:   {diag['pred_active_ratio']:.2%}  (上限: {args.max_active_ratio:.0%})")
                logger.info(f"  Best Precision: {diag['best_precision']:.4f}")
                logger.info(f"  Best F1:        {diag['best_f1']:.4f}")
                logger.info(f"  Val-Train gap:  {diag['gap']:.4f}")
                logger.info(f"  Total Epochs:   {diag['total_epochs']}")
                logger.info(f"  Recall OK: {diag['recall_satisfied']}  Active OK: {diag['active_ok']}")
                logger.info(f"  診断: {diag['diagnosis']}")

                # ベスト更新
                if diag['best_recall'] > best_recall_overall:
                    best_recall_overall = diag['best_recall']
                    best_f1_overall = diag['best_f1']
                    best_round = round_num - 1
                    logger.info(f"  [BEST] recall更新: {best_recall_overall:.4f} / F1: {best_f1_overall:.4f} (ラウンド{best_round})")

                # 終了条件: recall達成 & precision最低ラインOK
                if diag['recall_satisfied'] and not diag['precision_too_low']:
                    logger.info(f"\n[DONE] 目標達成 (recall={diag['best_recall']:.4f} >= {args.target_recall}, "
                                f"precision={diag['best_precision']:.4f} >= {args.min_precision})")
                    logger.info("自動ループ終了")
                    break

                # config調整
                cfg = load_config(config_path)
                cfg, changes = adjust_config(cfg, diag, round_num)

                logger.info(f"\n[CONFIG] 設定変更:")
                if changes:
                    for c in changes:
                        logger.info(f"  - {c}")
                else:
                    logger.info("  変更なし")

                save_config(config_path, cfg)
            else:
                logger.warning("metricsが読めなかったためconfigを変更せず再実行")

        # checkpoint削除
        clean_checkpoint(checkpoint_dir)

        # トレーニング実行
        logger.info(f"\n[START] トレーニング開始 (ラウンド {round_num})")
        start_time = time.time()
        ret = run_training()
        elapsed = time.time() - start_time
        logger.info(f"トレーニング完了 (所要時間: {elapsed/60:.1f}分, return code: {ret})")

        # バックアップ
        backup_round(checkpoint_dir, round_num)

        # return code判定: best_model.pthが存在すれば成功とみなす（Windows環境対応）
        best_model_exists = (checkpoint_dir / 'best_model.pth').exists()
        if ret != 0 and not best_model_exists:
            logger.error(f"トレーニングがエラーで終了 (code={ret}, best_model.pth なし)。ループを停止します。")
            break
        elif ret != 0:
            logger.warning(f"return code={ret} だが best_model.pth が存在するため続行します")

    # 最終サマリー
    df = load_metrics(metrics_csv)
    if df is not None:
        diag = diagnose(df, args.target_recall, args.min_precision, args.max_active_ratio)
        if diag['best_recall'] > best_recall_overall:
            best_recall_overall = diag['best_recall']
            best_f1_overall = diag['best_f1']
            best_round = args.max_rounds

    logger.info(f"\n{'='*60}")
    logger.info(f"  Auto Training Loop 完了")
    logger.info(f"  全ラウンド中のベストRecall: {best_recall_overall:.4f} (ラウンド{best_round})")
    logger.info(f"  全ラウンド中のベストF1:     {best_f1_overall:.4f}")
    logger.info(f"{'='*60}")


if __name__ == '__main__':
    main()
