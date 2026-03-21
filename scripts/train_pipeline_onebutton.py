#!/usr/bin/env python
"""
One-Button Training Pipeline

特徴量抽出からトレーニングまでを1コマンドで実行

Usage:
    python scripts/train_pipeline_onebutton.py [OPTIONS]

Examples:
    # 全プロセスを実行
    python scripts/train_pipeline_onebutton.py
    
    # 音声分離を有効化（Whisper精度向上）
    python scripts/train_pipeline_onebutton.py --enable-audio-separation
    
    # 音声分離を高品質モードで実行
    python scripts/train_pipeline_onebutton.py --enable-audio-separation --audio-separation-quality high
    
    # 特徴量抽出をスキップ
    python scripts/train_pipeline_onebutton.py --skip-extraction
    
    # トレーニングのみ
    python scripts/train_pipeline_onebutton.py --only-train
    
    # 前回から再開
    python scripts/train_pipeline_onebutton.py --resume
    
    # 実行内容を確認
    python scripts/train_pipeline_onebutton.py --dry-run
"""
import argparse
import logging
import sys
from pathlib import Path

# プロジェクトルートをパスに追加
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from scripts.pipeline.manager import PipelineManager
from scripts.pipeline.state import StateManager


def setup_logging(verbose: bool = False):
    """
    ロギングを設定
    
    Args:
        verbose: 詳細なログを表示する場合True
    """
    log_level = logging.DEBUG if verbose else logging.INFO
    
    # ロギング設定
    logging.basicConfig(
        level=log_level,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.StreamHandler(sys.stdout),
            logging.FileHandler('outputs/pipeline_log.txt', encoding='utf-8')
        ]
    )
    
    # サードパーティライブラリのログレベルを調整
    logging.getLogger('urllib3').setLevel(logging.WARNING)
    logging.getLogger('transformers').setLevel(logging.WARNING)
    logging.getLogger('torch').setLevel(logging.WARNING)


def parse_args():
    """
    コマンドライン引数を解析
    
    Returns:
        解析された引数
    """
    parser = argparse.ArgumentParser(
        description="One-button training pipeline for cut selection model",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # 全プロセスを実行
  python scripts/train_pipeline_onebutton.py
  
  # 音声分離を有効化（Whisper精度向上）
  python scripts/train_pipeline_onebutton.py --enable-audio-separation
  
  # 音声分離を高品質モードで実行
  python scripts/train_pipeline_onebutton.py --enable-audio-separation --audio-separation-quality high
  
  # 特徴量抽出をスキップ
  python scripts/train_pipeline_onebutton.py --skip-extraction
  
  # トレーニングのみ
  python scripts/train_pipeline_onebutton.py --only-train
  
  # 前回から再開
  python scripts/train_pipeline_onebutton.py --resume
  
  # 実行内容を確認
  python scripts/train_pipeline_onebutton.py --dry-run
        """
    )
    
    # 設定ファイル
    parser.add_argument(
        '--config',
        type=str,
        default='configs/config_cut_selection_fullvideo.yaml',
        help='Path to config file (default: configs/config_cut_selection_fullvideo.yaml)'
    )
    
    # ステップスキップオプション
    parser.add_argument(
        '--skip-extraction',
        action='store_true',
        help='Skip feature extraction step'
    )
    
    parser.add_argument(
        '--skip-labels',
        action='store_true',
        help='Skip label extraction step'
    )
    
    parser.add_argument(
        '--skip-temporal',
        action='store_true',
        help='Skip temporal features addition step'
    )
    
    parser.add_argument(
        '--skip-dataset',
        action='store_true',
        help='Skip dataset creation step'
    )
    
    parser.add_argument(
        '--only-train',
        action='store_true',
        help='Execute only training step (skip all preprocessing)'
    )
    
    # 実行制御オプション
    parser.add_argument(
        '--resume',
        action='store_true',
        help='Resume from previous state (skip completed steps)'
    )
    
    parser.add_argument(
        '--reset',
        action='store_true',
        help='Reset state and start from beginning'
    )
    
    parser.add_argument(
        '--dry-run',
        action='store_true',
        help='Show execution plan without actually running'
    )
    
    # ロギングオプション
    parser.add_argument(
        '--verbose',
        action='store_true',
        help='Enable verbose logging'
    )
    
    # 音声分離オプション
    parser.add_argument(
        '--enable-audio-separation',
        action='store_true',
        help='Enable audio separation for Whisper (improves transcription accuracy)'
    )
    
    parser.add_argument(
        '--audio-separation-quality',
        type=str,
        choices=['fast', 'balanced', 'high'],
        default='balanced',
        help='Audio separation quality preset (default: balanced)'
    )
    
    return parser.parse_args()


def main():
    """
    メイン処理
    """
    # 引数解析
    args = parse_args()
    
    # ロギング設定
    setup_logging(args.verbose)
    logger = logging.getLogger(__name__)
    
    logger.info("="*70)
    logger.info("One-Button Training Pipeline")
    logger.info("="*70)
    
    # 音声分離の設定
    if args.enable_audio_separation:
        import os
        os.environ['ENABLE_AUDIO_SEPARATION'] = 'true'
        os.environ['AUDIO_SEPARATION_QUALITY'] = args.audio_separation_quality
        logger.info(f"Audio separation enabled: quality={args.audio_separation_quality}")
        print(f"🎵 Audio separation enabled (quality: {args.audio_separation_quality})")
        print("   This will improve Whisper transcription accuracy by separating game audio from voice")
        print()
    
    # 出力ディレクトリを作成
    Path("outputs").mkdir(exist_ok=True)
    
    # StateManagerを初期化
    state_manager = StateManager()
    
    # リセットオプション
    if args.reset:
        logger.info("Resetting pipeline state...")
        state_manager.reset_state()
        print("[DONE] Pipeline state has been reset\n")
    
    # 現在の状態を表示
    if not args.dry_run:
        print(state_manager.get_summary())
        print()
    
    # スキップするステップのリストを作成
    skip_steps = []
    if args.skip_extraction:
        skip_steps.append("feature_extraction")
    if args.skip_labels:
        skip_steps.append("label_extraction")
    if args.skip_temporal:
        skip_steps.append("temporal_features")
    if args.skip_dataset:
        skip_steps.append("dataset_creation")
    
    # PipelineManagerを初期化
    pipeline_manager = PipelineManager(args.config, state_manager)
    
    # パイプライン実行
    try:
        success = pipeline_manager.run(
            skip_steps=skip_steps,
            resume=args.resume,
            only_train=args.only_train,
            dry_run=args.dry_run
        )
        
        # 終了コード
        if success:
            logger.info("Pipeline execution completed successfully")
            sys.exit(0)
        else:
            logger.error("Pipeline execution failed")
            sys.exit(1)
            
    except KeyboardInterrupt:
        logger.warning("\nPipeline interrupted by user")
        print("\n[WARNING] Pipeline interrupted")
        print("To resume, run: python scripts/train_pipeline_onebutton.py --resume\n")
        sys.exit(130)
    
    except Exception as e:
        logger.exception("Unexpected error in pipeline execution")
        print(f"\n[ERROR] Unexpected error: {e}\n")
        sys.exit(1)


if __name__ == "__main__":
    main()
