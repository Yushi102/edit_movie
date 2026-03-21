"""
Progress Reporter for Pipeline Execution

パイプラインの進捗を表示
"""
import time
from typing import Optional
import logging

logger = logging.getLogger(__name__)


class ProgressReporter:
    """パイプラインの進捗を表示"""
    
    def __init__(self, total_steps: int):
        """
        ProgressReporterを初期化
        
        Args:
            total_steps: 総ステップ数
        """
        self.total_steps = total_steps
        self.current_step = 0
        self.step_start_time = None
        self.pipeline_start_time = time.time()
    
    def start_step(self, step_name: str, estimated_time: str):
        """
        ステップ開始を表示
        
        Args:
            step_name: ステップ名
            estimated_time: 推定時間
        """
        self.current_step += 1
        self.step_start_time = time.time()
        
        print(f"\n{'='*70}")
        print(f"Step {self.current_step}/{self.total_steps}: {step_name}")
        print(f"Estimated time: {estimated_time}")
        print(f"{'='*70}\n")
        
        logger.info(f"Starting step {self.current_step}/{self.total_steps}: {step_name}")
    
    def complete_step(self, step_name: str, elapsed_time: float):
        """
        ステップ完了を表示
        
        Args:
            step_name: ステップ名
            elapsed_time: 経過時間（秒）
        """
        elapsed_str = self._format_time(elapsed_time)
        
        print(f"\n[DONE] {step_name} completed in {elapsed_str}\n")
        
        logger.info(f"Step completed: {step_name} ({elapsed_str})")
    
    def skip_step(self, step_name: str, reason: str):
        """
        ステップスキップを表示
        
        Args:
            step_name: ステップ名
            reason: スキップ理由
        """
        self.current_step += 1
        
        print(f"\n{'='*70}")
        print(f"Step {self.current_step}/{self.total_steps}: {step_name}")
        print(f"[SKIP] Skipped: {reason}")
        print(f"{'='*70}\n")
        
        logger.info(f"Step skipped: {step_name} ({reason})")
    
    def report_error(self, step_name: str, error: str):
        """
        エラーを表示
        
        Args:
            step_name: ステップ名
            error: エラーメッセージ
        """
        print(f"\n{'='*70}")
        print(f"[ERROR] Error in {step_name}")
        print(f"{'='*70}")
        print(f"Error: {error}")
        print(f"\nTo resume from this step, run:")
        print(f"  python scripts/train_pipeline_onebutton.py --resume")
        print(f"{'='*70}\n")
        
        logger.error(f"Error in {step_name}: {error}")
    
    def report_summary(self, success: bool, total_time: float):
        """
        最終サマリーを表示
        
        Args:
            success: 成功した場合True
            total_time: 総実行時間（秒）
        """
        total_time_str = self._format_time(total_time)
        
        print(f"\n{'='*70}")
        if success:
            print(f"[DONE] Pipeline completed successfully!")
        else:
            print(f"[FAILED] Pipeline failed")
        print(f"Total time: {total_time_str}")
        print(f"{'='*70}\n")
        
        if success:
            logger.info(f"Pipeline completed successfully in {total_time_str}")
        else:
            logger.error(f"Pipeline failed after {total_time_str}")
    
    def _format_time(self, seconds: float) -> str:
        """
        時間を読みやすい形式にフォーマット
        
        Args:
            seconds: 秒数
        
        Returns:
            フォーマットされた時間文字列
        """
        if seconds < 60:
            return f"{seconds:.1f}s"
        elif seconds < 3600:
            minutes = seconds / 60
            return f"{minutes:.1f}m"
        else:
            hours = seconds / 3600
            return f"{hours:.1f}h"
    
    def get_elapsed_time(self) -> float:
        """
        現在のステップの経過時間を取得
        
        Returns:
            経過時間（秒）
        """
        if self.step_start_time is None:
            return 0.0
        return time.time() - self.step_start_time
    
    def get_total_elapsed_time(self) -> float:
        """
        パイプライン全体の経過時間を取得
        
        Returns:
            経過時間（秒）
        """
        return time.time() - self.pipeline_start_time
