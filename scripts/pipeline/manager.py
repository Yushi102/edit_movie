"""
Pipeline Manager

パイプライン全体を管理するオーケストレーター
"""
import logging
import time
from typing import List, Optional, Dict, Callable
from .state import StateManager
from .progress import ProgressReporter
from .executor import StepExecutor

logger = logging.getLogger(__name__)


class PipelineManager:
    """パイプライン全体を管理"""
    
    # ステップ定義
    STEPS = [
        {
            "name": "feature_extraction",
            "display_name": "Feature Extraction",
            "estimated_time": "5-10 min per video",
            "executor": StepExecutor.execute_feature_extraction
        },
        {
            "name": "label_extraction",
            "display_name": "Label Extraction",
            "estimated_time": "few seconds",
            "executor": StepExecutor.execute_label_extraction
        },
        {
            "name": "temporal_features",
            "display_name": "Temporal Features Addition",
            "estimated_time": "few minutes",
            "executor": StepExecutor.execute_temporal_features
        },
        {
            "name": "dataset_creation",
            "display_name": "Dataset Creation",
            "estimated_time": "few minutes",
            "executor": StepExecutor.execute_dataset_creation
        },
        {
            "name": "training",
            "display_name": "Model Training",
            "estimated_time": "1-2 hours",
            "executor": lambda: StepExecutor.execute_training()
        }
    ]
    
    def __init__(self, config_path: str, state_manager: StateManager):
        """
        PipelineManagerを初期化
        
        Args:
            config_path: 設定ファイルのパス
            state_manager: StateManagerインスタンス
        """
        self.config_path = config_path
        self.state_manager = state_manager
        self.progress_reporter = ProgressReporter(len(self.STEPS))
    
    def run(self, skip_steps: List[str] = None, resume: bool = False, 
            only_train: bool = False, dry_run: bool = False) -> bool:
        """
        全パイプラインを実行
        
        Args:
            skip_steps: スキップするステップ名のリスト
            resume: 前回の状態から再開する場合True
            only_train: トレーニングのみを実行する場合True
            dry_run: 実行内容を表示するのみの場合True
        
        Returns:
            成功した場合True
        """
        if skip_steps is None:
            skip_steps = []
        
        # only_trainの場合、トレーニング以外をスキップ
        if only_train:
            skip_steps = ["feature_extraction", "label_extraction", 
                         "temporal_features", "dataset_creation"]
        
        logger.info("Starting pipeline execution")
        logger.info(f"Config: {self.config_path}")
        logger.info(f"Resume: {resume}")
        logger.info(f"Skip steps: {skip_steps}")
        logger.info(f"Dry run: {dry_run}")
        
        # Dry-runモードの場合
        if dry_run:
            return self._dry_run(skip_steps, resume)
        
        # 各ステップを実行
        success = True
        for step in self.STEPS:
            step_name = step["name"]
            
            # スキップ判定
            if step_name in skip_steps:
                self.progress_reporter.skip_step(
                    step["display_name"], 
                    "Specified in skip options"
                )
                continue
            
            # レジューム判定
            if resume and self.state_manager.is_step_completed(step_name):
                self.progress_reporter.skip_step(
                    step["display_name"], 
                    "Already completed"
                )
                continue
            
            # ステップ実行
            if not self.run_step(step):
                success = False
                break
        
        # サマリー表示
        total_time = self.progress_reporter.get_total_elapsed_time()
        self.progress_reporter.report_summary(success, total_time)
        
        return success
    
    def run_step(self, step: Dict) -> bool:
        """
        特定のステップを実行
        
        Args:
            step: ステップ定義辞書
        
        Returns:
            成功した場合True
        """
        step_name = step["name"]
        display_name = step["display_name"]
        estimated_time = step["estimated_time"]
        executor = step["executor"]
        
        # 進捗表示開始
        self.progress_reporter.start_step(display_name, estimated_time)
        
        # ステート更新（開始）
        self.state_manager.mark_step_started(step_name)
        
        # ステップ実行
        step_start_time = time.time()
        
        try:
            success = executor()
            
            if success:
                # 成功
                elapsed_time = time.time() - step_start_time
                self.progress_reporter.complete_step(display_name, elapsed_time)
                self.state_manager.mark_step_completed(step_name)
                return True
            else:
                # 失敗
                error_msg = f"{display_name} failed"
                self.progress_reporter.report_error(display_name, error_msg)
                self.state_manager.mark_step_failed(step_name, error_msg)
                return False
                
        except Exception as e:
            # 例外発生
            error_msg = f"Exception in {display_name}: {str(e)}"
            self.progress_reporter.report_error(display_name, error_msg)
            self.state_manager.mark_step_failed(step_name, error_msg)
            logger.exception(f"Exception in step {step_name}")
            return False
    
    def _dry_run(self, skip_steps: List[str], resume: bool) -> bool:
        """
        Dry-runモード（実行内容を表示するのみ）
        
        Args:
            skip_steps: スキップするステップ名のリスト
            resume: レジュームモードの場合True
        
        Returns:
            常にTrue
        """
        print("\n" + "="*70)
        print("DRY RUN MODE - No actual execution")
        print("="*70 + "\n")
        
        print("Pipeline execution plan:\n")
        
        for i, step in enumerate(self.STEPS, 1):
            step_name = step["name"]
            display_name = step["display_name"]
            estimated_time = step["estimated_time"]
            
            # ステータス判定
            if step_name in skip_steps:
                status = "[SKIP] (specified in options)"
            elif resume and self.state_manager.is_step_completed(step_name):
                status = "[SKIP] (already completed)"
            else:
                status = "[EXECUTE]"
            
            print(f"{i}. {display_name}")
            print(f"   Status: {status}")
            print(f"   Estimated time: {estimated_time}")
            print()
        
        print("="*70)
        print("To execute the pipeline, run without --dry-run option")
        print("="*70 + "\n")
        
        return True
    
    def get_step_by_name(self, step_name: str) -> Optional[Dict]:
        """
        ステップ名からステップ定義を取得
        
        Args:
            step_name: ステップ名
        
        Returns:
            ステップ定義辞書、見つからない場合None
        """
        for step in self.STEPS:
            if step["name"] == step_name:
                return step
        return None
