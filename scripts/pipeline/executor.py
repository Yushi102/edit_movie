"""
Step Executor for Pipeline

各ステップを実行するためのラッパー
"""
import subprocess
import sys
import logging
from pathlib import Path
from typing import Optional

logger = logging.getLogger(__name__)


class StepExecutor:
    """各ステップを実行"""
    
    # スクリプトパスの定義
    SCRIPTS = {
        "feature_extraction": "scripts/utilities/extract_all_features.py",
        "label_extraction": "scripts/extract_active_labels.py",
        "temporal_features": "scripts/add_temporal_features.py",
        "dataset_creation": "scripts/create_cut_selection_data_enhanced_fullvideo.py",
        "training": "src/cut_selection/training/train_cut_selection_fullvideo_v2.py"
    }
    
    @staticmethod
    def _run_script(script_path: str, args: list = None, cwd: Optional[str] = None) -> bool:
        """
        Pythonスクリプトを実行
        
        Args:
            script_path: スクリプトのパス
            args: コマンドライン引数のリスト
            cwd: 作業ディレクトリ
        
        Returns:
            成功した場合True、失敗した場合False
        """
        # プロジェクトルートを取得
        if cwd is None:
            project_root = Path(__file__).parent.parent.parent
            cwd = str(project_root)
        
        # スクリプトの絶対パスを構築
        script_full_path = Path(cwd) / script_path
        
        if not script_full_path.exists():
            logger.error(f"Script not found: {script_full_path}")
            return False
        
        # コマンドを構築
        cmd = [sys.executable, str(script_full_path)]
        if args:
            cmd.extend(args)
        
        logger.info(f"Executing: {' '.join(cmd)}")
        
        # 環境変数にPYTHONPATHを追加
        import os
        env = os.environ.copy()
        env['PYTHONPATH'] = cwd
        
        try:
            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                encoding='utf-8',
                errors='replace',
                cwd=cwd,
                env=env
            )
            
            # 標準出力をログに記録
            if result.stdout:
                for line in result.stdout.splitlines():
                    if line.strip():  # 空行をスキップ
                        logger.info(f"  {line}")
            
            # エラー出力をログに記録
            if result.stderr:
                for line in result.stderr.splitlines():
                    if line.strip():  # 空行をスキップ
                        logger.warning(f"  {line}")
            
            if result.returncode != 0:
                logger.error(f"Script failed with return code {result.returncode}")
                return False
            
            return True
            
        except Exception as e:
            logger.error(f"Failed to execute script: {e}")
            return False
    
    @staticmethod
    def execute_feature_extraction() -> bool:
        """
        特徴量抽出を実行
        
        Returns:
            成功した場合True
        """
        logger.info("Executing feature extraction...")
        script_path = StepExecutor.SCRIPTS["feature_extraction"]
        return StepExecutor._run_script(script_path)
    
    @staticmethod
    def execute_label_extraction() -> bool:
        """
        ラベル抽出を実行
        
        Returns:
            成功した場合True
        """
        logger.info("Executing label extraction...")
        
        # extract_active_labels.py を実行（引数なし）
        # スクリプトは以下のデフォルトパスを使用:
        # - XML: data/raw/editxml/
        # - 特徴量: data/processed/source_features/
        # - 出力: data/processed/active_labels/
        
        script_path = StepExecutor.SCRIPTS["label_extraction"]
        return StepExecutor._run_script(script_path)
    
    @staticmethod
    def execute_temporal_features() -> bool:
        """
        時系列特徴量追加を実行
        
        Returns:
            成功した場合True
        """
        logger.info("Executing temporal features addition...")
        script_path = StepExecutor.SCRIPTS["temporal_features"]
        return StepExecutor._run_script(script_path)
    
    @staticmethod
    def execute_dataset_creation() -> bool:
        """
        データセット作成を実行
        
        Returns:
            成功した場合True
        """
        logger.info("Executing dataset creation...")
        script_path = StepExecutor.SCRIPTS["dataset_creation"]
        return StepExecutor._run_script(script_path)
    
    @staticmethod
    def execute_training(config_path: str = "configs/config_cut_selection_fullvideo.yaml") -> bool:
        """
        トレーニングを実行
        
        Args:
            config_path: 設定ファイルのパス
        
        Returns:
            成功した場合True
        """
        logger.info("Executing training...")
        
        args = ["--config", config_path]
        
        script_path = StepExecutor.SCRIPTS["training"]
        return StepExecutor._run_script(script_path, args)
