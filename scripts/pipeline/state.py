"""
State Manager for Pipeline Execution

パイプラインの実行状態を管理し、中断と再開をサポート
"""
import json
import os
from datetime import datetime
from pathlib import Path
from typing import Dict, Optional
import logging

logger = logging.getLogger(__name__)


class StateManager:
    """パイプラインの実行状態を管理"""
    
    def __init__(self, state_file: str = "outputs/pipeline_state.json"):
        """
        StateManagerを初期化
        
        Args:
            state_file: ステートファイルのパス
        """
        self.state_file = Path(state_file)
        self.state = self._init_state()
        
        # ファイルが存在する場合は読み込み
        if self.state_file.exists():
            try:
                self.state = self.load_state()
                logger.info(f"Loaded existing state from {self.state_file}")
            except Exception as e:
                logger.warning(f"Failed to load state file, using default: {e}")
    
    def _init_state(self) -> dict:
        """
        初期ステートを作成
        
        Returns:
            初期ステート辞書
        """
        return {
            "pipeline_version": "1.0",
            "last_run": None,
            "steps": {
                "feature_extraction": {
                    "status": "pending",
                    "started_at": None,
                    "completed_at": None,
                    "error": None
                },
                "label_extraction": {
                    "status": "pending",
                    "started_at": None,
                    "completed_at": None,
                    "error": None
                },
                "temporal_features": {
                    "status": "pending",
                    "started_at": None,
                    "completed_at": None,
                    "error": None
                },
                "dataset_creation": {
                    "status": "pending",
                    "started_at": None,
                    "completed_at": None,
                    "error": None
                },
                "training": {
                    "status": "pending",
                    "started_at": None,
                    "completed_at": None,
                    "error": None
                }
            }
        }
    
    def load_state(self) -> dict:
        """
        ステートファイルを読み込み
        
        Returns:
            ステート辞書
        
        Raises:
            FileNotFoundError: ファイルが存在しない場合
            json.JSONDecodeError: JSONパースに失敗した場合
        """
        with open(self.state_file, 'r', encoding='utf-8') as f:
            return json.load(f)
    
    def save_state(self):
        """
        ステートファイルに保存
        """
        try:
            # 出力ディレクトリが存在しない場合は作成
            self.state_file.parent.mkdir(parents=True, exist_ok=True)
            
            # ステートを保存
            with open(self.state_file, 'w', encoding='utf-8') as f:
                json.dump(self.state, f, indent=2, ensure_ascii=False)
            
            logger.debug(f"State saved to {self.state_file}")
        except Exception as e:
            logger.error(f"Failed to save state: {e}")
    
    def mark_step_started(self, step_name: str):
        """
        ステップ開始をマーク
        
        Args:
            step_name: ステップ名
        """
        if step_name not in self.state["steps"]:
            logger.warning(f"Unknown step: {step_name}")
            return
        
        self.state["steps"][step_name]["status"] = "in_progress"
        self.state["steps"][step_name]["started_at"] = datetime.now().isoformat()
        self.state["steps"][step_name]["error"] = None
        self.state["last_run"] = datetime.now().isoformat()
        
        self.save_state()
        logger.info(f"Step started: {step_name}")
    
    def mark_step_completed(self, step_name: str):
        """
        ステップ完了をマーク
        
        Args:
            step_name: ステップ名
        """
        if step_name not in self.state["steps"]:
            logger.warning(f"Unknown step: {step_name}")
            return
        
        self.state["steps"][step_name]["status"] = "completed"
        self.state["steps"][step_name]["completed_at"] = datetime.now().isoformat()
        self.state["last_run"] = datetime.now().isoformat()
        
        self.save_state()
        logger.info(f"Step completed: {step_name}")
    
    def mark_step_failed(self, step_name: str, error: str):
        """
        ステップ失敗をマーク
        
        Args:
            step_name: ステップ名
            error: エラーメッセージ
        """
        if step_name not in self.state["steps"]:
            logger.warning(f"Unknown step: {step_name}")
            return
        
        self.state["steps"][step_name]["status"] = "failed"
        self.state["steps"][step_name]["error"] = error
        self.state["last_run"] = datetime.now().isoformat()
        
        self.save_state()
        logger.error(f"Step failed: {step_name}, Error: {error}")
    
    def is_step_completed(self, step_name: str) -> bool:
        """
        ステップが完了しているか確認
        
        Args:
            step_name: ステップ名
        
        Returns:
            完了している場合True
        """
        if step_name not in self.state["steps"]:
            return False
        
        return self.state["steps"][step_name]["status"] == "completed"
    
    def get_step_status(self, step_name: str) -> Optional[str]:
        """
        ステップの状態を取得
        
        Args:
            step_name: ステップ名
        
        Returns:
            ステータス文字列 (pending, in_progress, completed, failed)
        """
        if step_name not in self.state["steps"]:
            return None
        
        return self.state["steps"][step_name]["status"]
    
    def reset_state(self):
        """
        ステートをリセット
        """
        self.state = self._init_state()
        self.save_state()
        logger.info("State reset to initial values")
    
    def get_summary(self) -> str:
        """
        ステートのサマリーを取得
        
        Returns:
            サマリー文字列
        """
        lines = []
        lines.append("Pipeline State Summary:")
        lines.append(f"  Last run: {self.state.get('last_run', 'Never')}")
        lines.append("  Steps:")
        
        for step_name, step_info in self.state["steps"].items():
            status = step_info["status"]
            status_icon = {
                "pending": "[PENDING]",
                "in_progress": "[RUNNING]",
                "completed": "[DONE]",
                "failed": "[FAILED]"
            }.get(status, "[?]")
            
            lines.append(f"    {status_icon} {step_name}: {status}")
            
            if step_info.get("error"):
                lines.append(f"       Error: {step_info['error']}")
        
        return "\n".join(lines)
