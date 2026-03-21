"""
Training script for FULL VIDEO cut selection model with detailed visualization

1 VIDEO = 1 SAMPLE (no sequence splitting)
Applies 90s minimum constraint PER VIDEO
Includes real-time graph updates and HTML viewer
"""
import argparse
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import yaml
import logging
from pathlib import Path
from tqdm import tqdm
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import random
import pandas as pd

import sys
sys.path.append(str(Path(__file__).parent.parent.parent))

from src.cut_selection.datasets.cut_dataset_enhanced_fullvideo import EnhancedCutSelectionDatasetFullVideo, collate_fn_fullvideo
from src.cut_selection.models.cut_model_enhanced import EnhancedCutSelectionModel
from src.cut_selection.utils.losses import CombinedCutSelectionLoss

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# 日本語フォント対応
try:
    import japanize_matplotlib
    logger.info("✅ japanize_matplotlib loaded")
except ImportError:
    logger.warning("⚠️  japanize_matplotlib not installed")


class TrainingVisualizer:
    """学習状況を可視化（6個のグラフ + リアルタイム更新）"""
    
    def __init__(self, checkpoint_dir: Path, num_epochs: int, config: dict):
        self.checkpoint_dir = checkpoint_dir
        self.num_epochs = num_epochs
        self.config = config
        
        # 学習履歴を保存
        self.history = {
            'epoch': [],
            'train_loss': [],
            'train_ce_loss': [],
            'train_tv_loss': [],
            'val_loss': [],
            'val_ce_loss': [],
            'val_tv_loss': [],
            'val_accuracy': [],
            'val_precision': [],
            'val_recall': [],
            'val_f1': [],
            'val_specificity': [],
            'val_pred_active_ratio': [],
            'val_pred_inactive_ratio': [],
            'optimal_threshold': [],
            'pred_total_duration': []
        }
        
        self.fig = None
        self.axes = None
        self.setup_plot()
    
    def setup_plot(self):
        """Initialize graph settings"""
        self.fig, self.axes = plt.subplots(3, 2, figsize=(15, 14))
        self.fig.suptitle('Full Video Training - Training Progress', fontsize=16, fontweight='bold')
        plt.tight_layout()
    
    def update(self, epoch: int, train_losses: dict, val_metrics: dict):
        """学習データを更新してグラフを再描画"""
        # データを追加
        self.history['epoch'].append(epoch)
        self.history['train_loss'].append(train_losses['total_loss'])
        self.history['train_ce_loss'].append(train_losses['ce_loss'])
        self.history['train_tv_loss'].append(train_losses['tv_loss'])
        self.history['val_loss'].append(val_metrics['loss'])
        self.history['val_ce_loss'].append(val_metrics['ce_loss'])
        self.history['val_tv_loss'].append(val_metrics['tv_loss'])
        self.history['val_accuracy'].append(val_metrics['accuracy'])
        self.history['val_precision'].append(val_metrics['precision'])
        self.history['val_recall'].append(val_metrics['recall'])
        self.history['val_f1'].append(val_metrics['f1'])
        self.history['val_specificity'].append(val_metrics['specificity'])
        self.history['val_pred_active_ratio'].append(val_metrics.get('pred_active_ratio', 0.0) * 100)
        self.history['val_pred_inactive_ratio'].append(val_metrics.get('pred_inactive_ratio', 0.0) * 100)
        self.history['optimal_threshold'].append(val_metrics.get('optimal_threshold', 0.0))
        self.history['pred_total_duration'].append(val_metrics['avg_pred_duration'])

        # figを毎回作り直す（twin軸の積み重なりを防ぐ）
        plt.close(self.fig)
        self.fig, self.axes = plt.subplots(3, 2, figsize=(15, 14))

        epochs = self.history['epoch']
        
        # 1. Loss Function
        ax = self.axes[0, 0]
        ax.plot(epochs, self.history['train_loss'], 'b-', label='Train Loss', linewidth=2)
        ax.plot(epochs, self.history['val_loss'], 'r-', label='Val Loss', linewidth=2)
        ax.set_title('Loss Function')
        ax.set_xlabel('Epoch')
        ax.set_ylabel('Loss')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        # 2. Loss Breakdown (CE Loss on left axis, TV Loss on right axis)
        ax = self.axes[0, 1]
        ax2 = ax.twinx()
        
        # CE Loss（左軸）
        line1 = ax.plot(epochs, self.history['train_ce_loss'], 'b-', label='Train CE', linewidth=2.5, marker='o', markersize=5)
        line2 = ax.plot(epochs, self.history['val_ce_loss'], 'r-', label='Val CE', linewidth=2.5, marker='o', markersize=5)
        ax.set_xlabel('Epoch')
        ax.set_ylabel('CE Loss', color='black', fontweight='bold')
        ax.tick_params(axis='y')
        
        # TV Loss（右軸）
        line3 = ax2.plot(epochs, self.history['train_tv_loss'], 'g--', label='Train TV', linewidth=2.5, marker='s', markersize=4)
        line4 = ax2.plot(epochs, self.history['val_tv_loss'], 'orange', linestyle='--', label='Val TV', linewidth=2.5, marker='^', markersize=4)
        ax2.set_ylabel('TV Loss', color='green', fontweight='bold')
        ax2.tick_params(axis='y', labelcolor='green')
        
        # 凡例を統合
        lines = line1 + line2 + line3 + line4
        labels = [l.get_label() for l in lines]
        legend = ax.legend(lines, labels, loc='upper right', fontsize=9, framealpha=0.9)
        legend.get_frame().set_facecolor('white')
        
        ax.set_title('Loss Breakdown (CE=left, TV=right)', fontweight='bold')
        ax.grid(True, alpha=0.3, linestyle=':')
        
        # 3. Classification Performance
        ax = self.axes[1, 0]
        ax.plot(epochs, self.history['val_accuracy'], 'g-', label='Accuracy', linewidth=2, marker='o')
        ax.plot(epochs, self.history['val_f1'], 'purple', label='F1 Score', linewidth=2, marker='s')
        ax.set_title('Classification Performance (Accuracy & F1)')
        ax.set_xlabel('Epoch')
        ax.set_ylabel('Score')
        ax.set_ylim([0, 1])
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        # 4. Precision, Recall, Specificity
        ax = self.axes[1, 1]
        ax.plot(epochs, self.history['val_precision'], 'b-', label='Precision', linewidth=2, marker='o')
        ax.plot(epochs, self.history['val_recall'], 'r-', label='Recall', linewidth=2, marker='s')
        ax.plot(epochs, self.history['val_specificity'], 'orange', label='Specificity', linewidth=2, marker='^')
        ax.set_title('Precision, Recall, Specificity')
        ax.set_xlabel('Epoch')
        ax.set_ylabel('Score')
        ax.set_ylim([0, 1])
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        # 5. Average Predicted Duration (Duration Constraint)
        ax = self.axes[2, 0]
        ax.plot(epochs, self.history['pred_total_duration'], 'green', label='Avg Predicted Duration', linewidth=2, marker='o')
        
        # 制約線を追加
        min_duration = self.config.get('min_total_duration', 90.0)
        target_duration = self.config.get('max_total_duration', 180.0)
        hard_max_duration = self.config.get('hard_max_duration', 200.0)
        
        ax.axhline(y=min_duration, color='red', linestyle='--', alpha=0.5, label=f'Min: {min_duration:.0f}s')
        ax.axhline(y=target_duration, color='blue', linestyle='--', alpha=0.5, label=f'Target: {target_duration:.0f}s')
        ax.axhline(y=hard_max_duration, color='orange', linestyle='--', alpha=0.5, label=f'Max: {hard_max_duration:.0f}s')
        
        ax.set_title('Average Predicted Duration (per video)')
        ax.set_xlabel('Epoch')
        ax.set_ylabel('Time (seconds)')
        ax.legend(loc='best')
        ax.grid(True, alpha=0.3)
        
        # 現在の時間を色分けして表示
        if len(epochs) > 0:
            current_duration = self.history['pred_total_duration'][-1]
            duration_text = f'Current: {current_duration:.1f}s'
            if current_duration < min_duration:
                color = 'red'
                status = '⚠️ Too Short'
            elif current_duration <= target_duration:
                color = 'green'
                status = '✅ Ideal'
            elif current_duration <= hard_max_duration:
                color = 'orange'
                status = '⚠️ Slightly Over'
            else:
                color = 'red'
                status = '❌ Over Limit'
            
            ax.text(0.02, 0.02, f'{duration_text}\n{status}', 
                   transform=ax.transAxes, fontsize=11, verticalalignment='bottom',
                   bbox=dict(boxstyle='round', facecolor=color, alpha=0.3, edgecolor=color))
        
        # 6. Prediction Active/Inactive Ratio
        ax = self.axes[2, 1]
        ax.plot(epochs, self.history['val_pred_active_ratio'], 'g-', label='Active (Adopted)', linewidth=2, marker='o')
        ax.plot(epochs, self.history['val_pred_inactive_ratio'], 'r-', label='Inactive (Not Adopted)', linewidth=2, marker='s')
        if len(epochs) > 0:
            true_active_ratio = val_metrics.get('true_active_ratio', 0) * 100
            true_inactive_ratio = val_metrics.get('true_inactive_ratio', 0) * 100
            ax.axhline(y=true_active_ratio, color='g', linestyle='--', alpha=0.5, label=f'True Active ({true_active_ratio:.1f}%)')
            ax.axhline(y=true_inactive_ratio, color='r', linestyle='--', alpha=0.5, label=f'True Inactive ({true_inactive_ratio:.1f}%)')
        ax.set_title('Prediction Active/Inactive Ratio')
        ax.set_xlabel('Epoch')
        ax.set_ylabel('Ratio (%)')
        ax.set_ylim([0, 105])
        ax.legend(loc='best')
        ax.grid(True, alpha=0.3)
        
        # タイトル更新
        best_f1 = max(self.history['val_f1'])
        best_epoch = self.history['epoch'][self.history['val_f1'].index(best_f1)]
        self.fig.suptitle(
            f'Full Video Training - Epoch {epoch}/{self.num_epochs} | Best F1: {best_f1:.4f} (Epoch {best_epoch})',
            fontsize=16,
            fontweight='bold'
        )
        
        plt.tight_layout()
        
        # 保存
        save_path = self.checkpoint_dir / 'training_progress.png'
        self.fig.savefig(save_path, dpi=100, bbox_inches='tight')
        logger.info(f"Training visualization saved: {save_path}")

        # CSVにメトリクスを追記
        csv_path = self.checkpoint_dir / 'training_metrics.csv'
        import csv, os
        write_header = not os.path.exists(csv_path)
        with open(csv_path, 'a', newline='', encoding='utf-8') as f:
            writer = csv.writer(f)
            if write_header:
                writer.writerow(['epoch','train_loss','val_loss','val_f1','val_precision','val_recall','val_accuracy','pred_active_ratio'])
            writer.writerow([
                epoch,
                round(train_losses['total_loss'], 4),
                round(val_metrics['loss'], 4),
                round(val_metrics['f1'], 4),
                round(val_metrics['precision'], 4),
                round(val_metrics['recall'], 4),
                round(val_metrics['accuracy'], 4),
                round(val_metrics.get('pred_active_ratio', 0), 4),
            ])
    
    def save_final(self):
        """最終的なグラフを高解像度で保存"""
        final_path = self.checkpoint_dir / 'training_final.png'
        self.fig.savefig(final_path, dpi=300, bbox_inches='tight')
        logger.info(f"📊 Final training visualization saved: {final_path}")
        
        # CSVとしても保存
        df = pd.DataFrame(self.history)
        csv_path = self.checkpoint_dir / 'training_history.csv'
        df.to_csv(csv_path, index=False)
        logger.info(f"📊 Training history saved: {csv_path}")
        
        plt.close(self.fig)
    
    def generate_html_viewer(self):
        """HTMLビューアーを生成（自動更新付き）"""
        import time
        timestamp = int(time.time() * 1000)
        
        html_content = f"""<!DOCTYPE html>
<html>
<head>
    <meta charset="UTF-8">
    <meta http-equiv="refresh" content="5">
    <title>Full Video Training - Real-time Progress</title>
    <style>
        body {{
            font-family: Arial, sans-serif;
            margin: 20px;
            background-color: #f5f5f5;
        }}
        h1 {{
            color: #333;
            text-align: center;
        }}
        .container {{
            display: flex;
            flex-wrap: wrap;
            gap: 20px;
            justify-content: center;
        }}
        .main-graph {{
            width: 100%;
            max-width: 1400px;
            background: white;
            padding: 20px;
            border-radius: 10px;
            box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        }}
        img {{
            width: 100%;
            height: auto;
            border-radius: 5px;
        }}
        .info {{
            text-align: center;
            color: #666;
            margin-top: 20px;
        }}
        .timestamp {{
            text-align: center;
            color: #999;
            font-size: 14px;
            margin-top: 10px;
        }}
    </style>
</head>
<body>
    <h1>🔄 Full Video Training - Real-time Progress</h1>
    <div class="info">
        <p>This page auto-refreshes every 5 seconds</p>
        <p>1 video = 1 sample | 90s constraint applied per video</p>
    </div>
    
    <div class="container">
        <div class="main-graph">
            <h2>📊 Training Progress</h2>
            <img src="training_progress.png?t={timestamp}" alt="Training Progress">
        </div>
    </div>
    
    <div class="timestamp">
        <p>Last updated: <span id="timestamp"></span></p>
    </div>
    
    <script>
        document.getElementById('timestamp').textContent = new Date().toLocaleString('en-US');
    </script>
</body>
</html>
"""
        
        html_path = self.checkpoint_dir / 'view_training.html'
        with open(html_path, 'w', encoding='utf-8') as f:
            f.write(html_content)
        
        logger.info(f"📊 HTML viewer generated: {html_path}")


def set_seed(seed: int):
    """完全な再現性のためにシードを固定"""
    import os
    os.environ['PYTHONHASHSEED'] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    logger.info(f"🎲 Random seed set to {seed} for reproducibility")


def train_epoch(model, dataloader, optimizer, criterion, device, scaler=None, max_grad_norm=1.0):
    """Train for one epoch with full videos"""
    model.train()
    total_loss = 0
    total_ce_loss = 0
    total_tv_loss = 0
    num_videos = 0
    
    for batch in tqdm(dataloader, desc="Training", leave=False):
        audio = batch['audio'].to(device)
        visual = batch['visual'].to(device)
        temporal = batch['temporal'].to(device)
        active_labels = batch['active'].to(device)
        padding_mask = batch['padding_mask'].to(device)
        
        if scaler is not None:
            with torch.amp.autocast('cuda'):
                outputs = model(audio, visual, temporal, padding_mask=padding_mask)
                active_logits = outputs['active']
                loss, loss_dict = criterion(active_logits, active_labels)
            
            optimizer.zero_grad()
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)
            scaler.step(optimizer)
            scaler.update()
        else:
            outputs = model(audio, visual, temporal, padding_mask=padding_mask)
            active_logits = outputs['active']
            loss, loss_dict = criterion(active_logits, active_labels)
            
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)
            optimizer.step()
        
        total_loss += loss_dict['total_loss']
        total_ce_loss += loss_dict['ce_loss']
        total_tv_loss += loss_dict['tv_loss']
        num_videos += audio.shape[0]
    
    return {
        'total_loss': total_loss / num_videos,
        'ce_loss': total_ce_loss / num_videos,
        'tv_loss': total_tv_loss / num_videos
    }


def validate(model, dataloader, criterion, device, config=None):
    """Validate the model with full videos - apply 90s constraint PER VIDEO"""
    model.eval()
    total_loss = 0
    total_ce_loss = 0
    total_tv_loss = 0
    
    all_predictions = []
    all_labels = []
    all_confidence_scores = []
    all_video_names = []
    all_original_lengths = []
    
    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Validation", leave=False):
            audio = batch['audio'].to(device)
            visual = batch['visual'].to(device)
            temporal = batch['temporal'].to(device)
            active_labels = batch['active'].to(device)
            padding_mask = batch['padding_mask'].to(device)
            
            outputs = model(audio, visual, temporal, padding_mask=padding_mask)
            active_logits = outputs['active']
            
            loss, loss_dict = criterion(active_logits, active_labels)
            total_loss += loss_dict['total_loss']
            total_ce_loss += loss_dict['ce_loss']
            total_tv_loss += loss_dict['tv_loss']
            
            predictions = torch.argmax(active_logits, dim=-1)
            probs = torch.softmax(active_logits, dim=-1)
            confidence_scores = probs[..., 1] - probs[..., 0]
            
            all_predictions.append(predictions.cpu().numpy())
            all_labels.append(active_labels.cpu().numpy())
            all_confidence_scores.append(confidence_scores.cpu().numpy())
            all_video_names.extend(batch['video_names'])
            all_original_lengths.extend(batch['original_lengths'])
    
    # Calculate average losses
    num_videos = len(all_video_names)
    avg_total_loss = total_loss / num_videos
    avg_ce_loss = total_ce_loss / num_videos
    avg_tv_loss = total_tv_loss / num_videos
    
    # Get duration constraint parameters
    max_total_duration = config.get('max_total_duration', 180.0) if config else 180.0
    hard_max_duration = config.get('hard_max_duration', 200.0) if config else 200.0
    min_total_duration = config.get('min_total_duration', 90.0) if config else 90.0
    fps = config.get('inference_fps', 10.0) if config else 10.0
    min_clip_duration = config.get('min_clip_duration', 3.0) if config else 3.0
    
    def extract_clips_from_predictions(predictions, fps=10.0, min_duration=3.0):
        """Extract clips from binary predictions"""
        clips = []
        in_clip = False
        start_idx = 0
        
        for i, active in enumerate(predictions):
            if active == 1 and not in_clip:
                start_idx = i
                in_clip = True
            elif active == 0 and in_clip:
                end_idx = i
                start_time = start_idx / fps
                end_time = end_idx / fps
                if (end_time - start_time) >= min_duration:
                    clips.append((start_time, end_time))
                in_clip = False
        
        # Handle last clip
        if in_clip:
            end_idx = len(predictions)
            start_time = start_idx / fps
            end_time = end_idx / fps
            if (end_time - start_time) >= min_duration:
                clips.append((start_time, end_time))
        
        return clips
    
    # Process each video separately and apply 90s constraint PER VIDEO
    video_metrics = []
    total_tp = 0
    total_fp = 0
    total_fn = 0
    total_tn = 0
    total_pred_duration = 0.0
    
    # Store per-video ratios for averaging
    video_pred_active_ratios = []
    video_pred_inactive_ratios = []
    video_true_active_ratios = []
    video_true_inactive_ratios = []
    
    for video_idx in range(num_videos):
        video_name = all_video_names[video_idx]
        original_length = all_original_lengths[video_idx]
        
        # Extract data for this video (remove padding)
        video_predictions_raw = all_predictions[video_idx]
        video_labels_raw = all_labels[video_idx]
        video_confidence_raw = all_confidence_scores[video_idx]
        
        # Handle batch dimension
        if video_predictions_raw.ndim > 1:
            video_predictions_raw = video_predictions_raw.flatten()
        if video_labels_raw.ndim > 1:
            video_labels_raw = video_labels_raw.flatten()
        if video_confidence_raw.ndim > 1:
            video_confidence_raw = video_confidence_raw.flatten()
        
        video_predictions = video_predictions_raw[:original_length]
        video_labels = video_labels_raw[:original_length]
        video_confidence = video_confidence_raw[:original_length]
        
        # Calculate video duration
        video_duration = original_length / fps

        # ── 純粋なargmax予測（active率制約なし） ──
        conf = video_confidence  # shape: (original_length,)
        optimal_predictions = (conf > 0.0).astype(int)
        
        # Calculate per-video ratios
        video_total_samples = len(optimal_predictions)
        video_pred_active_ratio = np.sum(optimal_predictions == 1) / video_total_samples
        video_pred_inactive_ratio = np.sum(optimal_predictions == 0) / video_total_samples
        video_true_active_ratio = np.sum(video_labels == 1) / video_total_samples
        video_true_inactive_ratio = np.sum(video_labels == 0) / video_total_samples
        
        video_pred_active_ratios.append(video_pred_active_ratio)
        video_pred_inactive_ratios.append(video_pred_inactive_ratio)
        video_true_active_ratios.append(video_true_active_ratio)
        video_true_inactive_ratios.append(video_true_inactive_ratio)
        
        # Extract clips and calculate duration
        clips = extract_clips_from_predictions(optimal_predictions, fps, min_clip_duration)
        pred_duration = sum(e - s for s, e in clips)
        
        # Calculate metrics for this video
        tp = np.sum((optimal_predictions == 1) & (video_labels == 1))
        fp = np.sum((optimal_predictions == 1) & (video_labels == 0))
        fn = np.sum((optimal_predictions == 0) & (video_labels == 1))
        tn = np.sum((optimal_predictions == 0) & (video_labels == 0))
        
        total_tp += tp
        total_fp += fp
        total_fn += fn
        total_tn += tn
        total_pred_duration += pred_duration
        
        video_metrics.append({
            'video_name': video_name,
            'duration': video_duration,
            'pred_duration': pred_duration,
            'tp': tp,
            'fp': fp,
            'fn': fn,
            'tn': tn
        })
    
    # Calculate overall metrics
    accuracy = (total_tp + total_tn) / (total_tp + total_fp + total_fn + total_tn) if (total_tp + total_fp + total_fn + total_tn) > 0 else 0.0
    precision = total_tp / (total_tp + total_fp) if (total_tp + total_fp) > 0 else 0.0
    recall = total_tp / (total_tp + total_fn) if (total_tp + total_fn) > 0 else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
    specificity = total_tn / (total_tn + total_fp) if (total_tn + total_fp) > 0 else 0.0
    
    # Average predicted duration per video
    avg_pred_duration = total_pred_duration / num_videos
    
    # Calculate average ratios across videos
    pred_active_ratio = np.mean(video_pred_active_ratios) if len(video_pred_active_ratios) > 0 else 0.0
    pred_inactive_ratio = np.mean(video_pred_inactive_ratios) if len(video_pred_inactive_ratios) > 0 else 0.0
    true_active_ratio = np.mean(video_true_active_ratios) if len(video_true_active_ratios) > 0 else 0.0
    true_inactive_ratio = np.mean(video_true_inactive_ratios) if len(video_true_inactive_ratios) > 0 else 0.0
    
    # Calculate optimal threshold (average across videos for display)
    optimal_threshold = 0.0  # Placeholder since we use per-video thresholds
    
    metrics = {
        'loss': avg_total_loss,
        'ce_loss': avg_ce_loss,
        'tv_loss': avg_tv_loss,
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1': f1,
        'specificity': specificity,
        'avg_pred_duration': avg_pred_duration,
        'pred_active_ratio': pred_active_ratio,
        'pred_inactive_ratio': pred_inactive_ratio,
        'true_active_ratio': true_active_ratio,
        'true_inactive_ratio': true_inactive_ratio,
        'optimal_threshold': optimal_threshold,
        'video_metrics': video_metrics
    }
    
    return metrics


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, required=True)
    args = parser.parse_args()
    
    # Load config
    with open(args.config, 'r', encoding='utf-8') as f:
        config = yaml.safe_load(f)
    
    # Set seed
    random_state = config.get('random_state', 42)
    set_seed(random_state)
    
    # Device
    device = 'cuda' if torch.cuda.is_available() and not config.get('cpu', False) else 'cpu'
    logger.info(f"Using device: {device}")
    
    # Create checkpoint directory
    checkpoint_dir = Path(config['checkpoint_dir'])
    checkpoint_dir.mkdir(exist_ok=True, parents=True)
    
    # Load datasets
    logger.info(f"\nLoading training data from {config['train_data_path']}")
    train_dataset = EnhancedCutSelectionDatasetFullVideo(config['train_data_path'])
    
    logger.info(f"\nLoading validation data from {config['val_data_path']}")
    val_dataset = EnhancedCutSelectionDatasetFullVideo(config['val_data_path'])
    
    # Create data loaders (batch_size=1 for variable length videos)
    train_loader = DataLoader(
        train_dataset,
        batch_size=config.get('batch_size', 1),
        shuffle=True,
        num_workers=config.get('num_workers', 0),
        collate_fn=collate_fn_fullvideo,
        pin_memory=True if device == 'cuda' else False
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=config.get('batch_size', 1),
        shuffle=False,
        num_workers=config.get('num_workers', 0),
        collate_fn=collate_fn_fullvideo,
        pin_memory=True if device == 'cuda' else False
    )
    
    # Create model
    model = EnhancedCutSelectionModel(
        audio_features=config['audio_features'],
        visual_features=config['visual_features'],
        temporal_features=config.get('temporal_features', 6),
        d_model=config['d_model'],
        nhead=config['nhead'],
        num_encoder_layers=config['num_encoder_layers'],
        dim_feedforward=config['dim_feedforward'],
        dropout=config['dropout']
    ).to(device)
    
    # Calculate class weights from training data
    logger.info("Calculating class weights from training data...")
    train_active_count = 0
    train_inactive_count = 0
    
    for item in train_dataset:
        active_labels = item['active'].numpy()
        train_active_count += np.sum(active_labels == 1)
        train_inactive_count += np.sum(active_labels == 0)
    
    total_train_samples = train_active_count + train_inactive_count
    base_weight_active = total_train_samples / (2 * train_active_count) if train_active_count > 0 else 1.0
    base_weight_inactive = total_train_samples / (2 * train_inactive_count) if train_inactive_count > 0 else 1.0
    
    weight_active = base_weight_active * 1.0
    inactive_multiplier = config.get('inactive_weight_multiplier', 2.0)
    weight_inactive = base_weight_inactive * inactive_multiplier  # precision改善のためinactiveを重視
    class_weights = torch.tensor([weight_inactive, weight_active], device=device)
    
    logger.info(f"  Train Active: {train_active_count:,} ({train_active_count/total_train_samples*100:.2f}%)")
    logger.info(f"  Train Inactive: {train_inactive_count:,} ({train_inactive_count/total_train_samples*100:.2f}%)")
    logger.info(f"  Class weights: [inactive={weight_inactive:.4f}, active={weight_active:.4f}] (multiplier={inactive_multiplier})")
    
    # Loss and optimizer
    criterion = CombinedCutSelectionLoss(
        class_weights=class_weights,
        tv_weight=config.get('tv_weight', 0.0),
        label_smoothing=config.get('label_smoothing', 0.0),
        use_focal=config.get('use_focal_loss', False),
        focal_alpha=config.get('focal_alpha', 0.25),
        focal_gamma=config.get('focal_gamma', 2.0),
        target_duration=config.get('max_total_duration', 180.0),
        duration_penalty_weight=config.get('duration_penalty_weight', 0.5),
        recall_reward_weight=config.get('recall_reward_weight', 5.0),
        fps=config.get('inference_fps', 10.0),
        min_clip_duration=config.get('min_clip_duration', 3.0)
    )
    
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=config['learning_rate'],
        weight_decay=config.get('weight_decay', 0.01)
    )
    
    # LR Scheduler
    use_cosine_lr = config.get('use_cosine_lr', False)
    if use_cosine_lr:
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer,
            T_max=config['num_epochs'],
            eta_min=config['learning_rate'] * 0.01
        )
        logger.info(f"📈 Cosine Annealing LR scheduler enabled (T_max={config['num_epochs']}, eta_min={config['learning_rate']*0.01:.2e})")
    else:
        scheduler = None
    
    # Training setup
    # Recall-first: best model = recall >= target の中でF1最大、なければrecall最大
    target_recall = config.get('target_recall', 0.75)
    min_precision = config.get('min_precision', 0.35)
    best_val_f1 = 0.0
    best_val_recall = 0.0
    best_score = -1.0  # recall_ok時はF1、そうでなければrecall
    patience_counter = 0
    use_amp = config.get('use_amp', False) and device == 'cuda'
    scaler = torch.amp.GradScaler('cuda') if use_amp else None
    max_grad_norm = config.get('max_grad_norm', 1.0)
    
    if scaler:
        logger.info("📊 Mixed Precision Training enabled")
    
    # Initialize visualizer
    visualizer = TrainingVisualizer(checkpoint_dir, config['num_epochs'], config)
    logger.info("📊 Training visualizer initialized (6 graphs + HTML viewer)")
    
    # Training loop
    logger.info(f"\n{'='*80}")
    logger.info(f"Starting training for {config['num_epochs']} epochs")
    logger.info(f"{'='*80}\n")
    
    for epoch in range(config['num_epochs']):
        # Train
        train_losses = train_epoch(model, train_loader, optimizer, criterion, device, scaler, max_grad_norm)
        
        # Validate
        val_metrics = validate(model, val_loader, criterion, device, config)
        
        logger.info(f"Epoch {epoch+1}/{config['num_epochs']} - "
                   f"Train Loss: {train_losses['total_loss']:.4f}, "
                   f"Val Loss: {val_metrics['loss']:.4f}, "
                   f"Val F1: {val_metrics['f1']:.4f}, "
                   f"Val Recall: {val_metrics['recall']:.4f}, "
                   f"Avg Pred Duration: {val_metrics['avg_pred_duration']:.1f}s, "
                   f"Active Ratio: {val_metrics['pred_active_ratio']*100:.1f}%")
        
        # Update visualizer (real-time graph update)
        visualizer.update(epoch + 1, train_losses, val_metrics)
        
        # Generate HTML viewer (auto-refresh every 5 seconds)
        visualizer.generate_html_viewer()
        
        # Step LR scheduler
        if scheduler is not None:
            scheduler.step()
        
        # Save best model (recall-first, then F1 strategy)
        # recall >= target_recall かつ active率 <= val_max_active_ratio の中でF1最大を保存。
        # 条件未達の場合はrecall最大を保存。
        current_recall = val_metrics['recall']
        current_f1 = val_metrics['f1']
        current_precision = val_metrics['precision']

        recall_ok = current_recall >= target_recall and current_precision >= min_precision
        if recall_ok:
            current_score = current_f1  # recall達成済み → F1で競う
        else:
            current_score = current_recall  # recall未達 → recallで競う

        if current_score > best_score:
            best_score = current_score
            best_val_f1 = current_f1
            best_val_recall = current_recall
            patience_counter = 0

            model_path = checkpoint_dir / "best_model.pth"
            torch.save({
                'epoch': epoch + 1,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_metrics': val_metrics,
                'config': config
            }, model_path)
            status = "recall+F1" if recall_ok else "recall"
            logger.info(f"Best model saved [{status}] Recall: {best_val_recall:.4f}, F1: {best_val_f1:.4f}")
        else:
            patience_counter += 1
            if patience_counter >= config.get('early_stopping_patience', 15):
                logger.info(f"Early stopping at epoch {epoch+1}")
                break
    
    # Save final visualization
    visualizer.save_final()
    
    logger.info(f"\nTraining complete! Best Recall: {best_val_recall:.4f}, Best F1: {best_val_f1:.4f}")
    logger.info(f"Model saved to: {checkpoint_dir / 'best_model.pth'}")
    logger.info(f"View training progress: {checkpoint_dir / 'view_training.html'}")


if __name__ == "__main__":
    main()
