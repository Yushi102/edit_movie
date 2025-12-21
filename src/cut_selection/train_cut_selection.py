"""
Training script for cut selection model with visualization
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
matplotlib.use('Agg')  # Non-interactive backend for background plotting
import matplotlib.pyplot as plt
import threading
import time

from src.cut_selection.cut_dataset import CutSelectionDataset
from src.cut_selection.cut_model import CutSelectionModel
from src.cut_selection.losses import CombinedCutSelectionLoss, FocalLoss

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# 日本語フォント対応（オプション）
try:
    import japanize_matplotlib
    logger.info("✅ japanize_matplotlib loaded - Japanese characters will display correctly")
except ImportError:
    logger.warning("⚠️  japanize_matplotlib not installed. Japanese characters may not display correctly.")
    logger.warning("   Install with: pip install japanize-matplotlib")


class TrainingVisualizer:
    """リアルタイムで学習の様子を可視化"""
    
    def __init__(self, checkpoint_dir: Path, num_epochs: int):
        self.checkpoint_dir = checkpoint_dir
        self.num_epochs = num_epochs
        
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
            'val_pred_active_ratio': [],  # 予測で採用した割合
            'val_pred_inactive_ratio': [],  # 予測で不採用した割合
            'optimal_threshold': []  # 最適な閾値
        }
        
        # グラフの初期化
        self.fig = None
        self.axes = None
        self.setup_plot()
    
    def setup_plot(self):
        """グラフの初期設定"""
        self.fig, self.axes = plt.subplots(3, 2, figsize=(15, 14))
        self.fig.suptitle('カット選択モデル学習状況', fontsize=16, fontweight='bold')
        
        # サブプロットのタイトル
        self.axes[0, 0].set_title('損失関数（Loss）')
        self.axes[0, 0].set_xlabel('エポック')
        self.axes[0, 0].set_ylabel('Loss')
        self.axes[0, 0].grid(True, alpha=0.3)
        
        self.axes[0, 1].set_title('損失の内訳（CE vs TV）')
        self.axes[0, 1].set_xlabel('エポック')
        self.axes[0, 1].set_ylabel('Loss')
        self.axes[0, 1].grid(True, alpha=0.3)
        
        self.axes[1, 0].set_title('分類性能（Accuracy & F1）')
        self.axes[1, 0].set_xlabel('エポック')
        self.axes[1, 0].set_ylabel('スコア')
        self.axes[1, 0].set_ylim([0, 1])
        self.axes[1, 0].grid(True, alpha=0.3)
        
        self.axes[1, 1].set_title('Precision, Recall, Specificity')
        self.axes[1, 1].set_xlabel('エポック')
        self.axes[1, 1].set_ylabel('スコア')
        self.axes[1, 1].set_ylim([0, 1])
        self.axes[1, 1].grid(True, alpha=0.3)
        
        self.axes[2, 0].set_title('最適な閾値の推移')
        self.axes[2, 0].set_xlabel('エポック')
        self.axes[2, 0].set_ylabel('閾値 (Confidence Threshold)')
        self.axes[2, 0].set_ylim([-1, 1])
        self.axes[2, 0].axhline(y=0, color='gray', linestyle='--', alpha=0.5, label='0 (Active=Inactive)')
        self.axes[2, 0].grid(True, alpha=0.3)
        
        self.axes[2, 1].set_title('予測の採用/不採用割合')
        self.axes[2, 1].set_xlabel('エポック')
        self.axes[2, 1].set_ylabel('割合 (%)')
        self.axes[2, 1].set_ylim([0, 100])
        self.axes[2, 1].grid(True, alpha=0.3)
        
        plt.tight_layout()
    
    def update(self, epoch: int, train_losses: dict, val_metrics: dict):
        """学習データを更新してグラフを再描画"""
        # データを追加
        self.history['epoch'].append(epoch)
        self.history['train_loss'].append(train_losses['total_loss'])
        self.history['train_ce_loss'].append(train_losses['ce_loss'])
        self.history['train_tv_loss'].append(train_losses['tv_loss'])
        self.history['val_loss'].append(val_metrics['total_loss'])
        self.history['val_ce_loss'].append(val_metrics['ce_loss'])
        self.history['val_tv_loss'].append(val_metrics['tv_loss'])
        self.history['val_accuracy'].append(val_metrics['accuracy'])
        self.history['val_precision'].append(val_metrics['precision'])
        self.history['val_recall'].append(val_metrics['recall'])
        self.history['val_f1'].append(val_metrics['f1'])
        self.history['val_specificity'].append(val_metrics['specificity'])
        self.history['val_pred_active_ratio'].append(val_metrics['pred_active_ratio'] * 100)
        self.history['val_pred_inactive_ratio'].append(val_metrics['pred_inactive_ratio'] * 100)
        self.history['optimal_threshold'].append(val_metrics.get('optimal_threshold', 0.0))
        
        # グラフをクリア
        for ax in self.axes.flat:
            ax.clear()
        
        epochs = self.history['epoch']
        
        # 1. 損失関数（Total Loss）
        ax = self.axes[0, 0]
        ax.plot(epochs, self.history['train_loss'], 'b-', label='Train Loss', linewidth=2)
        ax.plot(epochs, self.history['val_loss'], 'r-', label='Val Loss', linewidth=2)
        ax.set_title('損失関数（Loss）')
        ax.set_xlabel('エポック')
        ax.set_ylabel('Loss')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        # 2. 損失の内訳（CE vs TV）
        ax = self.axes[0, 1]
        ax.plot(epochs, self.history['train_ce_loss'], 'b-', label='Train CE Loss', linewidth=2)
        ax.plot(epochs, self.history['train_tv_loss'], 'b--', label='Train TV Loss', linewidth=2)
        ax.plot(epochs, self.history['val_ce_loss'], 'r-', label='Val CE Loss', linewidth=2)
        ax.plot(epochs, self.history['val_tv_loss'], 'r--', label='Val TV Loss', linewidth=2)
        ax.set_title('損失の内訳（CE vs TV）')
        ax.set_xlabel('エポック')
        ax.set_ylabel('Loss')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        # 3. 分類性能（Accuracy & F1）
        ax = self.axes[1, 0]
        ax.plot(epochs, self.history['val_accuracy'], 'g-', label='Accuracy', linewidth=2, marker='o')
        ax.plot(epochs, self.history['val_f1'], 'purple', label='F1 Score', linewidth=2, marker='s')
        ax.set_title('分類性能（Accuracy & F1）')
        ax.set_xlabel('エポック')
        ax.set_ylabel('スコア')
        ax.set_ylim([0, 1])
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        # 4. Precision, Recall, Specificity
        ax = self.axes[1, 1]
        ax.plot(epochs, self.history['val_precision'], 'b-', label='Precision', linewidth=2, marker='o')
        ax.plot(epochs, self.history['val_recall'], 'r-', label='Recall', linewidth=2, marker='s')
        ax.plot(epochs, self.history['val_specificity'], 'orange', label='Specificity', linewidth=2, marker='^')
        ax.set_title('Precision, Recall, Specificity')
        ax.set_xlabel('エポック')
        ax.set_ylabel('スコア')
        ax.set_ylim([0, 1])
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        # 5. 最適な閾値の推移
        ax = self.axes[2, 0]
        ax.plot(epochs, self.history['optimal_threshold'], 'purple', label='最適閾値', linewidth=2, marker='o')
        ax.axhline(y=0, color='gray', linestyle='--', alpha=0.5, label='0 (Active=Inactive)')
        ax.set_title('最適な閾値の推移')
        ax.set_xlabel('エポック')
        ax.set_ylabel('閾値 (Confidence Threshold)')
        ax.set_ylim([-1, 1])
        ax.legend()
        ax.grid(True, alpha=0.3)
        # 現在の閾値を表示
        if len(epochs) > 0:
            current_threshold = self.history['optimal_threshold'][-1]
            ax.text(0.02, 0.98, f'現在: {current_threshold:.3f}', 
                   transform=ax.transAxes, fontsize=10, verticalalignment='top',
                   bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
        
        # 6. 予測の採用/不採用割合
        ax = self.axes[2, 1]
        ax.plot(epochs, self.history['val_pred_active_ratio'], 'g-', label='採用 (Active)', linewidth=2, marker='o')
        ax.plot(epochs, self.history['val_pred_inactive_ratio'], 'r-', label='不採用 (Inactive)', linewidth=2, marker='s')
        # 正解データの割合を水平線で表示
        if len(epochs) > 0:
            true_active_ratio = val_metrics.get('true_active_ratio', 0) * 100
            true_inactive_ratio = val_metrics.get('true_inactive_ratio', 0) * 100
            ax.axhline(y=true_active_ratio, color='g', linestyle='--', alpha=0.5, label=f'正解採用 ({true_active_ratio:.1f}%)')
            ax.axhline(y=true_inactive_ratio, color='r', linestyle='--', alpha=0.5, label=f'正解不採用 ({true_inactive_ratio:.1f}%)')
        ax.set_title('予測の採用/不採用割合')
        ax.set_xlabel('エポック')
        ax.set_ylabel('割合 (%)')
        ax.set_ylim([0, 105])  # 少し余裕を持たせる
        ax.legend(loc='best')  # 最適な位置に配置
        ax.grid(True, alpha=0.3)
        
        # 全体のタイトルを更新
        best_f1 = max(self.history['val_f1'])
        best_epoch = self.history['epoch'][self.history['val_f1'].index(best_f1)]
        self.fig.suptitle(
            f'カット選択モデル学習状況 - Epoch {epoch}/{self.num_epochs} | Best F1: {best_f1:.4f} (Epoch {best_epoch})',
            fontsize=16,
            fontweight='bold'
        )
        
        plt.tight_layout()
        
        # 画像を保存
        save_path = self.checkpoint_dir / 'training_progress.png'
        self.fig.savefig(save_path, dpi=100, bbox_inches='tight')
        logger.info(f"📊 Training visualization saved: {save_path}")
    
    def save_final(self):
        """最終的なグラフを高解像度で保存"""
        final_path = self.checkpoint_dir / 'training_final.png'
        self.fig.savefig(final_path, dpi=300, bbox_inches='tight')
        logger.info(f"📊 Final training visualization saved: {final_path}")
        
        # CSVとしても保存
        import pandas as pd
        df = pd.DataFrame(self.history)
        csv_path = self.checkpoint_dir / 'training_history.csv'
        df.to_csv(csv_path, index=False)
        logger.info(f"📊 Training history saved: {csv_path}")
        
        plt.close(self.fig)



def analyze_class_balance(dataloader, device):
    """Analyze class balance in the dataset"""
    logger.info("Analyzing class balance...")
    
    total_active = 0
    total_inactive = 0
    
    for batch in tqdm(dataloader, desc="Analyzing"):
        active_labels = batch['active'].to(device)
        total_active += (active_labels == 1).sum().item()
        total_inactive += (active_labels == 0).sum().item()
    
    total = total_active + total_inactive
    active_ratio = total_active / total if total > 0 else 0
    inactive_ratio = total_inactive / total if total > 0 else 0
    
    logger.info(f"Class balance:")
    logger.info(f"  Active: {total_active:,} ({active_ratio*100:.2f}%)")
    logger.info(f"  Inactive: {total_inactive:,} ({inactive_ratio*100:.2f}%)")
    
    # Calculate class weights (inverse frequency)
    if total_active > 0 and total_inactive > 0:
        weight_active = total / (2 * total_active)
        weight_inactive = total / (2 * total_inactive)
        class_weights = torch.tensor([weight_inactive, weight_active], device=device)
        logger.info(f"  Recommended class weights: [inactive={weight_inactive:.4f}, active={weight_active:.4f}]")
    else:
        class_weights = torch.tensor([1.0, 1.0], device=device)
        logger.info(f"  Using default weights: [1.0, 1.0]")
    
    return class_weights


def find_optimal_threshold(model, dataloader, device, thresholds=None):
    """Find optimal confidence threshold using validation data"""
    if thresholds is None:
        # Search from -0.5 to 0.5 (confidence score range)
        thresholds = np.arange(-0.5, 0.5, 0.05)
    
    logger.info("Finding optimal confidence threshold...")
    
    model.eval()
    all_confidence_scores = []
    all_labels = []
    
    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Collecting predictions"):
            audio = batch['audio'].to(device)
            visual = batch['visual'].to(device)
            active_labels = batch['active'].to(device)
            
            outputs = model(audio, visual)
            active_logits = outputs['active']
            
            # Get probabilities for both classes
            probs = torch.softmax(active_logits, dim=-1)  # (batch, seq_len, 2)
            inactive_probs = probs[..., 0]  # Inactive probability
            active_probs = probs[..., 1]    # Active probability
            
            # Confidence score: Active - Inactive
            confidence_scores = active_probs - inactive_probs
            
            all_confidence_scores.append(confidence_scores.cpu().numpy())
            all_labels.append(active_labels.cpu().numpy())
    
    all_confidence_scores = np.concatenate([s.flatten() for s in all_confidence_scores])
    all_labels = np.concatenate([l.flatten() for l in all_labels])
    
    # Find threshold with best F1 score
    best_threshold = 0.0
    best_f1 = 0.0
    
    for threshold in thresholds:
        predictions = (all_confidence_scores >= threshold).astype(int)
        
        tp = np.sum((predictions == 1) & (all_labels == 1))
        fp = np.sum((predictions == 1) & (all_labels == 0))
        fn = np.sum((predictions == 0) & (all_labels == 1))
        
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
        
        if f1 > best_f1:
            best_f1 = f1
            best_threshold = threshold
    
    logger.info(f"Optimal confidence threshold: {best_threshold:.3f} (F1={best_f1:.4f})")
    logger.info(f"  (threshold=0 means: active_prob > inactive_prob)")
    
    return best_threshold


def train_epoch(model, dataloader, optimizer, criterion, device, use_amp=False):
    """Train for one epoch"""
    model.train()
    total_loss = 0
    total_ce_loss = 0
    total_tv_loss = 0
    
    # GradScaler for mixed precision training
    scaler = torch.cuda.amp.GradScaler() if use_amp else None
    
    for batch in tqdm(dataloader, desc="Training"):
        audio = batch['audio'].to(device)
        visual = batch['visual'].to(device)
        active_labels = batch['active'].to(device)
        
        # Mixed precision training
        if use_amp and device != 'cpu':
            with torch.cuda.amp.autocast():
                # Forward pass
                outputs = model(audio, visual)
                active_logits = outputs['active']  # (batch, seq, 2)
                
                # Calculate loss (combined CE + TV)
                loss, loss_dict = criterion(active_logits, active_labels)
            
            # Backward pass with gradient scaling
            optimizer.zero_grad()
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(optimizer)
            scaler.update()
        else:
            # Standard training
            outputs = model(audio, visual)
            active_logits = outputs['active']
            loss, loss_dict = criterion(active_logits, active_labels)
            
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
        
        total_loss += loss_dict['total_loss']
        total_ce_loss += loss_dict['ce_loss']
        total_tv_loss += loss_dict['tv_loss']
    
    num_batches = len(dataloader)
    return {
        'total_loss': total_loss / num_batches,
        'ce_loss': total_ce_loss / num_batches,
        'tv_loss': total_tv_loss / num_batches
    }


def validate(model, dataloader, criterion, device):
    """Validate the model"""
    model.eval()
    total_loss = 0
    total_ce_loss = 0
    total_tv_loss = 0
    
    # For metrics calculation
    all_predictions = []
    all_labels = []
    all_confidence_scores = []
    
    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Validation"):
            audio = batch['audio'].to(device)
            visual = batch['visual'].to(device)
            active_labels = batch['active'].to(device)
            
            # Forward pass
            outputs = model(audio, visual)
            active_logits = outputs['active']
            
            # Calculate loss
            loss, loss_dict = criterion(active_logits, active_labels)
            total_loss += loss_dict['total_loss']
            total_ce_loss += loss_dict['ce_loss']
            total_tv_loss += loss_dict['tv_loss']
            
            # Get predictions
            predictions = torch.argmax(active_logits, dim=-1)  # (batch, seq_len)
            
            # Get confidence scores for threshold calculation
            probs = torch.softmax(active_logits, dim=-1)
            inactive_probs = probs[..., 0]
            active_probs = probs[..., 1]
            confidence_scores = active_probs - inactive_probs
            
            # Collect for metrics
            all_predictions.append(predictions.cpu().numpy())
            all_labels.append(active_labels.cpu().numpy())
            all_confidence_scores.append(confidence_scores.cpu().numpy())
    
    # Flatten all predictions and labels
    all_predictions = np.concatenate([p.flatten() for p in all_predictions])
    all_labels = np.concatenate([l.flatten() for l in all_labels])
    all_confidence_scores = np.concatenate([s.flatten() for s in all_confidence_scores])
    
    # Calculate metrics
    num_batches = len(dataloader)
    
    # Accuracy
    accuracy = (all_predictions == all_labels).mean()
    
    # Precision, Recall, F1 for Active class (class 1)
    tp = np.sum((all_predictions == 1) & (all_labels == 1))
    fp = np.sum((all_predictions == 1) & (all_labels == 0))
    fn = np.sum((all_predictions == 0) & (all_labels == 1))
    tn = np.sum((all_predictions == 0) & (all_labels == 0))
    
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
    
    # Specificity (for Inactive class)
    specificity = tn / (tn + fp) if (tn + fp) > 0 else 0.0
    
    # Calculate optimal threshold (quick version - only test a few thresholds)
    best_threshold = 0.0
    best_f1_threshold = 0.0
    for threshold in np.arange(-0.5, 0.5, 0.1):
        threshold_predictions = (all_confidence_scores >= threshold).astype(int)
        tp_t = np.sum((threshold_predictions == 1) & (all_labels == 1))
        fp_t = np.sum((threshold_predictions == 1) & (all_labels == 0))
        fn_t = np.sum((threshold_predictions == 0) & (all_labels == 1))
        
        precision_t = tp_t / (tp_t + fp_t) if (tp_t + fp_t) > 0 else 0
        recall_t = tp_t / (tp_t + fn_t) if (tp_t + fn_t) > 0 else 0
        f1_t = 2 * precision_t * recall_t / (precision_t + recall_t) if (precision_t + recall_t) > 0 else 0
        
        if f1_t > best_f1_threshold:
            best_f1_threshold = f1_t
            best_threshold = threshold
    
    # Prediction ratios
    total_samples = len(all_predictions)
    pred_active_ratio = np.sum(all_predictions == 1) / total_samples
    pred_inactive_ratio = np.sum(all_predictions == 0) / total_samples
    true_active_ratio = np.sum(all_labels == 1) / total_samples
    true_inactive_ratio = np.sum(all_labels == 0) / total_samples
    
    # Normalized confusion matrix elements
    tp_norm = tp / total_samples
    fp_norm = fp / total_samples
    fn_norm = fn / total_samples
    tn_norm = tn / total_samples
    
    metrics = {
        'total_loss': total_loss / num_batches,
        'ce_loss': total_ce_loss / num_batches,
        'tv_loss': total_tv_loss / num_batches,
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1': f1,
        'specificity': specificity,
        'optimal_threshold': best_threshold,
        'pred_active_ratio': pred_active_ratio,
        'pred_inactive_ratio': pred_inactive_ratio,
        'true_active_ratio': true_active_ratio,
        'true_inactive_ratio': true_inactive_ratio,
        'tp_norm': tp_norm,
        'fp_norm': fp_norm,
        'fn_norm': fn_norm,
        'tn_norm': tn_norm
    }
    
    return metrics


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, default='configs/config_cut_selection.yaml')
    args = parser.parse_args()
    
    # Load config
    with open(args.config, 'r', encoding='utf-8') as f:
        config = yaml.safe_load(f)
    
    # Device
    device = 'cuda' if torch.cuda.is_available() and not config.get('cpu', False) else 'cpu'
    logger.info(f"Using device: {device}")
    
    if device == 'cuda':
        logger.info(f"GPU: {torch.cuda.get_device_name(0)}")
        logger.info(f"GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.2f} GB")
    
    use_amp = config.get('use_amp', False) and device == 'cuda'
    if use_amp:
        logger.info("Mixed Precision Training: Enabled")
    
    # Create checkpoint directory
    checkpoint_dir = Path(config['checkpoint_dir'])
    checkpoint_dir.mkdir(exist_ok=True, parents=True)
    
    # Load datasets
    train_dataset = CutSelectionDataset(config['train_data'])
    val_dataset = CutSelectionDataset(config['val_data'])
    
    train_loader = DataLoader(
        train_dataset,
        batch_size=config['batch_size'],
        shuffle=True,
        num_workers=config.get('num_workers', 4),
        pin_memory=True if device == 'cuda' else False,
        persistent_workers=True if config.get('num_workers', 4) > 0 else False
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=config['batch_size'],
        shuffle=False,
        num_workers=config.get('num_workers', 4),
        pin_memory=True if device == 'cuda' else False,
        persistent_workers=True if config.get('num_workers', 4) > 0 else False
    )
    
    # Create model
    model = CutSelectionModel(
        audio_features=config['audio_features'],
        visual_features=config['visual_features'],
        d_model=config['d_model'],
        nhead=config['nhead'],
        num_encoder_layers=config['num_encoder_layers'],
        dim_feedforward=config['dim_feedforward'],
        dropout=config['dropout'],
        fusion_type=config.get('fusion_type', 'gated')
    ).to(device)
    
    logger.info(f"Model parameters: {model.count_parameters():,}")
    
    # Analyze class balance and get weights
    if config.get('auto_balance_weights', False):
        class_weights = analyze_class_balance(train_loader, device)
    else:
        class_weights = None
    
    # Loss function with temporal smoothness
    criterion = CombinedCutSelectionLoss(
        class_weights=class_weights,
        tv_weight=config.get('tv_weight', 0.1),
        label_smoothing=config.get('label_smoothing', 0.0),
        use_focal=config.get('use_focal_loss', False),
        focal_alpha=config.get('focal_alpha', 0.25),
        focal_gamma=config.get('focal_gamma', 2.0)
    )
    
    if config.get('use_focal_loss', False):
        logger.info(f"Using Focal Loss (alpha={config.get('focal_alpha', 0.25)}, gamma={config.get('focal_gamma', 2.0)})")
    else:
        logger.info(f"Using Weighted CrossEntropy Loss")
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=config['learning_rate'],
        weight_decay=config.get('weight_decay', 0.0)
    )
    
    # Training loop
    best_val_f1 = 0.0  # Track best F1 score (higher is better)
    patience_counter = 0
    
    # 可視化の初期化
    visualizer = TrainingVisualizer(checkpoint_dir, config['num_epochs'])
    logger.info("📊 Training visualizer initialized")
    
    for epoch in range(config['num_epochs']):
        logger.info(f"\nEpoch {epoch+1}/{config['num_epochs']}")
        
        # Train
        train_losses = train_epoch(model, train_loader, optimizer, criterion, device, use_amp)
        logger.info(f"Train Loss: {train_losses['total_loss']:.4f} "
                   f"(CE: {train_losses['ce_loss']:.4f}, TV: {train_losses['tv_loss']:.4f})")
        
        # Validate
        val_metrics = validate(model, val_loader, criterion, device)
        logger.info(f"Val Loss: {val_metrics['total_loss']:.4f} "
                   f"(CE: {val_metrics['ce_loss']:.4f}, TV: {val_metrics['tv_loss']:.4f})")
        logger.info(f"Val Metrics - Acc: {val_metrics['accuracy']:.4f}, "
                   f"Prec: {val_metrics['precision']:.4f}, "
                   f"Rec: {val_metrics['recall']:.4f}, "
                   f"F1: {val_metrics['f1']:.4f}, "
                   f"Spec: {val_metrics['specificity']:.4f}")
        
        # 可視化を更新
        visualizer.update(epoch + 1, train_losses, val_metrics)
        
        # Save checkpoint
        if (epoch + 1) % config.get('save_every', 5) == 0:
            checkpoint_path = checkpoint_dir / f"checkpoint_epoch_{epoch+1}.pth"
            torch.save({
                'epoch': epoch + 1,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'train_loss': train_losses['total_loss'],
                'val_metrics': val_metrics,
                'config': config
            }, checkpoint_path)
            logger.info(f"Checkpoint saved: {checkpoint_path}")
        
        # Save best model (based on F1 score)
        val_f1 = val_metrics['f1']
        if val_f1 > best_val_f1:
            best_val_f1 = val_f1
            patience_counter = 0
            best_model_path = checkpoint_dir / "best_model.pth"
            torch.save({
                'epoch': epoch + 1,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'train_loss': train_losses['total_loss'],
                'val_metrics': val_metrics,
                'config': config
            }, best_model_path)
            logger.info(f"🏆 New best model! Val F1: {val_f1:.4f}")
        else:
            patience_counter += 1
            if patience_counter >= config.get('early_stopping_patience', 15):
                logger.info(f"Early stopping triggered after {epoch+1} epochs")
                break
    
    logger.info("\n✅ Training complete!")
    logger.info(f"Best Val F1: {best_val_f1:.4f}")
    
    # 最終的な可視化を保存
    visualizer.save_final()
    
    # Find optimal threshold on validation set
    logger.info("\nFinding optimal confidence threshold...")
    optimal_threshold = find_optimal_threshold(model, val_loader, device)
    
    # Save inference parameters
    inference_params = {
        'confidence_threshold': float(optimal_threshold),
        'target_duration': 90.0,
        'max_duration': 150.0
    }
    
    inference_params_path = checkpoint_dir / 'inference_params.yaml'
    with open(inference_params_path, 'w', encoding='utf-8') as f:
        yaml.dump(inference_params, f)
    
    logger.info(f"Inference parameters saved to: {inference_params_path}")
    logger.info(f"  Confidence threshold: {optimal_threshold:.3f}")


if __name__ == "__main__":
    main()
