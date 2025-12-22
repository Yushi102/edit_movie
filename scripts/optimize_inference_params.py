"""
学習済みモデルで推論パラメータを最適化するスクリプト
"""
import sys
import torch
from pathlib import Path

# プロジェクトルートをパスに追加
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.model.model_persistence import load_model
from src.training.multimodal_dataset import MultimodalDataset
from src.training.parameter_optimizer import optimize_and_save_parameters
import yaml

def load_config(config_path):
    with open(config_path, 'r', encoding='utf-8') as f:
        return yaml.safe_load(f)

def main():
    # 設定をロード
    config_path = "configs/config_multimodal_experiment.yaml"
    config = load_config(config_path)
    
    # デバイス
    device = "cuda" if torch.cuda.is_available() and not config.get('cpu', False) else "cpu"
    print(f"Using device: {device}")
    
    # モデルをロード
    model_path = "checkpoints/best_model.pth"
    print(f"Loading model from {model_path}")
    result = load_model(model_path, device=device)
    model = result['model']
    model.eval()
    
    # 検証データをロード
    print(f"Loading validation data from {config['val_data']}")
    val_dataset = MultimodalDataset(
        sequences_npz=config['val_data'],
        enable_multimodal=config.get('enable_multimodal', True)
    )
    
    val_loader = torch.utils.data.DataLoader(
        val_dataset,
        batch_size=config.get('batch_size', 8),
        shuffle=False,
        num_workers=0
    )
    
    # パラメータを最適化して保存
    output_path = Path(model_path).parent / "inference_params.yaml"
    print(f"\nOptimizing inference parameters...")
    optimal_params = optimize_and_save_parameters(
        model=model,
        val_loader=val_loader,
        device=device,
        output_path=str(output_path),
        fps=config.get('fps', 10.0)
    )
    
    print(f"\n✅ Optimal parameters saved to: {output_path}")
    print(f"   Active threshold: {optimal_params['active_threshold']:.4f}")

if __name__ == "__main__":
    main()
