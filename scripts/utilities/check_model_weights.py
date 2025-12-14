import torch

# Load checkpoint
ckpt = torch.load('checkpoints/best_model.pth', map_location='cpu')
state = ckpt['model_state_dict']

# Check for NaN/Inf
has_nan = any(torch.isnan(v).any() for v in state.values())
has_inf = any(torch.isinf(v).any() for v in state.values())

print('Has NaN:', has_nan)
print('Has Inf:', has_inf)

# Sample weights
print('\nSample weights:')
for k, v in list(state.items())[:5]:
    print(f'  {k}: shape={v.shape}, min={v.min():.4f}, max={v.max():.4f}, mean={v.mean():.4f}')

# Check specific layers
print('\nAudio embedding weights:')
audio_proj = state['audio_embedding.projection.weight']
print(f'  Shape: {audio_proj.shape}')
print(f'  Min: {audio_proj.min():.4f}, Max: {audio_proj.max():.4f}')
print(f'  Mean: {audio_proj.mean():.4f}, Std: {audio_proj.std():.4f}')
print(f'  Has NaN: {torch.isnan(audio_proj).any()}')
print(f'  Has Inf: {torch.isinf(audio_proj).any()}')
