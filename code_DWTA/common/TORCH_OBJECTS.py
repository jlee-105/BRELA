# GPU/CPU Configuration
import torch

# Check for MPS (Apple Silicon) first, then CUDA, then CPU
if hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
    print("Using MPS (Apple Silicon GPU)")
    DEVICE = torch.device('mps')
    device = DEVICE
elif torch.cuda.is_available():
    print("Using CUDA GPU")
    DEVICE = torch.device('cuda')
    device = DEVICE
else:
    print("Using CPU")
    DEVICE = torch.device('cpu')
    device = DEVICE

# Tensor types based on device
if device.type == 'cuda':
    FloatTensor = torch.cuda.FloatTensor
    LongTensor = torch.cuda.LongTensor
    ByteTensor = torch.cuda.ByteTensor
    BoolTensor = torch.cuda.BoolTensor
else:
    FloatTensor = torch.FloatTensor
    LongTensor = torch.LongTensor
    ByteTensor = torch.ByteTensor
    BoolTensor = torch.BoolTensor

Tensor = FloatTensor