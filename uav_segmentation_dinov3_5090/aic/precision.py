import contextlib
import torch

def dtype(c):
    choice = c.get('precision', 'auto')
    if choice == 'fp32': return torch.float32
    if choice == 'fp16': return torch.float16
    supported = torch.cuda.is_available() and torch.cuda.is_bf16_supported()
    if choice == 'bf16' and not supported:
        raise RuntimeError('BF16 unavailable; choose fp16 explicitly with a new run')
    return torch.bfloat16 if supported else torch.float16

def autocast(c, device='cuda'):
    if device != 'cuda' or dtype(c) == torch.float32: return contextlib.nullcontext()
    return torch.amp.autocast('cuda',dtype=dtype(c))

def scaler(c):
    return torch.amp.GradScaler('cuda',enabled=dtype(c)==torch.float16, init_scale=8192.)
