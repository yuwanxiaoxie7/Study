"""Probe real kernels, not just CUDA device enumeration."""
import json, subprocess
import torch

if not torch.cuda.is_available():
    raise RuntimeError('CUDA unavailable. Start a GPU instance, not CPU-only mode.')
p = torch.cuda.get_device_properties(0)
print(json.dumps({'gpu':p.name,'memory_gib':p.total_memory/2**30,
    'torch':torch.__version__,'cuda':torch.version.cuda,
    'capability':torch.cuda.get_device_capability(),
    'compiled_architectures':torch.cuda.get_arch_list()},ensure_ascii=False),flush=True)
subprocess.run(['nvidia-smi','--query-gpu=name,driver_version,memory.total','--format=csv'],check=True)
if '5090' not in p.name:
    raise RuntimeError('This profile targets RTX 5090; verify the rented instance.')
if p.total_memory < 29*2**30:
    raise RuntimeError('This profile expects about 32 GB VRAM. Verify server hardware.')
q = torch.randn(1,4,64,64,device='cuda',dtype=torch.bfloat16,requires_grad=True)
torch.nn.functional.scaled_dot_product_attention(q,q,q).float().square().mean().backward()
conv = torch.nn.Conv2d(3,8,3,padding=1).cuda()
with torch.amp.autocast('cuda',dtype=torch.bfloat16):
    loss = conv(torch.randn(1,3,32,32,device='cuda')).float().square().mean()
loss.backward()
torch.cuda.synchronize()
assert torch.isfinite(q.grad).all() and torch.isfinite(conv.weight.grad).all()
print('CUDA BF16 attention/convolution forward and backward passed.',flush=True)
