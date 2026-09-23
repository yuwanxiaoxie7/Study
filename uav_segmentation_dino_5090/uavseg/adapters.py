"""Engineering simplifications, not a full CrossEarth-Gate/PBSeg reproduction."""
import torch
from torch import nn

class SpatialAdapter(nn.Module):
    def __init__(self,channels):
        super().__init__()
        self.dw=nn.Conv2d(channels,channels,3,padding=1,groups=channels)
        self.pw=nn.Conv2d(channels,channels,1)
        nn.init.zeros_(self.pw.weight); nn.init.zeros_(self.pw.bias)
    def forward(self,x): return x+self.pw(torch.nn.functional.gelu(self.dw(x)))

class FrequencyAdapter(nn.Module):
    def __init__(self,channels):
        super().__init__(); self.low=nn.Parameter(torch.zeros(1,channels,1,1)); self.high=nn.Parameter(torch.zeros(1,channels,1,1))
    def forward(self,x):
        # FFT must use FP32 for non-power-of-two grids under CUDA AMP.
        with torch.autocast(device_type=x.device.type,enabled=False):
            f=torch.fft.rfft2(x.float(),norm='ortho'); h,w=x.shape[-2:]
            fy=torch.fft.fftfreq(h,device=x.device)[:,None]; fx=torch.fft.rfftfreq(w,device=x.device)[None,:]
            low=(fy.square()+fx.square()<=.15**2).to(f.dtype)
            gain=.1*(self.low.tanh()*low+self.high.tanh()*(1-low))
            delta=torch.fft.irfft2(f*gain,s=(h,w),norm='ortho')
        return x+delta.to(x.dtype)

class ContextRefinement(nn.Module):
    def __init__(self,channels):
        super().__init__()
        self.dw=nn.Conv2d(channels,channels,3,padding=1,groups=channels)
        self.gate=nn.Sequential(nn.AdaptiveAvgPool2d(1),nn.Conv2d(channels,max(8,channels//4),1),nn.GELU(),nn.Conv2d(max(8,channels//4),channels,1),nn.Sigmoid())
        self.scale=nn.Parameter(torch.zeros(()))
    def forward(self,x): return x+self.scale.tanh()*self.dw(x)*(1+self.gate(x))
