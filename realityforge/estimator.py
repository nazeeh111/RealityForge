"""A compact learned initializer trained only on seeded synthetic trajectories."""
from pathlib import Path
import numpy as np
import torch
from torch import nn
from scipy.signal import savgol_filter

from .resources import ROOT
SAMPLES = 96


def features(t, y, mode):
    t, y = np.asarray(t), np.asarray(y)
    grid = np.linspace(0, t[-1], SAMPLES)
    signal = np.interp(grid, t, y)
    smooth = savgol_filter(signal, 11, 3)
    v = np.gradient(smooth, grid)
    acc = np.gradient(v, grid)
    if mode == 'pendulum':
        # Weak analytical features improve conditioning; raw signal preserves nonlinear content.
        design = np.stack([-np.sin(smooth), -v], -1)[5:-5]
        ls = np.linalg.lstsq(design, acc[5:-5], rcond=None)[0]
        extra = [t[-1]/6, np.clip(ls[0],0,60)/30, np.clip(ls[1],-2,3), v[0]/5]
        return np.r_[signal, extra].astype('float32')
    scale = max(np.ptp(y), 1e-4)
    signal = (signal-y[0])/scale
    poly = np.polyfit(grid, signal, 2)
    return np.r_[signal, t[-1]/6, poly[0], poly[1], 0].astype('float32')


class ParameterNet(nn.Module):
    def __init__(self, output=4):
        super().__init__()
        self.layers = nn.Sequential(nn.Linear(100,96),nn.SiLU(),nn.Linear(96,64),nn.SiLU(),nn.Linear(64,output))
    def forward(self,x): return self.layers(x)


def estimate(t, y, mode, model_dir=None):
    path = Path(model_dir or ROOT/'models') / f'{mode}.pt'
    if not path.exists():
        raise FileNotFoundError(f'Missing {path.name}. Run python scripts/train.py first.')
    checkpoint = torch.load(path, map_location='cpu', weights_only=True)
    net = ParameterNet(4 if mode=='pendulum' else 2)
    net.load_state_dict(checkpoint['state_dict'])
    net.eval()
    with torch.inference_mode(): out = net(torch.from_numpy(features(t,y,mode))).numpy()
    if mode=='pendulum':
        return np.array([np.clip(out[0]*20,1,60), np.clip(out[1],0,3), y[0], np.clip(out[3]*5,-12,12)])
    scale = max(np.ptp(y),1e-4)
    return np.array([out[0]*scale,out[1]*scale,y[0]])
