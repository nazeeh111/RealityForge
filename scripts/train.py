#!/usr/bin/env python3
"""Seeded synthetic pretraining. No IRIS footage or parameters enter training."""
import argparse
import json
import time
from pathlib import Path
import numpy as np
import torch
from realityforge.physics import pendulum_numpy
from realityforge.estimator import ParameterNet, features


def train(count=6000, epochs=100, seed=47, output='models'):
    torch.set_num_threads(2)
    torch.manual_seed(seed)
    rng = np.random.default_rng(seed)
    output=Path(output); output.mkdir(parents=True,exist_ok=True)
    metadata={'seed':seed,'train_count':count,'validation_count':800,'epochs':epochs,'domain':'synthetic only','models':{}}
    for mode in ['pendulum','sliding']:
        start=time.perf_counter(); n=count+800
        duration=rng.uniform(1.5,6,n)
        times=duration[:,None]*np.linspace(0,1,121)[None,:]
        if mode=='pendulum':
            p=np.stack([rng.uniform(3,38,n),rng.uniform(0,.9,n),rng.uniform(-1,1,n),rng.uniform(-2.5,2.5,n)],-1)
            ys=pendulum_numpy(p,times)
            ys+=rng.normal(0,rng.uniform(.0005,.015,n)[:,None],ys.shape)
            targets=p/np.array([20,1,1,5])
        else:
            # Normalized image displacement, both directions, accelerating or decelerating.
            p=np.stack([rng.uniform(-1.5,1.5,n),rng.uniform(-2,2,n),np.zeros(n)],-1)
            ys=p[:,2,None]+p[:,1,None]*times+.5*p[:,0,None]*times**2
            ys+=rng.normal(0,.003,ys.shape)*np.ptp(ys,axis=1)[:,None]
            scales=np.maximum(np.ptp(ys,axis=1),1e-4)
            targets=p[:,:2]/scales[:,None]
        x=torch.tensor(np.stack([features(t,y,mode) for t,y in zip(times,ys)]))
        target=torch.tensor(targets,dtype=torch.float32)
        net=ParameterNet(4 if mode=='pendulum' else 2)
        opt=torch.optim.AdamW(net.parameters(),lr=.002,weight_decay=.0001)
        best=float('inf'); state=None
        for epoch in range(epochs):
            net.train()
            for indices in torch.randperm(count).split(256):
                pred=net(x[indices]); loss=(pred-target[indices]).square().mean()
                opt.zero_grad(); loss.backward(); opt.step()
            net.eval()
            with torch.no_grad(): val=float((net(x[count:])-target[count:]).square().mean())
            if val<best:
                best=val; state={k:v.clone() for k,v in net.state_dict().items()}
        net.load_state_dict(state)
        torch.save({'state_dict':state,'seed':seed,'train_count':count,'validation_mse_normalized':best},output/f'{mode}.pt')
        traced=torch.jit.trace(net,torch.zeros(1,100))
        traced.save(str(output/f'{mode}.torchscript'))
        metadata['models'][mode]={'parameters':sum(p.numel() for p in net.parameters()),'validation_mse_normalized':best,'runtime_s':time.perf_counter()-start}
        print(mode,metadata['models'][mode],flush=True)
    (output/'training.json').write_text(json.dumps(metadata,indent=2)+'\n')
    from export_models import export_models
    export_models(output)

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--count',type=int,default=6000);p.add_argument('--epochs',type=int,default=100);p.add_argument('--seed',type=int,default=47);p.add_argument('--output',default='models');a=p.parse_args();train(a.count,a.epochs,a.seed,a.output)
