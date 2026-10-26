"""Export portable neural estimators and verify outputs against each checkpoint."""
import argparse
from pathlib import Path
import json,hashlib
import torch
from realityforge.estimator import ParameterNet


def export_models(directory):
    directory=Path(directory);report={}
    for name,outputs in [('pendulum',4),('sliding',2)]:
        model=ParameterNet(outputs)
        state=torch.load(directory/f'{name}.pt',map_location='cpu',weights_only=True)
        model.load_state_dict(state['state_dict']);model.eval()
        example=torch.linspace(-1,1,100).reshape(1,100)
        exported=torch.export.export(model,(example,))
        path=directory/f'{name}.pt2';torch.export.save(exported,path)
        loaded=torch.export.load(path).module()
        with torch.no_grad(): error=float((loaded(example)-model(example)).abs().max())
        if error>1e-6:raise RuntimeError(f'{name} export output mismatch: {error}')
        report[name]={'max_absolute_output_difference':error,'checkpoint_sha256':hashlib.sha256((directory/f'{name}.pt').read_bytes()).hexdigest(),'export_sha256':hashlib.sha256(path.read_bytes()).hexdigest()}
    (directory/'exports.json').write_text(json.dumps(report,indent=2)+'\n');print(json.dumps(report,indent=2))

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--models',default='models');a=p.parse_args();export_models(a.models)
