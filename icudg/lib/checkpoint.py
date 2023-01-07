import os
import torch
from pathlib import Path

# functions for checkpoint/reload in case of job pre-emption on our slurm cluster
# will have to customize if you desire this functionality
# otherwise, the training script will still work fine as-is
def save_checkpoint(model, optimizer, sampler_dicts, start_step, es, rng, path='.'):   
    if isinstance(model.optimizer, dict):
        optimizer_dict = {k: opt.state_dict() for k, opt in model.optimizer.items()}
    else:
        optimizer_dict = optimizer.state_dict()

    torch.save(
        {
            'model_dict': model.state_dict(),
            'optimizer_dict': optimizer_dict,
            'sampler_dicts': sampler_dicts,
            'start_step': start_step,
            'es': es,
            'rng': rng
        }, 
        (Path(path)/'chkpt').open('wb')                  
    )
        
def has_checkpoint(path):
    if (Path(path)/'chkpt').exists():
        return True
    return False           

def load_checkpoint(path):   
    fname = (Path(path)/'chkpt')
    if fname.exists():
        return torch.load(fname)       
