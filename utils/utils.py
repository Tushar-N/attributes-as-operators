import os
import re
import torch
import torchvision.transforms as transforms
import torchvision.datasets as torchdata
import numpy as np
from torch.autograd import Variable
import itertools
import copy

# Save the training script and all the arguments to a file so that you 
# don't feel like an idiot later when you can't replicate results
import shutil
def save_args(args):
    shutil.copy('train.py', args.cv_dir)
    shutil.copy('models/models.py', args.cv_dir)
    with open(args.cv_dir+'/args.txt','w') as f:
        f.write(str(args))

class UnNormalizer:
    def __init__(self, mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]):
        self.mean = mean
        self.std = std
        
    def __call__(self, tensor):
        for b in range(tensor.size(0)):
            for t, m, s in zip(tensor[b], self.mean, self.std):
                t.mul_(s).add_(m)
        return tensor

def chunks(l, n):
    """Yield successive n-sized chunks from l."""
    for i in range(0, len(l), n):
        yield l[i:i + n]

def flatten(l):
    return list(itertools.chain.from_iterable(l))
    
