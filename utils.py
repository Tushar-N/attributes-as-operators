import os
import re
import torch
import torchvision.transforms as transforms
import torchvision.datasets as torchdata
import numpy as np
from torch.autograd import Variable
import itertools
import copy
from sklearn.metrics import average_precision_score

# Save the training script and all the arguments to a file so that you 
# don't feel like an idiot later when you can't replicate results
import shutil
def save_args(args):
    shutil.copy('train.py', args.cv_dir)
    shutil.copy('models.py', args.cv_dir)
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
    
def performance_stats(attr_pred, obj_pred, data):
  
    _, attr_truth, obj_truth, _ = data[:4]
    attr_truth, obj_truth = attr_truth.cpu(), obj_truth.cpu()

    zsl_match = (attr_pred[:,0]==attr_truth).data.float()*(obj_pred[:,0]==obj_truth).data.float()
    attr_match = (attr_pred[:,1]==attr_truth).data.float()
    obj_match = (obj_pred[:,1]==obj_truth).data.float()
    gzsl_match = attr_match*obj_match
    fixobj_match = (attr_pred[:,2]==attr_truth).data.float()*(obj_pred[:,2]==obj_truth).data.float()

    return attr_match, obj_match, zsl_match, gzsl_match, fixobj_match

def generate_prediction_tensors(scores, dset, obj_truth, is_distance=False, source='manifold'):
    pairs, test_pairs = dset.pairs, dset.test_pairs
    pairs = torch.LongTensor(pairs)
    batch_size = obj_truth.size(0)
    obj_truth = obj_truth.cpu()

    if source=='manifold':

        scores = {k:v.cpu() for k,v in scores.items()}

        if is_distance:
            scores = {k:-v for k,v in scores.items()}

        gzsl_scores = torch.cat([scores[(attr, obj)] for attr, obj in pairs], 1)
        zsl_scores = torch.cat([scores[(attr, obj)] for attr, obj in test_pairs], 1)

        fixobj_scores = copy.deepcopy(scores)
        for attr, obj in scores:
            fixobj_scores[(attr, obj)][obj_truth!=obj] = -1e10
        fixobj_scores = torch.cat([fixobj_scores[(attr, obj)] for attr, obj in pairs], 1)

        _, zsl_pred = zsl_scores.max(1)
        _, gzsl_pred = gzsl_scores.max(1)
        _, fixobj_pred = fixobj_scores.max(1)

        attr_pred = [[test_pairs[zsl_pred[i]][0], pairs[gzsl_pred[i]][0], pairs[fixobj_pred[i]][0]] for i in range(batch_size)]
        obj_pred = [[test_pairs[zsl_pred[i]][1], pairs[gzsl_pred[i]][1], pairs[fixobj_pred[i]][1]] for i in range(batch_size)]

        attr_pred = Variable(torch.Tensor(attr_pred)).long() # (B,3)
        obj_pred = Variable(torch.Tensor(obj_pred)).long() # (B,3)

    elif source=='classification':

        scores = [s.cpu() for s in scores]
        attr_pred, obj_pred = scores

        # gzsl
        attr_subset = attr_pred.index_select(1, pairs[:,0])
        obj_subset = obj_pred.index_select(1, pairs[:,1])
        gzsl_scores = (attr_subset*obj_subset)
        _, pair_pred = gzsl_scores.max(1)
        gzsl_attr_pred, gzsl_obj_pred = pairs[pair_pred][:,0], pairs[pair_pred][:,1]

        # zsl
        attr_subset = attr_pred.index_select(1, test_pairs[:,0])
        obj_subset = obj_pred.index_select(1, test_pairs[:,1])
        zsl_scores = (attr_subset*obj_subset)
        _, pair_pred = zsl_scores.max(1)
        zsl_attr_pred, zsl_obj_pred = test_pairs[pair_pred][:,0], test_pairs[pair_pred][:,1]

        # fix obj
        attr_set = set(range(len(dset.attrs)))
        fixobj_attr_pred = attr_pred.clone()
        for i in range(batch_size):
            afforded = set(dset.obj_affordance[obj_truth[i]])
            if len(attr_set - afforded)==0:
                # all attributes are potential candidates
                continue
            remove = torch.LongTensor(list(attr_set - afforded))
            fixobj_attr_pred[i][remove] = -1e10

        _, fixobj_attr_pred = fixobj_attr_pred.max(1)

        attr_pred = Variable(torch.stack([zsl_attr_pred, gzsl_attr_pred, fixobj_attr_pred], 1))
        obj_pred = Variable(torch.stack([zsl_obj_pred, gzsl_obj_pred, obj_truth], 1))

    return attr_pred, obj_pred, gzsl_scores

