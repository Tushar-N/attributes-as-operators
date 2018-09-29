import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

import numpy as np
import tqdm
from data import dataset as dset
import torchvision.models as tmodels
import tqdm
from models import models
import os
import itertools
import glob

from utils import utils

import torch.backends.cudnn as cudnn
cudnn.benchmark = True

import argparse
parser = argparse.ArgumentParser()
parser.add_argument('--dataset', default='mitstates', help='mitstates|zappos')
parser.add_argument('--data_dir', default='data/mit-states/', help='data root dir')
parser.add_argument('--cv_dir', default='cv/tmp/', help='dir to save checkpoints to')
parser.add_argument('--load', default=None, help='path to checkpoint to load from')

# model parameters
parser.add_argument('--model', default='visprodNN', help='visprodNN|redwine|labelembed+|attributeop')
parser.add_argument('--emb_dim', type=int, default=300, help='dimension of common embedding space')
parser.add_argument('--nlayers', type=int, default=2, help='number of layers for labelembed+')
parser.add_argument('--glove_init', action='store_true', default=False, help='initialize inputs with word vectors')
parser.add_argument('--clf_init', action='store_true', default=False, help='initialize inputs with SVM weights')
parser.add_argument('--static_inp', action='store_true', default=False, help='do not optimize input representations')

# regularizers
parser.add_argument('--lambda_aux', type=float, default=0.0)
parser.add_argument('--lambda_inv', type=float, default=0.0)
parser.add_argument('--lambda_comm', type=float, default=0.0)
parser.add_argument('--lambda_ant', type=float, default=0.0)

# optimization
parser.add_argument('--batch_size', type=int, default=512)
parser.add_argument('--lr', type=float, default=1e-4)
parser.add_argument('--wd', type=float, default=5e-5)
parser.add_argument('--save_every', type=int, default=100)
parser.add_argument('--eval_val_every', type=int, default=20)
parser.add_argument('--max_epochs', type=int, default=1000)
args = parser.parse_args()

def test(epoch):

    model.eval()

    accuracies = []
    for idx, data in tqdm.tqdm(enumerate(testloader), total=len(testloader)):

        data = [d.cuda() for d in data]
        _, predictions = model(data)
        
        attr_truth, obj_truth = data[1], data[2]
        results = evaluator.score_model(predictions, obj_truth)
        match_stats = evaluator.evaluate_predictions(results, attr_truth, obj_truth)
        accuracies.append(match_stats)

    accuracies = zip(*accuracies)
    accuracies = map(torch.mean, map(torch.cat, accuracies))
    attr_acc, obj_acc, closed_acc, open_acc, objoracle_acc = accuracies

    print ('(test) E: %d | A: %.3f | O: %.3f | Cl: %.3f | Op: %.4f | OrO: %.4f'%(epoch, attr_acc, obj_acc, closed_acc, open_acc, objoracle_acc))

#----------------------------------------------------------------#

testset = dset.CompositionDatasetActivations(root=args.data_dir, phase='test', split='compositional-split')
testloader = torch.utils.data.DataLoader(testset, batch_size=args.batch_size, shuffle=False, num_workers=2)

if args.model == 'visprodNN':
    model = models.VisualProductNN(testset, args)
elif args.model == 'redwine':
    model = models.RedWine(testset, args)
elif args.model =='labelembed+':
    model = models.LabelEmbedPlus(testset, args)
elif args.model =='attributeop':
    model = models.AttributeOperator(testset, args)
model.cuda()

evaluator = models.Evaluator(testset, model)

checkpoint = torch.load(args.load)
model.load_state_dict(checkpoint['net'])
start_epoch = checkpoint['epoch']
print ('loaded model from', os.path.basename(args.load))

with torch.no_grad():
    test(0)
