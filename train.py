import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.autograd import Variable

import numpy as np
import tqdm
import cPickle as pickle
import dataset as dset
import torchvision.models as tmodels
import tqdm
import models
import os
import itertools
import glob

from tensorboard_logger import configure, log_value
from utils import utils

import torch.backends.cudnn as cudnn
cudnn.benchmark = True

import argparse
parser = argparse.ArgumentParser()
parser.add_argument('--dataset', default='mitstates', help='mitstates|zappos')
parser.add_argument('--data_dir', default='data/mit-states/', help='data root dir')
parser.add_argument('--cv_dir', default='cv/tmp/', help='dir to save checkpoints to')
parser.add_argument('--load', default=None, help='path to checkpoint to load from')
parser.add_argument('--val', action='store_true', default=False, help='use the train/val splits instead of the train/test splits')

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


if not os.path.exists(args.cv_dir):
    os.system('mkdir ' + args.cv_dir)
utils.save_args(args)

#----------------------------------------------------------------#

def train(epoch):

    model.train()

    train_loss = 0.0
    for idx, data in tqdm.tqdm(enumerate(trainloader), total=len(trainloader)):

        data = [Variable(d).cuda() for d in data]
        loss, _ = model(data)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        train_loss += loss.data[0]

    train_loss = train_loss/len(trainloader)
    log_value('train_loss', train_loss, epoch)
    print 'E: %d | L: %.2E'%(epoch, train_loss)


def test(epoch):

    model.eval()

    accuracies = []
    for idx, data in tqdm.tqdm(enumerate(testloader), total=len(testloader)):

        data = [Variable(d, volatile=True).cuda() for d in data]
        _, [attr_pred, obj_pred, _] = model(data)
        
        match_stats = utils.performance_stats(attr_pred, obj_pred, data)
        accuracies.append(match_stats)

    accuracies = zip(*accuracies)
    accuracies = map(torch.mean, map(torch.cat, accuracies))
    attr_acc, obj_acc, closed_acc, open_acc, objoracle_acc = accuracies

    log_value('test_attr_acc', attr_acc, epoch)
    log_value('test_obj_acc', obj_acc, epoch)
    log_value('test_closed_acc', closed_acc, epoch)
    log_value('test_open_acc', open_acc, epoch)
    log_value('test_objoracle_acc', objoracle_acc, epoch)
    print '(test) E: %d | A: %.3f | O: %.3f | Cl: %.3f | Op: %.4f | OrO: %.4f'%(epoch, attr_acc, obj_acc, closed_acc, open_acc, objoracle_acc)

    if epoch>0 and epoch%args.save_every==0:
        state = {
            'net': model.state_dict(),
            'epoch': epoch,
        }
        torch.save(state, args.cv_dir+'/ckpt_E_%d_A_%.3f_O_%.3f_Cl_%.3f_Op_%.3f.t7'%(epoch, attr_acc, obj_acc, closed_acc, open_acc))

#----------------------------------------------------------------#
if args.dataset == 'mitstates':
    DSet = dset.MITStatesActivations
elif args.dataset == 'zappos':
    DSet = dset.UTZapposActivations

split = 'compositional-split-val' if args.val else 'compositional-split'
trainset = DSet(root=args.data_dir, phase='train', split=split)
trainloader = torch.utils.data.DataLoader(trainset, batch_size=args.batch_size, shuffle=True, num_workers=2)
testset = DSet(root=args.data_dir, phase='test', split=split)
testloader = torch.utils.data.DataLoader(testset, batch_size=args.batch_size, shuffle=False, num_workers=2)

if args.model == 'visprodNN':
    model = models.VisualProductNN(trainset, args)
elif args.model == 'redwine':
    model = models.RedWine(trainset, args)
elif args.model =='labelembed+':
    model = models.LabelEmbedPlus(trainset, args)
elif args.model =='attributeop':
    model = models.AttributeOperator(trainset, args)

if args.model=='redwine':
    params = filter(lambda p: p.requires_grad, model.parameters())
    optimizer = optim.SGD(params, lr=0.01, weight_decay=args.wd, momentum=0.9)
elif args.model=='attributeop':
    attr_params = [param for name, param in model.named_parameters() if 'attr_op' in name and param.requires_grad]
    other_params = [param for name, param in model.named_parameters() if 'attr_op' not in name and param.requires_grad]
    optim_params = [{'params':attr_params, 'lr':0.1*args.lr}, {'params':other_params}]
    optimizer = optim.Adam(optim_params, lr=args.lr, weight_decay=args.wd)
else:
    params = filter(lambda p: p.requires_grad, model.parameters())
    optimizer = optim.Adam(params, lr=args.lr, weight_decay=args.wd)

model.cuda()
print model

start_epoch = 0
if args.load is not None:
    checkpoint = torch.load(args.load)
    model.load_state_dict(checkpoint['net'])
    start_epoch = checkpoint['epoch']
    print 'loaded model from', os.path.basename(args.load)

configure(args.cv_dir+'/log', flush_secs=5)
for epoch in range(start_epoch, start_epoch+args.max_epochs+1):
    train(epoch)
    if epoch%args.eval_val_every==0:
        test(epoch)