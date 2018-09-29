import numpy as np
import torch.utils.data as tdata
import torch
import torchvision.transforms as transforms
from PIL import Image
import glob
import os
import tqdm
import torchvision.models as tmodels
import torch.nn as nn
from torch.autograd import Variable
import torch
import bz2
from utils import utils
import h5py
import models
import itertools
import os 
import collections
import scipy.io
from sklearn.model_selection import train_test_split


class ImageLoader:
    def __init__(self, root):
        self.img_dir = root

    def __call__(self, img):
        file = '%s/%s'%(self.img_dir, img)
        img = Image.open(file).convert('RGB')
        return img

def imagenet_transform(phase):
    mean, std = [0.485, 0.456, 0.406], [0.229, 0.224, 0.225]

    if phase=='train':
        transform = transforms.Compose([
                        transforms.RandomResizedCrop(224),
                        transforms.RandomHorizontalFlip(),
                        transforms.ToTensor(),
                        transforms.Normalize(mean, std)
                    ])
    elif phase=='test':
        transform = transforms.Compose([
                        transforms.Resize(256),
                        transforms.CenterCrop(224),
                        transforms.ToTensor(),
                        transforms.Normalize(mean, std)
                    ])

    return transform

#------------------------------------------------------------------------------------------------------------------------------------#

class CompositionDataset(tdata.Dataset):

    def __init__(self, root, phase, split='compositional-split'):
        self.root = root
        self.phase = phase
        self.split = split

        self.feat_dim = None
        self.transform = imagenet_transform(phase)
        self.loader = ImageLoader(self.root+'/images/')

        self.attrs, self.objs, self.pairs, self.train_pairs, self.test_pairs = self.parse_split()
        assert len(set(self.train_pairs)&set(self.test_pairs))==0, 'train and test are not mutually exclusive'

        self.train_data, self.test_data = self.get_split_info()
        self.data = self.train_data if self.phase=='train' else self.test_data

        self.attr2idx = {attr: idx for idx, attr in enumerate(self.attrs)}
        self.obj2idx = {obj: idx for idx, obj in enumerate(self.objs)}
        self.pair2idx = {pair: idx for idx, pair in enumerate(self.pairs)}

        print ('# train pairs: %d | # test pairs: %d'%(len(self.train_pairs), len(self.test_pairs)))

        # fix later -- affordance thing
        # return {object: all attrs that occur with obj}
        self.obj_affordance = {}
        self.train_obj_affordance = {}
        for _obj in self.objs:
            candidates = [attr for (_, attr, obj) in self.train_data+self.test_data if obj==_obj]
            self.obj_affordance[_obj] = list(set(candidates))

            candidates = [attr for (_, attr, obj) in self.train_data if obj==_obj]
            self.train_obj_affordance[_obj] = list(set(candidates))
        
    def get_split_info(self):

        data = torch.load(self.root+'/metadata.t7')
        train_pair_set = set(self.train_pairs)
        train_data, test_data = [], []
        for instance in data:

            image, attr, obj = instance['image'], instance['attr'], instance['obj']

            if attr=='NA' or (attr, obj) not in self.pairs:
                # ignore instances with unlabeled attributes
                # ignore instances that are not in current split
                continue

            data_i = [image, attr, obj]
            if (attr, obj) in train_pair_set:
                train_data.append(data_i)
            else:
                test_data.append(data_i)

        return train_data, test_data

    def parse_split(self):

        def parse_pairs(pair_list):
            with open(pair_list,'r') as f:
                pairs = f.read().strip().split('\n')
                pairs = [t.split() for t in pairs]
                pairs = list(map(tuple, pairs))
            attrs, objs = zip(*pairs)
            return attrs, objs, pairs

        tr_attrs, tr_objs, tr_pairs = parse_pairs('%s/%s/train_pairs.txt'%(self.root, self.split))
        ts_attrs, ts_objs, ts_pairs = parse_pairs('%s/%s/test_pairs.txt'%(self.root, self.split))

        all_attrs, all_objs =  sorted(list(set(tr_attrs+ts_attrs))), sorted(list(set(tr_objs+ts_objs)))    
        all_pairs = sorted(list(set(tr_pairs + ts_pairs)))

        return all_attrs, all_objs, all_pairs, tr_pairs, ts_pairs

    def sample_negative(self, attr, obj):
        new_attr, new_obj = self.train_pairs[np.random.choice(len(self.train_pairs))]
        if new_attr==attr and new_obj==obj:
            return self.sample_negative(attr, obj)
        return (self.attr2idx[new_attr], self.obj2idx[new_obj])

    def sample_affordance(self, attr, obj):
        new_attr = np.random.choice(self.obj_affordance[obj])
        if new_attr==attr:
            return self.sample_affordance(attr, obj)
        return self.attr2idx[new_attr]

    def sample_train_affordance(self, attr, obj):
        new_attr = np.random.choice(self.train_obj_affordance[obj])
        if new_attr==attr:
            return self.sample_train_affordance(attr, obj)
        return self.attr2idx[new_attr]

    def __getitem__(self, index):
        image, attr, obj = self.data[index]
        img = self.loader(image)
        img = self.transform(img)

        data = [img, self.attr2idx[attr], self.obj2idx[obj], self.pair2idx[(attr, obj)]]

        if self.phase=='train':
            neg_attr, neg_obj = self.sample_negative(attr, obj) # negative example for triplet loss
            inv_attr = self.sample_train_affordance(attr, obj)  # attribute for inverse regularizer
            comm_attr = self.sample_affordance(inv_attr, obj)   # attribute for commutative regularizer
            data += [neg_attr, neg_obj, inv_attr, comm_attr]
        return data

    def __len__(self):
        return len(self.data)

#------------------------------------------------------------------------------------------------------------------------------------#

class CompositionDatasetActivations(CompositionDataset):

    def __init__(self, root, phase, split):
        super(CompositionDatasetActivations, self).__init__(root, phase, split)

        # precompute the activations -- weird. Fix pls
        feat_file = '%s/features.t7'%root
        if not os.path.exists(feat_file):
            with torch.no_grad():
                self.generate_features(feat_file)

        activation_data = torch.load(feat_file)
        self.activations = dict(zip(activation_data['files'], activation_data['features']))
        self.feat_dim = activation_data['features'].size(1)

        print ('%d activations loaded'%(len(self.activations)))

    def generate_features(self, out_file):

        data = self.train_data+self.test_data
        transform = imagenet_transform('test')
        feat_extractor = tmodels.resnet18(pretrained=True)
        feat_extractor.fc = nn.Sequential()
        feat_extractor.eval().cuda()

        image_feats = []
        image_files = []
        for chunk in tqdm.tqdm(utils.chunks(data, 512), total=len(data)//512):
            files, attrs, objs = zip(*chunk)
            imgs = list(map(self.loader, files))
            imgs = list(map(transform, imgs)) 
            feats = feat_extractor(torch.stack(imgs, 0).cuda())
            image_feats.append(feats.data.cpu())
            image_files += files
        image_feats = torch.cat(image_feats, 0)
        print ('features for %d images generated'%(len(image_files)))

        torch.save({'features': image_feats, 'files': image_files}, out_file)
      

    def __getitem__(self, index):
        data = super(CompositionDatasetActivations, self).__getitem__(index)
        image, attr, obj = self.data[index]
        data[0] = self.activations[image]
        return data
