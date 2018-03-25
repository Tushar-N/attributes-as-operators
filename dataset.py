import numpy as np
import cPickle as pickle
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
import utils
import h5py
import models
import itertools
import os 
import collections
import scipy.io
from sklearn.model_selection import train_test_split


def generate_hdf5(out_file, dset):

    def load_img(index):
        img = dset.img_dir+'/'+dset.images[index]
        img = Image.open(img).convert('RGB')
        img = transform(img)
        return img
    transform = imagenet_transform('test')

    feat_extractor = tmodels.resnet18(pretrained=True)
    feat_extractor.fc = nn.Sequential()
    feat_extractor.eval().cuda()

    image_feats = []
    for chunk in tqdm.tqdm(utils.chunks(range(len(dset.images)), 512), total=len(dset.images)/512):
        imgs = map(load_img, chunk)
        imgs = Variable(torch.stack(imgs), volatile=True).cuda()
        feats = feat_extractor(imgs).data.cpu()
        image_feats.append(feats)
    image_feats = torch.cat(image_feats, 0).numpy()
    print image_feats.shape

    hf = h5py.File(out_file, 'w')
    hf.create_dataset('feats', data=image_feats)
    hf.close()

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

    def __init__(self, root, split_dir, phase):
        self.root = root
        self.phase = phase
        self.activ, self.feat_dim = None, None
        self.transform = imagenet_transform(phase)
        meta_dir = os.path.join(root, split_dir)

        self.attrs, self.objs, self.pairs, self.train_pairs, self.test_pairs = self.parse_split(meta_dir)
        assert len(set(self.train_pairs)&set(self.test_pairs))==0, 'train and test are not mutually exclusive'

        self.images, self.train_data, self.test_data = self.get_split_info()
        self.data = self.train_data if self.phase=='train' else self.test_data

        # convert all pair information into split index space
        self.pairs = [(self.attrs.index(p[0]), self.objs.index(p[1])) for p in self.pairs]
        self.train_pairs = [(self.attrs.index(p[0]), self.objs.index(p[1])) for p in self.train_pairs]
        self.test_pairs = [(self.attrs.index(p[0]), self.objs.index(p[1])) for p in self.test_pairs]
        self.train_pairs = torch.LongTensor(self.train_pairs)
        self.test_pairs = torch.LongTensor(self.test_pairs)

        # generate affordance maps for training
        self.obj_affordance, self.attr_affordance = {}, {}
        self.train_obj_affordance = {}
        for ob_id in range(len(self.objs)):
            candidates = [attr for _,attr,obj,_ in self.train_data+self.test_data if obj==ob_id]
            self.obj_affordance[ob_id] = list(set(candidates))

            candidates = [attr for _,attr,obj,_ in self.train_data if obj==ob_id]
            self.train_obj_affordance[ob_id] = list(set(candidates))

        for at_id in range(len(self.attrs)):
            candidates = [obj for _,attr,obj,_ in self.train_data+self.test_data if attr==at_id] 
            self.attr_affordance[at_id] = list(set(candidates))


        self.inv_images = collections.defaultdict(list)
        for im_id, attr, obj, _ in self.train_data:
            self.inv_images[(attr, obj)].append(im_id)

        
        # write out metadata
        with open(meta_dir+'/objs.txt','w') as f:
            f.write('\n'.join(self.objs))
        with open(meta_dir+'/attrs.txt','w') as f:
            f.write('\n'.join(self.attrs))
        with open(meta_dir+'/all_pairs.txt','w') as f:
            all_pairs = torch.cat([self.train_pairs, self.test_pairs], 0)
            all_pairs = ['%s %s'%(self.attrs[attr], self.objs[obj]) for attr, obj in all_pairs]
            f.write('\n'.join(all_pairs))


    # Overwrite in subclasses
    def get_split_info(self):
        pass

    def load_img(self, index):
        img = self.img_dir+'/'+self.images[index]
        img = Image.open(img).convert('RGB')
        img = self.transform(img)
        return img

    def parse_split(self, meta_dir):

        def parse_pairs(pair_list):
            with open(pair_list,'r') as f:
                pairs = f.read().strip().split('\n')
                pairs = [t.split() for t in pairs]
                pairs = map(tuple, pairs)
            attrs, objs = zip(*pairs)
            return attrs, objs, pairs

        tr_attrs, tr_objs, tr_pairs = parse_pairs(meta_dir+'/train_pairs.txt')
        ts_attrs, ts_objs, ts_pairs = parse_pairs(meta_dir+'/test_pairs.txt')

        all_attrs, all_objs =  sorted(list(set(tr_attrs+ts_attrs))), sorted(list(set(tr_objs+ts_objs)))    
        all_pairs = sorted(list(set(tr_pairs + ts_pairs)))

        # text attrs, objs, pairs
        return all_attrs, all_objs, all_pairs, tr_pairs, ts_pairs

    def sample_negative(self, attr, obj):
        new_pair = self.train_pairs[np.random.choice(len(self.train_pairs))]
        if new_pair[0]==attr and new_pair[1]==obj: # AND/OR?
            return self.sample_negative(attr, obj)
        return new_pair

    def sample_affordance(self, attr, obj):
        new_attr = np.random.choice(self.obj_affordance[obj])
        while new_attr==attr:
            new_attr = np.random.choice(self.obj_affordance[obj])
        return new_attr

    def sample_train_affordance(self, attr, obj):
        new_attr = np.random.choice(self.train_obj_affordance[obj])
        while new_attr==attr:
            new_attr = np.random.choice(self.train_obj_affordance[obj])
        return new_attr

    def __getitem__(self, index):
        img_id, attr_label, obj_label, pair_label = self.data[index]
        img = self.load_img(img_id)
        data = [img, attr_label, obj_label, pair_label]

        if self.phase=='train':
            neg_attr, neg_obj = self.sample_negative(attr_label, obj_label)
            inv_attr = self.sample_train_affordance(attr_label, obj_label)
            comm_attr = self.sample_affordance(inv_attr, obj_label)
            data += [neg_attr, neg_obj, inv_attr, comm_attr]

        return data

    def __len__(self):
        return len(self.data)

#------------------------------------------------------------------------------------------------------------------------------------#

class MITStates(CompositionDataset):

    def __init__(self, root, split_dir, phase):
        super(MITStates, self).__init__(root, split_dir, phase)
        self.img_dir = self.root + '/images/'

    # retrieve dataset level info: all images, all attrs, all objs
    # use this to pull out the relevant information for the split in question
    def get_split_info(self):
        data = pickle.load(open(self.root+'/metadata.pkl'))
        images, all_annots, all_attrs, all_objs = data['files'], data['annots'], data['attributes'], data['objects']

        train_pair_set = set(self.train_pairs)
        train_data, test_data = [], []
        for idx in range(len(images)):

            attr_id, obj_id = all_annots[idx]['attr'], all_annots[idx]['obj']

            # X obj images. Ignore.
            if attr_id==-1:
                continue

            attr, obj = all_attrs[attr_id], all_objs[obj_id]

            # pair not in current split
            if (attr, obj) not in self.pairs:
                continue

            data_i = [idx, self.attrs.index(attr), self.objs.index(obj), self.pairs.index((attr, obj))]
            if (attr, obj) in train_pair_set:
                train_data.append(data_i)
            else:
                test_data.append(data_i)

        self.all_attrs, self.all_objs = all_attrs, all_objs

        return images, train_data, test_data


class MITStatesActivations(MITStates):

    def __init__(self, root, split_dir, phase, activ='resnet'):
        super(MITStatesActivations, self).__init__(root, split_dir, phase)
        self.activ = activ

        # precompute the activations
        feat_file = 'data/mitstates_%s.h5'%activ
        if not os.path.exists(feat_file):
            generate_hdf5(feat_file, self)

        hf = h5py.File(feat_file, 'r')
        self.activations = torch.from_numpy(np.array(hf['feats'])) # h5py is a MESS. Convert to np.array
        self.feat_dim = self.activations.size(1)
        hf.close()

    def load_img(self, img_id):
        feats = self.activations[img_id]
        return feats

#------------------------------------------------------------------------------------------------------------------------------------#

class UTZappos(CompositionDataset):


    def __init__(self, root, split_dir, phase):
        super(UTZappos, self).__init__(root, split_dir, phase)
        self.img_dir = self.root + '/images/'
    
    # retrieve dataset level info: all images, all attrs, all objs
    # use this to pull out the relevant information for the split in question
    def get_split_info(self):
        images, all_attrs, all_objs, all_pairs, all_attr_labels, all_obj_labels = pickle.load(open(self.root+'/metadata.pkl', 'rb'))

        # fix the missing fullstops
        fixed_images = []
        for fl in images:
            fl = fl.split('/')
            fl[-2] = fl[-2].strip('.')
            fl = '/'.join(fl)
            fixed_images.append(fl)
        images = fixed_images

        train_pair_set = set(self.train_pairs)
        train_data, test_data = [], []
        for idx in range(len(images)):

            attr, obj = all_attrs[all_attr_labels[idx]], all_objs[all_obj_labels[idx]]

            # image not in relevant split
            if (attr, obj) not in self.pairs:
                continue

            data_i = [idx, self.attrs.index(attr), self.objs.index(obj), self.pairs.index((attr, obj))]
            if (attr, obj) in train_pair_set:
                train_data.append(data_i)
            else:
                test_data.append(data_i)

        self.all_attrs, self.all_objs = all_attrs, all_objs

        return images, train_data, test_data


class UTZapposActivations(UTZappos):

    def __init__(self, root, split_dir, phase, activ='resnet'):
        super(UTZapposActivations, self).__init__(root, split_dir, phase)
        self.activ = activ

        feat_file = 'data/zappos_%s.h5'%activ
        if not os.path.exists(feat_file):
            generate_hdf5(feat_file, self)

        hf = h5py.File(feat_file, 'r')
        self.activations = torch.from_numpy(np.array(hf['feats'])) # h5py is a MESS. Convert to np.array
        self.feat_dim = self.activations.size(1)
        hf.close()

    def load_img(self, img_id):
        feats = self.activations[img_id]
        return feats
