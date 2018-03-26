import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import torchvision.models as tmodels
import numpy as np
from utils import utils
import itertools
import cPickle as pickle
import math
import collections


def load_word_embeddings(emb_file, vocab):

    vocab = [v.lower() for v in vocab]

    embeds = {}
    for line in open(emb_file, 'r'):
        line = line.strip().split(' ')
        wvec = torch.Tensor(map(float, line[1:]))
        embeds[line[0]] = wvec

    # for zappos (should account for everything)
    custom_map = {'Faux.Fur':'fur', 'Faux.Leather':'leather', 'Full.grain.leather':'leather', 'Hair.Calf':'hair', 'Patent.Leather':'leather', 'Nubuck':'leather', 
                'Boots_Ankle':'boots', 'Boots_Knee_High':'knee-high', 'Boots_Mid-Calf':'midcalf', 'Shoes_Boat_Shoes':'shoes', 'Shoes_Clogs_and_Mules':'clogs',
                'Shoes_Flats':'flats', 'Shoes_Heels':'heels', 'Shoes_Loafers':'loafers', 'Shoes_Oxfords':'oxfords', 'Shoes_Sneakers_and_Athletic_Shoes':'sneakers'}
    for k in custom_map:
        embeds[k.lower()] = embeds[custom_map[k]]

    embeds = [embeds[k] for k in vocab]
    embeds = torch.stack(embeds)
    print 'loaded embeddings', embeds.size()

    return embeds

#--------------------------------------------------------------------------------#

class MLP(nn.Module):
    def __init__(self, inp_dim, out_dim, num_layers=1, relu=True, bias=True):
        super(MLP, self).__init__()
        mod = []
        for L in range(num_layers-1):
            mod.append(nn.Linear(inp_dim, inp_dim, bias=bias))
            mod.append(nn.ReLU(True))

        mod.append(nn.Linear(inp_dim, out_dim, bias=bias))
        if relu:
            mod.append(nn.ReLU(True))

        self.mod = nn.Sequential(*mod)

    def forward(self, x):
        output = self.mod(x)
        return output


class VisualProductNN(nn.Module):
    def __init__(self, dset, args):
        super(VisualProductNN, self).__init__()
        self.attr_clf = MLP(dset.feat_dim, len(dset.attrs), 2, relu=False)
        self.obj_clf = MLP(dset.feat_dim, len(dset.objs), 2, relu=False)
        self.dset = dset

    def train_forward(self, x):
        img, attr_label, obj_label = x[0], x[1], x[2]

        attr_pred = self.attr_clf(img)
        obj_pred = self.obj_clf(img)

        attr_loss = F.cross_entropy(attr_pred, attr_label)
        obj_loss = F.cross_entropy(obj_pred, obj_label)
        loss = attr_loss + obj_loss

        return loss, None

    def val_forward(self, x):
        img = x[0]
        attr_pred = F.softmax(self.attr_clf(img), dim=1).data
        obj_pred = F.softmax(self.obj_clf(img), dim=1).data

        attr_pred, obj_pred, score_tensor = utils.generate_prediction_tensors([attr_pred, obj_pred], self.dset, x[2].data, source='classification')
        return None, [attr_pred, obj_pred, score_tensor]

    def forward(self, x):
        if self.training:
            loss, pred = self.train_forward(x)
        else:
            loss, pred = self.val_forward(x)
        return loss, pred


class RedWine(nn.Module):

    def __init__(self, dset, args):
        super(RedWine, self).__init__()
        self.args = args
        self.dset = dset
        in_dim = dset.feat_dim if not args.glove_init else 300

        self.pairs = dset.pairs
        self.test_pairs = dset.test_pairs
        self.train_pairs = dset.train_pairs

        self.T = nn.Sequential(
            nn.Linear(2*in_dim, 3*in_dim),
            nn.LeakyReLU(0.1, True),
            nn.Linear(3*in_dim, 3*in_dim/2),
            nn.LeakyReLU(0.1, True),
            nn.Linear(3*in_dim/2, dset.feat_dim))

        # eye init
        for mod in self.T:
            if type(mod)==nn.Linear:
                torch.nn.init.eye(mod.weight)
                mod.bias.data.fill_(0)

        self.attr_embedder = nn.Embedding(len(dset.attrs), in_dim)
        self.obj_embedder = nn.Embedding(len(dset.objs), in_dim)

        # initialize the weights of the embedders with the svm weights
        if args.glove_init:
            pretrained_weight = load_word_embeddings('data/glove/glove.6B.300d.txt', dset.attrs)
            self.attr_embedder.weight.data.copy_(pretrained_weight)
            pretrained_weight = load_word_embeddings('data/glove/glove.6B.300d.txt', dset.objs)
            self.obj_embedder.weight.data.copy_(pretrained_weight)

        elif args.clf_init:
            for idx, attr in enumerate(dset.attrs):
                at_id = idx
                weight = pickle.load(open('%s/svm/attr_%d'%(args.data_dir, at_id))).coef_.squeeze()
                self.attr_embedder.weight[idx].data.copy_(torch.from_numpy(weight))
            for idx, obj in enumerate(dset.objs):
                obj_id = idx
                weight = pickle.load(open('%s/svm/obj_%d'%(args.data_dir, obj_id))).coef_.squeeze()
                self.obj_embedder.weight[idx].data.copy_(torch.from_numpy(weight))
        else:
            print 'init must be either glove or clf'
            return

        if args.static_inp:
            for param in self.attr_embedder.parameters():
                param.requires_grad = False
            for param in self.obj_embedder.parameters():
                param.requires_grad = False
        
    def compose(self, attrs, objs):
        attr_wt = self.attr_embedder(attrs)
        obj_wt = self.obj_embedder(objs)
        inp_wts = torch.cat([attr_wt, obj_wt], 1) # 2D
        composed_clf = self.T(inp_wts)
        return composed_clf

    def sample(self, attrs, objs):
        
        def rand_pair(attr, obj):
            new_pair = self.train_pairs[np.random.choice(len(self.train_pairs))]
            if new_pair[0]==attr and new_pair[1]==obj:
                return rand_pair(attr, obj)
            return new_pair

        sample_pairs = []
        match = []
        for i in range(attrs.size(0)):
            # 25% of the examples retain their original pairs
            if np.random.rand()<=0.25:
                sample_pairs.append((attrs.data[i], objs.data[i]))
                match.append(1)
            else:
                sample_pairs.append(rand_pair(attrs.data[i], objs.data[i]))
                match.append(0)

        sample_attrs, sample_objs = zip(*sample_pairs)
        sample_attrs = Variable(torch.LongTensor(sample_attrs)).cuda()
        sample_objs = Variable(torch.LongTensor(sample_objs)).cuda()
        match = Variable(torch.Tensor(match)).cuda()

        return sample_attrs, sample_objs, match


    def train_forward(self, x):

        img, attr_label, obj_label = x[0], x[1], x[2]

        attrs, objs, match = self.sample(attr_label, obj_label)
        composed_clf = self.compose(attrs, objs)
        p = F.sigmoid((img*composed_clf).sum(1))
        loss = F.binary_cross_entropy(p, match)

        return loss, None

    def val_forward(self, x):
        img, obj_label = x[0], x[2]
        batch_size = img.size(0)

        attrs, objs = zip(*self.pairs)
        attrs = Variable(torch.LongTensor(attrs), volatile=True).cuda()
        objs = Variable(torch.LongTensor(objs), volatile=True).cuda()
        composed_clfs = self.compose(attrs, objs)

        scores = {}
        for i, (attr, obj) in enumerate(self.pairs):
            composed_clf = composed_clfs[i, None].expand(batch_size, composed_clfs.size(1))
            score = F.sigmoid((img*composed_clf).sum(1)).unsqueeze(1)
            scores[(attr, obj)] = score.data
        attr_pred, obj_pred, score_tensor = utils.generate_prediction_tensors(scores, self.dset, obj_label.data, is_distance=False, source='manifold')

        return None, [attr_pred, obj_pred, score_tensor]

    def forward(self, x):
        if self.training:
            loss, pred = self.train_forward(x)
        else:
            loss, pred = self.val_forward(x)
        return loss, pred


# Base Manifold Learning model for LabelEmbed+ and AttributeOperator
class ManifoldModel(nn.Module):

    def __init__(self, dset, args):
        super(ManifoldModel, self).__init__()
        self.args = args
        self.dset = dset
        self.margin = 0.5
        self.num_attrs, self.num_objs = len(dset.attrs), len(dset.objs)

        self.pairs = dset.pairs
        self.test_pairs = dset.test_pairs
        self.train_pairs = dset.train_pairs

        self.pdist_func = F.pairwise_distance

        if args.lambda_aux>0:
            self.obj_clf = nn.Linear(args.emb_dim, len(dset.objs))
            self.attr_clf = nn.Linear(args.emb_dim, len(dset.attrs))

    def train_forward(self, x):
        img, attr_label, obj_label = x[0], x[1], x[2]
        neg_attrs, neg_objs = x[4], x[5]

        img_feat = self.image_embedder(img)
        positive = self.compose(attr_label, obj_label)
        negative = self.compose(neg_attrs, neg_objs)
        loss = F.triplet_margin_loss(img_feat, positive, negative, margin=self.margin)

        # Auxiliary object/attribute prediction loss
        if self.args.lambda_aux>0:
            obj_pred = self.obj_clf(positive)
            attr_pred = self.attr_clf(positive)
            loss_aux = F.cross_entropy(attr_pred, attr_label) + F.cross_entropy(obj_pred, obj_label)
            loss += self.args.lambda_aux*loss_aux

        return loss, None

    def val_forward(self, x):
        img, obj_label = x[0], x[2]
        batch_size = img.size(0)

        img_feat = self.image_embedder(img)

        attrs, objs = zip(*self.pairs)
        attrs = Variable(torch.LongTensor(attrs), volatile=True).cuda()
        objs = Variable(torch.LongTensor(objs), volatile=True).cuda()
        attr_embeds = self.compose(attrs, objs)

        dists = {}
        for i, (attr, obj) in enumerate(self.pairs):
            attr_embed = attr_embeds[i, None].expand(batch_size, attr_embeds.size(1))
            dist = self.pdist_func(img_feat, attr_embed) # (B, 1)
            dists[(attr, obj)] = dist.data
        attr_pred, obj_pred, score_tensor = utils.generate_prediction_tensors(dists, self.dset, obj_label.data, is_distance=True, source='manifold')

        return None, [attr_pred, obj_pred, score_tensor]

    def forward(self, x):
        if self.training:
            loss, pred = self.train_forward(x)
        else:
            loss, pred = self.val_forward(x)
        return loss, pred


class LabelEmbedPlus(ManifoldModel):
    def __init__(self, dset, args):
        super(LabelEmbedPlus, self).__init__(dset, args)
        self.image_embedder = MLP(dset.feat_dim, args.emb_dim)

        input_dim = dset.feat_dim if args.clf_init else args.emb_dim
        self.attr_embedder = nn.Embedding(len(dset.attrs), input_dim)
        self.obj_embedder = nn.Embedding(len(dset.objs), input_dim)
        self.T = MLP(2*input_dim, args.emb_dim, num_layers=args.nlayers)

        # init with word embeddings
        if args.glove_init:
            pretrained_weight = load_word_embeddings('data/glove/glove.6B.300d.txt', dset.attrs)
            self.attr_embedder.weight.data.copy_(pretrained_weight)
            pretrained_weight = load_word_embeddings('data/glove/glove.6B.300d.txt', dset.objs)
            self.obj_embedder.weight.data.copy_(pretrained_weight)

        # init with classifier weights
        elif args.clf_init:
            for idx, attr in enumerate(dset.attrs):
                at_id = dset.all_attrs.index(attr)
                weight = pickle.load(open('%s/svm/attr_%d'%(args.data_dir, at_id))).coef_.squeeze()
                self.attr_embedder.weight[idx].data.copy_(torch.from_numpy(weight))
            for idx, obj in enumerate(dset.objs):
                obj_id = dset.all_objs.index(obj)
                weight = pickle.load(open('%s/svm/obj_%d'%(args.data_dir, obj_id))).coef_.squeeze()
                self.obj_emb.weight[idx].data.copy_(torch.from_numpy(weight))

        # static inputs
        if args.static_inp:
            for param in self.attr_embedder.parameters():
                param.requires_grad = False
            for param in self.obj_embedder.parameters():
                param.requires_grad = False

    def compose(self, attrs, objs):
        inputs = [self.attr_embedder(attrs), self.obj_embedder(objs)]
        inputs = torch.cat(inputs, 1)
        output = self.T(inputs)
        return output


class AttributeOperator(ManifoldModel):
    def __init__(self, dset, args):
        super(AttributeOperator, self).__init__(dset, args)
        self.image_embedder = MLP(dset.feat_dim, args.emb_dim)

        self.attr_ops = nn.ParameterList([nn.Parameter(torch.eye(args.emb_dim)) for _ in range(self.num_attrs)])
        self.obj_embedder = nn.Embedding(len(dset.objs), args.emb_dim)

        if args.glove_init:
            pretrained_weight = load_word_embeddings('data/glove/glove.6B.300d.txt', dset.objs)
            self.obj_embedder.weight.data.copy_(pretrained_weight)
            
        self.inverse_cache = {}

        if args.lambda_ant>0 and args.dataset=='mitstates':
            antonym_list = open('data/antonyms.txt').read().strip().split('\n')
            antonym_list = [l.split() for l in antonym_list]
            antonym_list = [[self.dset.attrs.index(a1), self.dset.attrs.index(a2)] for a1, a2 in antonym_list]
            antonyms = {}
            antonyms.update({a1:a2 for a1, a2 in antonym_list})
            antonyms.update({a2:a1 for a1, a2 in antonym_list})
            self.antonyms, self.antonym_list = antonyms, antonym_list

        if args.static_inp:
            for param in self.obj_embedder.parameters():
                param.requires_grad = False

    def apply_ops(self, ops, rep):
        out = torch.bmm(ops, rep.unsqueeze(2)).squeeze(2)
        out = F.relu(out)
        return out

    def compose(self, attrs, objs):
        obj_rep = self.obj_embedder(objs)
        attr_ops = torch.stack([self.attr_ops[attr] for attr in attrs.data])
        embedded_reps = self.apply_ops(attr_ops, obj_rep)
        return embedded_reps

    def apply_inverse(self, img_rep, attrs):
        inverse_ops = []
        for i in range(img_rep.size(0)):
            attr = attrs[i]
            if attr not in self.inverse_cache:
                self.inverse_cache[attr] = self.attr_ops[attr].inverse()
            inverse_ops.append(self.inverse_cache[attr])
        inverse_ops = torch.stack(inverse_ops) # (B,512,512)
        obj_rep = self.apply_ops(inverse_ops, img_rep)
        return obj_rep

    def train_forward(self, x):
        img, attr_label, obj_label = x[0], x[1], x[2]
        neg_attrs, neg_objs, inv_attr, comm_attr = x[4], x[5], x[6], x[7]
        batch_size = img.size(0)

        loss = []

        anchor = self.image_embedder(img)

        obj_emb = self.obj_embedder(obj_label)
        pos_ops = torch.stack([self.attr_ops[attr] for attr in attr_label.data])
        positive = self.apply_ops(pos_ops, obj_emb)

        neg_obj_emb = self.obj_embedder(neg_objs)
        neg_ops = torch.stack([self.attr_ops[attr] for attr in neg_attrs.data])
        negative = self.apply_ops(neg_ops, neg_obj_emb)

        loss_triplet = F.triplet_margin_loss(anchor, positive, negative, margin=self.margin)
        loss.append(loss_triplet)

        #-----------------------------------------------------------------------------#

        # Auxiliary object/attribute loss
        if self.args.lambda_aux>0:
            obj_pred = self.obj_clf(positive)
            attr_pred = self.attr_clf(positive)
            loss_aux = F.cross_entropy(attr_pred, attr_label) + F.cross_entropy(obj_pred, obj_label)
            loss.append(self.args.lambda_aux*loss_aux)

        # Inverse Consistency
        if self.args.lambda_inv>0:
            obj_rep = self.apply_inverse(anchor, attr_label.data)
            new_ops = torch.stack([self.attr_ops[attr] for attr in inv_attr.data])
            new_rep = self.apply_ops(new_ops, obj_rep)
            new_positive = self.apply_ops(new_ops, obj_emb)
            loss_inv = F.triplet_margin_loss(new_rep, new_positive, positive, margin=self.margin)
            loss.append(self.args.lambda_inv*loss_inv)

        # Commutative Operators
        if self.args.lambda_comm>0:
            B = torch.stack([self.attr_ops[attr] for attr in comm_attr.data])
            BA = self.apply_ops(B, positive)
            AB = self.apply_ops(pos_ops, self.apply_ops(B, obj_emb))
            loss_comm = ((AB-BA)**2).sum(1).mean()
            loss.append(self.args.lambda_comm*loss_comm)

        # Antonym Consistency
        if self.args.lambda_ant>0:

            select_idx = [i for i in range(batch_size) if attr_label.data[i] in self.antonyms]
            if len(select_idx)>0:
                select_idx = torch.LongTensor(select_idx).cuda()
                attr_subset = attr_label[select_idx].data
                antonym_ops = torch.stack([self.attr_ops[self.antonyms[attr]] for attr in attr_subset])

                Ao = anchor[select_idx]
                if self.args.lambda_inv>0:
                    o = obj_rep[select_idx]
                else:
                    o = self.apply_inverse(Ao, attr_subset)
                BAo = self.apply_ops(antonym_ops, Ao)

                loss_cycle = ((BAo-o)**2).sum(1).mean()
                loss.append(self.args.lambda_ant*loss_cycle)

        #-----------------------------------------------------------------------------#

        loss = sum(loss)
        return loss, None

    def forward(self, x):
        if self.training:
            loss, pred = self.train_forward(x)
        else:
            loss, pred = self.val_forward(x)
        self.inverse_cache = {}
        return loss, pred

#--------------------------------------------------------------------------------------------------------------#
