import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as tmodels
import numpy as np
from utils import utils
import itertools
import math
import collections
from torch.distributions.bernoulli import Bernoulli


def load_word_embeddings(emb_file, vocab):

    vocab = [v.lower() for v in vocab]

    embeds = {}
    for line in open(emb_file, 'r'):
        line = line.strip().split(' ')
        wvec = torch.FloatTensor(list(map(float, line[1:])))
        embeds[line[0]] = wvec

    # for zappos (should account for everything)
    custom_map = {'Faux.Fur':'fur', 'Faux.Leather':'leather', 'Full.grain.leather':'leather', 'Hair.Calf':'hair', 'Patent.Leather':'leather', 'Nubuck':'leather', 
                'Boots.Ankle':'boots', 'Boots.Knee.High':'knee-high', 'Boots.Mid-Calf':'midcalf', 'Shoes.Boat.Shoes':'shoes', 'Shoes.Clogs.and.Mules':'clogs',
                'Shoes.Flats':'flats', 'Shoes.Heels':'heels', 'Shoes.Loafers':'loafers', 'Shoes.Oxfords':'oxfords', 'Shoes.Sneakers.and.Athletic.Shoes':'sneakers'}
    for k in custom_map:
        embeds[k.lower()] = embeds[custom_map[k]]

    embeds = [embeds[k] for k in vocab]
    embeds = torch.stack(embeds)
    print ('loaded embeddings', embeds.size())

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

class Evaluator:

    def __init__(self, dset, model):

        self.dset = dset 

        # convert text pairs to idx tensors: [('sliced', 'apple'), ('ripe', 'apple'), ...] --> torch.LongTensor([[0,1],[1,1], ...])
        pairs = [(dset.attr2idx[attr], dset.obj2idx[obj]) for attr, obj in dset.pairs]
        self.pairs = torch.LongTensor(pairs)

        # mask over pairs that occur in closed world 
        test_pair_set = set(dset.test_pairs)
        mask = [1 if pair in test_pair_set else 0 for pair in dset.pairs]
        self.closed_mask = torch.ByteTensor(mask)

        # object specific mask over which pairs occur in the object oracle setting
        oracle_obj_mask = []
        for _obj in dset.objs:
            mask = [1 if _obj==obj else 0 for attr, obj in dset.pairs]
            oracle_obj_mask.append(torch.ByteTensor(mask))
        self.oracle_obj_mask = torch.stack(oracle_obj_mask, 0)

        # decide if the model being evaluated is a manifold model or not
        mname = model.__class__.__name__
        if 'VisualProduct' in mname:
            self.score_model = self.score_clf_model
        else:
            self.score_model = self.score_manifold_model


    # generate masks for each setting, mask scores, and get prediction labels    
    def generate_predictions(self, scores, obj_truth): # (B, #pairs)

        def get_pred_from_scores(_scores):
            _, pair_pred = _scores.max(1)
            attr_pred, obj_pred = self.pairs[pair_pred][:,0], self.pairs[pair_pred][:,1]
            return (attr_pred, obj_pred)

        results = {}

        # open world setting -- no mask
        results.update({'open': get_pred_from_scores(scores)})

        # closed world setting - set the score for all NON test-pairs to -1e10
        mask = self.closed_mask.repeat(scores.shape[0], 1)
        closed_scores = scores.clone()
        closed_scores[1-mask] = -1e10
        results.update({'closed': get_pred_from_scores(closed_scores)})

        # object_oracle setting - set the score to -1e10 for all pairs where the true object does NOT participate
        mask = self.oracle_obj_mask[obj_truth]
        oracle_obj_scores = scores.clone()
        oracle_obj_scores[1-mask] = -1e10
        results.update({'object_oracle': get_pred_from_scores(oracle_obj_scores)})

        return results

    def score_clf_model(self, scores, obj_truth):

        attr_pred, obj_pred = scores

        # put everything on CPU
        attr_pred, obj_pred, obj_truth = attr_pred.cpu(), obj_pred.cpu(), obj_truth.cpu() 

        # - gather scores (P(a), P(o)) for all relevant (a,o) pairs
        # - multiply P(a)*P(o) to get P(pair)
        attr_subset = attr_pred.index_select(1, self.pairs[:,0])
        obj_subset = obj_pred.index_select(1, self.pairs[:,1])
        scores = (attr_subset*obj_subset) # (B, #pairs)
        
        results = self.generate_predictions(scores, obj_truth)
        return results

    def score_manifold_model(self, scores, obj_truth):

        # put everything on CPU
        scores = {k:v.cpu() for k,v in scores.items()}
        obj_truth = obj_truth.cpu()

        # gather scores for all relevant (a,o) pairs
        scores = torch.stack([scores[(attr, obj)] for attr, obj in self.dset.pairs], 1) # (B, #pairs)
        results = self.generate_predictions(scores, obj_truth)
        return results

    def evaluate_predictions(self, predictions, attr_truth, obj_truth):
  
        # put everything on cpu
        attr_truth, obj_truth = attr_truth.cpu(), obj_truth.cpu()

        # top 1 pair accuracy
        # open world: attribute, object and pair
        attr_match = (attr_truth==predictions['open'][0]).float()
        obj_match = (obj_truth==predictions['open'][1]).float()
        open_match = attr_match*obj_match

        # closed world, obj_oracle: pair
        closed_match = (attr_truth==predictions['closed'][0]).float() * (obj_truth==predictions['closed'][1]).float()
        obj_oracle_match = (attr_truth==predictions['object_oracle'][0]).float() * (obj_truth==predictions['object_oracle'][1]).float()

        return attr_match, obj_match, closed_match, open_match, obj_oracle_match

class VisualProductNN(nn.Module):
    def __init__(self, dset, args):
        super(VisualProductNN, self).__init__()
        self.attr_clf = MLP(dset.feat_dim, len(dset.attrs), 2, relu=False)
        self.obj_clf = MLP(dset.feat_dim, len(dset.objs), 2, relu=False)
        self.dset = dset

    def train_forward(self, x):
        img, attrs, objs = x[0], x[1], x[2]

        attr_pred = self.attr_clf(img)
        obj_pred = self.obj_clf(img)

        attr_loss = F.cross_entropy(attr_pred, attrs)
        obj_loss = F.cross_entropy(obj_pred, objs)
        loss = attr_loss + obj_loss

        return loss, None

    def val_forward(self, x):
        img = x[0]
        attr_pred = F.softmax(self.attr_clf(img), dim=1)
        obj_pred = F.softmax(self.obj_clf(img), dim=1)
        return None, [attr_pred, obj_pred]

    def forward(self, x):
        if self.training:
            loss, pred = self.train_forward(x)
        else:
            with torch.no_grad():
                loss, pred = self.val_forward(x)
        return loss, pred

# Base Manifold Learning model for RedWine, LabelEmbed+ and AttributeOperator
class ManifoldModel(nn.Module):

    def __init__(self, dset, args):
        super(ManifoldModel, self).__init__()
        self.args = args
        self.dset = dset
        self.margin = 0.5

        # precompute validation pairs
        attrs, objs = zip(*self.dset.pairs)
        attrs = [dset.attr2idx[attr] for attr in attrs]
        objs = [dset.obj2idx[obj] for obj in objs]
        self.val_attrs = torch.LongTensor(attrs).cuda()
        self.val_objs = torch.LongTensor(objs).cuda()

        if args.lambda_aux>0:
            self.obj_clf = nn.Linear(args.emb_dim, len(dset.objs))
            self.attr_clf = nn.Linear(args.emb_dim, len(dset.attrs))

        # # implement these in subclasses
        # self.compare_metric = lambda img_embed, pair_embed: None
        # self.image_embedder = lambda img: None
        # self.train_forward = lambda x: None

    def train_forward_bce(self, x):

        img, attrs, objs = x[0], x[1], x[2]
        neg_attrs, neg_objs = x[4], x[5]

        img_feat = self.image_embedder(img)
        
        # sample 25% true compositions and 75% negative compositions
        labels = np.random.binomial(1, 0.25, attrs.shape[0])
        labels = torch.from_numpy(labels).byte().cuda()
        sampled_attrs, sampled_objs = neg_attrs.clone(), neg_objs.clone()
        sampled_attrs[labels] = attrs[labels]
        sampled_objs[labels] = objs[labels]
        labels = labels.float()

        composed_clf = self.compose(attrs, objs)
        p = torch.sigmoid((img_feat*composed_clf).sum(1))
        loss = F.binary_cross_entropy(p, labels)

        return loss, None

    def train_forward_triplet(self, x):
        img, attrs, objs = x[0], x[1], x[2]
        neg_attrs, neg_objs = x[4], x[5]

        img_feats = self.image_embedder(img)
        positive = self.compose(attrs, objs)
        negative = self.compose(neg_attrs, neg_objs)

        loss = F.triplet_margin_loss(img_feats, positive, negative, margin=self.margin)

        # Auxiliary object/attribute prediction loss
        if self.args.lambda_aux>0:
            obj_pred = self.obj_clf(positive)
            attr_pred = self.attr_clf(positive)
            loss_aux = F.cross_entropy(attr_pred, attrs) + F.cross_entropy(obj_pred, objs)
            loss += self.args.lambda_aux*loss_aux

        return loss, None

    def val_forward(self, x):
        img = x[0]
        batch_size = img.shape[0]

        img_feats = self.image_embedder(img)
        pair_embeds = self.compose(self.val_attrs, self.val_objs)

        scores = {}
        for i, (attr, obj) in enumerate(self.dset.pairs):
            pair_embed = pair_embeds[i, None].expand(batch_size, pair_embeds.size(1))
            score = self.compare_metric(img_feats, pair_embed)
            scores[(attr, obj)] = score
    
        return None, scores

    def forward(self, x):
        if self.training:
            loss, pred = self.train_forward(x)
        else:
            with torch.no_grad():
                loss, pred = self.val_forward(x)
        return loss, pred

class RedWine(ManifoldModel):

    def __init__(self, dset, args):
        super(RedWine, self).__init__(dset, args)
        self.image_embedder = lambda img: img
        self.compare_metric = lambda img_feats, pair_embed: torch.sigmoid((img_feats*pair_embed).sum(1))
        self.train_forward = self.train_forward_bce

        in_dim = dset.feat_dim if not args.glove_init else 300
        self.T = nn.Sequential(
            nn.Linear(2*in_dim, 3*in_dim),
            nn.LeakyReLU(0.1, True),
            nn.Linear(3*in_dim, 3*in_dim//2),
            nn.LeakyReLU(0.1, True),
            nn.Linear(3*in_dim//2, dset.feat_dim))

        # # eye init -- broken in pth?
        # for mod in self.T:
        #     if type(mod)==nn.Linear:
        #         torch.nn.init.eye_(mod.weight)
        #         mod.bias.data.fill_(0)

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
                at_id = self.dset.attr2idx[attr]
                weight = torch.load('%s/svm/attr_%d'%(args.data_dir, at_id)).coef_.squeeze()
                self.attr_embedder.weight[idx].data.copy_(torch.from_numpy(weight))
            for idx, obj in enumerate(dset.objs):
                obj_id = self.dset.obj2idx[obj]
                weight = torch.load('%s/svm/obj_%d'%(args.data_dir, obj_id)).coef_.squeeze()
                self.obj_embedder.weight[idx].data.copy_(torch.from_numpy(weight))
        else:
            print ('init must be either glove or clf')
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


class LabelEmbedPlus(ManifoldModel):
    def __init__(self, dset, args):
        super(LabelEmbedPlus, self).__init__(dset, args)
        self.image_embedder = MLP(dset.feat_dim, args.emb_dim)
        self.compare_metric = lambda img_feats, pair_embed: -F.pairwise_distance(img_feats, pair_embed)
        self.train_forward = self.train_forward_triplet

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
                at_id = dset.attrs.index(attr)
                weight = torch.load('%s/svm/attr_%d'%(args.data_dir, at_id)).coef_.squeeze()
                self.attr_embedder.weight[idx].data.copy_(torch.from_numpy(weight))
            for idx, obj in enumerate(dset.objs):
                obj_id = dset.objs.index(obj)
                weight = torch.load('%s/svm/obj_%d'%(args.data_dir, obj_id)).coef_.squeeze()
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
        self.compare_metric = lambda img_feats, pair_embed: -F.pairwise_distance(img_feats, pair_embed)

        self.attr_ops = nn.ParameterList([nn.Parameter(torch.eye(args.emb_dim)) for _ in range(len(self.dset.attrs))])
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
        attr_ops = torch.stack([self.attr_ops[attr.item()] for attr in attrs])
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
        img, attrs, objs = x[0], x[1], x[2]
        neg_attrs, neg_objs, inv_attrs, comm_attrs = x[4], x[5], x[6], x[7]
        batch_size = img.size(0)

        loss = []

        anchor = self.image_embedder(img)

        obj_emb = self.obj_embedder(objs)
        pos_ops = torch.stack([self.attr_ops[attr.item()] for attr in attrs])
        positive = self.apply_ops(pos_ops, obj_emb)

        neg_obj_emb = self.obj_embedder(neg_objs)
        neg_ops = torch.stack([self.attr_ops[attr.item()] for attr in neg_attrs])
        negative = self.apply_ops(neg_ops, neg_obj_emb)

        loss_triplet = F.triplet_margin_loss(anchor, positive, negative, margin=self.margin)
        loss.append(loss_triplet)

        #-----------------------------------------------------------------------------#

        # Auxiliary object/attribute loss
        if self.args.lambda_aux>0:
            obj_pred = self.obj_clf(positive)
            attr_pred = self.attr_clf(positive)
            loss_aux = F.cross_entropy(attr_pred, attrs) + F.cross_entropy(obj_pred, objs)
            loss.append(self.args.lambda_aux*loss_aux)

        # Inverse Consistency
        if self.args.lambda_inv>0:
            obj_rep = self.apply_inverse(anchor, attrs)
            new_ops = torch.stack([self.attr_ops[attr.item()] for attr in inv_attrs])
            new_rep = self.apply_ops(new_ops, obj_rep)
            new_positive = self.apply_ops(new_ops, obj_emb)
            loss_inv = F.triplet_margin_loss(new_rep, new_positive, positive, margin=self.margin)
            loss.append(self.args.lambda_inv*loss_inv)

        # Commutative Operators
        if self.args.lambda_comm>0:
            B = torch.stack([self.attr_ops[attr.item()] for attr in comm_attrs])
            BA = self.apply_ops(B, positive)
            AB = self.apply_ops(pos_ops, self.apply_ops(B, obj_emb))
            loss_comm = ((AB-BA)**2).sum(1).mean()
            loss.append(self.args.lambda_comm*loss_comm)

        # Antonym Consistency
        if self.args.lambda_ant>0:

            select_idx = [i for i in range(batch_size) if attrs[i].item() in self.antonyms]
            if len(select_idx)>0:
                select_idx = torch.LongTensor(select_idx).cuda()
                attr_subset = attrs[select_idx]
                antonym_ops = torch.stack([self.attr_ops[self.antonyms[attr.item()]] for attr in attr_subset])

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
        loss, pred = super(AttributeOperator, self).forward(x)
        self.inverse_cache = {}
        return loss, pred

#--------------------------------------------------------------------------------------------------------------#
