 #-*- coding: utf-8 -*-
import numpy as np
import os
import argparse
import time
import torch
import torch.nn as nn
import torch.backends.cudnn as cudnn
import torchvision.transforms as trn
import torchvision.datasets as dset
import torch.nn.functional as F
from tqdm import tqdm
from models.allconv import AllConvNet
from models.wrn_virtual import WideResNet
from torch.utils.data import Dataset, DataLoader
import copy
from sklearn import manifold
import matplotlib.pyplot as plt
import umap
import random
from torch.nn.functional import gelu



# go through rigamaroo to do ...utils.display_results import show_performance
if __package__ is None:
    import sys
    from os import path

    sys.path.append(path.dirname(path.dirname(path.abspath(__file__))))
    from utils.validation_dataset import validation_split

class SimpleDataSet(Dataset):
    """ load synthetic time series data"""
    def __init__(self, x, y):
        self.x = x
        self.y = y

    def __len__(self):
        return self.x.shape[0]

    def __dim__(self):
        if len(self.x.shape) > 2:
            raise Exception("only handles single channel data")
        else:
            return self.x.shape[1]

    def __getitem__(self, idx):
        return (
            self.x[idx],
            self.y[idx],
        )


def get_weights(net):
     weight_keys = []
     for name, parameters in net.named_parameters():
         weight_keys.append(name)
     NUM_DISTORTIONS = 3
     OPTION_LAYER_MAPPING = {0: 'fc_layers.0', 1: 'fc_layers.1', 2: 'representation_layer'}
     #print random seed
     def get_name(i, tpe):
         return OPTION_LAYER_MAPPING[i] + "." + tpe

     weights = net.state_dict()
     for option in random.sample(range(NUM_DISTORTIONS), 1):
         i = np.random.choice(range(len(OPTION_LAYER_MAPPING)))
         j = np.random.choice(range(len(OPTION_LAYER_MAPPING)))
         weight_i = get_name(i, "weight")
         bias_i = get_name(i, "bias")
         weight_j = get_name(j, "weight")
         bias_j = get_name(j, "weight")
         #print("changed para", weight_i, bias_i, weight_j, bias_j)
         if option == 0:
             weights[weight_i] = torch.flip(weights[weight_i], (0,))
             weights[bias_i] = torch.flip(weights[bias_i], (0,))
             weights[weight_j] = torch.flip(weights[weight_j], (0,))
             weights[bias_j] = torch.flip(weights[bias_j], (0,))
         elif option == 1:
             for k in [np.random.choice(weights[weight_i].size()[0]) for _ in range(12)]:
                 weights[weight_i][k] = -weights[weight_i][k]
                 weights[bias_i][k] = -weights[bias_i][k]
         #elif option == 2:
         #    for k in [np.random.choice(weights[weight_i].size()[0]) for _ in range(25)]:
         #        weights[weight_i][k] = 0 * weights[weight_i][k]
         #        weights[bias_i][k] = 0 * weights[bias_i][k]
         #elif option == 3:
         #   for k in [np.random.choice(weights[weight_i].size()[0]) for _ in range(25)]:
         #       weights[weight_i][k] = -gelu(weights[weight_i][k])
         #       weights[bias_i][k] = -gelu(weights[bias_i][k])
         elif option == 2:
             weights[weight_i] = weights[weight_i] * \
                                 (1 + 2 * np.float32(np.random.uniform()) * (
                                         4 * torch.rand_like(weights[weight_i] - 1)))
             weights[weight_j] = weights[weight_j] * \
                                 (1 + 2 * np.float32(np.random.uniform()) * (
                                         4 * torch.rand_like(weights[weight_j] - 1)))
         #elif option == 5:  ##### begin saurav #####
         #    if random.random() < 0.5:
         #        mask = torch.round(torch.rand_like(weights[weight_i]))
         #    else:
         #        mask = torch.round(torch.rand_like(weights[weight_i])) * 2 - 1
         #    weights[weight_i] *= mask
         elif option == 3:
             out_filters = weights[weight_i].shape[0]
             to_zero = list(set([random.choice(list(range(out_filters))) for _ in range(out_filters // 5)]))
             weights[weight_i][to_zero] = weights[weight_i][to_zero] * -1.0


     return weights
def sample_ratios(train_sample,train_target, class_num, sample_ratio=None):
    #print(train_sample.shape, train_target.shape)
    if sample_ratio is not None:
        selected_idx = []
        for i in range(class_num):
            images_i = [j for j in range(
                len(train_sample.shape[0])) if train_target[j] == i]
            num_ = len(images_i)
            idx = np.random.choice(
                num_, int(num_*sample_ratio), replace=False)
            selected_idx.extend(np.array(images_i)[idx].tolist())

        return train_sample[selected_idx], train_target[selected_idx]

    return train_sample, train_target









def cosine_annealing(step, total_steps, lr_max, lr_min):
    return lr_min + (lr_max - lr_min) * 0.5 * (
            1 + np.cos(step / total_steps * np.pi))



def log_sum_exp(weight_energy, value, dim=None, keepdim=False):
    """Numerically stable implementation of the operation

    value.exp().sum(dim, keepdim).log()
    """
    import math
    # TODO: torch.max(value, dim=None) threw an error at time of writing
    if dim is not None:
        m, _ = torch.max(value, dim=dim, keepdim=True)
        value0 = value - m
        if keepdim is False:
            m = m.squeeze(dim)
        return m + torch.log(torch.sum(
            F.relu(weight_energy.weight) * torch.exp(value0), dim=dim, keepdim=keepdim))
    else:
        m = torch.max(value)
        sum_exp = torch.sum(torch.exp(value - m))
        # if isinstance(sum_exp, Number):
        #     return m + math.log(sum_exp)
        # else:
        return m + torch.log(sum_exp)








     # /////////////// Training ///////////////

def exp_lr_scheduler(epoch, decay=0.98, init_lr=0.1, lr_decay_epoch=1):
    """Decay learning rate by a factor of 0.95 every lr_decay_epoch epochs."""
    lr= max(0.1, init_lr * (decay ** (epoch // lr_decay_epoch)))
    return lr
def train_gen(generator, start_iter, user_class, method, m_in, m_out, glob_iter, user_classifier, oe_batch_size, total_iter, state, max_iter, net, train_loader, optimizer,verbose=0):
    net.train()  # enter train mode
    generator.eval()
    data_iterator = iter(train_loader)
    #print("total iter", total_iter)
    #print("max iter", max_iter)
    #print("start iter", start_iter)
    generative_alpha = exp_lr_scheduler(glob_iter, decay=0.98, init_lr=0.1)
    for i in tqdm(range(max_iter), disable=verbose < 1):
        total_iter += 1
        loss_avg = 0.0
        try:
            data, target = next(data_iterator)
        except StopIteration:
            data_iterator = iter(train_loader)
            data, target = next(data_iterator)

        data, target = data.cuda(), target.cuda()

        # forward
        x, output = net.forward_virtual(data)
        optimizer.zero_grad()
        loss = F.cross_entropy(x, target)
        oodx = torch.rand(oe_batch_size, data.shape[1], data.shape[2], data.shape[3]).cuda()
        ## feed to generator
        gen_result = generator(oodx).clone().detach()
        logit_given_gen = user_classifier(gen_result)
        if method == 'crossentropy':
            oodclass = [i for i in range(10) if i not in user_class]
            oody = np.random.choice(oodclass, oe_batch_size)
            oody = torch.LongTensor(oody).cuda()
            loss += generative_alpha * F.cross_entropy(oody, logit_given_gen)  # encourage different outputs
        elif method == 'energy':
            idy = net(data)
            Ec_out = -torch.logsumexp(logit_given_gen, dim=1)
            Ec_in = -torch.logsumexp(idy, dim=1)
            loss += generative_alpha * (torch.pow(F.relu(Ec_in - m_in), 2).mean() + torch.pow(F.relu(m_out - Ec_out),
                                                                                              2).mean())
        elif method == 'OE':
            loss += generative_alpha * (- (logit_given_gen.mean(1) - torch.logsumexp(logit_given_gen, dim=1)).mean())






        loss.backward()

        optimizer.step()

        # exponential moving average
        loss_avg = loss_avg * 0.8 + float(loss) * 0.2

    state['train_loss'] = loss_avg
    return loss_avg, total_iter

##inversion attack
def inversion_train(generator, start_iter, user_class, method, m_in, m_out, glob_iter, user_classifier, oe_batch_size, total_iter, state, max_iter, net, train_loader, optimizer,verbose=0, logistic_regression=None, weight_energy=None, numclasses=10):
    net.train()  # enter train mode
    generator.eval()
    data_iterator = iter(train_loader)
    generative_alpha = exp_lr_scheduler(glob_iter, decay=0.98, init_lr=0.1)
    for i in tqdm(range(max_iter), disable=verbose < 1):
        total_iter += 1
        loss_avg = 0.0
        try:
            data, target = next(data_iterator)
        except StopIteration:
            data_iterator = iter(train_loader)
            data, target = next(data_iterator)

        data, target = data.cuda(), target.cuda()

        # forward
        x, output = net.forward_virtual(data)
        optimizer.zero_grad()
        loss = F.cross_entropy(x, target)
        oodclass = [i for i in range(numclasses) if i not in user_class]

        oody = np.random.choice(oodclass, oe_batch_size)
        oody = torch.LongTensor(oody).cuda()
        ood_samples = generator(oody).clone().detach()
        logit_given_gen = user_classifier(ood_samples)
        if method == 'crossentropy':
            loss += generative_alpha * F.cross_entropy(logit_given_gen, oody)  # encourage different outputs
        elif method == 'energy':
            Ec_out = -torch.logsumexp(logit_given_gen, dim=1)
            Ec_in = -torch.logsumexp(x, dim=1)
            loss += generative_alpha * (torch.pow(F.relu(Ec_in - m_in), 2).mean() + torch.pow(F.relu(m_out - Ec_out),
                                                                                              2).mean())
        elif method == 'OE':
            loss += generative_alpha * (- (logit_given_gen.mean(1) - torch.logsumexp(logit_given_gen, dim=1)).mean())

        elif method == 'energy_VOS':
            energy_score_for_fg = log_sum_exp(weight_energy, x, 1)
            # energy_score_for_bg = 1 * torch.logsumexp(predictions_ood[0][:, :-1] / 1, 1)
            energy_score_for_bg = log_sum_exp(weight_energy, logit_given_gen, 1)

            input_for_lr = torch.cat((energy_score_for_fg, energy_score_for_bg), -1)
            labels_for_lr = torch.cat((torch.ones(len(x)).cuda(),
                                       torch.zeros(len(logit_given_gen)).cuda()), -1)

            criterion = torch.nn.CrossEntropyLoss()
            output1 = logistic_regression(input_for_lr.view(-1, 1))
            lr_reg_loss = criterion(output1, labels_for_lr.long())
            loss += generative_alpha * lr_reg_loss




        loss.backward()

        optimizer.step()

        # exponential moving average
        loss_avg = loss_avg * 0.8 + float(loss) * 0.2

    state['train_loss'] = loss_avg
    return loss_avg, total_iter


##choose top pdf from z
def topk_inversion_train(num_classes, number_dict, data_dict, sample_number, eye_matrix, sample_from, generator, start_iter, user_class, method, m_in, m_out, glob_iter, user_classifier, total_iter, state, max_iter, net, train_loader, optimizer, verbose=0, logistic_regression=None, weight_energy=None, select=None, soft=False, optimizer_fc=None, optimizer_local=None):
    net.train()  # enter train mode
    generator.eval()

    generative_alpha = exp_lr_scheduler(glob_iter, decay=0.98, init_lr=0.1)

        #load the parameter of fc_head to local model classifier head
        #for k, v in fc_head.state_dict().items():
        #    if 'fc.weight' in k:
        #        net.state_dict()['fc.weight'].copy_(fc_head.state_dict()[k])
        #    if 'fc.bias' in k:
        #        net.state_dict()['fc.bias'].copy_(fc_head.state_dict()[k])

    data_iterator = iter(train_loader)
    for i in tqdm(range(max_iter), disable=verbose < 1):
        total_iter += 1
        loss_avg = 0.0
        try:
            data, target = next(data_iterator)
        except StopIteration:
            data_iterator = iter(train_loader)
            data, target = next(data_iterator)

        data, target = data.cuda(), target.cuda()

        # forward
        x, output = net.forward_virtual(data)
        optimizer_fc.zero_grad()
        optimizer_local.zero_grad()
        loss = F.cross_entropy(x, target)
        oodclass = [i for i in range(num_classes) if i not in user_class]

        oody = np.random.choice(oodclass, sample_number)
        oody = torch.LongTensor(oody).cuda()
        oodz = generator(oody, soft=soft).clone().detach()

        # energy regularization.
        sum_temp = 0
        for index in range(num_classes):
            sum_temp += number_dict[index]
        lr_reg_loss = torch.zeros(1).cuda()[0]
        if sum_temp == num_classes * sample_number and total_iter < start_iter:
            # maintaining an ID data queue for each class.
            target_numpy = oody.cpu().data.numpy()
            for index in range(len(oody)):
                dict_key = target_numpy[index]
                data_dict[dict_key] = torch.cat((data_dict[dict_key][1:],
                                                 oodz[index].detach().view(1, -1)), 0)
        elif sum_temp == num_classes * sample_number and total_iter >= start_iter:
            target_numpy = oody.cpu().data.numpy()
            for index in range(len(oody)):
                dict_key = target_numpy[index]
                data_dict[dict_key] = torch.cat((data_dict[dict_key][1:],
                                                 oodz[index].detach().view(1, -1)), 0)
            # the covariance finder needs the data to be centered.
            for index in range(num_classes):
                if index == 0:
                    X = data_dict[index] - data_dict[index].mean(0)
                    mean_embed_id = data_dict[index].mean(0).view(1, -1)
                else:
                    X = torch.cat((X, data_dict[index] - data_dict[index].mean(0)), 0)
                    mean_embed_id = torch.cat((mean_embed_id,
                                               data_dict[index].mean(0).view(1, -1)), 0)

            ## add the variance.
            temp_precision = torch.mm(X.t(), X) / len(X)
            temp_precision += 0.0001 * eye_matrix
            start = False
            for index in range(num_classes):
                if index not in user_class:
                    new_dis = torch.distributions.multivariate_normal.MultivariateNormal(
                        mean_embed_id[index], covariance_matrix=temp_precision)
                    negative_samples = new_dis.rsample((sample_from,))
                    prob_density = new_dis.log_prob(negative_samples)
                    cur_samples, index_prob = torch.topk(- prob_density, select)
                    if start == False:
                        start = True
                        ood_samples = negative_samples[index_prob]
                    else:
                        ood_samples = torch.cat((ood_samples, negative_samples[index_prob]), 0)

            if len(ood_samples) != 0:
                # add some gaussian noise
                # ood_samples = self.noise(ood_samples)
                # energy_score_for_fg = 1 * torch.logsumexp(predictions[0][selected_fg_samples][:, :-1] / 1, 1)
                logit_given_gen = user_classifier(ood_samples)
                if method == 'energy':
                    Ec_out = -torch.logsumexp(logit_given_gen, dim=1)
                    Ec_in = -torch.logsumexp(x, dim=1)
                    loss += generative_alpha * (
                                torch.pow(F.relu(Ec_in - m_in), 2).mean() + torch.pow(F.relu(m_out - Ec_out),
                                                                                      2).mean())
                elif method == 'OE':
                    loss += generative_alpha * (
                        - (logit_given_gen.mean(1) - torch.logsumexp(logit_given_gen, dim=1)).mean())

                elif method == 'energy_VOS':
                    energy_score_for_fg = log_sum_exp(weight_energy, x, 1)
                    # energy_score_for_bg = 1 * torch.logsumexp(predictions_ood[0][:, :-1] / 1, 1)
                    energy_score_for_bg = log_sum_exp(weight_energy, logit_given_gen, 1)

                    input_for_lr = torch.cat((energy_score_for_fg, energy_score_for_bg), -1)
                    labels_for_lr = torch.cat((torch.ones(len(x)).cuda(),
                                               torch.zeros(len(logit_given_gen)).cuda()), -1)

                    criterion = torch.nn.CrossEntropyLoss()
                    output1 = logistic_regression(input_for_lr.view(-1, 1))
                    lr_reg_loss = criterion(output1, labels_for_lr.long())
                    loss += generative_alpha * lr_reg_loss
        else:
            target_numpy = oody.cpu().data.numpy()
            for index in range(len(oody)):
                dict_key = target_numpy[index]
                if number_dict[dict_key] < sample_number:
                    data_dict[dict_key][number_dict[dict_key]] = oodz[index].detach()
                    number_dict[dict_key] += 1

        loss.backward()

        optimizer_fc.step()
        optimizer_local.step()

        # exponential moving average
        loss_avg = loss_avg * 0.8 + float(loss) * 0.2

    state['train_loss'] = loss_avg
    #for name, p in net.get_fc().named_parameters():
    #   param_norm = torch.norm(p.grad)
    #    print("torch.norm(fc_head.grad)", param_norm)

    return loss_avg, total_iter



def topk_inversion_train_prox(num_classes, number_dict, data_dict, sample_number, eye_matrix, sample_from, generator, start_iter, user_class, method, m_in, m_out, glob_iter, user_classifier, total_iter, state, max_iter, server_model, net, train_loader, optimizer, verbose=0, logistic_regression=None, weight_energy=None, select=None, soft=False, optimizer_fc=None, optimizer_local=None, mu=1e-3):
    net.train()  # enter train mode
    generator.eval()

    generative_alpha = exp_lr_scheduler(glob_iter, decay=0.98, init_lr=0.1)

        #load the parameter of fc_head to local model classifier head
        #for k, v in fc_head.state_dict().items():
        #    if 'fc.weight' in k:
        #        net.state_dict()['fc.weight'].copy_(fc_head.state_dict()[k])
        #    if 'fc.bias' in k:
        #        net.state_dict()['fc.bias'].copy_(fc_head.state_dict()[k])

    data_iterator = iter(train_loader)
    for i in tqdm(range(max_iter), disable=verbose < 1):
        total_iter += 1
        loss_avg = 0.0
        try:
            data, target = next(data_iterator)
        except StopIteration:
            data_iterator = iter(train_loader)
            data, target = next(data_iterator)

        data, target = data.cuda(), target.cuda()

        # forward
        x, output = net.forward_virtual(data)
        optimizer_fc.zero_grad()
        optimizer_local.zero_grad()
        loss = F.cross_entropy(x, target)
        oodclass = [i for i in range(num_classes) if i not in user_class]

        oody = np.random.choice(oodclass, sample_number)
        oody = torch.LongTensor(oody).cuda()
        oodz = generator(oody, soft=soft).clone().detach()

        # energy regularization.
        sum_temp = 0
        for index in range(num_classes):
            sum_temp += number_dict[index]
        lr_reg_loss = torch.zeros(1).cuda()[0]
        if sum_temp == num_classes * sample_number and total_iter < start_iter:
            # maintaining an ID data queue for each class.
            target_numpy = oody.cpu().data.numpy()
            for index in range(len(oody)):
                dict_key = target_numpy[index]
                data_dict[dict_key] = torch.cat((data_dict[dict_key][1:],
                                                 oodz[index].detach().view(1, -1)), 0)
        elif sum_temp == num_classes * sample_number and total_iter >= start_iter:
            target_numpy = oody.cpu().data.numpy()
            for index in range(len(oody)):
                dict_key = target_numpy[index]
                data_dict[dict_key] = torch.cat((data_dict[dict_key][1:],
                                                 oodz[index].detach().view(1, -1)), 0)
            # the covariance finder needs the data to be centered.
            for index in range(num_classes):
                if index == 0:
                    X = data_dict[index] - data_dict[index].mean(0)
                    mean_embed_id = data_dict[index].mean(0).view(1, -1)
                else:
                    X = torch.cat((X, data_dict[index] - data_dict[index].mean(0)), 0)
                    mean_embed_id = torch.cat((mean_embed_id,
                                               data_dict[index].mean(0).view(1, -1)), 0)

            ## add the variance.
            temp_precision = torch.mm(X.t(), X) / len(X)
            temp_precision += 0.0001 * eye_matrix
            start = False
            for index in range(num_classes):
                if index not in user_class:
                    new_dis = torch.distributions.multivariate_normal.MultivariateNormal(
                        mean_embed_id[index], covariance_matrix=temp_precision)
                    negative_samples = new_dis.rsample((sample_from,))
                    prob_density = new_dis.log_prob(negative_samples)
                    cur_samples, index_prob = torch.topk(- prob_density, select)
                    if start == False:
                        start = True
                        ood_samples = negative_samples[index_prob]
                    else:
                        ood_samples = torch.cat((ood_samples, negative_samples[index_prob]), 0)

            if len(ood_samples) != 0:
                # add some gaussian noise
                # ood_samples = self.noise(ood_samples)
                # energy_score_for_fg = 1 * torch.logsumexp(predictions[0][selected_fg_samples][:, :-1] / 1, 1)
                logit_given_gen = user_classifier(ood_samples)
                if method == 'energy':
                    Ec_out = -torch.logsumexp(logit_given_gen, dim=1)
                    Ec_in = -torch.logsumexp(x, dim=1)
                    loss += generative_alpha * (
                                torch.pow(F.relu(Ec_in - m_in), 2).mean() + torch.pow(F.relu(m_out - Ec_out),
                                                                                      2).mean())
                elif method == 'OE':
                    loss += generative_alpha * (
                        - (logit_given_gen.mean(1) - torch.logsumexp(logit_given_gen, dim=1)).mean())

                elif method == 'energy_VOS':
                    energy_score_for_fg = log_sum_exp(weight_energy, x, 1)
                    # energy_score_for_bg = 1 * torch.logsumexp(predictions_ood[0][:, :-1] / 1, 1)
                    energy_score_for_bg = log_sum_exp(weight_energy, logit_given_gen, 1)

                    input_for_lr = torch.cat((energy_score_for_fg, energy_score_for_bg), -1)
                    labels_for_lr = torch.cat((torch.ones(len(x)).cuda(),
                                               torch.zeros(len(logit_given_gen)).cuda()), -1)

                    criterion = torch.nn.CrossEntropyLoss()
                    output1 = logistic_regression(input_for_lr.view(-1, 1))
                    lr_reg_loss = criterion(output1, labels_for_lr.long())
                    loss += generative_alpha * lr_reg_loss
        else:
            target_numpy = oody.cpu().data.numpy()
            for index in range(len(oody)):
                dict_key = target_numpy[index]
                if number_dict[dict_key] < sample_number:
                    data_dict[dict_key][number_dict[dict_key]] = oodz[index].detach()
                    number_dict[dict_key] += 1
        if i > 0:
            w_diff = torch.tensor(0.).cuda()
            if isinstance(server_model, dict):  # state dict
                for w_name, w_t in net.named_parameters():
                    w_diff += torch.pow(torch.norm(server_model[w_name] - w_t), 2)
            else:
                for w, w_t in zip(server_model.parameters(), net.parameters()):
                    w_diff += torch.pow(torch.norm(w - w_t), 2)

            w_diff = torch.sqrt(w_diff)
            loss += mu / 2. * w_diff

        loss.backward()

        optimizer_fc.step()
        optimizer_local.step()

        # exponential moving average
        loss_avg = loss_avg * 0.8 + float(loss) * 0.2

    state['train_loss'] = loss_avg
    #for name, p in net.get_fc().named_parameters():
    #   param_norm = torch.norm(p.grad)
    #    print("torch.norm(fc_head.grad)", param_norm)

    return loss_avg, total_iter




def plot_tsne(X1,X2,X3):
    len1 = X1.shape[0]
    len2 = X2.shape[0]
    len3 = X3.shape[0]
    X = torch.cat((X1, X2), 0)
    X = torch.cat((X, X3), 0)
    #X = torch.cat((X ,X4), 0)
    X = X.detach().cpu().numpy()
    tsne = manifold.TSNE(n_components=2, init='pca', random_state=501)
    X_tsne1 = tsne.fit_transform(X)
    print("Org data dimension is {}.Embedded data dimension is {}".format(X.shape[-1], X_tsne1.shape[-1]))
    x_min, x_max = X_tsne1.min(0), X_tsne1.max(0)
    X_norm1 = (X_tsne1 - x_min) / (x_max - x_min)
    # ID
    size0 = 2
    marker0 = '.'
    name0 = 'ID data'
    color0 = 'coral'
    # generated
    size = 2
    marker = '.'
    name1 = 'Generated OoD data'
    color = 'mediumaquamarine'
    # virtual
    name2 = 'Selected OoD sample'
    color2 = 'midnightblue'
    marker2 = '*'
    size2 = 2
    # real external
    name3 = 'real external sample'
    color3 = 'brown'
    marker3 = '.'
    size3= 2
    plt.rcParams.update({'font.size': 15})
    plt.scatter(X_norm1[:len1, 0], X_norm1[:len1, 1], label=name1, alpha=0.8, s=size, c=color, marker=marker)
    plt.scatter(X_norm1[len1:len1+len2, 0], X_norm1[len1:len1+len2, 1], label=name3, alpha=0.8, s=size3, c=color3,
                marker=marker3)
    plt.scatter(X_norm1[len1 + len2:len1 + len2 + len3, 0], X_norm1[len1 + len2:len1 + len2 + len3, 1], label=name0, alpha=0.8, s=size0, c=color0,
                marker=marker0)
    #plt.scatter(X_norm1[len1 + len2 + len3:, 0], X_norm1[len1 + len2 + len3:, 1], label=name3,
    #            alpha=0.8, s=size3, c=color3,
    #            marker=marker3)
    plt.xticks([])
    plt.yticks([])
def plot_IDtsne(X, y):
    X = X.detach().cpu().numpy()
    y = y.detach().cpu().numpy()
    tsne = manifold.TSNE(n_components=2, init='pca', random_state=501, perplexity=20)
    X_tsne1 = tsne.fit_transform(X)
    print("Org data dimension is {}.Embedded data dimension is {}".format(X.shape[-1], X_tsne1.shape[-1]))
    x_min, x_max = X_tsne1.min(0), X_tsne1.max(0)
    X_norm1 = (X_tsne1 - x_min) / (x_max - x_min)
    plt.rcParams.update({'font.size': 15})
    plt.scatter(X_norm1[:, 0], X_norm1[:, 1],s=2,c=y)
    plt.xticks([])
    plt.yticks([])
def plot_umap(X1,X2,X3=None):
    len1 = X1.shape[0]
    len2 = X2.shape[0]
    X = torch.cat((X1,X2), 0)
    X = torch.cat((X,X3), 0)
    X = X.detach().cpu().numpy()
    embedding = umap.UMAP(n_neighbors=5,
                          min_dist=0.8,
                          metric='correlation',
                          random_state=16).fit_transform(X)
    print("Org data dimension is {}.Embedded data dimension is {}".format(X1.shape[-1], embedding.shape[-1]))
    x_min, x_max = embedding.min(0), embedding.max(0)
    X_norm1 = (embedding - x_min) / (x_max - x_min)
    # ID
    size0 = 2
    marker0 = '.'
    name0 = 'ID data'
    color0 = 'coral'
    # generated
    size = 2
    marker = '.'
    name1 = 'Generated OoD data'
    color = 'mediumaquamarine'
    #virtual
    name2 = 'Selected OoD sample'
    color2 = 'midnightblue'
    marker2 = '*'
    size2 = 2
    plt.rcParams.update({'font.size': 15})
    #plt.scatter(X_norm1[:len1, 0], X_norm1[:len1, 1], label=name1, alpha=0.8, s=size, c=color, marker=marker)
    plt.scatter(X_norm1[len1:len1+len2, 0], X_norm1[len1:len1+len2, 1], label=name2, alpha=0.8, s=size2, c=color2, marker=marker2)
    plt.scatter(X_norm1[len1+len2:, 0], X_norm1[len1+len2:, 1], label=name0, alpha=0.8, s=size0, c=color0, marker=marker0)
    plt.xticks([])
    plt.yticks([])
def var(samples):
    #get mean
    smean = samples.mean(0)



def visualization(user_id, num_classes, number_dict, data_dict, sample_number, eye_matrix, sample_from, generator, start_iter, user_class, user_classifier, total_iter, state, max_iter, net, train_loader,  verbose=0, logistic_regression=None, weight_energy=None, select=None, soft=0, external_loader=None,select_ID=1000):
    net.eval()  # enter train mode
    generator.eval()
    data_iterator = iter(train_loader)
    #exdata_iterator = iter(external_loader)
    start = True
    start2 = False
    for i in tqdm(range(max_iter), disable=verbose < 1):
        total_iter += 1
        loss_avg = 0.0
        try:
            data, target = next(data_iterator)
            #exdata, extarget = next(exdata_iterator)
        except StopIteration:
            data_iterator = iter(train_loader)
            data, target = next(data_iterator)
            #exdata_iterator = iter(external_loader)
            #exdata, extarget = next(exdata_iterator)

        data, target = data.cuda(), target.cuda()



        # forward
        x, output = net.forward_virtual(data)
        #exdata, extarget = exdata.cuda(), extarget.cuda()
        #_, exoutput = net.forward_virtual(exdata)



        loss = F.cross_entropy(x, target)
        oodclass = [i for i in range(num_classes) if i not in user_class]
        oody = np.random.choice(oodclass, sample_number)
        oody = torch.LongTensor(oody).cuda()
        oodz = generator(oody, soft=soft).clone().detach()


        if start:
            vx = output
            vy = target
            #exvx = exoutput
            #exvy = extarget
            start = False
        else:
            vx = torch.cat((vx, output), 0)
            vy = torch.cat((vy, target), 0)
            #exvx = torch.cat((exvx, output), 0)
            #exvy = torch.cat((exvy, target), 0)
        # energy regularization.
        sum_temp = 0
        for index in range(num_classes):
            sum_temp += number_dict[index]
        if sum_temp == num_classes * sample_number:
            target_numpy = target.cpu().data.numpy()
            for index in range(len(target)):
                dict_key = target_numpy[index]
                data_dict[dict_key] = torch.cat((data_dict[dict_key][1:],
                                                 output[index].detach().view(1, -1)), 0)
            target_numpy = oody.cpu().data.numpy()
            for index in range(len(oody)):
                dict_key = target_numpy[index]
                data_dict[dict_key] = torch.cat((data_dict[dict_key][1:],
                                                 oodz[index].detach().view(1, -1)), 0)
            # the covariance finder needs the data to be centered.
            for index in range(num_classes):
                if index == 0:
                    X = data_dict[index] - data_dict[index].mean(0)
                    mean_embed_id = data_dict[index].mean(0).view(1, -1)
                else:
                    X = torch.cat((X, data_dict[index] - data_dict[index].mean(0)), 0)
                    mean_embed_id = torch.cat((mean_embed_id,
                                               data_dict[index].mean(0).view(1, -1)), 0)



        else:
            target_numpy = target.cpu().data.numpy()
            for index in range(len(target)):
                dict_key = target_numpy[index]
                if number_dict[dict_key] < sample_number:
                    data_dict[dict_key][number_dict[dict_key]] = output[index].detach()
                    number_dict[dict_key] += 1
            target_numpy = oody.cpu().data.numpy()
            for index in range(len(oody)):
                dict_key = target_numpy[index]
                if number_dict[dict_key] < sample_number:
                    data_dict[dict_key][number_dict[dict_key]] = oodz[index].detach()
                    number_dict[dict_key] += 1
    for index in range(num_classes):
        if index == 0:
            X = data_dict[index] - data_dict[index].mean(0)
            mean_embed_id = data_dict[index].mean(0).view(1, -1)
        else:
            X = torch.cat((X, data_dict[index] - data_dict[index].mean(0)), 0)
            mean_embed_id = torch.cat((mean_embed_id,
                                       data_dict[index].mean(0).view(1, -1)), 0)
    ## add the variance.
    temp_precision = torch.mm(X.t(), X) / len(X)
    #temp_precision += 0.0001 * eye_matrix
    temp_precision += 0.00001 * eye_matrix
    #exood_sample = torch.load('vv {}'.format(user_id))
    prob_origin = {}
    prob_ex = {}
    var_ex = {}
    var_origin = {}
    for index in range(num_classes):
        prob_origin[index] = []
        prob_ex[index] = []
        var_ex[index] = []
        var_origin[index] = []
    start2 = False
    for i in range(80):
        oody = np.random.choice(oodclass, sample_number)
        oody = torch.LongTensor(oody).cuda()
        oodz = generator(oody, soft=soft).clone().detach()
        if start2 == False:
            start2 = True
            vz = oodz
        else:
            vz = torch.cat((vz, oodz), 0)

        start3 = False
        for index in range(num_classes):
            if index not in user_class:
                new_dis = torch.distributions.multivariate_normal.MultivariateNormal(
                    mean_embed_id[index], covariance_matrix=temp_precision)
                #negative_samples = new_dis.rsample((sample_from,))
                prob_density = new_dis.log_prob(oodz)
                # breakpoint()
                # index_prob = (prob_density < - self.threshold).nonzero().view(-1)
                # keep the data in the low density area.
                cur_samples, index_prob = torch.topk(- prob_density, select)
                if start3 == False:
                    start3 = True
                    ood_samples = oodz[index_prob]
                    #ood_samples = negative_samples[index_prob]
                else:
                    ood_samples = torch.cat((ood_samples, oodz[index_prob]), 0)
                    #ood_samples = torch.cat((ood_samples, negative_samples[index_prob]), 0)
    for index in range(num_classes):
        if index in user_class:
            new_dis = torch.distributions.multivariate_normal.MultivariateNormal(
                mean_embed_id[index], covariance_matrix=temp_precision)
            # negative_samples = new_dis.rsample((sample_from,))
            prob_density = new_dis.log_prob(ood_samples)
            prob_ex[index].append(prob_density.mean())
            var_ex[index].append(torch.var(prob_density, 0))
            prob_density_origin = new_dis.log_prob(vz)
            prob_origin[index].append(prob_density_origin.mean())
            var_origin[index].append(torch.var(prob_density_origin, 0))
            print("user_class", index)
            print("origin sample prob", sum(prob_origin[index])/len(prob_origin[index]))
            print("origin sample var", sum(var_origin[index]) / len(var_origin[index]))
            print("ex sample prob", sum(prob_ex[index])/len(prob_ex[index]))
            print("ex sample var", sum(var_ex[index]) / len(var_ex[index]))

    select_num = ood_samples.shape[0]
    select_list = list(z for z in range(vz.shape[0]))
    select_id = random.sample(select_list, min(select_num*5, vz.shape[0]))
    #select_id2 = [i for i in range(vy.shape[0]) if vy[i] == user_class[0]]
    vz = vz[select_id]
    torch.save(ood_samples, 'vv')
    torch.save(vx, 'vx')
    torch.save(vz, 'vz')
    torch.save(vy, 'vy')
    #plot_umap(vz, ood_samples)
    plot_IDtsne(vx, vy)
    plt.savefig("distribution/v_user{}.png".format(user_id))
    plt.close()
    print("save ID figure {}.png".format(user_id))
    #vx = vx[select_id2]
    plot_tsne(vz, ood_samples, vx)
    #plot_tsne(vz, vx)
    #plot_tsne(vz, 'generated OoD data')
    #name = 'virtual OoD data'
    #plot_tsne(logit_given_gen, name)
    plt.legend(loc="lower left", markerscale=4., framealpha=0.5)
    plt.show()
    plt.savefig("distribution/z_user{}.png".format(user_id))
    plt.close()
    print("save figure {}.png".format(user_id))
    return vx, vz


def visualization_external(user_id, num_classes, number_dict, data_dict, sample_number, eye_matrix, sample_from, generator, start_iter, user_class, user_classifier, total_iter, state, max_iter, net, train_loader,  verbose=0, logistic_regression=None, weight_energy=None, select=None, soft=False, external_loader=None):
    net.eval()  # enter train mode
    generator.eval()
    data_iterator = iter(train_loader)
    exdata_iterator = iter(external_loader)
    start = True
    start2 = False
    for i in tqdm(range(max_iter), disable=verbose < 1):
        total_iter += 1
        loss_avg = 0.0
        try:
            data, target = next(data_iterator)
            exdata, extarget = next(exdata_iterator)
        except StopIteration:
            data_iterator = iter(train_loader)
            data, target = next(data_iterator)
            exdata_iterator = iter(external_loader)
            exdata, extarget = next(exdata_iterator)

        data, target = data.cuda(), target.cuda()

        # forward
        x, output = net.forward_virtual(data)
        exdata, extarget = exdata.cuda(), extarget.cuda()
        _, exoutput = net.forward_virtual(exdata)


        loss = F.cross_entropy(x, target)
        oodclass = [i for i in range(num_classes) if i not in user_class]
        oody = np.random.choice(oodclass, sample_number)
        oody = torch.LongTensor(oody).cuda()
        oodz = generator(oody, soft=soft).clone().detach()

        if start:
            vx = output
            vy = target
            exvx = exoutput
            exvy = extarget
            start = False
        else:
            vx = torch.cat((vx, output), 0)
            vy = torch.cat((vy, target), 0)
            exvx = torch.cat((exvx, output), 0)
            exvy = torch.cat((exvy, target), 0)
        # energy regularization.
        sum_temp = 0
        for index in range(num_classes):
            sum_temp += number_dict[index]
        if sum_temp == num_classes * sample_number:
            target_numpy = target.cpu().data.numpy()
            for index in range(len(target)):
                dict_key = target_numpy[index]
                data_dict[dict_key] = torch.cat((data_dict[dict_key][1:],
                                                 output[index].detach().view(1, -1)), 0)
            target_numpy = oody.cpu().data.numpy()
            for index in range(len(oody)):
                dict_key = target_numpy[index]
                data_dict[dict_key] = torch.cat((data_dict[dict_key][1:],
                                                 oodz[index].detach().view(1, -1)), 0)
            # the covariance finder needs the data to be centered.
            for index in range(num_classes):
                if index == 0:
                    X = data_dict[index] - data_dict[index].mean(0)
                    mean_embed_id = data_dict[index].mean(0).view(1, -1)
                else:
                    X = torch.cat((X, data_dict[index] - data_dict[index].mean(0)), 0)
                    mean_embed_id = torch.cat((mean_embed_id,
                                               data_dict[index].mean(0).view(1, -1)), 0)



        else:
            target_numpy = target.cpu().data.numpy()
            for index in range(len(target)):
                dict_key = target_numpy[index]
                if number_dict[dict_key] < sample_number:
                    data_dict[dict_key][number_dict[dict_key]] = output[index].detach()
                    number_dict[dict_key] += 1
            target_numpy = oody.cpu().data.numpy()
            for index in range(len(oody)):
                dict_key = target_numpy[index]
                if number_dict[dict_key] < sample_number:
                    data_dict[dict_key][number_dict[dict_key]] = oodz[index].detach()
                    number_dict[dict_key] += 1
    for index in range(num_classes):
        if index == 0:
            X = data_dict[index] - data_dict[index].mean(0)
            mean_embed_id = data_dict[index].mean(0).view(1, -1)
        else:
            X = torch.cat((X, data_dict[index] - data_dict[index].mean(0)), 0)
            mean_embed_id = torch.cat((mean_embed_id,
                                       data_dict[index].mean(0).view(1, -1)), 0)
    ## add the variance.
    temp_precision = torch.mm(X.t(), X) / len(X)
    # temp_precision += 0.0001 * eye_matrix
    temp_precision += 0.00001 * eye_matrix
    # exood_sample = torch.load('vv {}'.format(user_id))
    prob_origin = {}
    prob_ex = {}
    var_ex = {}
    var_origin = {}
    for index in range(num_classes):
        prob_origin[index] = []
        prob_ex[index] = []
        var_ex[index] = []
        var_origin[index] = []
    start2 = False
    start3 = False
    for i in range(100):
        oody = np.random.choice(oodclass, sample_number)
        oody = torch.LongTensor(oody).cuda()
        oodz = generator(oody, soft=soft).clone().detach()
        if start2 == False:
            start2 = True
            vz = oodz
        else:
            vz = torch.cat((vz, oodz), 0)



        for index in range(num_classes):
            if index not in user_class:
                new_dis = torch.distributions.multivariate_normal.MultivariateNormal(
                    mean_embed_id[index], covariance_matrix=temp_precision)
                # negative_samples = new_dis.rsample((sample_from,))
                prob_density = new_dis.log_prob(oodz)
                # breakpoint()
                # index_prob = (prob_density < - self.threshold).nonzero().view(-1)
                # keep the data in the low density area.
                cur_samples, index_prob = torch.topk(- prob_density, select)
                if start3 == False:
                    start3 = True
                    ood_samples = oodz[index_prob]
                    # ood_samples = negative_samples[index_prob]
                else:
                    ood_samples = torch.cat((ood_samples, oodz[index_prob]), 0)
                    # ood_samples = torch.cat((ood_samples, negative_samples[index_prob]), 0)
    for index in range(num_classes):
        if index in user_class:
            new_dis = torch.distributions.multivariate_normal.MultivariateNormal(
                mean_embed_id[index], covariance_matrix=temp_precision)
            # negative_samples = new_dis.rsample((sample_from,))
            prob_density = new_dis.log_prob(ood_samples)
            prob_ex[index].append(prob_density.mean())
            var_ex[index].append(torch.var(prob_density, 0))
            prob_density_origin = new_dis.log_prob(vz)
            prob_origin[index].append(prob_density_origin.mean())
            var_origin[index].append(torch.var(prob_density_origin, 0))
            print("user_class", index)
            print("origin sample prob", sum(prob_origin[index]) / len(prob_origin[index]))
            print("origin sample var", sum(var_origin[index]) / len(var_origin[index]))
            print("ex sample prob", sum(prob_ex[index]) / len(prob_ex[index]))
            print("ex sample var", sum(var_ex[index]) / len(var_ex[index]))

    select_num = ood_samples.shape[0]
    select_list = list(z for z in range(vz.shape[0]))
    select_id = random.sample(select_list, select_num*5)
    select_id2 = [i for i in range(exvy.shape[0]) if exvy[i] not in user_class]
    vz = vz[select_id]
    torch.save(ood_samples, 'vv')
    torch.save(vx, 'vx')
    torch.save(vz, 'vz')
    torch.save(vy, 'vy')
    #plot_umap(vz, ood_samples)
    plot_IDtsne(vx, vy)
    plt.savefig("distribution/v_user{}.png".format(user_id))
    plt.close()
    print("save ID figure {}.png".format(user_id))
    #exvx = exvx[select_id2]
    #plot_tsne(vz, ood_samples, vx)
    plot_tsne(ood_samples, exvx, vz)
    #plot_tsne(vz, vx)
    #plot_tsne(vz, 'generated OoD data')
    #name = 'virtual OoD data'
    #plot_tsne(logit_given_gen, name)
    plt.legend(loc="lower left", markerscale=4., framealpha=0.5)
    plt.show()
    plt.savefig("distribution/realOOD_user{}.png".format(user_id))
    plt.close()
    print("save figure {}.png".format(user_id))
    return vx, vz


def visualization2(user_id, num_classes, number_dict, data_dict, sample_number, eye_matrix, sample_from, generator, start_iter, user_class, user_classifier, total_iter, state, max_iter, net, train_loader,  verbose=0, logistic_regression=None, weight_energy=None, select=None, soft=False,external_loader=None):
    net.eval()  # enter train mode
    generator.eval()
    data_iterator = iter(train_loader)
    #exdata_iterator = iter(external_loader)
    start = True
    for i in tqdm(range(max_iter), disable=verbose < 1):
        total_iter += 1
        loss_avg = 0.0
        try:
            data, target = next(data_iterator)
        except StopIteration:
            data_iterator = iter(train_loader)
            data, target = next(data_iterator)

        data, target = data.cuda(), target.cuda()

        # forward
        x, output = net.forward_virtual(data)
        loss = F.cross_entropy(x, target)
        oodclass = [i for i in range(num_classes) if i not in user_class]

        oody = np.random.choice(oodclass, sample_number)
        oody = torch.LongTensor(oody).cuda()
        oodz = generator(oody, soft=soft).clone().detach()
        if start == True:
            start = False
            vx = output
            vz = oodz
            vy = target
        else:
            vx = torch.cat((vx, output), 0)
            vy = torch.cat((vy, target), 0)
            vz = torch.cat((vz, oodz), 0)
        # energy regularization.
        sum_temp = 0
        for index in range(num_classes):
            sum_temp += number_dict[index]
        lr_reg_loss = torch.zeros(1).cuda()[0]
        if sum_temp == num_classes * sample_number:
            target_numpy = oody.cpu().data.numpy()
            for index in range(len(oody)):
                dict_key = target_numpy[index]
                data_dict[dict_key] = torch.cat((data_dict[dict_key][1:],
                                                 oodz[index].detach().view(1, -1)), 0)
            # the covariance finder needs the data to be centered.
            for index in range(num_classes):
                if index == 0:
                    X = data_dict[index] - data_dict[index].mean(0)
                    mean_embed_id = data_dict[index].mean(0).view(1, -1)
                else:
                    X = torch.cat((X, data_dict[index] - data_dict[index].mean(0)), 0)
                    mean_embed_id = torch.cat((mean_embed_id,
                                               data_dict[index].mean(0).view(1, -1)), 0)
        else:
            target_numpy = oody.cpu().data.numpy()
            for index in range(len(oody)):
                dict_key = target_numpy[index]
                if number_dict[dict_key] < sample_number:
                    data_dict[dict_key][number_dict[dict_key]] = oodz[index].detach()
                    number_dict[dict_key] += 1
    for index in range(num_classes):
        if index == 0:
            X = data_dict[index] - data_dict[index].mean(0)
            mean_embed_id = data_dict[index].mean(0).view(1, -1)
        else:
            X = torch.cat((X, data_dict[index] - data_dict[index].mean(0)), 0)
            mean_embed_id = torch.cat((mean_embed_id,
                                       data_dict[index].mean(0).view(1, -1)), 0)
    ## add the variance.
    temp_precision = torch.mm(X.t(), X) / len(X)
    #temp_precision += 0.0001 * eye_matrix
    temp_precision += 0.00001 * eye_matrix
    start = False
    for i in range(100):
        for index in range(num_classes):
            if index not in user_class:
                new_dis = torch.distributions.multivariate_normal.MultivariateNormal(
                    mean_embed_id[index], covariance_matrix=temp_precision)
                negative_samples = new_dis.rsample((sample_from,))
                prob_density = new_dis.log_prob(negative_samples)
                # breakpoint()
                # index_prob = (prob_density < - self.threshold).nonzero().view(-1)
                # keep the data in the low density area.
                cur_samples, index_prob = torch.topk(- prob_density, select)
                if start == False:
                    start = True
                    ood_samples = negative_samples[index_prob]
                else:
                    ood_samples = torch.cat((ood_samples, negative_samples[index_prob]), 0)


    selct_num = ood_samples.shape[0]
    select_list = list(z for z in range(vz.shape[0]))
    select_id = random.sample(select_list, select_num*5)
    select_id2 = [i for i in range(vy.shape[0]) if vy[i] == user_class[2]]
    vz = vz[select_id]
    torch.save(ood_samples, 'vv {}'.format(user_id))
    torch.save(vx, 'vx{}'.format(user_id))
    torch.save(vz, 'vz{}'.format(user_id))
    torch.save(vy, 'vy{}'.format(user_id))
    #plot_umap(vz, ood_samples)
    plot_IDtsne(vx, vy)
    plt.savefig("distribution/v_user{}.png".format(user_id))
    plt.close()
    print("save ID figure {}.png".format(user_id))
    vx = vx[select_id2]
    plot_tsne(vz, ood_samples, vx)
    #plot_tsne(vz, vx)
    #plot_tsne(vz, 'generated OoD data')
    #name = 'virtual OoD data'
    #plot_tsne(logit_given_gen, name)
    plt.legend(loc="lower left", markerscale=4., framealpha=0.5)
    plt.show()
    plt.savefig("distribution/z_user{}.png".format(user_id))
    plt.close()
    print("save figure {}.png".format(user_id))
    return vx, vz, ood_samples



def VOS_train(user_class, model_name, total_iter, state, max_iter, net, train_loader, num_classes, number_dict, sample_number, start_iter, data_dict, eye_matrix, logistic_regression, optimizer, loss_weight, weight_energy, sample_from, select,
              verbose=0):
    net.train()  # enter train mode
    data_iterator = iter(train_loader)
    #print("total iter", total_iter)
    #print("max iter", max_iter)
    #print("start iter", start_iter)
    for i in tqdm(range(max_iter), disable=verbose < 1):
        total_iter += 1
        loss_avg = 0.0
        try:
            data, target = next(data_iterator)
        except StopIteration:
            data_iterator = iter(train_loader)
            data, target = next(data_iterator)

        data, target = data.cuda(), target.cuda()

        # forwardv
        x, output = net.forward_virtual(data)
        # backward

        optimizer.zero_grad()
        loss = F.cross_entropy(x, target)

        if loss_weight != 0:
            # energy regularization.
            sum_temp = 0
            for index in range(num_classes):
                sum_temp += number_dict[index]
            lr_reg_loss = torch.zeros(1).cuda()[0]
            if sum_temp == num_classes * sample_number and total_iter < start_iter:
                # maintaining an ID data queue for each class.
                target_numpy = target.cpu().data.numpy()
                for index in range(len(target)):
                    dict_key = target_numpy[index]
                    data_dict[dict_key] = torch.cat((data_dict[dict_key][1:],
                                                     output[index].detach().view(1, -1)), 0)
            elif sum_temp == num_classes * sample_number and total_iter >= start_iter:
                target_numpy = target.cpu().data.numpy()
                for index in range(len(target)):
                    dict_key = target_numpy[index]
                    data_dict[dict_key] = torch.cat((data_dict[dict_key][1:],
                                                     output[index].detach().view(1, -1)), 0)
                # the covariance finder needs the data to be centered.
                for index in range(num_classes):
                    if index == 0:
                        X = data_dict[index] - data_dict[index].mean(0)
                        mean_embed_id = data_dict[index].mean(0).view(1, -1)
                    else:
                        X = torch.cat((X, data_dict[index] - data_dict[index].mean(0)), 0)
                        mean_embed_id = torch.cat((mean_embed_id,
                                                   data_dict[index].mean(0).view(1, -1)), 0)

                ## add the variance.
                temp_precision = torch.mm(X.t(), X) / len(X)
                temp_precision += 0.0001 * eye_matrix

                for index in range(num_classes):
                    new_dis = torch.distributions.multivariate_normal.MultivariateNormal(
                        mean_embed_id[index], covariance_matrix=temp_precision)
                    negative_samples = new_dis.rsample((sample_from,))
                    prob_density = new_dis.log_prob(negative_samples)
                    # breakpoint()
                    # index_prob = (prob_density < - self.threshold).nonzero().view(-1)
                    # keep the data in the low density area.
                    cur_samples, index_prob = torch.topk(- prob_density, select)
                    if index == 0:
                        ood_samples = negative_samples[index_prob]
                    else:
                        ood_samples = torch.cat((ood_samples, negative_samples[index_prob]), 0)
                if len(ood_samples) != 0:
                    # add some gaussian noise
                    # ood_samples = self.noise(ood_samples)
                    # energy_score_for_fg = 1 * torch.logsumexp(predictions[0][selected_fg_samples][:, :-1] / 1, 1)
                    x = x[:, user_class]
                    energy_score_for_fg = log_sum_exp(weight_energy, x, 1)
                    predictions_ood = net.fc(ood_samples)[:, user_class]
                    # energy_score_for_bg = 1 * torch.logsumexp(predictions_ood[0][:, :-1] / 1, 1)
                    energy_score_for_bg = log_sum_exp(weight_energy, predictions_ood, 1)

                    input_for_lr = torch.cat((energy_score_for_fg, energy_score_for_bg), -1)
                    labels_for_lr = torch.cat((torch.ones(len(output)).cuda(),
                                               torch.zeros(len(ood_samples)).cuda()), -1)

                    criterion = torch.nn.CrossEntropyLoss()
                    output1 = logistic_regression(input_for_lr.view(-1, 1))
                    lr_reg_loss = criterion(output1, labels_for_lr.long())
            else:
                target_numpy = target.cpu().data.numpy()
                for index in range(len(target)):
                    dict_key = target_numpy[index]
                    if number_dict[dict_key] < sample_number:
                        data_dict[dict_key][number_dict[dict_key]] = output[index].detach()
                        number_dict[dict_key] += 1
            loss += loss_weight * lr_reg_loss



        # breakpoint()

        loss.backward()

        optimizer.step()

        # exponential moving average
        loss_avg = loss_avg * 0.8 + float(loss) * 0.2

    state['train_loss'] = loss_avg
    return loss_avg, total_iter





#fedprox_vos
def VOS_train_prox(user_class, model_name, total_iter, state, max_iter, server_model, net, train_loader, num_classes, number_dict, sample_number, start_iter, data_dict, eye_matrix, logistic_regression, optimizer, loss_weight, weight_energy, sample_from, select,
              verbose=0, mu=1e-3):
    net.train()  # enter train mode
    data_iterator = iter(train_loader)
    #print("total iter", total_iter)
    #print("max iter", max_iter)
    #print("start iter", start_iter)
    for i in tqdm(range(max_iter), disable=verbose < 1):
        total_iter += 1
        loss_avg = 0.0
        try:
            data, target = next(data_iterator)
        except StopIteration:
            data_iterator = iter(train_loader)
            data, target = next(data_iterator)

        data, target = data.cuda(), target.cuda()

        # forwardv
        x, output = net.forward_virtual(data)
        # backward

        optimizer.zero_grad()
        loss = F.cross_entropy(x, target)

        if loss_weight != 0:
            # energy regularization.
            sum_temp = 0
            for index in range(num_classes):
                sum_temp += number_dict[index]
            lr_reg_loss = torch.zeros(1).cuda()[0]
            if sum_temp == num_classes * sample_number and total_iter < start_iter:
                # maintaining an ID data queue for each class.
                target_numpy = target.cpu().data.numpy()
                for index in range(len(target)):
                    dict_key = target_numpy[index]
                    data_dict[dict_key] = torch.cat((data_dict[dict_key][1:],
                                                     output[index].detach().view(1, -1)), 0)
            elif sum_temp == num_classes * sample_number and total_iter >= start_iter:
                target_numpy = target.cpu().data.numpy()
                for index in range(len(target)):
                    dict_key = target_numpy[index]
                    data_dict[dict_key] = torch.cat((data_dict[dict_key][1:],
                                                     output[index].detach().view(1, -1)), 0)
                # the covariance finder needs the data to be centered.
                for index in range(num_classes):
                    if index == 0:
                        X = data_dict[index] - data_dict[index].mean(0)
                        mean_embed_id = data_dict[index].mean(0).view(1, -1)
                    else:
                        X = torch.cat((X, data_dict[index] - data_dict[index].mean(0)), 0)
                        mean_embed_id = torch.cat((mean_embed_id,
                                                   data_dict[index].mean(0).view(1, -1)), 0)

                ## add the variance.
                temp_precision = torch.mm(X.t(), X) / len(X)
                temp_precision += 0.0001 * eye_matrix

                for index in range(num_classes):
                    new_dis = torch.distributions.multivariate_normal.MultivariateNormal(
                        mean_embed_id[index], covariance_matrix=temp_precision)
                    negative_samples = new_dis.rsample((sample_from,))
                    prob_density = new_dis.log_prob(negative_samples)
                    # breakpoint()
                    # index_prob = (prob_density < - self.threshold).nonzero().view(-1)
                    # keep the data in the low density area.
                    cur_samples, index_prob = torch.topk(- prob_density, select)
                    if index == 0:
                        ood_samples = negative_samples[index_prob]
                    else:
                        ood_samples = torch.cat((ood_samples, negative_samples[index_prob]), 0)
                if len(ood_samples) != 0:
                    # add some gaussian noise
                    # ood_samples = self.noise(ood_samples)
                    # energy_score_for_fg = 1 * torch.logsumexp(predictions[0][selected_fg_samples][:, :-1] / 1, 1)
                    x = x[:, user_class]
                    energy_score_for_fg = log_sum_exp(weight_energy, x, 1)
                    predictions_ood = net.fc(ood_samples)[:, user_class]
                    # energy_score_for_bg = 1 * torch.logsumexp(predictions_ood[0][:, :-1] / 1, 1)
                    energy_score_for_bg = log_sum_exp(weight_energy, predictions_ood, 1)

                    input_for_lr = torch.cat((energy_score_for_fg, energy_score_for_bg), -1)
                    labels_for_lr = torch.cat((torch.ones(len(output)).cuda(),
                                               torch.zeros(len(ood_samples)).cuda()), -1)

                    criterion = torch.nn.CrossEntropyLoss()
                    output1 = logistic_regression(input_for_lr.view(-1, 1))
                    lr_reg_loss = criterion(output1, labels_for_lr.long())
            else:
                target_numpy = target.cpu().data.numpy()
                for index in range(len(target)):
                    dict_key = target_numpy[index]
                    if number_dict[dict_key] < sample_number:
                        data_dict[dict_key][number_dict[dict_key]] = output[index].detach()
                        number_dict[dict_key] += 1
            loss += loss_weight * lr_reg_loss



        # breakpoint()
        if i > 0:
            w_diff = torch.tensor(0.).cuda()
            if isinstance(server_model, dict):  # state dict
                for w_name, w_t in net.named_parameters():
                    w_diff += torch.pow(torch.norm(server_model[w_name] - w_t), 2)
            else:
                for w, w_t in zip(server_model.parameters(), net.parameters()):
                    w_diff += torch.pow(torch.norm(w - w_t), 2)

            w_diff = torch.sqrt(w_diff)
            loss += mu / 2. * w_diff

        loss.backward()

        optimizer.step()

        # exponential moving average
        loss_avg = loss_avg * 0.8 + float(loss) * 0.2

    state['train_loss'] = loss_avg
    return loss_avg, total_iter
#use data from other user as to generate virtual OOD sample

#use data from other user as OOD sample+VOS
def VOS_train1(model_name, total_iter, state, max_iter, net, train_loader,train_loader2, num_classes, number_dict, sample_number, start_iter, data_dict, eye_matrix, logistic_regression, optimizer, loss_weight, weight_energy, sample_from, select,
               ext_class_dis_dict=None, verbose=0):
    class_dis_dict = {}
    net.train()  # enter train mode
    data_iterator = iter(train_loader)
    data_iterator2 = iter(train_loader2)
    #print("total iter", total_iter)
    #print("max iter", max_iter)
    #print("start iter", start_iter)
    for i in tqdm(range(max_iter), disable=verbose < 1):
        total_iter += 1
        loss_avg = 0.0
        try:
            data, target = next(data_iterator)
            OODdata, OODtarget = next(data_iterator2)
        except StopIteration:
            data_iterator = iter(train_loader)
            data, target = next(data_iterator)

            data_iterator2 = iter(train_loader2)
            OODdata, OODtarget = next(data_iterator2)


        data, target = data.cuda(), target.cuda()

        # forward
        x, output = net.forward_virtual(data)



        # energy regularization.
        sum_temp = 0
        for index in range(num_classes):
            sum_temp += number_dict[index]
        lr_reg_loss = torch.zeros(1).cuda()[0]
        if sum_temp == num_classes * sample_number and total_iter < start_iter:
            # maintaining an ID data queue for each class.
            target_numpy = target.cpu().data.numpy()
            for index in range(len(target)):
                dict_key = target_numpy[index]
                data_dict[dict_key] = torch.cat((data_dict[dict_key][1:],
                                                 output[index].detach().view(1, -1)), 0)
        elif sum_temp == num_classes * sample_number and total_iter >= start_iter:
            target_numpy = target.cpu().data.numpy()
            for index in range(len(target)):
                dict_key = target_numpy[index]
                data_dict[dict_key] = torch.cat((data_dict[dict_key][1:],
                                                 output[index].detach().view(1, -1)), 0)
            # the covariance finder needs the data to be centered.
            for index in range(num_classes):
                if index == 0:
                    X = data_dict[index] - data_dict[index].mean(0)
                    mean_embed_id = data_dict[index].mean(0).view(1, -1)
                else:
                    X = torch.cat((X, data_dict[index] - data_dict[index].mean(0)), 0)
                    mean_embed_id = torch.cat((mean_embed_id,
                                               data_dict[index].mean(0).view(1, -1)), 0)

            ## add the variance.
            temp_precision = torch.mm(X.t(), X) / len(X)
            temp_precision += 0.0001 * eye_matrix

            for index in range(num_classes):
                new_dis = torch.distributions.multivariate_normal.MultivariateNormal(
                    mean_embed_id[index], covariance_matrix=temp_precision)
                class_dis_dict[index] = new_dis
                negative_samples = new_dis.rsample((sample_from,))
                prob_density = new_dis.log_prob(negative_samples)
                # breakpoint()
                # index_prob = (prob_density < - self.threshold).nonzero().view(-1)
                # keep the data in the low density area.
                cur_samples, index_prob = torch.topk(- prob_density, select)
                if index == 0:
                    ood_samples = negative_samples[index_prob]
                else:
                    ood_samples = torch.cat((ood_samples, negative_samples[index_prob]), 0)
            if len(ood_samples) != 0:
                # add some gaussian noise
                # ood_samples = self.noise(ood_samples)
                # energy_score_for_fg = 1 * torch.logsumexp(predictions[0][selected_fg_samples][:, :-1] / 1, 1)
                energy_score_for_fg = log_sum_exp(weight_energy, x, 1)
                #prediction ood for sample from other client
                if ext_class_dis_dict is None:
                    predictions_ood_other, _ = net.forward_virtual(OODdata)
                else:
                    if len(ext_class_dis_dict) > 0:
                        for index in range(num_classes):
                            other_samples = ext_class_dis_dict[index].rsample((sample_from,))
                            prob_density = class_dis_dict[index].log_prob(other_samples)
                            # select most OoD samples
                            cur_samples, index_prob = torch.topk(- prob_density, select, largest=False)
                            other_samples = other_samples[index_prob]
                            if model_name == 'preresnet18':
                                _pred_ood = net.linear(other_samples)
                            else:
                                _pred_ood = net.fc(other_samples)
                            if index == 0:
                                predictions_ood_other = _pred_ood
                            else:
                                predictions_ood_other = torch.cat((predictions_ood_other, _pred_ood), 0)
                if model_name == 'preresnet18':
                    predictions_ood = net.linear(ood_samples)
                else:
                    predictions_ood = net.fc(ood_samples)
                # energy_score_for_bg = 1 * torch.logsumexp(predictions_ood[0][:, :-1] / 1, 1)
                energy_score_for_bg = log_sum_exp(weight_energy, predictions_ood, 1)
                energy_score_for_bg_other = log_sum_exp(weight_energy, predictions_ood_other, 1)

                input_for_lr = torch.cat((energy_score_for_fg, energy_score_for_bg), -1)
                input_for_lr = torch.cat((input_for_lr, energy_score_for_bg_other), -1)
                labels_for_lr = torch.cat((torch.ones(len(output)).cuda(),
                                           torch.zeros(len(ood_samples)+OODdata.shape[0]).cuda()), -1)

                criterion = torch.nn.CrossEntropyLoss()
                output1 = logistic_regression(input_for_lr.view(-1, 1))
                lr_reg_loss = criterion(output1, labels_for_lr.long())
        else:
            target_numpy = target.cpu().data.numpy()
            for index in range(len(target)):
                dict_key = target_numpy[index]
                if number_dict[dict_key] < sample_number:
                    data_dict[dict_key][number_dict[dict_key]] = output[index].detach()
                    number_dict[dict_key] += 1

        # backward

        optimizer.zero_grad()
        loss = F.cross_entropy(x, target)
        # breakpoint()
        loss += loss_weight * lr_reg_loss
        loss.backward()

        optimizer.step()

        # exponential moving average
        loss_avg = loss_avg * 0.8 + float(loss) * 0.2

    state['train_loss'] = loss_avg
    return loss_avg, total_iter, class_dis_dict

def VOS_train2(model_name, total_iter, state, max_iter, net, train_loader,train_loader2, num_classes, number_dict, sample_number, start_iter, data_dict, eye_matrix, logistic_regression, optimizer, loss_weight, weight_energy, sample_from, select,
               verbose=0):
    net.train()  # enter train mode
    data_iterator = iter(train_loader)
    data_iterator2 = iter(train_loader2)
    #print("total iter", total_iter)
    #print("max iter", max_iter)
    #print("start iter", start_iter)
    for i in tqdm(range(max_iter), disable=verbose < 1):
        total_iter += 1
        loss_avg = 0.0
        try:
            data, target = next(data_iterator)
            OODdata, OODtarget = next(data_iterator2)
        except StopIteration:
            data_iterator = iter(train_loader)
            data, target = next(data_iterator)

            data_iterator2 = iter(train_loader2)
            OODdata, OODtarget = next(data_iterator2)


        data, target = data.cuda(), target.cuda()

        # forward
        x, output = net.forward_virtual(data)



        # energy regularization.
        sum_temp = 0
        for index in range(num_classes):
            sum_temp += number_dict[index]
        lr_reg_loss = torch.zeros(1).cuda()[0]
        if sum_temp == num_classes * sample_number and total_iter < start_iter:
            # maintaining an ID data queue for each class.
            target_numpy = target.cpu().data.numpy()
            for index in range(len(target)):
                dict_key = target_numpy[index]
                data_dict[dict_key] = torch.cat((data_dict[dict_key][1:],
                                                 output[index].detach().view(1, -1)), 0)
        elif sum_temp == num_classes * sample_number and total_iter >= start_iter:
            target_numpy = target.cpu().data.numpy()
            for index in range(len(target)):
                dict_key = target_numpy[index]
                data_dict[dict_key] = torch.cat((data_dict[dict_key][1:],
                                                 output[index].detach().view(1, -1)), 0)
            # the covariance finder needs the data to be centered.
            for index in range(num_classes):
                if index == 0:
                    X = data_dict[index] - data_dict[index].mean(0)
                    mean_embed_id = data_dict[index].mean(0).view(1, -1)
                else:
                    X = torch.cat((X, data_dict[index] - data_dict[index].mean(0)), 0)
                    mean_embed_id = torch.cat((mean_embed_id,
                                               data_dict[index].mean(0).view(1, -1)), 0)

            ## add the variance.
            temp_precision = torch.mm(X.t(), X) / len(X)
            temp_precision += 0.0001 * eye_matrix

            for index in range(num_classes):
                new_dis = torch.distributions.multivariate_normal.MultivariateNormal(
                    mean_embed_id[index], covariance_matrix=temp_precision)
                negative_samples = new_dis.rsample((sample_from,))
                prob_density = new_dis.log_prob(negative_samples)
                # breakpoint()
                # index_prob = (prob_density < - self.threshold).nonzero().view(-1)
                # keep the data in the low density area.
                cur_samples, index_prob = torch.topk(- prob_density, select)
                if index == 0:
                    ood_samples = negative_samples[index_prob]
                else:
                    ood_samples = torch.cat((ood_samples, negative_samples[index_prob]), 0)
            if len(ood_samples) != 0:
                # add some gaussian noise
                # ood_samples = self.noise(ood_samples)
                # energy_score_for_fg = 1 * torch.logsumexp(predictions[0][selected_fg_samples][:, :-1] / 1, 1)
                energy_score_for_fg = log_sum_exp(weight_energy, x, 1)
                #prediction ood for sample from other client
                predictions_ood_other, _ = net.forward_virtual(OODdata)
                if model_name == 'preresnet18':
                    predictions_ood = net.linear(ood_samples)
                else:
                    predictions_ood = net.fc(ood_samples)
                # energy_score_for_bg = 1 * torch.logsumexp(predictions_ood[0][:, :-1] / 1, 1)
                energy_score_for_bg = log_sum_exp(weight_energy, predictions_ood, 1)
                energy_score_for_bg_other = log_sum_exp(weight_energy, predictions_ood_other, 1)

                input_for_lr = torch.cat((energy_score_for_fg, energy_score_for_bg_other), -1)
                labels_for_lr = torch.cat((torch.ones(len(output)).cuda(),
                                           torch.zeros(OODdata.shape[0]).cuda()), -1)

                criterion = torch.nn.CrossEntropyLoss()
                output1 = logistic_regression(input_for_lr.view(-1, 1))
                lr_reg_loss = criterion(output1, labels_for_lr.long())
        else:
            target_numpy = target.cpu().data.numpy()
            for index in range(len(target)):
                dict_key = target_numpy[index]
                if number_dict[dict_key] < sample_number:
                    data_dict[dict_key][number_dict[dict_key]] = output[index].detach()
                    number_dict[dict_key] += 1

        # backward

        optimizer.zero_grad()
        loss = F.cross_entropy(x, target)
        # breakpoint()
        loss += loss_weight * lr_reg_loss
        loss.backward()

        optimizer.step()

        # exponential moving average
        loss_avg = loss_avg * 0.8 + float(loss) * 0.2

    state['train_loss'] = loss_avg
    return loss_avg, total_iter



# test function
def VOS_test(net, test_loader):
    net.eval()
    loss_avg = 0.0
    correct = 0
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.cuda(), target.cuda()

            # forward
            output = net(data)
            loss = F.cross_entropy(output, target)

            # accuracy
            pred = output.data.max(1)[1]
            correct += pred.eq(target.data).sum().item()

            # test loss average
            loss_avg += float(loss.data)

    test_loss = loss_avg / len(test_loader)
    test_accuracy = correct / len(test_loader.dataset)
    return test_loss, test_accuracy




