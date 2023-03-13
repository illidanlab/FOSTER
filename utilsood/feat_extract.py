#! /usr/bin/env python3

import torch

import os



import numpy as np
import torch.nn.functional as F
import time




FORCE_RUN = False


def feature_extract_in(client_id, loss_weight, in_loader, split, model, batch_size, num_classes=10, in_dataset='Cifar10', m_name='FOSTER'):
    if in_dataset in ['Cifar10', 'Cifar100']:
        dummy_input = torch.zeros((1, 3, 32, 32)).cuda()
    elif in_dataset == 'stl':
        dummy_input = torch.zeros((1, 3, 96, 96)).cuda()
    elif in_dataset in ['DomainNet', 'ImageNet']:
        dummy_input = torch.zeros((1, 3, 256, 256)).cuda()
    else:
        raise ValueError(f"Invalid dataname: {in_dataset}")
    if in_dataset in ['Cifar10', 'stl']:
        num_classes = 10
    elif in_dataset == 'Cifar100':
        num_classes = 100
    elif in_dataset == 'ImageNet':
        num_classes = 12
    else:
        raise ValueError(f"Invalid dataname: {in_dataset}")
    score, feature_list = model.feature_list(dummy_input)
    featdims = [feat.shape[1] for feat in feature_list]
    cache_name = f"cache/{in_dataset}_{split}_{m_name}_{client_id}_{loss_weight}_in_alllayers.npy"
    print("start {}".format(cache_name))
    if not os.path.exists(cache_name):

        feat_log = np.zeros((len(in_loader.dataset), sum(featdims)))

        score_log = np.zeros((len(in_loader.dataset), num_classes))
        label_log = np.zeros(len(in_loader.dataset))

        model.eval()
        for batch_idx, (inputs, targets) in enumerate(in_loader):
            inputs, targets = inputs.cuda(), targets.cuda()
            start_ind = batch_idx * batch_size
            end_ind = min((batch_idx + 1) * batch_size, len(in_loader.dataset))

            score, feature_list = model.feature_list(inputs)
            start = True
            for layer_feat in feature_list:
                temp = F.adaptive_avg_pool2d(layer_feat, 1).squeeze()
                if len(temp.shape) < 2:
                    temp = temp.unsqueeze(0)
                if start:
                    out = temp
                    start = False
                else:
                    out = torch.cat((out, temp), dim=1)
                #print(F.adaptive_avg_pool2d(layer_feat, 1).shape)
            #out = torch.cat([F.adaptive_avg_pool2d(layer_feat, 1).squeeze() for layer_feat in feature_list], dim=1)

            feat_log[start_ind:end_ind, :] = out.data.cpu().numpy()
            label_log[start_ind:end_ind] = targets.data.cpu().numpy()
            score_log[start_ind:end_ind] = score.data.cpu().numpy()
            # if batch_idx % 100 == 0:
            #print(f"{batch_idx}/{len(in_loader)}")
        np.save(cache_name, (feat_log.T, score_log.T, label_log))
    else:
        feat_log, score_log, label_log = np.load(cache_name, allow_pickle=True)
        feat_log, score_log = feat_log.T, score_log.T

        return feat_log, score_log
def feature_extract_out(client_id, loss_weight, out_loader, model,batch_size, num_classes=10, out_datasets='SVHN', in_dataset='Cifar10', m_name='FOSTER'):
    if in_dataset in ['Cifar10', 'Cifar100']:
        dummy_input = torch.zeros((1, 3, 32, 32)).cuda()
    elif in_dataset == 'stl':
        dummy_input = torch.zeros((1, 3, 96, 96)).cuda()
    elif in_dataset in ['DomainNet', 'ImageNet']:
        dummy_input = torch.zeros((1, 3, 256, 256)).cuda()
    else:
        raise ValueError(f"Invalid dataname: {in_dataset}")
    if in_dataset in ['Cifar10', 'stl']:
        num_classes = 10
    elif in_dataset == 'Cifar100':
        num_classes = 100
    elif in_dataset == 'ImageNet':
        num_classes = 12
    else:
        raise ValueError(f"Invalid dataname: {in_dataset}")
    score, feature_list = model.feature_list(dummy_input)
    featdims = [feat.shape[1] for feat in feature_list]
    for ood_dataset in out_datasets:
        cache_name = f"cache/{ood_dataset}vs{in_dataset}_{m_name}_{client_id}_{loss_weight}_out_alllayers.npy"
        if not os.path.exists(cache_name):
            ood_feat_log = np.zeros((len(out_loader.dataset), sum(featdims)))
            ood_score_log = np.zeros((len(out_loader.dataset), num_classes))

            model.eval()
            for batch_idx, (inputs, _) in enumerate(out_loader):
                inputs = inputs.cuda()
                start_ind = batch_idx * batch_size
                end_ind = min((batch_idx + 1) * batch_size, len(out_loader.dataset))

                score, feature_list = model.feature_list(inputs)
                out = torch.cat([F.adaptive_avg_pool2d(layer_feat, 1).squeeze() for layer_feat in feature_list], dim=1)

                ood_feat_log[start_ind:end_ind, :] = out.data.cpu().numpy()
                ood_score_log[start_ind:end_ind] = score.data.cpu().numpy()
                #if batch_idx % 10 == 0:
                #print(f"{batch_idx}/{len(out_loader)}")
            np.save(cache_name, (ood_feat_log.T, ood_score_log.T))
        else:
            ood_feat_log, ood_score_log = np.load(cache_name, allow_pickle=True)
            ood_feat_log, ood_score_log = ood_feat_log.T, ood_score_log.T
        return ood_feat_log, ood_score_log


