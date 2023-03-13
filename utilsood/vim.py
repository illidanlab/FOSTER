#!/usr/bin/env python
import argparse
import torch
import numpy as np
from tqdm import tqdm
from numpy.linalg import norm, pinv
from scipy.special import softmax
from sklearn import metrics
from sklearn.metrics import pairwise_distances_argmin_min
from sklearn.covariance import EmpiricalCovariance
from os.path import basename, splitext
from scipy.special import logsumexp
import pandas as pd
from .extract_feature_bit import get_wb
from .feat_extract import feature_extract_in, feature_extract_out
import os


#region Helper
def num_fp_at_recall(ind_conf, ood_conf, tpr):
    num_ind = len(ind_conf)

    if num_ind == 0 and len(ood_conf) == 0:
        return 0, 0.
    if num_ind == 0:
        return 0, np.max(ood_conf) + 1

    recall_num = int(np.floor(tpr * num_ind))
    thresh = np.sort(ind_conf)[-recall_num]
    num_fp = np.sum(ood_conf >= thresh)
    return num_fp, thresh

def fpr_recall(ind_conf, ood_conf, tpr):
    num_fp, thresh = num_fp_at_recall(ind_conf, ood_conf, tpr)
    num_ood = len(ood_conf)
    fpr = num_fp / max(1, num_ood)
    return fpr, thresh

def auc(ind_conf, ood_conf):
    conf = np.concatenate((ind_conf, ood_conf))
    ind_indicator = np.concatenate((np.ones_like(ind_conf), np.zeros_like(ood_conf)))

    fpr, tpr, _ = metrics.roc_curve(ind_indicator, conf)
    precision_in, recall_in, _ = metrics.precision_recall_curve(
        ind_indicator, conf)
    precision_out, recall_out, _ = metrics.precision_recall_curve(
        1 - ind_indicator, 1 - conf)

    auroc = metrics.auc(fpr, tpr)
    aupr_in = metrics.auc(recall_in, precision_in)
    aupr_out = metrics.auc(recall_out, precision_out)

    return auroc, aupr_in, aupr_out

def kl(p, q):
    return np.sum(np.where(p != 0, p * np.log(p / q), 0))

#endregion

#region OOD

def gradnorm(x, w, b):
    fc = torch.nn.Linear(*w.shape[::-1])
    fc.weight.data[...] = torch.from_numpy(w)
    fc.bias.data[...] = torch.from_numpy(b)
    fc.cuda()

    x = torch.from_numpy(x).float().cuda()
    logsoftmax = torch.nn.LogSoftmax(dim=-1).cuda()

    confs = []

    for i in tqdm(x):
        targets = torch.ones((1, 1000)).cuda()
        fc.zero_grad()
        loss = torch.mean(torch.sum(-targets * logsoftmax(fc(i[None])), dim=-1))
        loss.backward()
        layer_grad_norm = torch.sum(torch.abs(fc.weight.grad.data)).cpu().numpy()
        confs.append(layer_grad_norm)

    return np.array(confs)

#endregion

def get_vim(client_id, loss_weight, train_loader, test_loader,out_loader, model, in_dataset,out_datasets, train_batch_size, batch_size, num_classes, m_name):


    w,b = get_wb(model, in_dataset, m_name)
    print('w.shape={}, b.shape={}'.format(w.shape, b.shape))
    print('load features')
    cache_name = f"cache/{in_dataset}_train_{m_name}_{client_id}_{loss_weight}_in_alllayers.npy"
    if not os.path.exists(cache_name):
        feature_extract_in(client_id, loss_weight, train_loader, 'train', model, train_batch_size, num_classes, in_dataset, m_name)
    feat_log, score_log, label_log = np.load(cache_name, allow_pickle=True)
    feature_id_train , score_log = feat_log.T.astype(np.float32), score_log.T.astype(np.float32)
    class_num = score_log.shape[1]

    cache_name = f"cache/{in_dataset}_val_{m_name}_{client_id}_{loss_weight}_in_alllayers.npy"
    if not os.path.exists(cache_name):
        feature_extract_in(client_id, loss_weight, test_loader, 'val', model, batch_size, num_classes, in_dataset, m_name)
    feat_log_val, score_log_val, label_log_val = np.load(cache_name, allow_pickle=True)
    feature_id_val, score_log_val = feat_log_val.T.astype(np.float32), score_log_val.T.astype(np.float32)

    feature_oods = {}
    for ood_dataset in out_datasets:
        cache_name = f"cache/{ood_dataset}vs{in_dataset}_{m_name}_{client_id}_{loss_weight}_out_alllayers.npy"
        if not os.path.exists(cache_name):
            feature_extract_out(client_id, loss_weight, out_loader, model, batch_size, num_classes, out_datasets, in_dataset, m_name)
        ood_feat_log, ood_score_log = np.load(cache_name, allow_pickle=True)
        ood_feat_log, ood_score_log = ood_feat_log.T.astype(np.float32), ood_score_log.T.astype(np.float32)
        feature_oods[ood_dataset] = ood_feat_log

    for name, ood in feature_oods.items():
        print(f'{name} {ood.shape}')
    print('computing logits...')
    logit_id_train = feature_id_train @ w.T + b
    logit_id_val = feature_id_val @ w.T + b
    logit_oods = {name: feat @ w.T + b for name, feat in feature_oods.items()}


    print('computing softmax...')
    softmax_id_train = softmax(logit_id_train, axis=-1)
    softmax_id_val = softmax(logit_id_val, axis=-1)
    softmax_oods = {name: softmax(logit, axis=-1) for name, logit in logit_oods.items()}

    u = -np.matmul(pinv(w), b)

    df = pd.DataFrame(columns = ['method', 'oodset', 'auroc', 'fpr'])

    dfs = []
    recall = 0.95


    # ---------------------------------------
    method = 'ViM'
    print(f'\n{method}')
    result = []
    DIM = 1000 if feature_id_val.shape[-1] >= 2048 else 512


    print('computing principal space...')
    ec = EmpiricalCovariance(assume_centered=True)
    ec.fit(feature_id_train - u)
    eig_vals, eigen_vectors = np.linalg.eig(ec.covariance_)
    #print("eigen_vectors", eigen_vectors.shape)
    NS = np.ascontiguousarray((eigen_vectors.T[np.argsort(eig_vals * -1)]).T)

    print('computing alpha...')
    vlogit_id_train = norm(np.matmul(feature_id_train - u, NS), axis=-1)
    #print("NS", NS)
    alpha = logit_id_train.max(axis=-1).mean() / vlogit_id_train.mean()


    vlogit_id_val = norm(np.matmul(feature_id_val - u, NS), axis=-1) * alpha
    energy_id_val = logsumexp(logit_id_val, axis=-1)
    score_id = -vlogit_id_val + energy_id_val

    for name, logit_ood, feature_ood in zip(out_datasets, logit_oods.values(), feature_oods.values()):
        energy_ood = logsumexp(logit_ood, axis=-1)
        vlogit_ood = norm(np.matmul(feature_ood - u, NS), axis=-1) * alpha
        score_ood = -vlogit_ood + energy_ood
        auc_ood = auc(score_id, score_ood)[0]
        fpr_ood, _ = fpr_recall(score_id, score_ood, recall)
        result.append(dict(method=method, oodset=name, auroc=auc_ood, fpr=fpr_ood))
        print(f'{method}: {name} auroc {auc_ood:.2%}, fpr {fpr_ood:.2%}')
    df = pd.DataFrame(result)
    dfs.append(df)
    print(f'mean auroc {df.auroc.mean():.2%}, {df.fpr.mean():.2%}')
    return score_ood, score_id

