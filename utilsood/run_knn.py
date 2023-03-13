import os
import time

from .metrics import cal_metric, print_all_results
import torch
import faiss
import numpy as np
from .feat_extract import feature_extract_in, feature_extract_out


def run_knn_func(client_id, loss_weight, train_loader, test_loader, out_loader, model, train_batch_size, batch_size, num_classes, in_dataset, out_datasets, m_name='FOSTER'):
    cache_name = f"cache/{in_dataset}_train_{m_name}_{client_id}_{loss_weight}_in_alllayers.npy"
    if ~os.path.exists(cache_name):
        print("not exist train")
        feature_extract_in(client_id, loss_weight, train_loader, 'train', model, train_batch_size, num_classes, in_dataset, m_name)
    feat_log, score_log, label_log = np.load(cache_name, allow_pickle=True)
    feat_log, score_log = feat_log.T.astype(np.float32), score_log.T.astype(np.float32)
    class_num = score_log.shape[1]

    cache_name = f"cache/{in_dataset}_val_{m_name}_{client_id}_{loss_weight}_in_alllayers.npy"
    if ~os.path.exists(cache_name):
        print("not exist test")
        feature_extract_in(client_id, loss_weight, test_loader, 'val', model, batch_size, num_classes, in_dataset, m_name)
    feat_log_val, score_log_val, label_log_val = np.load(cache_name, allow_pickle=True)
    feat_log_val, score_log_val = feat_log_val.T.astype(np.float32), score_log_val.T.astype(np.float32)

    ood_feat_log_all = {}
    for ood_dataset in out_datasets:
        cache_name = f"cache/{ood_dataset}vs{in_dataset}_{m_name}_{client_id}_{loss_weight}_out_alllayers.npy"
        if ~os.path.exists(cache_name):
            print("not exist ood")
            feature_extract_out(client_id, loss_weight, out_loader, model, batch_size, num_classes, out_datasets, in_dataset, m_name)
        ood_feat_log, ood_score_log = np.load(cache_name, allow_pickle=True)
        ood_feat_log, ood_score_log = ood_feat_log.T.astype(np.float32), ood_score_log.T.astype(np.float32)
        ood_feat_log_all[ood_dataset] = ood_feat_log

    normalizer = lambda x: x / (np.linalg.norm(x, ord=2, axis=-1, keepdims=True) + 1e-10)

    prepos_feat = lambda x: np.ascontiguousarray(normalizer(x))  # Last Layer only
    ftrain = prepos_feat(feat_log)
    ftest = prepos_feat(feat_log_val)
    food_all = {}
    for ood_dataset in out_datasets:
        food_all[ood_dataset] = prepos_feat(ood_feat_log_all[ood_dataset])

    #################### KNN score OOD detection #################

    index = faiss.IndexFlatL2(ftrain.shape[1])
    index.add(ftrain)
    for K in [200]:

        D, _ = index.search(ftest, K)
        scores_in = -D[:, -1]
        all_results = []
        all_score_ood = []
        for ood_dataset, food in food_all.items():
            D, _ = index.search(food, K)
            scores_ood_test = -D[:, -1]
            all_score_ood.append(scores_ood_test)
            results = cal_metric(scores_in, scores_ood_test)
            all_results.append(results)

        print_all_results(all_results, out_datasets, f'KNN k={K}')
    return np.array(all_score_ood), scores_in

# #################### SSD+ score OOD detection #################
# begin = time.time()
# mean_feat = ftrain.mean(0)
# std_feat = ftrain.std(0)
# prepos_feat_ssd = lambda x: (x - mean_feat) / (std_feat + 1e-10)
# ftrain_ssd = prepos_feat_ssd(ftrain)
# ftest_ssd = prepos_feat_ssd(ftest)
# food_ssd_all = {}
# for ood_dataset in args.out_datasets:
#     food_ssd_all[ood_dataset] = prepos_feat_ssd(food_all[ood_dataset])
#
# inv_sigma_cls = [None for _ in range(class_num)]
# covs_cls = [None for _ in range(class_num)]
# mean_cls = [None for _ in range(class_num)]
# cov = lambda x: np.cov(x.T, bias=True)
# for cls in range(class_num):
#     mean_cls[cls] = ftrain_ssd[label_log == cls].mean(0)
#     feat_cls_center = ftrain_ssd[label_log == cls] - mean_cls[cls]
#     inv_sigma_cls[cls] = np.linalg.pinv(cov(feat_cls_center))
#
# def maha_score(X):
#     score_cls = np.zeros((class_num, len(X)))
#     for cls in range(class_num):
#         inv_sigma = inv_sigma_cls[cls]
#         mean = mean_cls[cls]
#         z = X - mean
#         score_cls[cls] = -np.sum(z * (inv_sigma.dot(z.T)).T, axis=-1)
#     return score_cls.max(0)
#
# dtest = maha_score(ftest_ssd)
# all_results = []
# for name, food in food_ssd_all.items():
#     print(f"SSD+: Evaluating {name}")
#     dood = maha_score(food)
#     results = metrics.cal_metric(dtest, dood)
#     all_results.append(results)
#
# metrics.print_all_results(all_results, args.out_datasets, 'SSD+')
# print(time.time() - begin)

