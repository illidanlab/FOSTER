import torch

def accuracy(output, target, topk=(1,)):
    """Computes the accuracy over the k top predictions for the specified values of k"""
    maxk = max(topk)
    batch_size = target.size(0)
    _, pred = output.topk(maxk, 1, True, True)
    pred = pred.t()
    correct = pred.eq(target.reshape(1, -1).expand_as(pred))
    return [correct[:k].reshape(-1).float().sum(0) * 100. / batch_size for k in topk]


def Acc(targets, preds):
    '''
    PyTorch operation: Accuracy. 

    Args:
        targets: Tensor. Ground truth targets of data.
        preds: Tensor. Predictions on data.

    Returns:
        acc: float
    '''
    correct = preds.eq(targets.view_as(preds)).sum().item()
    total = torch.numel(preds)
    acc = correct / total
    return acc


def FPR(targets, preds):
    '''
    PyTorch operation: False positive rate. 

    Args:
        targets: Tensor. Ground truth targets of data.
        preds: Tensor. Predictions on data.

    Returns:
        FPR: float
    '''
    N = (targets == 0).sum().item()  # negative sample number 
    FP = torch.logical_and(targets == 0, preds.squeeze() == 1).sum().item()  # FP sample number
    FPR = FP / N
    return FPR


def FNR(targets, preds):
    '''
    PyTorch operation: False negative rate. 

    Args:
        targets: Tensor. Ground truth targets of data.
        preds: Tensor. Predictions on data.

    Returns:
        FNR: float
    '''
    P = (targets == 1).sum().item()  # positive sample number 
    FN = torch.logical_and(targets == 1, preds.squeeze() == 0).sum().item()  # FP sample number
    FNR = FN / P
    return FNR
