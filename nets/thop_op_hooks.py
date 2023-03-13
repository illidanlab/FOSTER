import torch
from torch import nn
from thop.vision.basic_hooks import count_convNd, count_linear, count_bn, zero_ops
from .slimmable_ops import SlimmableConv2d, SlimmableLinear, SlimmableBatchNorm2d, \
    SlimmableBatchNorm1d

__all__ = ['thop_hooks']

# extra profile functions from newer version of thop
def count_ln(m, x, y):
    """layer norm"""
    x = x[0]
    if not m.training:
        m.total_ops += counter_norm(x.numel())

def counter_norm(input_size):
    """input is a number not a array or tensor"""
    return torch.DoubleTensor([2 * input_size])

# def count_embedding(m: nn.Embedding, x, y):
#     # per output element
#     total_mul = m.embedding_dim
#     num_elements = y.numel()
#     total_ops = total_mul * num_elements
#
#     m.total_ops += torch.DoubleTensor([int(total_ops)])

def count_softmax(m, x, y):
    x = x[0]
    nfeatures = x.size()[m.dim]
    batch_size = x.numel() // nfeatures

    m.total_ops += counter_softmax(batch_size, nfeatures)

def counter_softmax(batch_size, nfeatures):
    total_exp = nfeatures
    total_add = nfeatures - 1
    total_div = nfeatures
    total_ops = batch_size * (total_exp + total_add + total_div)
    return torch.DoubleTensor([int(total_ops)])

thop_hooks = {
    SlimmableConv2d: count_convNd,
    SlimmableLinear: count_linear,
    SlimmableBatchNorm2d: count_bn,
    SlimmableBatchNorm1d: count_bn,
    # SlimmableEmbedding: zero_ops, # count_embedding,  # seems no cost for index accessing
    # SlimmableLayerNorm: count_ln,
    nn.LayerNorm: count_ln,
}
