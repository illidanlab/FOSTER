import torch
import numpy as np


def element_wise_sample_lambda(lambda_choices, batch_size=128, zero_probs=-1,
                               min_n_sample=5):
    """Elementwise sample and encode lambda.

    Args:
        lambda_choices (list): list of floats. Useful when sampling is 'disc'.
        encoding_mat (str): encoding scheme. Useful when sampling is 'disc'.
        batch_size (int): batch size
        zero_probs (float): If probs<=0, ignored and use uniform sample.

    Returns:
        lambda_vals: Tensor. size=(batch_size). For loss.
        encoded_lambda: Tensor. size=(batch_size, 1). FiLM input tensor.
        num_zeros: int. How many 0s are sampled in this batch.
    """
    # if sampling == 'disc':
    lambda_none_zero_choices = list(set(lambda_choices) - set([0]))
    if 0 in lambda_choices:
        if zero_probs > 0:
            num_zeros = np.ceil(zero_probs * batch_size).astype(int)
        else:  # uniform sample
            num_zeros = np.ceil(batch_size / len(lambda_choices)).astype(int)
            # print('num_zeros %d/batch_size %d' % (num_zeros, batch_size))
        num_zeros = max((min_n_sample, num_zeros))
        if batch_size - num_zeros < min_n_sample:
            num_zeros = batch_size
    else:
        num_zeros = 0

    num_none_zeros = int(batch_size - num_zeros)
    _lambda_zeros = np.zeros((num_zeros, 1))
    _lambda_none_zeros = np.random.choice(lambda_none_zero_choices, size=(num_none_zeros, 1))
    encoded_lambda = np.concatenate([_lambda_zeros, _lambda_none_zeros], axis=0)

    lambda_vals = np.squeeze(encoded_lambda) / 1.0
    assert np.amax(lambda_vals) <= 1 and np.amin(lambda_vals) >= 0

    return lambda_vals
