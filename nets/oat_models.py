"""Extend models with lambda-conditioned models (FiLM-based)."""
from typing import Union

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as func
from torch.nn.parallel import DistributedDataParallel as DDP

from .FiLM import FiLMLayer, FilmDualNormLayer, FiLM2D, FiLM1D
from .bn_ops import get_bn_layer, is_film_dual_norm
from .dual_bn import DualNormLayer
from .models import BaseModule
from .oat_utils import element_wise_sample_lambda
from .dual_ops import DualConv2d


class OATBaseModel(BaseModule):
    lambda_choices = [0.0, 0.1, 0.2, 0.3, 0.4, 1.0]

    def __init__(self, lmbd_enc_dim):
        super(BaseModule, self).__init__()

        # lambda encoding matrix by random
        self.lmbd_enc_dim = lmbd_enc_dim
        if lmbd_enc_dim > 0:
            assert lmbd_enc_dim > len(
                self.lambda_choices), f"Too small dim. Require at  least be {len(self.lambda_choices)}."
            rand_mat = np.random.randn(lmbd_enc_dim, lmbd_enc_dim)
            rand_otho_mat, _ = np.linalg.qr(rand_mat)
            self.register_buffer('lambda_encoding_mat', torch.tensor(rand_otho_mat).float())
        else:
            self.register_parameter('lambda_encoding_mat', None)

    def set_bn_mode(self, oat_lambdas: Union[None, torch.Tensor, float],
                    is_noised: Union[bool, torch.Tensor, None] = None):
        """Set BN mode to be noised or clean. This is only effective for StackedNormLayer
        or DualNormLayer.

        Args:
            oat_lambdas: The encoded lambda.
            is_noised: Bool or bool tensor chooses BNs. If None is provided, this will be inferred
                from `oat_lambdas`. Note, you may set `is_noised=True` to enforce BN selection even
                if 0 (no noise) presents in `oat_lambdas`.
        """
        set_oat_bn_mode(self, oat_lambdas, is_noised)

    def sample_lambdas(self, n_sample):
        """Sample `n_sample` lambdas."""
        if self.lmbd_enc_dim > 0:
            adv_lmbd = element_wise_sample_lambda(self.lambda_choices, batch_size=n_sample)
            adv_lmbd = torch.from_numpy(adv_lmbd).float().to(self.lambda_encoding_mat.device)
            return adv_lmbd
        else:
            raise RuntimeError(f"Cannot sample lambdas since FiLM layers are disabled by zero encoding dimension.")


class DigitOATModel(OATBaseModel):
    """
    Model for benchmark experiment on Digits.

    Different from DigitModel:
    * Not share affine
    * Add FiLM layer after the BN layer.
    """
    input_shape = [None, 3, 28, 28]

    def __init__(self, num_classes=10, bn_type='bn', track_running_stats=True,
                 lmbd_enc_dim=64, share_affine=False, bn_affine=True, **kwargs):
        super(DigitOATModel, self).__init__(lmbd_enc_dim=lmbd_enc_dim)
        self.bn_type = bn_type
        bn_class = get_bn_layer(bn_type)
        # share_affine
        bn_kwargs = dict(
            track_running_stats=track_running_stats,
            affine=bn_affine,
        )
        if bn_type.startswith('d'):  # dual BN
            bn_kwargs['share_affine'] = share_affine
        is_fdbn = is_film_dual_norm(self.bn_type)
        affine_comb_mode = kwargs.setdefault('affine_comb_mode', 'af_lmbd')

        # feature layers
        self.conv1 = nn.Conv2d(3, 64, 5, 1, 2)
        self.bn1 = bn_class['2d'](64, **bn_kwargs)
        self.film1 = FiLM2D(64, lmbd_enc_dim, is_fdbn=is_fdbn, affine_comb_mode=affine_comb_mode) if lmbd_enc_dim > 0 else None

        self.conv2 = nn.Conv2d(64, 64, 5, 1, 2)
        self.bn2 = bn_class['2d'](64, **bn_kwargs)
        self.film2 = FiLM2D(64, lmbd_enc_dim, is_fdbn=is_fdbn, affine_comb_mode=affine_comb_mode) if lmbd_enc_dim > 0 else None

        self.conv3 = nn.Conv2d(64, 128, 5, 1, 2)
        self.bn3 = bn_class['2d'](128, **bn_kwargs)
        self.film3 = FiLM2D(128, lmbd_enc_dim, is_fdbn=is_fdbn, affine_comb_mode=affine_comb_mode) if lmbd_enc_dim > 0 else None

        # decoding layers
        self.fc1 = nn.Linear(6272, 2048)
        self.bn4 = bn_class['1d'](2048, **bn_kwargs)
        self.film4 = FiLM1D(2048, lmbd_enc_dim, is_fdbn=is_fdbn, affine_comb_mode=affine_comb_mode) if lmbd_enc_dim > 0 else None

        self.fc2 = nn.Linear(2048, 512)
        self.bn5 = bn_class['1d'](512, **bn_kwargs)
        self.film5 = FiLM1D(512, lmbd_enc_dim, is_fdbn=is_fdbn, affine_comb_mode=affine_comb_mode) if lmbd_enc_dim > 0 else None

        self.fc3 = nn.Linear(512, num_classes)

    def forward(self, x):
        z = self.encode(x)
        return self.decode_clf(z)

    def encode(self, x):
        x = self.bn1(self.conv1(x))
        if self.film1:
            x = self.film1(x)
        x = func.max_pool2d(func.relu(x), 2)

        x = self.bn2(self.conv2(x))
        if self.film2:
            x = self.film2(x)
        x = func.max_pool2d(func.relu(x), 2)

        x = self.bn3(self.conv3(x))
        if self.film3:
            x = self.film3(x)
        x = func.relu(x)

        x = x.view(x.shape[0], -1)
        return x

    def decode_clf(self, x):
        x = self.fc1(x)
        x = self.bn4(x)
        if self.film4:
            x = self.film4(x)
        x = func.relu(x)

        x = self.fc2(x)
        x = self.bn5(x)
        if self.film5:
            x = self.film5(x)
        x = func.relu(x)

        logits = self.fc3(x)
        return logits


def set_oat_bn_mode(self: OATBaseModel, oat_lambdas: Union[None, torch.Tensor, float],
                    is_noised: Union[bool, torch.Tensor, None] = None):
    """Set BN mode to be noised or clean. This is only effective for StackedNormLayer
    or DualNormLayer."""
    # Set FiLM layers
    oat_module = self.module if isinstance(self, DDP) else self
    if isinstance(oat_lambdas, float):
        encoded_lambdas = oat_module.lambda_encoding_mat[oat_module.lambda_choices.index(oat_lambdas), :].detach()
    elif isinstance(oat_lambdas, torch.Tensor):
        encoded_lambdas = torch.stack(
            [oat_module.lambda_encoding_mat[oat_module.lambda_choices.index(lmbd), :] for lmbd in
             oat_lambdas],
            dim=0
        ).detach()
        assert encoded_lambdas.dim() == 2
    else:
        assert oat_lambdas is None, f"Unexpected type: {type(oat_lambdas)}"
        encoded_lambdas = None

    def set_film_state_(m):
        if isinstance(m, FiLMLayer):
            m.lmbd = encoded_lambdas
            m.lmbd_values = oat_lambdas

    self.apply(set_film_state_)

    # Set dual BN
    if is_noised is None:
        is_noised = oat_lambdas > 0

    def set_dbn_state_(m):
        if isinstance(m, (DualNormLayer, FilmDualNormLayer)):
            if isinstance(is_noised, torch.Tensor):
                m.clean_input = ~is_noised
            else:
                m.clean_input = not is_noised
        elif isinstance(m, (DualConv2d,)):
            m.mode = 1 if is_noised else 0

    self.apply(set_dbn_state_)
