"""FiLM layers from https://github.com/VITA-Group/Once-for-All-Adversarial-Training/blob/26152796290113da2c2a97fa5f4221befccb57d7/models/FiLM.py"""
import torch
import torch.nn as nn
import torch.nn.functional as F


class FiLMLayer(nn.Module):
    def __init__(self, data_channels, lmbd_channels, alpha=1, activation=F.leaky_relu,
                 is_fdbn=False, affine_comb_mode='af_lmbd'):
        """Layer conditioned on the lambda. Use after BN layers.

        input size: (N, in_channels). output size: (N, channels*2)

        Args:
            data_channels (int): Output channel will be `alpha * data_channels * 2`.
            lmbd_channels (int): Channels for lambda vector. Use 1 for scale lambda.
            alpha (int): scalar. Expand ratio for FiLM hidden layer. Hidden layer has
                `alpha * data_channels * 2` channels.
            is_dfbn (bool): True if the former layer is FDBN.
            share_affine: useless if not fdbn

        Example:
            >>> film = FiLMLayer(...)
            >>> x = torch.rand(10, 6)
            >>> lmbd = torch.rand(10, 6*2)
            >>> out = film(x, lmbd)
        """
        super(FiLMLayer, self).__init__()
        self.channels = data_channels
        self.activation = activation
        self.num_affine = 1
        if is_fdbn:
            self.num_affine = 2
        self.lmbd_encoder = nn.Sequential(
            nn.Linear(lmbd_channels, alpha * data_channels * 2, bias=True),
            nn.LeakyReLU(inplace=True),
            nn.Linear(alpha * data_channels * 2, data_channels * 2 * self.num_affine, bias=True),
        )
        self.lmbd = None  # embedding
        self.lmbd_values = None
        self.is_fdbn = is_fdbn
        self.affine_comb_mode = affine_comb_mode

    def forward(self, x, a=None):
        assert self.lmbd is not None, f"lambda has not been set, yet."

        if self.lmbd.dim() < 2:
            self.lmbd = torch.unsqueeze(self.lmbd, 0)

        out = self.lmbd_encoder(self.lmbd)
        w, b = torch.split(out, [self.channels*self.num_affine]*2, dim=-1)
        if self.activation is not None:
            w, b = self.activation(w), self.activation(b)

        if self.is_fdbn:
            assert self.lmbd_values is not None, f"lambda has not been set, yet."
            clean_x, noise_x = torch.split(x, [self.channels] * 2, dim=1)
            w_c, w_n = torch.split(w, [self.channels] * 2, dim=1)
            b_c, b_n = torch.split(b, [self.channels] * 2, dim=1)
            if self.affine_comb_mode == 'af_lmbd':
                out = self._channel_wise_linear(clean_x, w_c, b_c, 1 - self.lmbd_values) + \
                      self._channel_wise_linear(noise_x, w_n, b_n, self.lmbd_values)
            elif self.affine_comb_mode == 'af':
                out = self._channel_wise_linear(clean_x, w_c, b_c, 0.5) + \
                      self._channel_wise_linear(noise_x, w_n, b_n, 0.5)
            elif self.affine_comb_mode == 'lmbd':
                out = self._channel_mul(clean_x, 1 - self.lmbd_values) \
                      + self._channel_mul(noise_x, self.lmbd_values)
            elif self.affine_comb_mode == 'afx':
                out = clean_x + \
                      self._channel_wise_linear(noise_x - clean_x, w_n, b_n, 1.)
            elif self.affine_comb_mode == 'afx_lmbd':
                out = clean_x + \
                      self._channel_wise_linear(noise_x - clean_x, w_n, b_n, self.lmbd_values)
            else:
                raise ValueError(f"Invalid affine_comb_mode: {self.affine_comb_mode}")
        else:
            out = self._channel_wise_linear(x, w, b)
        return out

    # def _channel_wise_linear(self, x, w, b):
    #     raise NotImplementedError("Use subclass that implements this.")

    def _channel_wise_linear(self, x, w, b, a=1.):
        if isinstance(a, torch.Tensor):
            a_shape = [1] * x.dim()
            a_shape[1] = -1  # channel
            a_shape[0] = a.size(0)
            a = a.view(*a_shape)

        shape = [1] * x.dim()
        shape[1] = -1  # channel
        shape[0] = w.size(0)
        if a != 1.:
            out = a * (x * w.view(*shape) + b.view(*shape))
        else:
            out = x * w.view(*shape) + b.view(*shape)
        return out

    def _channel_mul(self, x, a):
        if isinstance(a, torch.Tensor):
            a_shape = [1] * x.dim()
            a_shape[1] = -1  # channel
            a_shape[0] = a.size(0)
            a = a.view(*a_shape)
        return a * x


class FiLM2D(FiLMLayer):
    pass
    # def _channel_wise_linear(self, x, w, b):
    #     N, C, H, W = x.size()
    #     if w.size(0) == 1:
    #         z = x * w.view(1, C, 1, 1).expand_as(x) \
    #                   + b.view(1, C, 1, 1).expand_as(x)
    #     else:
    #         z = x * w.view(N, C, 1, 1).expand_as(x) \
    #                   + b.view(N, C, 1, 1).expand_as(x)
    #     return z


class FiLM1D(FiLMLayer):
    pass
    # def _channel_wise_linear(self, x, w, b):
    #     # N, C = x.size()
    #     if w.size(0) == 1:
    #         z = x * w.view(1, -1).expand_as(x) + b.view(1, -1).expand_as(x)
    #     else:
    #         z = x * w + b
    #     return z



class FilmDualNormLayer(nn.Module):
    """Dual BN layer with lambda conditioned FiLM for each BN."""
    _version = 2

    def __init__(self, num_features, track_running_stats=True, bn_class=None, affine=False,
                 **kwargs):
        if 'share_affine' in kwargs:
            assert not kwargs['share_affine']
            del kwargs['share_affine']
        # if 'affine' in kwargs:
        #     assert not kwargs['affine'], "Affine BN layer is not allowed. " \
        #                                  "Do not use BN with trainable params."
        #     del kwargs['affine']
        super().__init__()

        # dual BN
        if bn_class is None:
            bn_class = nn.BatchNorm2d
        self.bn_class = bn_class
        self.clean_bn = bn_class(num_features, track_running_stats=track_running_stats, affine=affine, **kwargs)
        self.noise_bn = bn_class(num_features, track_running_stats=track_running_stats, affine=affine, **kwargs)

        # condition variables
        self.clean_input = True  # indicate if the input is clean. This will be ignored at test.

    def forward(self, inp: torch.Tensor) -> torch.Tensor:
        if isinstance(self.clean_input, bool):
            if self.training:
                if self.clean_input:
                    self.clean_bn.train(True)
                    self.noise_bn.train(False)
                else:
                    self.clean_bn.train(False)
                    self.noise_bn.train(True)
            else:
                self.clean_bn.train(False)
                self.noise_bn.train(False)
        else:
            raise TypeError(f"Invalid self.clean_input: {type(self.clean_input)}")
        clean_out = self.clean_bn(inp)
        noise_out = self.noise_bn(inp)

        return torch.cat([clean_out, noise_out], dim=1)


class FilmDualBatchNorm2d(FilmDualNormLayer):
    def __init__(self, *args, **kwargs):
        super(FilmDualBatchNorm2d, self).__init__(*args, bn_class=nn.BatchNorm2d, **kwargs)


class FilmDualBatchNorm1d(FilmDualNormLayer):
    def __init__(self, *args, **kwargs):
        super(FilmDualBatchNorm1d, self).__init__(*args, bn_class=nn.BatchNorm1d, **kwargs)
