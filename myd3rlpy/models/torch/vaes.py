from abc import ABCMeta, abstractmethod
from typing import List, Optional, Sequence, Tuple

import torch
import torch.nn.functional as F
from torch import nn


class VAE(nn.Module, metaclass=ABCMeta):
    @abstractmethod
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        pass

    @abstractmethod
    def encode(self, x: torch.Tensor) -> torch.Tensor:
        pass

    @abstractmethod
    def decode(self, h: torch.Tensor) -> torch.Tensor:
        pass

    @abstractmethod
    def generate(self, num: int, x: torch.Tensor) -> torch.Tensor:
        pass

    # @abstractmethod
    # def get_feature_size(self) -> int:
    #     pass

    # @property
    # def observation_shape(self) -> Sequence[int]:
    #     pass

    # @abstractmethod
    # def __call__(self, x: torch.Tensor) -> torch.Tensor:
    #     pass

    # @property
    # def encoder_last_layer(self) -> nn.Linear:
    #     raise NotImplementedError

    # @property
    # def decoder_last_layer(self) -> nn.Linear:
    #     raise NotImplementedError

class VectorVAE(VAE):

    _observation_shape: Sequence[int]
    _use_batch_norm: bool
    _dropout_rate: Optional[float]
    _use_dense: bool
    _activation: nn.Module
    _feature_size: int
    _fcs: nn.ModuleList
    _bns: nn.ModuleList
    _dropouts: nn.ModuleList

    def __init__(
        self,
        observation_shape: Sequence[int],
        feature_size: int,
        hidden_units: Optional[Sequence[int]] = None,
        use_batch_norm: bool = False,
        dropout_rate: Optional[float] = None,
        use_dense: bool = False,
        activation: nn.Module = nn.ReLU(),
    ):
        super().__init__()
        self._observation_shape = observation_shape
        self._feature_size = feature_size

        if hidden_units is None:
            hidden_units = [256, 256]

        self._use_batch_norm = use_batch_norm
        self._dropout_rate = dropout_rate
        self._activation = activation
        self._use_dense = use_dense

        encoder_in_units = [observation_shape[0]] + list(hidden_units[:-1])
        encoder_out_units = list(hidden_units)
        self._mu = nn.Linear(hidden_units[-1], feature_size)
        self._logvar = nn.Linear(hidden_units[-1], feature_size)
        self._encoder_fcs = nn.ModuleList()
        self._encoder_bns = nn.ModuleList()
        self._encoder_dropouts = nn.ModuleList()
        for i, (in_unit, out_unit) in enumerate(zip(encoder_in_units, encoder_out_units)):
            if use_dense and i > 0:
                in_unit += observation_shape[0]
            self._encoder_fcs.append(nn.Linear(in_unit, out_unit))
            if use_batch_norm:
                self._encoder_bns.append(nn.BatchNorm1d(out_unit))
            if dropout_rate is not None:
                self._encoder_dropouts.append(nn.Dropout(dropout_rate))

        decoder_in_units = list(reversed(hidden_units))
        decoder_out_units = list(reversed(hidden_units))[1:] + [observation_shape[0]]
        self._decoder_fcs = nn.ModuleList()
        self._decoder_bns = nn.ModuleList()
        self._decoder_dropouts = nn.ModuleList()
        for i, (in_unit, out_unit) in enumerate(zip(decoder_in_units, decoder_out_units)):
            self._decoder_fcs.append(nn.Linear(in_unit, out_unit))
            if use_batch_norm:
                self._decoder_bns.append(nn.BatchNorm1d(out_unit))
            if dropout_rate is not None:
                self._decoder_dropouts.append(nn.Dropout(dropout_rate))
        self._latent_mapping = nn.Linear(feature_size, hidden_units[-1])

    def encode(self, x: torch.Tensor) -> Sequence[torch.Tensor]:
        h = x
        for i, fc in enumerate(self._encoder_fcs):
            if self._use_dense and i > 0:
                h = torch.cat([h, x], dim=1)
            h = self._activation(fc(h))
            if self._use_batch_norm:
                h = self._encoder_bns[i](h)
            if self._dropout_rate is not None:
                h = self._encoder_dropouts[i](h)
        mu, logvar = self._mu(h), self._logvar(h)
        return mu, logvar

    def sample_z(self, mu, logvar):
        eps = torch.randn_like(mu)
        return mu + eps * torch.exp(0.5 * logvar)

    def decode(self, z: torch.Tensor) -> torch.Tensor:
        h = self._latent_mapping(z)
        for i, fc in enumerate(self._decoder_fcs):
            h = self._activation(fc(h))
            if self._use_batch_norm:
                h = self._decoder_bns[i](h)
            if self._dropout_rate is not None:
                h = self._decoder_dropouts[i](h)
        return h

    @property
    def observation_shape(self) -> Sequence[int]:
        return self._observation_shape

    def forward(self, x: torch.Tensor) -> Sequence[torch.Tensor]:
        mu, logvar = self.encode(x)
        z = self.sample_z(mu, logvar)
        x_ = self.decode(z)
        return x_, mu, logvar

    def generate(self, num: int, x: torch.Tensor) -> torch.Tensor:
        z = torch.randn(num, self._feature_size).to(x.device).to(x.dtype)
        x_ = self.decode(z)
        return x_
