import copy
import math
from typing import Optional, Sequence, Tuple

import numpy as np
import torch
from torch import nn
from torch.optim import Optimizer
from d3rlpy.gpu import Device
from d3rlpy.models.builders import create_continuous_q_function, create_squashed_normal_policy
from d3rlpy.torch_utility import TorchMiniBatch, soft_sync, train_api, torch_api
from d3rlpy.models.encoders import EncoderFactory
from d3rlpy.models.optimizers import OptimizerFactory
from d3rlpy.models.q_functions import QFunctionFactory
from d3rlpy.models.torch import (
    CategoricalPolicy,
    EnsembleDiscreteQFunction,
    EnsembleQFunction,
    Parameter,
    Policy,
    SquashedNormalPolicy,
)
from myd3rlpy.algos.torch.st_sacn_impl import STImpl as SACNImpl


class STImpl(SACNImpl):

    _policy: Optional[SquashedNormalPolicy]
    _targ_policy: Optional[SquashedNormalPolicy]
    _temp_learning_rate: float
    _temp_optim_factory: OptimizerFactory
    _initial_temperature: float
    _log_temp: Optional[Parameter]
    _temp_optim: Optional[Optimizer]

    def __init__(self, eta, **kwargs):
        super().__init__(
            **kwargs
        )
        self._eta = eta

    def compute_critic_loss(self, batch, q_tpn, clone_critic: bool=False, online: bool=False, replay=False, first_time=False):
        q_loss = super().compute_critic_loss(batch, q_tpn)

        obs_tile = batch.observations.unsqueeze(dim=0).repeat(self._n_ensemble, 1, 1)
        actions_tile = batch.actions.unsqueeze(dim=0).repeat(self._n_ensemble, 1, 1).requires_grad_(True)
        qs_preds_tile = self._q_func(obs_tile, actions_tile)
        qs_pred_grads, = torch.autograd.grad(qs_preds_tile.sum(), actions_tile, retain_graph=True, create_graph=True)
        qs_pred_grads = qs_pred_grads / (torch.norm(qs_pred_grads, p=2, dim=2).unsqueeze(-1) + 1e-10)
        qs_pred_grads = qs_pred_grads.transpose(0, 1)

        qs_pred_grads = torch.einsum('bik,bjk->bij', qs_pred_grads, qs_pred_grads)
        masks = torch.eye(self._n_ensemble, device=self.device).unsqueeze(dim=0).repeat(qs_pred_grads.size(0), 1, 1)
        qs_pred_grads = (1 - masks) * qs_pred_grads
        grad_loss = torch.mean(torch.sum(qs_pred_grads, dim=(1, 2))) / (self._n_ensemble - 1)

        q_loss += self._eta * grad_loss
        return q_loss
