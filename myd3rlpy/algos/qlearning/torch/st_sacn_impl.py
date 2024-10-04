import copy
import math
from typing import Optional, Sequence, Tuple

import numpy as np
import torch
from torch import nn
import torch.nn.functional as F
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

from myd3rlpy.algos.torch.st_sac_impl import STSACImpl
from utils.networks import ParallelizedEnsembleFlattenMLP


class STSACNImpl(STSACImpl):

    def compute_critic_loss(self, batch, q_tpn, clone_critic: bool=False, online: bool=False, replay=False, first_time=False):
        value, _ = torch.min(self._q_func(batch.observations, batch.actions), dim=0)
        y = batch.rewards + self._gamma * q_tpn * (1 - batch.terminals)
        loss = F.mse_loss(value, y, reduction="mean")
        return loss

    def compute_target(self, batch: TorchMiniBatch) -> torch.Tensor:
        with torch.no_grad():
            # v_t = []
            # for value_func in self._value_func:
            #     v_t.append(value_func(batch.next_observations))
            # v_t, _ = torch.min(torch.stack(v_t, dim=0), dim=0)
            action, log_prob = self._policy.sample_with_log_prob(batch.next_observations)
            q_t, _ = torch.min(self._targ_q_func(batch.next_observations, action), dim=0)
            return q_t

    def compute_actor_loss(self, batch: TorchMiniBatch, clone_actor=False, online: bool=False, replay=False) -> torch.Tensor:
    # def compute_actor_loss(self, batch: TorchMiniBatch, online: bool=False) -> torch.Tensor:
        assert self._policy is not None
        assert self._q_func is not None
        action, log_prob = self._policy.sample_with_log_prob(batch.observations)
        entropy = self._log_temp().exp() * log_prob
        q_t, _ = torch.min(self._q_func(batch.observations, action), dim=0)
        loss = (entropy - q_t).mean()
        return loss
