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
from d3rlpy.algos.torch.sac_impl import SACImpl
from myd3rlpy.algos.torch.st_impl import STImpl


class STImpl(STImpl, SACImpl):

    _policy: Optional[SquashedNormalPolicy]
    _targ_policy: Optional[SquashedNormalPolicy]
    _temp_learning_rate: float
    _temp_optim_factory: OptimizerFactory
    _initial_temperature: float
    _log_temp: Optional[Parameter]
    _temp_optim: Optional[Optimizer]

    def _build_actor(self) -> None:
        self._policy = create_squashed_normal_policy(
            self._observation_shape,
            self._action_size,
            self._actor_encoder_factory,
            use_std_parameter=True
        )

    def compute_critic_loss(self, batch, q_tpn, clone_critic: bool=False, online: bool=False, replay=False, first_time=False):
        return super().compute_critic_loss(batch, q_tpn)

    def compute_actor_loss(self, batch, clone_actor: bool=False, online: bool=False):
        return super().compute_actor_loss(batch)
