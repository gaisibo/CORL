from copy import deepcopy
import inspect
import types
import time
import math
import copy
from typing import Optional, Sequence, List, Any, Tuple, Dict, Union, cast

import numpy as np
import torch
from torch import nn
import torch.nn.functional as F

from d3rlpy.argument_utility import check_encoder
from d3rlpy.gpu import Device
from d3rlpy.models.encoders import EncoderFactory
from d3rlpy.models.optimizers import OptimizerFactory
from d3rlpy.models.q_functions import QFunctionFactory
from d3rlpy.models.torch.policies import squash_action
from d3rlpy.preprocessing import ActionScaler, RewardScaler, Scaler
from d3rlpy.torch_utility import TorchMiniBatch, soft_sync, train_api, torch_api
from d3rlpy.dataset import TransitionMiniBatch
from d3rlpy.algos.torch.sac_impl import SACImpl
from d3rlpy.models.builders import create_probabilistic_ensemble_dynamics_model

from myd3rlpy.models.builders import create_phi, create_psi
from myd3rlpy.algos.torch.gem import overwrite_grad, store_grad, project2cone2
from myd3rlpy.algos.torch.agem import project
from myd3rlpy.algos.torch.st_impl import STImpl
from myd3rlpy.algos.torch.st_sac_impl import STSACImpl
from myd3rlpy.algos.torch.co_deterministic_impl import CODeterministicImpl
from myd3rlpy.models.builders import create_parallel_continuous_q_function
from utils.utils import Struct


replay_name = ['observations', 'actions', 'rewards', 'next_observations', 'terminals', 'policy_actions', 'qs']
class STTD3Impl(STImpl, SACImpl):

    def _build_critic(self) -> None:
        self._q_func = create_parallel_continuous_q_function(
            self._observation_shape,
            self._action_size,
            n_ensembles=self._n_critics,
            reduction='min',
        )

    def compute_critic_loss(
        self, batch: TorchMiniBatch, q_tpn: torch.Tensor, clone_critic: bool = False, online: bool = False
    ) -> torch.Tensor:
        return super().compute_critic_loss(batch, q_tpn)

    def _compute_actor_loss(self, batch: TorchMiniBatch, clone_actor: bool = False, online: bool = False) -> torch.Tensor:
        assert self._policy is not None
        assert self._q_func is not None
        action = self._policy(batch.observations)
        q_t = self._q_func(batch.observations, action)[0]
        if clone_actor and not online:
            clone_q_t = self._q_func(batch.observations, action)[0]
            return - torch.where(q_t > clone_q_t, q_t, torch.zeros_like(q_t)).mean()
        return -q_t.mean()

    def compute_actor_loss(self, batch, clone_actor: bool = False, online: bool = False, replay: bool = False):
        loss = self._compute_actor_loss(batch, clone_actor=clone_actor, online=online)
        return loss

    def copy_from_sac(self, sac_impl: STSACImpl):
        self._q_func = sac_impl._q_func
        self._policy = sac_impl._policy
        self._targ_q_func = sac_impl._targ_q_func
        self._targ_policy = sac_impl._targ_policy
        self._actor_optim = sac_impl._actor_optim
        self._critic_optim = sac_impl._critic_optim