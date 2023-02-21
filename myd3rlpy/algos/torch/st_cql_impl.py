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
from d3rlpy.models.builders import create_squashed_normal_policy, create_parameter, create_continuous_q_function
from d3rlpy.models.encoders import EncoderFactory
from d3rlpy.models.optimizers import OptimizerFactory
from d3rlpy.models.q_functions import QFunctionFactory
from d3rlpy.models.torch.policies import squash_action
from d3rlpy.preprocessing import ActionScaler, RewardScaler, Scaler
from d3rlpy.torch_utility import TorchMiniBatch, soft_sync, train_api, torch_api
from d3rlpy.dataset import TransitionMiniBatch
from d3rlpy.algos.torch.cql_impl import CQLImpl

from myd3rlpy.algos.torch.st_impl import STImpl
from myd3rlpy.algos.torch.gem import overwrite_grad, store_grad, project2cone2
from myd3rlpy.algos.torch.agem import project
from utils.utils import Struct


replay_name = ['observations', 'actions', 'rewards', 'next_observations', 'terminals', 'policy_actions', 'qs', 'phis', 'psis']
class STImpl(STImpl, CQLImpl):

    def compute_critic_loss(
            self, batch: TorchMiniBatch, q_tpn: torch.Tensor, clone_critic: bool = False, online: bool = False
    ) -> torch.Tensor:
        assert self._q_func is not None
        td_sum = 0
        for q_func in self._q_func._q_funcs:
            loss = q_func.compute_error(
                observations=batch.observations,
                actions=batch.actions,
                rewards=batch.rewards,
                target=q_tpn,
                terminals=batch.terminals,
                gamma=self._gamma ** batch.n_steps,
                reduction="none",
            )
            td_sum += loss.mean(dim=1)
        loss = td_sum
        if online:
            return loss.mean()
        if clone_critic:
            action = self._policy(batch.observations)
            q_t = self._q_func(batch.observations, action, "min")
            clone_action = self._clone_policy.sample(batch.observations)
            clone_q_t = self._clone_q_func(batch.observations, clone_action, "min")
            loss = torch.where(clone_q_t > q_t, loss, torch.zeros_like(loss)).mean()
        else:
            loss = loss.mean()

        conservative_loss = self._compute_conservative_loss(
            batch.observations, batch.actions, batch.next_observations
        )
        return loss + conservative_loss

    def compute_actor_loss(self, batch: TorchMiniBatch, clone_actor=False, online: bool=False) -> torch.Tensor:
    # def compute_actor_loss(self, batch: TorchMiniBatch, online: bool=False) -> torch.Tensor:
        assert self._policy is not None
        assert self._q_func is not None
        action, log_prob = self._policy.sample_with_log_prob(batch.observations)
        entropy = self._log_temp().exp() * log_prob
        q_t = self._q_func(batch.observations, action, "min")
        if clone_actor:
            clone_action, clone_log_prob = self._clone_policy.sample_with_log_prob(batch.observations)
            clone_q_t = self._clone_q_func(batch.observations, clone_action, "min")
            loss = (entropy - q_t)
            loss = torch.where(clone_q_t > q_t, loss, torch.zeros_like(loss)).mean()
        else:
            loss = (entropy - q_t).mean()
        return loss

    # def reinit_network(self):
    #     initial_val = math.log(self._initial_temperature)
    #     self._log_temp = create_parameter((1, 1), initial_val)
    #     initial_val = math.log(self._initial_alpha)
    #     self._log_alpha = create_parameter((1, 1), initial_val)
    #     if self._use_gpu:
    #         self.to_gpu(self._use_gpu)
    #     else:
    #         self.to_cpu()
    #     self._actor_optim = self._actor_optim_factory.create(
    #         self._policy.parameters(), lr=self._actor_learning_rate
    #     )
    #     if self._temp_learning_rate > 0:
    #         self._temp_optim = self._temp_optim_factory.create(
    #             self._log_temp.parameters(), lr=self._temp_learning_rate
    #         )
    #     self._critic_optim = self._critic_optim_factory.create(
    #         self._q_func.parameters(), lr=self._critic_learning_rate
    #     )
    #     if self._alpha_learning_rate > 0:
    #         self._alpha_optim = self._alpha_optim_factory.create(
    #             self._log_alpha.parameters(), lr=self._alpha_learning_rate
    #         )

    def _build_actor(self) -> None:
        self._policy = create_squashed_normal_policy(
            self._observation_shape,
            self._action_size,
            self._actor_encoder_factory,
            use_std_parameter=True
        )
