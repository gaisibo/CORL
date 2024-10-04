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


replay_name = ['observations', 'actions', 'rewards', 'next_observations', 'terminals', 'policy_actions', 'qs']
class STImpl(STImpl, CQLImpl):
    def __init__(self, std_time=1, std_type='clamp', **kwargs):
        super().__init__(
            **kwargs
        )
        self._std_threshold = 0
        self._std_time = std_time
        self._std_type = std_type
        if self._std_type != 'none':
            self._qs_stds = []

    def compute_critic_loss(
        self, batch: TorchMiniBatch, q_tpn: torch.Tensor, clone_critic: bool = False, online: bool = False, replay: bool=False, first_time = False
    ) -> torch.Tensor:
        assert self._q_func is not None
        loss = self._q_func.compute_error(
            observations=batch.observations,
            actions=batch.actions,
            rewards=batch.rewards,
            target=q_tpn,
            terminals=batch.terminals,
            gamma=self._gamma**batch.n_steps,
        )
        if online:
            return loss.mean()
        # if clone_critic:
        #     action = self._policy(batch.observations)
        #     q_t = self._q_func(batch.observations, action, "min")
        #     clone_action = self._clone_policy.sample(batch.observations)
        #     clone_q_t = self._clone_q_func(batch.observations, clone_action, "min")
        #     conservative_loss = self._compute_conservative_loss(
        #         batch.observations, batch.actions, batch.next_observations
        #     )
        #     loss += conservative_loss
        #     loss = torch.where(q_t > clone_q_t, loss, torch.zeros_like(loss)).mean()
        # else:
        conservative_loss = self._compute_conservative_loss(
            batch.observations, batch.actions, batch.next_observations
        )
        loss += conservative_loss.mean()
        return loss

    def compute_actor_loss(self, batch: TorchMiniBatch, clone_actor=False, replay: bool=False, online: bool=False) -> torch.Tensor:
    # def compute_actor_loss(self, batch: TorchMiniBatch, online: bool=False) -> torch.Tensor:
        assert self._policy is not None
        assert self._q_func is not None
        action, log_prob = self._policy.sample_with_log_prob(batch.observations)
        entropy = self._log_temp().exp() * log_prob
        q_t = self._q_func(batch.observations, action, "min")
        new_q_t = self._q_func(batch.observations, batch.actions, "min")
        if clone_actor and self._clone_policy is not None:
            clone_action, clone_log_prob = self._clone_policy.sample_with_log_prob(batch.observations)
            clone_q_t = self._clone_q_func(batch.observations, clone_action, "min")
            loss = (entropy - q_t)
            adv = q_t - clone_q_t
            weight = (3.0 * adv).exp().clamp(max=100.0)
            loss *= weight
            loss = torch.where(new_q_t > clone_q_t, loss, torch.zeros_like(loss)).mean()
        else:
            loss = (entropy - q_t).mean()
        return loss

    def _build_actor(self) -> None:
        self._policy = create_squashed_normal_policy(
            self._observation_shape,
            self._action_size,
            self._actor_encoder_factory,
            min_logstd=-20.0,
            max_logstd=2.0,
            use_std_parameter=False,
        )

    def _compute_conservative_loss(
        self, obs_t: torch.Tensor, act_t: torch.Tensor, obs_tp1: torch.Tensor
    ) -> torch.Tensor:
        assert self._policy is not None
        assert self._q_func is not None
        assert self._log_alpha is not None

        policy_values_t = self._compute_policy_is_values(obs_t, obs_t)
        policy_values_tp1 = self._compute_policy_is_values(obs_tp1, obs_t)
        random_values = self._compute_random_is_values(obs_t)

        # compute logsumexp
        # (n critics, batch, 3 * n samples) -> (n critics, batch, 1)
        target_values = torch.cat(
            [policy_values_t, policy_values_tp1, random_values], dim=2
        )
        logsumexp = torch.logsumexp(target_values, dim=2, keepdim=True)

        # estimate action-values for data actions
        data_values = self._q_func(obs_t, act_t, "none")

        loss = logsumexp.mean(dim=0).mean(dim=-1) - data_values.mean(dim=0).mean(dim=-1)
        scaled_loss = self._conservative_weight * loss

        # clip for stability
        clipped_alpha = self._log_alpha().exp().clamp(0, 1e6)[0][0]

        return clipped_alpha * (scaled_loss - self._alpha_threshold)
