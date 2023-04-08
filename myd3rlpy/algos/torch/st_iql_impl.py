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
from d3rlpy.models.builders import create_non_squashed_normal_policy, create_value_function
from d3rlpy.models.encoders import EncoderFactory
from d3rlpy.models.optimizers import OptimizerFactory
from d3rlpy.models.q_functions import QFunctionFactory
from d3rlpy.models.torch.policies import squash_action
from d3rlpy.preprocessing import ActionScaler, RewardScaler, Scaler
from d3rlpy.torch_utility import TorchMiniBatch, soft_sync, train_api, torch_api
from d3rlpy.dataset import TransitionMiniBatch
from d3rlpy.algos.torch.iql_impl import IQLImpl

from myd3rlpy.algos.torch.st_impl import STImpl
from utils.utils import Struct


replay_name = ['observations', 'actions', 'rewards', 'next_observations', 'terminals', 'policy_actions', 'qs', 'phis', 'psis']
class STImpl(STImpl, IQLImpl):

    def _build_actor(self) -> None:
        self._policy = create_non_squashed_normal_policy(
            self._observation_shape,
            self._action_size,
            self._actor_encoder_factory,
            min_logstd=-5.0,
            max_logstd=2.0,
            use_std_parameter=True,
        )

    def _build_critic(self) -> None:
        super()._build_critic()
        self._value_func = create_value_function(self._observation_shape, self._value_encoder_factory)

    def _build_critic_optim(self) -> None:
        assert self._q_func is not None
        assert self._value_func is not None
        q_func_params = list(self._q_func.parameters())
        v_func_params = list(self._value_func.parameters())
        self._critic_optim = self._critic_optim_factory.create(
            q_func_params + v_func_params, lr=self._critic_learning_rate
        )

    def _compute_actor_loss(self, batch: TorchMiniBatch, clone_actor: bool=False, online: bool=False, replay: bool = False) -> torch.Tensor:
        assert self._policy

        # compute log probability
        dist = self._policy.dist(batch.observations)
        log_probs = dist.log_prob(batch.actions)

        # compute weight
        with torch.no_grad():
            weight = self._compute_weight(batch.observations, batch.actions, clone_actor=clone_actor, online=online, replay=replay)
        ret = -(weight * log_probs).mean()
        if not replay:
            self._log_probs = log_probs.mean()
        else:
            self._replay_log_probs = log_probs.mean()
        # if clone_actor:
        #     # compute log probability
        #     dist = self._clone_policy.dist(batch.observations)
        #     log_probs = dist.log_prob(batch.actions)

        #     # compute weight
        #     with torch.no_grad():
        #         weight = self._compute_weight(batch.observations, self._clone_policy(batch.actions), clone_actor=clone_actor, online=online)
        #     ret += -(weight * log_probs).mean()

        return ret

    def _compute_weight(self, observations, actions, clone_actor: bool=False, online: bool=False, replay=False) -> torch.Tensor:
        assert self._targ_q_func
        assert self._value_func
        q_t = self._targ_q_func(observations, actions, "min")
        v_t = self._value_func(observations)
        if clone_actor and not online and '_clone_value_func' in self.__dict__.keys():
            # clone_q_t = self._clone_q_func(observations, self._clone_policy(observations), "min")
            # v_t = torch.where(v_t > clone_q_t, v_t, clone_q_t)
            clone_v_t = self._clone_value_func(observations)
            v_t = torch.where(v_t > clone_v_t, v_t, clone_v_t)
        adv = q_t - v_t
        weight = (self._weight_temp * adv).exp().clamp(max=self._max_weight)
        if not replay:
            self._q_t = q_t.mean()
            self._v_t = v_t.mean()
            self._weight = weight.mean()
        else:
            self._replay_q_t = q_t.mean()
            self._replay_v_t = v_t.mean()
            self._replay_weight = weight.mean()
        return weight

    def compute_target(self, batch: TorchMiniBatch) -> torch.Tensor:
        assert self._value_func
        with torch.no_grad():
            v_t = self._value_func(batch.next_observations)
            return v_t

    def _compute_critic_loss(
        self, batch: TorchMiniBatch, q_tpn: torch.Tensor
    ) -> torch.Tensor:
        assert self._q_func is not None
        return self._q_func.compute_error(
            observations=batch.observations,
            actions=batch.actions,
            rewards=batch.rewards,
            target=q_tpn,
            terminals=batch.terminals,
            gamma=self._gamma**batch.n_steps,
        )

    def _compute_value_loss(self, batch: TorchMiniBatch, clone_critic=False, replay: bool = False) -> torch.Tensor:
        assert self._targ_q_func
        assert self._value_func
        q_t = self._targ_q_func(batch.observations, batch.actions, "min")
        v_t = self._value_func(batch.observations)
        diff = q_t.detach() - v_t
        if clone_critic and '_clone_value_func' in self.__dict__.keys() and '_clone_q_func' in self.__dict__.keys():
            clone_v_t = self._clone_value_func(batch.observations).detach()
            diff_clone = (clone_v_t - v_t)
            diff = torch.max(diff, diff_clone)
        # else:
        weight = (self._expectile - (diff < 0.0).float()).abs().detach()
        ret = (weight * (diff ** 2)).mean()
        return ret

    def compute_critic_loss(self, batch, q_tpn, clone_critic: bool=True, online: bool = False, replay=False, first_time=False):
        assert self._q_func is not None
        if not online:
            critic_loss = self._compute_critic_loss(batch, q_tpn)
            value_loss = self._compute_value_loss(batch, clone_critic=clone_critic, replay=replay)
            # if not replay:
            #     self._q_loss = critic_loss.mean()
            #     self._v_loss = value_loss.mean()
            # else:
            #     self._replay_q_loss = critic_loss.mean()
            #     self._replay_v_loss = value_loss.mean()
            return critic_loss + value_loss
        else:
            return self._compute_critic_loss(batch, q_tpn)

    def compute_generate_critic_loss(self, batch, clone_critic: bool=True):
        assert self._q_func is not None
        return self._compute_value_loss(batch, clone_critic=clone_critic)

    def compute_actor_loss(self, batch, clone_actor: bool = False, online: bool = False, replay: bool = False):
        return self._compute_actor_loss(batch, clone_actor=clone_actor, online=online, replay=replay)
