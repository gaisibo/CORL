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
from functorch import make_functional

from d3rlpy.argument_utility import check_encoder
from d3rlpy.gpu import Device
from d3rlpy.models.builders import create_non_squashed_normal_policy, create_value_function
from d3rlpy.models.encoders import EncoderFactory
from d3rlpy.models.optimizers import OptimizerFactory
from d3rlpy.models.q_functions import QFunctionFactory
from d3rlpy.models.torch.policies import squash_action
from d3rlpy.models.torch.distributions import SquashedGaussianDistribution, GaussianDistribution
from d3rlpy.preprocessing import ActionScaler, RewardScaler, Scaler
from d3rlpy.torch_utility import TorchMiniBatch, soft_sync, train_api, torch_api
from d3rlpy.dataset import TransitionMiniBatch
from d3rlpy.algos.torch.iql_impl import IQLImpl

from myd3rlpy.algos.torch.fs_impl import FSImpl
from utils.utils import Struct


replay_name = ['observations', 'actions', 'rewards', 'next_observations', 'terminals', 'policy_actions', 'qs', 'phis', 'psis']
class FSImpl(FSImpl, IQLImpl):

    def __init__(
        self,
        **kwargs
    ):
        super().__init__(
            **kwargs
        )
        self._inner = True

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

    def calc_with_state_dict(self, x, network, state_dict):
        network_function, network_params = make_functional(network)
        # for key, value in state_dict.items():
        #     print(f"{key}: {torch.mean(value)=}")
        # print(f"{torch.mean(x[0])=}")
        # for name, param in state_dict.items():
        #     print(f"{name}: {torch.mean(param)=}")
        return network_function(state_dict.values(), *x)

        # # print(inspect.getsource(network.forward))
        # temp_state_dict = {}
        # for n, p in state_dict.items():
        #     names = n.split('.')
        #     name_str = 'network'
        #     for name in names:
        #         if '0' <= name <= '9':
        #             name_str += f'[{name}]'
        #         else:
        #             name_str += f'.{name}'
        #     temp_state_dict[n] = eval(name_str)
        #     name_str += ' = state_dict[n]'
        #     network._encoder._fcs[0].weight = cast(nn.Parameter, state_dict[n])
        #     eval(name_str)
        #     assert False
        # network(*x)
        # for n, p in state_dict.items():
        #     names = n.split('.')
        #     name_str = 'network'
        #     for name in names:
        #         if '0' <= name <= '9':
        #             name_str += f'[{name}]'
        #         else:
        #             name_str += f'.{name}'
        #     name_str += ' = temp_state_dict[n]'
        #     eval(name_str)
        # assert False

    def policy_dist(self, observations):
        assert self._policy
        if not hasattr(self, '_actor_state_dict'):
            return self._policy.dist(observations)
        else:
            # action, log_std = self.calc_with_state_dict((observations, True, True), self._policy, self._actor_state_dict)
            # dist = GaussianDistribution(action, torch.exp(log_std))
            forward_tmp = self._policy.forward
            self._policy.forward = self._policy.dist
            dist = self.calc_with_state_dict((observations, ), self._policy, self._actor_state_dict)
            self._policy.forward = forward_tmp
            # dist = GaussianDistribution(action, torch.exp(log_std))
            return dist

    def q_(self, observations, actions, mix='min'):
        assert self._q_func
        if not hasattr(self, '_critic_state_dict'):
            return self._q_func(observations, actions, mix)
        else:
            return self.calc_with_state_dict((observations, actions, mix), self._q_func, self._critic_state_dict)

    def targ_q_(self, observations, actions, mix='min'):
        assert self._targ_q_func
        if not hasattr(self, '_targ_critic_state_dict'):
            return self._targ_q_func(observations, actions, mix)
        else:
            return self.calc_with_state_dict((observations, actions, mix), self._targ_q_func, self._targ_critic_state_dict)

    def v_(self, observations):
        assert self._value_func
        if not hasattr(self, '_value_state_dict'):
            return self._value_func(observations)
        else:
            return self.calc_with_state_dict((observations,), self._value_func, self._value_state_dict)

    def _compute_actor_loss(self, batch: TorchMiniBatch, state_dict=None) -> torch.Tensor:

        # compute log probability
        dist = self.policy_dist(batch.observations)
        log_probs = dist.log_prob(batch.actions)

        # compute weight
        with torch.no_grad():
            weight = self._compute_weight(batch.observations, batch.actions)
        if self._inner:
            ret = -(weight * log_probs).mean()
        else:
            ret = -(log_probs).mean()

        return ret, torch.mean(weight), torch.mean(log_probs)

    def _compute_weight(self, observations, actions) -> torch.Tensor:
        q_t = self.targ_q_(observations, actions, "min")
        v_t = self.v_(observations)
        adv = q_t - v_t
        weight = (self._weight_temp * adv).exp().clamp(max=self._max_weight)
        return weight

    def compute_target(self, batch: TorchMiniBatch) -> torch.Tensor:
        with torch.no_grad():
            v_t = self.v_(batch.next_observations)
            return v_t

    def _compute_critic_loss(
        self, batch: TorchMiniBatch, q_tpn: torch.Tensor
    ) -> torch.Tensor:
        value = self.q_(batch.observations, batch.actions, "mean")
        y = batch.rewards + self._gamma * q_tpn * (1 - batch.terminals)
        loss = F.mse_loss(value, y)
        return loss, torch.mean(value), torch.mean(y)

    def _compute_value_loss(self, batch: TorchMiniBatch) -> torch.Tensor:
        q_t = self.targ_q_(batch.observations, batch.actions, "min")
        v_t = self.v_(batch.observations)
        diff = q_t.detach() - v_t
        # if clone_critic and '_clone_value_func' in self.__dict__.keys() and '_clone_q_func' in self.__dict__.keys():
        #     clone_v_t = self._clone_value_func(batch.observations).detach()
        #     diff_clone = (clone_v_t - v_t)
        #     diff = torch.max(diff, diff_clone)
        # else:
        weight = (self._expectile - (diff < 0.0).float()).abs().detach()
        ret = (weight * (diff ** 2)).mean()
        return ret, torch.mean(diff)

    def compute_critic_loss(self, batch, q_tpn):
        critic_loss, value, y = self._compute_critic_loss(batch, q_tpn)
        value_loss, diff = self._compute_value_loss(batch)
        return critic_loss, value_loss, value, y, diff

    def compute_generate_critic_loss(self, batch):
        return self._compute_value_loss(batch)

    def compute_actor_loss(self, batch, state_dict=None):
        return self._compute_actor_loss(batch, state_dict)
