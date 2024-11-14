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
from d3rlpy.torch_utility import TorchMiniBatch, soft_sync, train_api, torch_api, eval_api
from d3rlpy.dataset import TransitionMiniBatch
from d3rlpy.algos.torch.iql_impl import IQLImpl

from myd3rlpy.algos.torch.st_impl import STImpl
from myd3rlpy.models.builders import create_parallel_continuous_q_function
from myd3rlpy.torch_utility import torch_api
from utils.utils import Struct
from utils.networks import ParallelizedEnsembleFlattenMLP


replay_name = ['observations', 'actions', 'rewards', 'next_observations', 'terminals', 'policy_actions', 'qs']
class STIQLNImpl(STImpl, IQLImpl):
    def __init__(
        self,
        n_ensemble = 10,
        std_time = 1,
        std_type = 'clamp',
        expectile_max = 0.7,
        expectile_min = 0.7,
        entropy_time = 0.2,
        **kwargs
    ):
        super().__init__(
            **kwargs
        )
        self._n_ensemble = n_ensemble
        self._test_i = 0
        self._str = ''
        self._std_time = std_time
        self._std_type = std_type
        self._expectile_max = expectile_max
        self._expectile_min = expectile_min
        self._entropy_time = entropy_time

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
        # self._value_func = nn.ModuleList([create_value_function(self._observation_shape, self._value_encoder_factory) for _ in range(self._n_ensemble)])
        self._q_func = create_parallel_continuous_q_function(
            self._observation_shape,
            self._action_size,
            n_ensembles=self._n_critics,
            reduction='min',
        )
        self._value_func = ParallelizedEnsembleFlattenMLP(self._n_ensemble, [256, 256], self._observation_shape[0], 1, device=self.device)

    def _build_critic_optim(self) -> None:
        assert self._q_func is not None
        assert self._value_func is not None
        q_func_params = list(self._q_func.parameters())
        v_func_params = list(self._value_func.parameters())
        self._critic_optim = self._critic_optim_factory.create(
            q_func_params + v_func_params, lr=self._critic_learning_rate
        )

    def _compute_actor_loss(self, batch: TorchMiniBatch, clone_actor: bool=False, online: bool=False, replay:bool = False) -> torch.Tensor:
        assert self._policy

        # compute log probability
        dist = self._policy.dist(batch.observations)
        log_probs = dist.log_prob(batch.actions)

        # compute weight
        with torch.no_grad():
            weight = self._compute_weight(batch.observations, batch.actions)
        ret = -(weight * log_probs).mean()
        if clone_actor and not online and self._clone_policy is not None:
            # compute log probability
            dist = self._clone_policy.dist(batch.observations)
            log_probs = dist.log_prob(self._clone_policy(batch.observations))

            # compute weight
            with torch.no_grad():
                weight = self._compute_weight(batch.observations, self._clone_policy(batch.observations))
            ret_clone = -(weight * log_probs).mean()
            ret += ret_clone

        return ret

    def _compute_weight(self, observations, actions) -> torch.Tensor:
        assert self._targ_q_func
        assert self._value_func
        q_t = self._targ_q_func(observations, actions, "min")
        # v_t = []
        # for value_func in self._value_func:
        #     v_t.append(value_func(observations))
        # v_t, _ = torch.min(torch.stack(v_t, dim=0), dim=0)
        v_t, _ = torch.min(self._value_func(observations), dim=0)
        adv = q_t - v_t
        weight = (self._weight_temp * adv).exp().clamp(max=self._max_weight)

        return weight
        # self._test_i += 1
        # self._str += str(q_t.mean()) + ' ' + str(v_t.mean()) + ' ' + str(adv.max()) + ' ' + str(adv.min()) + '\n'
        # if self._test_i > 3000:
        #     raise NotImplementedError(self._str)

    def compute_target(self, batch: TorchMiniBatch) -> torch.Tensor:
        assert self._value_func
        with torch.no_grad():
            # v_t = []
            # for value_func in self._value_func:
            #     v_t.append(value_func(batch.next_observations))
            # v_t, _ = torch.min(torch.stack(v_t, dim=0), dim=0)
            v_t, _ = torch.min(self._value_func(batch.next_observations), dim=0)
            return v_t

    def _compute_critic_loss(
        self, batch: TorchMiniBatch, q_tpn: torch.Tensor
    ) -> torch.Tensor:
        assert self._q_func is not None
        error = self._q_func.compute_error(
            observations=batch.observations,
            actions=batch.actions,
            rewards=batch.rewards,
            target=q_tpn,
            terminals=batch.terminals,
            gamma=self._gamma**batch.n_steps,
        )
        # self._str += str((self._q_func(batch.observations, batch.actions) + batch.rewards - q_tpn * self._gamma).mean()) + ' ' + str(error.mean()) + '\n'
        return error

    def _compute_value_loss(self, batch: TorchMiniBatch, clone_critic=False, replay=False, first_time=False) -> torch.Tensor:
        assert self._targ_q_func
        assert self._value_func
        q_t = self._targ_q_func(batch.observations, batch.actions, "min")
        v_ts = self._value_func(batch.observations)
        v_ts_std = torch.std(v_ts, dim=0)
        diff = q_t.detach().expand(self._n_ensemble, -1, -1) - v_ts
        weight = (self._expectile - (diff < 0.0).float()).abs().detach()
        ret = torch.mean(torch.sum(weight * (diff ** 2), dim=0))
        if self._entropy_time != 0:
            entropy = 0.5 * torch.log(2 * math.pi * math.e * v_ts_std)
            ret -= torch.mean(entropy) / self._entropy_time
        # if clone_critic and '_clone_value_func' in self.__dict__.keys() and '_clone_q_func' in self.__dict__.keys():
        #     ret += (weight * (diff_clone ** 2)).mean()
        # else:
        return ret

    def compute_critic_loss(self, batch, q_tpn, clone_critic: bool=True, online: bool = False, replay=False, first_time=False):
        assert self._q_func is not None
        critic_loss = self._compute_critic_loss(batch, q_tpn)
        value_loss = self._compute_value_loss(batch, clone_critic=clone_critic, replay=replay, first_time=False)
        self._q_loss = critic_loss.mean()
        self._v_loss = value_loss.mean()
        return critic_loss + value_loss

    def compute_generate_critic_loss(self, batch, clone_critic: bool=True):
        assert self._q_func is not None
        return self._compute_value_loss(batch, clone_critic=clone_critic)

    def compute_actor_loss(self, batch, clone_actor: bool = False, online: bool = False, replay=False):
        return self._compute_actor_loss(batch, clone_actor=clone_actor, online=online, replay=replay)

    #@eval_api
    #@torch_api(scaler_targets=["x"])
    #def _sample_action(self, x: torch.Tensor) -> torch.Tensor:
    #    action = super()._sample_action(x)
    #    noise = torch.randn_like(action) * self._policy_noise
    #    action = torch.clamp(action + noise, -1.0, 1.0)
    #    return action#.cpu().detach().numpy()
