
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
from d3rlpy.models.encoders import EncoderFactory
from d3rlpy.models.optimizers import OptimizerFactory
from d3rlpy.models.q_functions import QFunctionFactory
from d3rlpy.models.torch.policies import squash_action
from d3rlpy.preprocessing import ActionScaler, RewardScaler, Scaler
from d3rlpy.torch_utility import TorchMiniBatch, soft_sync, train_api, torch_api
from d3rlpy.dataset import TransitionMiniBatch
from d3rlpy.algos.torch.td3_plus_bc_impl import TD3PlusBCImpl
from d3rlpy.models.builders import create_probabilistic_ensemble_dynamics_model

from myd3rlpy.algos.torch.gem import overwrite_grad, store_grad, project2cone2
from myd3rlpy.algos.torch.agem import project
# from myd3rlpy.algos.torch.co_deterministic_impl import CODeterministicImpl
from myd3rlpy.algos.torch.fs_impl import FSImpl
from utils.utils import Struct


replay_name = ['observations', 'actions', 'rewards', 'next_observations', 'terminals', 'policy_actions', 'qs']
class FSTD3PlusBCImpl(FSImpl, TD3PlusBCImpl):

    def __init__(
        self,
        **kwargs
    ):
        super().__init__(
            **kwargs
        )

    def calc_with_state_dict(self, x, network, state_dict):
        network_function, network_params = make_functional(network)
        return network_function(state_dict.values(), *x)

    def p_(self, observations):
        assert self._policy
        if not hasattr(self, '_actor_state_dict'):
            return self._policy(observations)
        else:
            return self.calc_with_state_dict((observations, ), self._policy, self._actor_state_dict)

    def q_(self, observations, actions, mix='min'):
        assert self._q_func
        if not hasattr(self, '_critic_state_dict'):
            return self._q_func(observations, actions, mix)
        else:
            return self.calc_with_state_dict((observations, actions, mix), self._q_func, self._critic_state_dict)

    def _targ_q(self, next_observations, clipped_action, reduction):
        assert self._targ_q_func
        if not hasattr(self, '_critic_state_dict'):
            return self._q_func.compute_target(next_observations, clipped_action, reduction)
        else:
            tmp_save = self._q_func.forward
            self._q_func.forward = self._q_func.compute_target
            ret = self.calc_with_state_dict((next_observations, clipped_action, reduction), self._q_func, self._critic_state_dict)
            self._q_func.forward = tmp_save
            return ret

    def _compute_actor_loss(self, batch: TorchMiniBatch, clone_actor: bool=False, online: bool=False) -> torch.Tensor:
        assert self._policy is not None
        assert self._q_func is not None
        action = torch.tanh(self.p_(batch.observations))
        q_t = self.q_(batch.observations, action, "none")[0]
        if online:
            return -q_t.mean()
        # if clone_actor:
        #     clone_actions = self._clone_policy(batch.observations)
        #     clone_q_t = self._clone_q_func(batch.observations, clone_actions, "none")[0]
        #     new_q_t = self._q_func(batch.observations, batch.actions, "none")[0]
        #     max_q_t = new_q_t > clone_q_t
        #     select_q_t = q_t[max_q_t]
        #     select_action = action[max_q_t]
        #     select_batch_action = batch.actions[max_q_t]
        # else:
        select_q_t = q_t
        select_action = action
        select_batch_action = batch.actions
        lam = self._alpha / (select_q_t.abs().mean()).detach()
        return lam * -q_t.mean() + 10 * ((select_batch_action - select_action) ** 2).mean()

    def _compute_critic_loss(
        self, batch: TorchMiniBatch, q_tpn: torch.Tensor
    ) -> torch.Tensor:
        value = self.q_(batch.observations, batch.actions, "mean")
        y = batch.rewards + self._gamma * q_tpn * (1 - batch.terminals)
        loss = F.mse_loss(value, y)
        print(f"torch.mean(value): {torch.mean(value)}")
        print(f"torch.mean(y): {torch.mean(y)}")
        print(f"torch.mean(batch.rewards): {torch.mean(batch.rewards)}")
        print(f"torch.mean(q_tpn): {torch.mean(q_tpn)}")
        print()
        return loss, torch.mean(value), torch.mean(y)

    def compute_target(self, batch: TorchMiniBatch) -> torch.Tensor:
        with torch.no_grad():
            action = self.p_(batch.next_observations)
            # smoothing target
            noise = torch.randn(action.shape, device=batch.device)
            scaled_noise = self._target_smoothing_sigma * noise
            clipped_noise = scaled_noise.clamp(
                -self._target_smoothing_clip, self._target_smoothing_clip
            )
            smoothed_action = torch.tanh(action) + clipped_noise
            clipped_action = smoothed_action.clamp(-1.0, 1.0)
            return self._targ_q(
                batch.next_observations,
                clipped_action,
                "min",
            )

    def compute_critic_loss(
        self, batch: TorchMiniBatch, q_tpn: torch.Tensor, clone_critic: bool=False, online: bool=False, replay: bool=False, first_time: bool=False
    ) -> torch.Tensor:
        loss, value, y = self._compute_critic_loss(batch, q_tpn)
        return loss, loss, value, y, y

    def compute_actor_loss(self, batch, clone_actor: bool = False, online: bool = False, replay: bool = False):
        loss = self._compute_actor_loss(batch, clone_actor=clone_actor, online=online)
        return loss, loss, loss
