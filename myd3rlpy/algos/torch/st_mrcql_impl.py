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
from d3rlpy.models.builders import create_probabilistic_ensemble_dynamics_model

from myd3rlpy.algos.torch.st_mcql_impl import STImpl
from utils.utils import Struct


replay_name = ['observations', 'actions', 'rewards', 'next_observations', 'terminals', 'policy_actions', 'qs', 'phis', 'psis']
class STImpl(STImpl):
    def __init__(
        self,
        random_sample_times: int,
        **kwargs,
    ):
        super().__init__(
            **kwargs
        )
        self._random_sample_times = random_sample_times
        self._save_diff_q_values = None

    def compute_critic_loss(
            self, batch: TorchMiniBatch, q_tpn: torch.Tensor, clone_critic: bool=False, online: bool=False
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
            loss = loss.to(torch.float32)
            return loss.mean()
        # start_time = time.time()
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
        if self._save_diff_q_values is not None:
            new_diff_q_values = self.match_prop(batch)
            zero_loss = torch.zeros_like(conservative_loss)
            conservative_loss = torch.where(new_diff_q_values > self._save_diff_q_values, conservative_loss, zero_loss).mean()
            # conservative_nums = torch.where(new_diff_q_values > self._save_diff_q_values + self._match_epsilon, torch.ones_like(conservative_loss), torch.zeros_like(conservative_loss)).sum()
            # print(f"conservative_nums: {conservative_nums}")
            # print(f"zero_loss.shape: {zero_loss.shape}")
        else:
            conservative_loss = conservative_loss.mean()
        return loss + conservative_loss

    def match_prop_post_train_process(self, iterator: TransitionMiniBatch):
        start_time = time.time()
        iterator.reset()

        save_diff_q_values = []
        for batch_num, itr in enumerate(range(len(iterator))):
            batch = next(iterator)
            batch = TorchMiniBatch(
                batch,
                self.device,
                scaler=self.scaler,
                action_scaler=self.action_scaler,
                reward_scaler=self.reward_scaler,
            )
            diff_q_values = self.match_prop(batch)
            save_diff_q_values.append(diff_q_values)
        save_diff_q_values = torch.cat(save_diff_q_values, dim=0)
        save_diff_q_values = torch.quantile(save_diff_q_values, self._match_prop_quantile)
        if self._save_diff_q_values is None:
            assert self._save_diff_q_values is None
            self._save_diff_q_values = save_diff_q_values
        else:
            self._save_diff_q_values = (self._save_diff_q_values * (self._impl_id - 1) + save_diff_q_values) / self._impl_id

        print("match prop time: {time.time() - start_time}")

    def match_prop(self, batch: TorchMiniBatch):
        assert self._policy is not None
        assert self._actor_optim is not None
        assert self._q_func is not None
        assert self._log_temp is not None

        observations = batch.observations

        with torch.no_grad():
            policy_actions, _ = self._policy.sample_with_log_prob(observations)
            q_values = self._q_func(observations, policy_actions, "min")

        diff_q_values = torch.zeros_like(q_values)
        for _ in range(self._random_sample_times):
            with torch.no_grad():
                new_policy_actions = policy_actions + 0.1 * torch.rand_like(policy_actions)
                new_q_values = self._q_func(observations, new_policy_actions, "min")
            new_diff_q_values = new_q_values - q_values
            diff_q_values = torch.where(diff_q_values > new_diff_q_values, diff_q_values, new_diff_q_values)

        return diff_q_values
