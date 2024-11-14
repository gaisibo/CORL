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
        **kwargs
    ):
        super().__init__(
            **kwargs
        )
        self._save_match_prop = None

    def compute_critic_loss(
        self, batch: TorchMiniBatch, q_tpn: torch.Tensor
    ) -> torch.Tensor:
        loss = super().compute_critic_loss(batch, q_tpn).mean()
        start_time = time.time()
        conservative_loss = self._compute_conservative_loss(
            batch.observations, batch.actions, batch.next_observations
        )
        if self._save_match_prop is not None:
            new_match_prop = self.match_prop(batch)
            print(f"one time match time: {time.time() - start_time}")
            zero_loss = torch.zeros_like(conservative_loss)
            conservative_loss = torch.where(new_match_prop < self._save_match_prop + self._match_epsilon, conservative_loss, zero_loss).mean()
        else:
            conservative_loss = conservative_loss.mean()
        return loss + conservative_loss

    def match_prop_post_train_process(self, iterator: TransitionMiniBatch):
        start_time = time.time()
        iterator.reset()

        save_match_prop = []
        for batch_num, itr in enumerate(range(len(iterator))):
            batch = next(iterator)
            batch = TorchMiniBatch(
                batch,
                self.device,
                scaler=self.scaler,
                action_scaler=self.action_scaler,
                reward_scaler=self.reward_scaler,
            )
            match_prop = self.match_prop(batch)
            save_match_prop.append(match_prop)
        save_match_prop = torch.cat(save_match_prop, dim=0)
        save_match_prop = torch.quantile(save_match_prop, self._match_prop_quantile)
        if self._save_match_prop is None:
            assert self._save_match_prop is None
            self._save_match_prop = save_match_prop
        else:
            self._save_match_prop = (self._save_match_prop * (self._impl_id - 1) + save_match_prop) / self._impl_id

        print("match prop time: {time.time() - start_time}")

    def match_prop(self, batch: TorchMiniBatch):
        assert self._policy is not None
        assert self._actor_optim is not None
        assert self._q_func is not None
        assert self._log_temp is not None

        observations = batch.observations

        diff_q_values = []
        policy_state_dict = copy.deepcopy(self._policy.state_dict())
        policy_optim_state_dict = copy.deepcopy(self._actor_optim.state_dict())
        for observation in observations:
            observation = observation.unsqueeze(dim=0)
            policy_action, log_prob = self._policy.sample_with_log_prob(observation)
            # (batch)
            print(f"observation.shape: {observation.shape}")
            print(f"policy_action.shape: {policy_action.shape}")
            q_value = self._q_func(observation, policy_action, "min")

            entropy = self._log_temp().exp() * log_prob

            # (batch)
            policy_value_loss = entropy - q_value
            self._actor_optim.zero_grad()
            policy_value_loss.backward()
            self._actor_optim.step()
            new_policy_action, new_log_prob = self._policy.sample_with_log_prob(observation)
            q_value = self._q_func(observation, new_policy_action, "min")
            diff_q_value = q_value.unsqueeze(dim=0) - q_value
            diff_q_values.append(diff_q_value)
            self._policy.load_state_dict(policy_state_dict)
            self._actor_optim.load_state_dict(policy_optim_state_dict)
        diff_q_values = torch.stack(diff_q_values, dim=0)
        return diff_q_values
