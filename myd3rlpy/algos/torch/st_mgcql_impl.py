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
from myd3rlpy.algos.torch.gem import overwrite_grad, store_grad, project2cone2
from myd3rlpy.algos.torch.agem import project
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
        self._save_grads_mean = None
        self._save_grads = None

    def compute_critic_loss(
        self, batch: TorchMiniBatch, q_tpn: torch.Tensor
    ) -> torch.Tensor:
        loss = super().compute_critic_loss(batch, q_tpn).mean()
        start_time = time.time()
        conservative_loss = self._compute_conservative_loss(
            batch.observations, batch.actions, batch.next_observations
        )
        # For test:
        if self._save_grads_mean is not None:
            new_match_prop = self.match_prop(batch)
            print(f"one time match time: {time.time() - start_time}")
            mean_grads = (new_match_prop / self._save_grads_mean).mean(dim=1)
            print(f"mean_grads.shape: {mean_grads.shape}")
            zero_loss = torch.zeros_like(conservative_loss)
            conservative_loss = torch.where(mean_grads < self._save_grads + self._match_epsilon, conservative_loss, zero_loss).mean()
        else:
            conservative_loss = conservative_loss.mean()
        return loss + conservative_loss

    def match_prop_post_train_process(self, iterator: TransitionMiniBatch):
        start_time = time.time()
        iterator.reset()

        save_grads = []
        for batch_num, itr in enumerate(range(len(iterator))):
            batch = next(iterator)
            batch = TorchMiniBatch(
                batch,
                self.device,
                scaler=self.scaler,
                action_scaler=self.action_scaler,
                reward_scaler=self.reward_scaler,
            )
            grad = self.match_prop(batch)
            save_grads.append(grad)
        save_grads = torch.cat(save_grads, dim=0)
        save_grads_mean = torch.mean(save_grads, dim=1)
        save_grads = (save_grads / save_grads_mean).mean(dim=1)
        save_grads = torch.quantile(save_grads, self._match_prop_quantile)
        if self._save_grads is None:
            assert self._save_grads_mean is None
            self._save_grads = save_grads
            self._save_grads_mean = save_grads_mean
        else:
            self._save_grads = (self._save_grads * (self._impl_id - 1) + save_grads) / self._impl_id
            self._save_grads_mean = (self._save_grads_mean * (self._impl_id - 1) + save_grads_mean) / self._impl_id

        print("match prop time: {time.time() - start_time}")

    def match_prop(self, batch: TorchMiniBatch):
        assert self._policy is not None
        assert self._actor_optim is not None
        assert self._q_func is not None
        assert self._log_temp is not None

        observations = batch.observations

        grads = []
        policy_optim_state_dict = copy.deepcopy(self._actor_optim.state_dict())
        for observation in observations:
            observation = observation.unsqueeze(dim=0)
            policy_action, log_prob = self._policy.sample_with_log_prob(observation)
            # (batch)
            q_value = self._q_func(observation, policy_action, "min")

            entropy = self._log_temp().exp() * log_prob

            # (batch)
            policy_value_loss = entropy - q_value
            observation = observation.unsqueeze(dim=0)
            self._actor_optim.zero_grad()
            policy_value_loss.backward()
            grad = torch.cat([param.grad.reshape(-1) for param in self._policy.parameters() if param.requires_grad], dim=0)
            grads.append(grad)
        grads = torch.stack(grads)
        self._actor_optim.load_state_dict(policy_optim_state_dict)

        return grads
