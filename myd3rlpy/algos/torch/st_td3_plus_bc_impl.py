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
from d3rlpy.algos.torch.td3_plus_bc_impl import TD3PlusBCImpl
from d3rlpy.models.builders import create_probabilistic_ensemble_dynamics_model

from myd3rlpy.models.builders import create_phi, create_psi
from myd3rlpy.algos.torch.gem import overwrite_grad, store_grad, project2cone2
from myd3rlpy.algos.torch.agem import project
from myd3rlpy.algos.torch.co_deterministic_impl import CODeterministicImpl
from myd3rlpy.algos.torch.st_impl import STImpl
from utils.utils import Struct


replay_name = ['observations', 'actions', 'rewards', 'next_observations', 'terminals', 'policy_actions', 'qs', 'phis', 'psis']
class STTD3PlusBCImpl(STImpl, TD3PlusBCImpl):

    def compute_actor_loss(self, batch: TorchMiniBatch, clone_actor: bool=False, online: bool=False) -> torch.Tensor:
        assert self._policy is not None
        assert self._q_func is not None
        action = self._policy(batch.observations)
        q_t = self._q_func(batch.observations, action, "none")[0]
        if online:
            return -q_t.mean()
        if clone_actor:
            clone_actions = self._clone_policy(batch.observations)
            clone_q_t = self._clone_q_func(batch.observations, clone_actions, "none")[0]
            new_q_t = self._q_func(batch.observations, batch.actions, "none")[0]
            max_q_t = new_q_t > clone_q_t
            select_q_t = q_t[max_q_t]
            select_action = action[max_q_t]
            select_batch_action = batch.actions[max_q_t]
        else:
            select_q_t = q_t
            select_action = action
            select_batch_action = batch.actions
        lam = self._alpha / (select_q_t.abs().mean()).detach()
        return lam * -q_t.mean() + ((select_batch_action - select_action) ** 2).mean()

    def compute_critic_loss(
        self, batch: TorchMiniBatch, q_tpn: torch.Tensor, clone_critic: bool=False, online: bool=False
    ) -> torch.Tensor:
        return super().compute_critic_loss(batch, q_tpn)
