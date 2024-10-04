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
from d3rlpy.algos.base import AlgoImplBase
from d3rlpy.algos.torch.base import TorchImplBase
from d3rlpy.torch_utility import hard_sync

from myd3rlpy.algos.torch.gem import overwrite_grad, store_grad, project2cone2
from myd3rlpy.algos.torch.agem import project
from utils.utils import Struct


replay_name = ['observations', 'actions', 'rewards', 'next_observations', 'terminals', 'policy_actions', 'qs']
class BaseModelImpl():
    def __init__(
        self,
        model_learning_rate: float,
        model_optim_factory: OptimizerFactory,
        model_encoder_factory: EncoderFactory,
        model_n_ensembles: int,
        retrain_model_alpha: float,
    ):

        self._model_learning_rate = model_learning_rate
        self._model_optim_factory = model_optim_factory
        self._model_encoder_factory = model_encoder_factory
        self._model_n_ensembles = model_n_ensembles
        self._retrain_model_alpha = retrain_model_alpha

    def build(self, task_id):
        self._dynamic = create_probabilistic_ensemble_dynamics_model(
            self._observation_shape,
            self._action_size,
            check_encoder(self._model_encoder_factory),
            n_ensembles=self._model_n_ensembles,
            discrete_action=False,
        )
        self._dynamic.to(self.device)
        for model in self._dynamic._models:
            model.to(self.device)

    @train_api
    def update_model(self, batch: TransitionMiniBatch):
        assert self._dynamic is not None
        assert self._model_optim is not None
        batch = TorchMiniBatch(
            batch,
            self.device,
            scaler=self.scaler,
            action_scaler=self.action_scaler,
            reward_scaler=self.reward_scaler,
        )

        self._q_func.eval()
        self._policy.eval()
        self._dynamic.train()

        self._model_optim.zero_grad()
        loss = self._dynamic.compute_error(
            observations=batch.observations,
            actions=batch.actions[:, :self._action_size],
            rewards=batch.rewards,
            next_observations=batch.next_observations,
        )

        self._model_optim.zero_grad()
        loss.backward()
        self._model_optim.step()

        loss = loss.cpu().detach().numpy()

        return loss