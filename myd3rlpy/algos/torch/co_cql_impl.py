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
from d3rlpy.algos.torch.cql_impl import CQLImpl
from d3rlpy.models.builders import create_probabilistic_ensemble_dynamics_model

from myd3rlpy.models.builders import create_phi, create_psi
from myd3rlpy.algos.torch.gem import overwrite_grad, store_grad, project2cone2
from myd3rlpy.algos.torch.agem import project
from myd3rlpy.algos.torch.co_impl import COImpl
from utils.utils import Struct


replay_name = ['observations', 'actions', 'rewards', 'next_observations', 'terminals', 'policy_actions', 'qs', 'phis', 'psis']
class COCQLImpl(COImpl, CQLImpl):
    def __init__(
        self,
        observation_shape: Sequence[int],
        action_size: int,
        actor_learning_rate: float,
        critic_learning_rate: float,
        temp_learning_rate: float,
        alpha_learning_rate: float,
        phi_learning_rate: float,
        psi_learning_rate: float,
        model_learning_rate: float,
        actor_optim_factory: OptimizerFactory,
        critic_optim_factory: OptimizerFactory,
        temp_optim_factory: OptimizerFactory,
        alpha_optim_factory: OptimizerFactory,
        phi_optim_factory: OptimizerFactory,
        psi_optim_factory: OptimizerFactory,
        model_optim_factory: OptimizerFactory,
        actor_encoder_factory: EncoderFactory,
        critic_encoder_factory: EncoderFactory,
        model_encoder_factory: EncoderFactory,
        q_func_factory: QFunctionFactory,
        replay_type: str,
        gamma: float,
        gem_alpha: float,
        agem_alpha: float,
        ewc_r_walk_alpha: float,
        damping: float,
        epsilon: float,
        tau: float,
        n_critics: int,
        initial_temperature: float,
        initial_alpha: float,
        alpha_threshold: float,
        conservative_weight: float,
        n_action_samples: int,
        soft_q_backup: bool,
        use_gpu: Optional[Device],
        scaler: Optional[Scaler],
        action_scaler: Optional[ActionScaler],
        reward_scaler: Optional[RewardScaler],
        model_n_ensembles: int,
        use_phi: bool,
        use_model: bool,
        replay_model: bool,
        replay_critic: bool,
        replay_alpha: float,
        retrain_model_alpha: float,
        single_head: bool,
    ):
        super().__init__(
            observation_shape = observation_shape,
            action_size = action_size,
            actor_learning_rate = actor_learning_rate,
            critic_learning_rate = critic_learning_rate,
            temp_learning_rate = temp_learning_rate,
            alpha_learning_rate = alpha_learning_rate,
            actor_optim_factory = actor_optim_factory,
            critic_optim_factory = critic_optim_factory,
            temp_optim_factory = temp_optim_factory,
            alpha_optim_factory = alpha_optim_factory,
            actor_encoder_factory = actor_encoder_factory,
            critic_encoder_factory = critic_encoder_factory,
            q_func_factory = q_func_factory,
            gamma = gamma,
            tau = tau,
            n_critics = n_critics,
            initial_temperature = initial_temperature,
            initial_alpha = initial_alpha,
            alpha_threshold = alpha_threshold,
            conservative_weight = conservative_weight,
            n_action_samples = n_action_samples,
            soft_q_backup = soft_q_backup,
            use_gpu = use_gpu,
            scaler = scaler,
            action_scaler = action_scaler,
            reward_scaler = reward_scaler,
        )
        self._replay_type = replay_type

        self._gem_alpha = gem_alpha
        self._agem_alpha = agem_alpha
        self._ewc_r_walk_alpha = ewc_r_walk_alpha
        self._damping = damping
        self._epsilon = epsilon

        self._use_phi = use_phi
        self._use_model = use_model
        self._replay_alpha = replay_alpha
        self._replay_critic = replay_critic
        self._replay_model = replay_model
        self._phi_learning_rate = phi_learning_rate
        self._psi_learning_rate = psi_learning_rate
        self._phi_optim_factory = phi_optim_factory
        self._psi_optim_factory = psi_optim_factory

        self._model_learning_rate = model_learning_rate
        self._model_optim_factory = model_optim_factory
        self._model_encoder_factory = model_encoder_factory
        self._model_n_ensembles = model_n_ensembles
        self._retrain_model_alpha = retrain_model_alpha

        single_head = False
        self._single_head = single_head
        if single_head:
            self.change_task = self.change_task_singlehead
        else:
            self.change_task = self.change_task_multihead

        self._first = True

    def compute_critic_loss(
        self, batch: TorchMiniBatch, q_tpn: torch.Tensor
    ) -> torch.Tensor:
        assert self._q_func is not None
        loss =  self._q_func.compute_error(
            observations=batch.observations,
            actions=batch.actions[:, :self._action_size],
            rewards=batch.rewards,
            target=q_tpn,
            terminals=batch.terminals,
            gamma=self._gamma ** batch.n_steps,
        )
        loss += self._compute_conservative_loss(batch.observations, batch.actions[:, :self._action_size], batch.next_observations)
        return loss

    def change_task_multihead(self, task_id):
        assert self._policy is not None
        assert self._targ_policy is not None
        assert self._q_func is not None
        assert self._targ_q_func is not None
        assert self._dynamic is not None
        if self._impl_id is not None and self._impl_id == task_id:
            return
        if "_mus" not in self._policy.__dict__.keys():
            self._policy._mus = dict()
            self._policy._mus[task_id] = deepcopy(self._policy._mu.state_dict())
            if isinstance(self._policy._logstd, torch.nn.parameter.Parameter):
                self._policy._logstds = dict()
                self._policy._logstds[task_id] = deepcopy(self._policy._logstd)
            else:
                self._policy._logstds = dict()
                self._policy._logstds[task_id] = deepcopy(self._policy._logstd.state_dict())
            if self._replay_critic:
                for q_func in self._q_func._q_funcs:
                    q_func._fcs = dict()
                    q_func._fcs[task_id] = deepcopy(q_func._fc.state_dict())
            if self._replay_type == 'orl':
                self._targ_policy._mus = dict()
                self._targ_policy._mus[task_id] = deepcopy(self._targ_policy._mu.state_dict())
                if isinstance(self._targ_policy._logstd, torch.nn.parameter.Parameter):
                    self._targ_policy._logstds = dict()
                    self._targ_policy._logstds[task_id] = deepcopy(self._targ_policy._logstd)
                else:
                    self._targ_policy._logstds = dict()
                    self._targ_policy._logstds[task_id] = deepcopy(self._targ_policy._logstd.state_dict())
                if self._replay_critic:
                    for q_func in self._targ_q_func._q_funcs:
                        q_func._fcs = dict()
                        q_func._fcs[task_id] = deepcopy(q_func._fc.state_dict())
            if self._use_model and self._replay_model:
                for model in self._dynamic._models:
                    model._mus = dict()
                    model._mus[task_id] = deepcopy(model._mu.state_dict())
                    model._logstds = dict()
                    model._logstds[task_id] = deepcopy(model._logstd.state_dict())
                    model._max_logstds = dict()
                    model._max_logstds[task_id] = deepcopy(model._max_logstd)
                    model._min_logstds = dict()
                    model._min_logstds[task_id] = deepcopy(model._min_logstd)
            self._impl_id = task_id
        # self._using_id = task_id
        if task_id not in self._policy._mus.keys():
            print(f'add new id: {task_id}')
            if self._replay_critic:
                for q_func in self._q_func._q_funcs:
                    assert task_id not in q_func._fcs.keys()
            self._policy._mus[task_id] = deepcopy(nn.Linear(self._policy._mu.weight.shape[1], self._policy._mu.weight.shape[0], bias=self._policy._mu.bias is not None).to(self.device).state_dict())
            if isinstance(self._policy._logstd, torch.nn.parameter.Parameter):
                self._policy._logstds[task_id] = deepcopy(nn.Parameter(torch.empty(torch.zeros(1, self._policy._logstd.weight.shape[0], dtype=torch.float32).to(self.device))))
            else:
                self._policy._logstds[task_id] = deepcopy(nn.Linear(self._policy._logstd.weight.shape[1], self._policy._logstd.weight.shape[0], bias=self._policy._logstd.bias is not None).to(self.device).state_dict())
            if self._replay_type == 'orl':
                self._targ_policy._mus[task_id] = deepcopy(nn.Linear(self._targ_policy._mu.weight.shape[1], self._targ_policy._mu.weight.shape[0], bias=self._targ_policy._mu.bias is not None).to(self.device).state_dict())
                if isinstance(self._policy._logstd, torch.nn.parameter.Parameter):
                    self._targ_policy._logstds[task_id] = deepcopy(nn.Parameter(torch.empty(torch.zeros(1, self._targ_policy._logstd.weight.shape[0], dtype=torch.float32).to(self.device))))
                else:
                    self._targ_policy._logstds[task_id] = deepcopy(nn.Linear(self._targ_policy._logstd.weight.shape[1], self._targ_policy._logstd.weight.shape[0], bias=self._targ_policy._logstd.bias is not None).to(self.device).state_dict())

            if self._replay_critic:
                for q_func in self._q_func._q_funcs:
                    q_func._fcs[task_id] = deepcopy(nn.Linear(q_func._fc.weight.shape[1], q_func._fc.weight.shape[0], bias=q_func._fc.bias is not None).to(self.device).state_dict())
                if self._replay_type == 'orl':
                    for q_func in self._targ_q_func._q_funcs:
                        q_func._fcs[task_id] = deepcopy(nn.Linear(q_func._fc.weight.shape[1], q_func._fc.weight.shape[0], bias=q_func._fc.bias is not None).to(self.device).state_dict())
            if self._use_model and self._replay_model:
                for model in self._dynamic._models:
                    model._mus[task_id] = deepcopy(nn.Linear(model._mu.weight.shape[1], model._mu.weight.shape[0], bias=model._mu.bias is not None).to(self.device).state_dict())
                    model._logstds[task_id] = deepcopy(nn.Linear(model._logstd.weight.shape[1], model._logstd.weight.shape[0], bias=model._logstd.bias is not None).to(self.device).state_dict())
                    model._max_logstds[task_id] = deepcopy(nn.Parameter(torch.empty(1, model._logstd.weight.shape[0], dtype=torch.float32).fill_(2.0).to(self.device)))
                    model._min_logstds[task_id] = deepcopy(nn.Parameter(torch.empty(1, model._logstd.weight.shape[0], dtype=torch.float32).fill_(-10.0).to(self.device)))
        if self._impl_id != task_id:
            self._policy._mus[task_id] = deepcopy(self._policy._mu.state_dict())
            self._policy._mu.load_state_dict(self._policy._mus[task_id])
            if isinstance(self._policy._logstd, torch.nn.parameter.Parameter):
                self._policy._logstds[task_id] = deepcopy(self._policy._logstd)
                self._policy._logstd.copy_(self._policy._logstds[task_id])
            else:
                self._policy._logstds[task_id] = deepcopy(self._policy._logstd.state_dict())
                self._policy._logstd.load_state_dict(self._policy._logstds[task_id])
            if self._replay_type == 'orl':
                self._targ_policy._mus[task_id] = deepcopy(self._targ_policy._mu.state_dict())
                self._targ_policy._mu.load_state_dict(self._targ_policy._mus[task_id])
                if isinstance(self._targ_policy._logstd, torch.nn.parameter.Parameter):
                    self._targ_policy._logstds[task_id] = deepcopy(self._targ_policy._logstd)
                    self._targ_policy._logstd.copy_(self._targ_policy._logstds[task_id])
                else:
                    self._targ_policy._logstds[task_id] = deepcopy(self._targ_policy._logstd.state_dict())
                    self._targ_policy._logstd.load_state_dict(self._targ_policy._logstds[task_id])
            if self._replay_critic:
                for q_func in self._q_func._q_funcs:
                    q_func._fcs[self._impl_id] = deepcopy(q_func._fc.state_dict())
                    q_func._fc.load_state_dict(q_func._fcs[task_id])
                if self._replay_type == 'orl':
                    for q_func in self._targ_q_func._q_funcs:
                        q_func._fcs[self._impl_id] = deepcopy(q_func._fc.state_dict())
                        q_func._fc.load_state_dict(q_func._fcs[task_id])
            if self._use_model and self._replay_model:
                for model in self._dynamic._models:
                    model._mus[self._impl_id] = deepcopy(model._mu.state_dict())
                    model._mu.load_state_dict(model._mus[task_id].state_dict())
                    model._logstds[self._impl_id] = deepcopy(model._logstd.state_dict())
                    model._logstd.load_state_dict(model._logstds[task_id].state_dict())
                    model._max_logstds[self._impl_id] =  deepcopy(model._max_logstd)
                    model._max_logstd.copy_(model._max_logstds[task_id])
                    model._min_logstds[self._impl_id] = deepcopy(model._min_logstd)
                    model._min_logstd.copy_(model._min_logstds[task_id])
        self._build_actor_optim()
        self._build_critic_optim()
        if self._use_model and self._replay_model:
            self._model_optim = self._model_optim_factory.create(
                self._dynamic.parameters(), lr=self._model_learning_rate
            )
        self._impl_id = task_id
