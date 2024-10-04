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
from d3rlpy.algos.torch.combo_impl import COMBOImpl
from d3rlpy.models.builders import create_probabilistic_ensemble_dynamics_model

from myd3rlpy.algos.torch.gem import overwrite_grad, store_grad, project2cone2
from myd3rlpy.algos.torch.agem import project
from myd3rlpy.algos.torch.co_cql_impl import COCQLImpl
from utils.utils import Struct


replay_name = ['observations', 'actions', 'rewards', 'next_observations', 'terminals', 'policy_actions', 'qs']
class COCOMBOImpl(COCQLImpl):
    def __init__(
        self,
        observation_shape: Sequence[int],
        action_size: int,
        actor_learning_rate: float,
        critic_learning_rate: float,
        temp_learning_rate: float,
        model_learning_rate: float,
        actor_optim_factory: OptimizerFactory,
        critic_optim_factory: OptimizerFactory,
        temp_optim_factory: OptimizerFactory,
        model_optim_factory: OptimizerFactory,
        actor_encoder_factory: EncoderFactory,
        critic_encoder_factory: EncoderFactory,
        model_encoder_factory: EncoderFactory,
        q_func_factory: QFunctionFactory,
        replay_type: str,
        gamma: float,
        gem_alpha: float,
        agem_alpha: float,
        ewc_rwalk_alpha: float,
        damping: float,
        epsilon: float,
        tau: float,
        n_critics: int,
        initial_temperature: float,
        conservative_weight: float,
        n_action_samples: int,
        real_ratio: float,
        soft_q_backup: bool,
        use_gpu: Optional[Device],
        scaler: Optional[Scaler],
        action_scaler: Optional[ActionScaler],
        reward_scaler: Optional[RewardScaler],
        model_n_ensembles: int,
        clone_actor: bool,
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
            actor_optim_factory = actor_optim_factory,
            critic_optim_factory = critic_optim_factory,
            temp_optim_factory = temp_optim_factory,
            actor_encoder_factory = actor_encoder_factory,
            critic_encoder_factory = critic_encoder_factory,
            q_func_factory = q_func_factory,
            gamma = gamma,
            tau = tau,
            n_critics = n_critics,
            initial_temperature = initial_temperature,
            conservative_weight = conservative_weight,
            n_action_samples = n_action_samples,
            real_ratio = real_ratio,
            soft_q_backup = soft_q_backup,
            use_gpu = use_gpu,
            scaler = scaler,
            action_scaler = action_scaler,
            reward_scaler = reward_scaler,
        )
        self._replay_type = replay_type

        self._gem_alpha = gem_alpha
        self._agem_alpha = agem_alpha
        self._ewc_rwalk_alpha = ewc_rwalk_alpha
        self._damping = damping
        self._epsilon = epsilon

        self._clone_actor = clone_actor
        self._replay_alpha = replay_alpha
        self._replay_critic = replay_critic
        self._replay_model = replay_model

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

    def build(self, task_id):
        super().build(task_id)
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
        self._model_optim = self._model_optim_factory.create(
            self._dynamic.parameters(), lr=self._model_learning_rate
        )
        # To save the actor grad for DBC and BC
        if self._replay_type in ['ewc', 'rwalk', 'si']:
            if self._replay_model:
                # Store current parameters for the next task
                self._model_older_params = {n: p.clone().detach() for n, p in self._dynamic.named_parameters() if p.requires_grad}
            if self._replay_type in ['ewc', 'rwalk']:
                if self._replay_critic:
                    # Store fisher information weight importance
                    self._critic_fisher = {n: torch.zeros(p.shape).to(self.device) for n, p in self._q_func.named_parameters() if p.requires_grad and ('_fc' not in n or '_fcs' in n)}
                if self._replay_actor:
                    # Store fisher information weight importance
                    self._actor_fisher = {n: torch.zeros(p.shape).to(self.device) for n, p in self._policy.named_parameters() if p.requires_grad and ('_fc' not in n or '_fcs' in n)}
                if self._replay_model:
                    # Store fisher information weight importance
                    self._model_fisher = {n: torch.zeros(p.shape).to(self.device) for n, p in self._dynamic.named_parameters() if p.requires_grad}
                if self._replay_type == 'rwalk':
                    if self._replay_critic:
                        # Page 7: "task-specific parameter importance over the entire training trajectory."
                        self._critic_w = {n: torch.zeros(p.shape).to(self.device) for n, p in self._q_func.named_parameters() if p.requires_grad and ('_fc' not in n or '_fcs' in n)}
                        self._critic_scores = {n: torch.zeros(p.shape).to(self.device) for n, p in self._q_func.named_parameters() if p.requires_grad and ('_fc' not in n or '_fcs' in n)}
                    if self._replay_actor:
                        # Page 7: "task-specific parameter importance over the entire training trajectory."
                        self._actor_w = {n: torch.zeros(p.shape).to(self.device) for n, p in self._policy.named_parameters() if p.requires_grad and ('_fc' not in n or '_fcs' in n)}
                        self._actor_scores = {n: torch.zeros(p.shape).to(self.device) for n, p in self._policy.named_parameters() if p.requires_grad and ('_fc' not in n or '_fcs' in n)}
                    if self._replay_model:
                        # Page 7: "task-specific parameter importance over the entire training trajectory."
                        self._model_w = {n: torch.zeros(p.shape).to(self.device) for n, p in self._dynamic.named_parameters() if p.requires_grad}
                        self._model_scores = {n: torch.zeros(p.shape).to(self.device) for n, p in self._dynamic.named_parameters() if p.requires_grad}
                elif self._replay_type == 'si':
                    if self._replay_critic:
                        self._critic_W = {n: p.clone().detach().zero_() for n, p in self._q_func.named_parameters() if p.requires_grad and ('_fc' not in n or '_fcs' in n)}
                        self._critic_omega = {n: p.clone().detach().zero_() for n, p in self._q_func.named_parameters() if p.requires_grad and ('_fc' not in n or '_fcs' in n)}
                    if self._replay_actor:
                        self._actor_W = {n: p.clone().detach().zero_() for n, p in self._policy.named_parameters() if p.requires_grad and ('_fc' not in n or '_fcs' in n)}
                        self._actor_omega = {n: p.clone().detach().zero_() for n, p in self._policy.named_parameters() if p.requires_grad and ('_fc' not in n or '_fcs' in n)}
                    if self._replay_model:
                        self._model_W = {n: p.clone().detach().zero_() for n, p in self._dynamic.named_parameters() if p.requires_grad}
                        self._model_omega = {n: p.clone().detach().zero_() for n, p in self._dynamic.named_parameters() if p.requires_grad}
        elif self._replay_type == 'gem':
            if self._replay_critic:
                # Allocate temporary synaptic memory
                self._critic_grad_dims = []
                for name, pp in self._q_func.named_parameters():
                    if ('_fc' not in name or '_fcs' in name):
                        self._critic_grad_dims.append(pp.data.numel())
                self._critic_grad_cs = {}
                self._critic_grad_da = torch.zeros(np.sum(self._critic_grad_dims)).to(self.device)

            if self._replay_actor:
                self._actor_grad_dims = []
                for name, pp in self._policy.named_parameters():
                    if ('_fc' not in name or '_fcs' in name):
                        self._actor_grad_dims.append(pp.data.numel())
                self._actor_grad_cs = {}
                self._actor_grad_da = torch.zeros(np.sum(self._actor_grad_dims)).to(self.device)

            if self._replay_model:
                self._model_grad_dims = []
                for pp in self._dynamic.parameters():
                    self._model_grad_dims.append(pp.data.numel())
                self._model_grad_cs = {}
                self._model_grad_da = torch.zeros(np.sum(self._model_grad_dims)).to(self.device)
        elif self._replay_type == 'agem':
            if self._replay_critic:
                self._critic_grad_dims = []
                for name, param in self._q_func.named_parameters():
                    if ('_fc' not in name or '_fcs' in name):
                        self._critic_grad_dims.append(param.data.numel())
                self._critic_grad_xy = torch.Tensor(np.sum(self._critic_grad_dims)).to(self.device)
                self._critic_grad_er = torch.Tensor(np.sum(self._critic_grad_dims)).to(self.device)
            if self._replay_actor:
                self._actor_grad_dims = []
                for name, param in self._policy.named_parameters():
                    if ('_fc' not in name or '_fcs' in name):
                        self._actor_grad_dims.append(param.data.numel())
                self._actor_grad_xy = torch.Tensor(np.sum(self._actor_grad_dims)).to(self.device)
                self._actor_grad_er = torch.Tensor(np.sum(self._actor_grad_dims)).to(self.device)
            if self._replay_model:
                self._model_grad_dims = []
                for param in self._q_func.parameters():
                    self._model_grad_dims.append(param.data.numel())
                self._model_grad_xy = torch.Tensor(np.sum(self._model_grad_dims)).to(self.device)
                self._model_grad_er = torch.Tensor(np.sum(self._model_grad_dims)).to(self.device)

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
        super().change_task_multihead(task_id)
        if self._replay_model:
            if "_mus" not in self._dynamic._models[0].__dict__.keys():
                for model in self._dynamic._models:
                    model._mus = dict()
                    model._mus[task_id] = deepcopy(model._mu.state_dict())
                    model._logstds = dict()
                    model._logstds[task_id] = deepcopy(model._logstd.state_dict())
                    model._max_logstds = dict()
                    model._max_logstds[task_id] = deepcopy(model._max_logstd)
                    model._min_logstds = dict()
                    model._min_logstds[task_id] = deepcopy(model._min_logstd)
        if self._replay_model:
            if task_id not in self._dynamic._models[0]._mus.keys():
                for model in self._dynamic._models:
                    model._mus[task_id] = deepcopy(nn.Linear(model._mu.weight.shape[1], model._mu.weight.shape[0], bias=model._mu.bias is not None).to(self.device).state_dict())
                    model._logstds[task_id] = deepcopy(nn.Linear(model._logstd.weight.shape[1], model._logstd.weight.shape[0], bias=model._logstd.bias is not None).to(self.device).state_dict())
                    model._max_logstds[task_id] = deepcopy(nn.Parameter(torch.empty(1, model._logstd.weight.shape[0], dtype=torch.float32).fill_(2.0).to(self.device)))
                    model._min_logstds[task_id] = deepcopy(nn.Parameter(torch.empty(1, model._logstd.weight.shape[0], dtype=torch.float32).fill_(-10.0).to(self.device)))
        if self._impl_id != task_id:
            if self._clone_actor and self._replay_type == 'bc':
                self._clone_policy._mus[task_id] = deepcopy(self._clone_policy._mu.state_dict())
                self._clone_policy._mu.load_state_dict(self._clone_policy._mus[task_id])
                if isinstance(self._clone_policy._logstd, torch.nn.parameter.Parameter):
                    self._clone_policy._logstds[task_id] = deepcopy(self._clone_policy._logstd)
                    self._clone_policy._logstd.copy_(self._clone_policy._logstds[task_id])
                else:
                    self._clone_policy._logstds[task_id] = deepcopy(self._clone_policy._logstd.state_dict())
                    self._clone_policy._logstd.load_state_dict(self._clone_policy._logstds[task_id])
            else:
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
            if self._replay_model:
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
        if self._replay_model:
            self._model_optim = self._model_optim_factory.create(
                self._dynamic.parameters(), lr=self._model_learning_rate
            )
        self._impl_id = task_id
