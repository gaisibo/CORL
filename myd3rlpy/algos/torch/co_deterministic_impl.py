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

from myd3rlpy.models.builders import create_phi, create_psi
from myd3rlpy.algos.torch.gem import overwrite_grad, store_grad, project2cone2
from myd3rlpy.algos.torch.agem import project
from myd3rlpy.algos.torch.co_impl import COImpl
from utils.utils import Struct


replay_name = ['observations', 'actions', 'rewards', 'next_observations', 'terminals', 'policy_actions', 'qs', 'phis', 'psis']
class CODeterministicImpl(COImpl):
    def change_task_multihead(self, task_id):
        assert self._policy is not None
        assert self._targ_policy is not None
        if self._impl_id is not None and self._impl_id == task_id:
            return
        if not self._clone_actor:
            if "_fcs" not in self._policy.__dict__.keys():
                self._policy._fcs = dict()
                self._policy._fcs[task_id] = deepcopy(self._policy._fc.state_dict())
                if self._replay_type == 'orl':
                    self._targ_policy._fcs = dict()
                    self._targ_policy._fcs[task_id] = deepcopy(self._targ_policy._fc.state_dict())
                self._impl_id = task_id
        else:
            # 第一个任务直接复制policy就行。
            if "_fcs" not in self._clone_policy.__dict__.keys():
                self._clone_policy._fcs = dict()
                self._clone_policy._fcs[task_id] = deepcopy(self._policy._fc.state_dict())
                self._impl_id = task_id
        if self._replay_critic:
            if "_fcs" not in self._q_func._q_funcs[0].__dict__.keys():
                for q_func in self._q_func._q_funcs:
                    q_func._fcs = dict()
                    q_func._fcs[task_id] = deepcopy(q_func._fc.state_dict())
            if self._replay_type == 'orl':
                for q_func in self._targ_q_func._q_funcs:
                    q_func._fcs = dict()
                    q_func._fcs[task_id] = deepcopy(q_func._fc.state_dict())
        if self._use_model and self._replay_model:
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
    # self._using_id = task_id
        if self._clone_actor:
            if task_id not in self._clone_policy._fcs.keys():
                self._clone_policy._fcs[task_id] = deepcopy(nn.Linear(self._clone_policy._fc.weight.shape[1], self._clone_policy._fc.weight.shape[0], bias=self._clone_policy._fc.bias is not None).to(self.device).state_dict())
                self._targ_policy = copy.deepcopy(self._policy)
        else:
            if task_id not in self._policy._fcs.keys():
                self._policy._fcs[task_id] = deepcopy(nn.Linear(self._policy._fc.weight.shape[1], self._policy._fc.weight.shape[0], bias=self._policy._fc.bias is not None).to(self.device).state_dict())
                if self._replay_type == 'orl':
                    self._targ_policy._fcs[task_id] = deepcopy(nn.Linear(self._targ_policy._fc.weight.shape[1], self._targ_policy._fc.weight.shape[0], bias=self._targ_policy._fc.bias is not None).to(self.device).state_dict())

        if self._replay_critic:
            if task_id not in self._q_func._q_funcs[0]._fcs.keys():
                for q_func in self._q_func._q_funcs:
                    q_func._fcs[task_id] = deepcopy(nn.Linear(q_func._fc.weight.shape[1], q_func._fc.weight.shape[0], bias=q_func._fc.bias is not None).to(self.device).state_dict())
                if self._replay_type == 'orl':
                    for q_func in self._targ_q_func._q_funcs:
                        q_func._fcs[task_id] = deepcopy(nn.Linear(q_func._fc.weight.shape[1], q_func._fc.weight.shape[0], bias=q_func._fc.bias is not None).to(self.device).state_dict())
                else:
                    self._targ_q_func = copy.deepcopy(self._q_func)
            else:
                self._targ_q_func = copy.deepcopy(self._q_func)
        if self._use_model and self._replay_model:
            if task_id not in self._dynamic._models[0]._mus.keys():
                for model in self._dynamic._models:
                    model._mus[task_id] = deepcopy(nn.Linear(model._mu.weight.shape[1], model._mu.weight.shape[0], bias=model._mu.bias is not None).to(self.device).state_dict())
                    model._logstds[task_id] = deepcopy(nn.Linear(model._logstd.weight.shape[1], model._logstd.weight.shape[0], bias=model._logstd.bias is not None).to(self.device).state_dict())
                    model._max_logstds[task_id] = deepcopy(nn.Parameter(torch.empty(1, model._logstd.weight.shape[0], dtype=torch.float32).fill_(2.0).to(self.device)))
                    model._min_logstds[task_id] = deepcopy(nn.Parameter(torch.empty(1, model._logstd.weight.shape[0], dtype=torch.float32).fill_(-10.0).to(self.device)))
        if self._impl_id != task_id:
            if self._clone_actor and self._replay_type == 'bc':
                self._clone_policy._fcs[self._impl_id] = deepcopy(self._clone_policy._fc.state_dict())
                self._clone_policy._fc.load_state_dict(self._clone_policy._fcs[task_id])
            else:
                self._policy._fcs[self._impl_id] = deepcopy(self._policy._fc.state_dict())
                self._policy._fc.load_state_dict(self._policy._fcs[task_id])
            if self._replay_type == 'orl':
                self._targ_policy._fcs[self._impl_id] = deepcopy(self._targ_policy._fc.state_dict())
                self._targ_policy._fc.load_state_dict(self._targ_policy._fcs[task_id])
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
        print(f"self._impl_id:{self._impl_id}")
        print(f"task_id: {task_id}")
