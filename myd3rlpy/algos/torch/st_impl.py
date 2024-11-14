from copy import deepcopy
import inspect
import types
import time
import math
import copy
from typing import Optional, Sequence, List, Any, Tuple, Dict, Union

import numpy as np
import torch
import torch.nn.functional as F

from d3rlpy.torch_utility import train_api, eval_api
from myd3rlpy.torch_utility import torch_api
from d3rlpy.dataset import TransitionMiniBatch

from myd3rlpy.algos.torch.plug.ewc import EWC
from myd3rlpy.algos.torch.plug.rwalk import RWalk
from myd3rlpy.algos.torch.plug.si import SI
from myd3rlpy.algos.torch.plug.gem import GEM
from myd3rlpy.algos.torch.plug.agem import AGEM
from myd3rlpy.algos.torch.plug.piggyback import Piggyback


replay_name = ['observations', 'actions', 'rewards', 'next_observations', 'terminals', 'policy_actions', 'qs']
class STImpl():
    _q_func: torch.nn.Module
    _policy: torch.nn.Module
    device: torch.device
    def __init__(
        self,
        critic_replay_type: str,
        critic_replay_lambda: float,
        actor_replay_type: str,
        actor_replay_lambda: float,
        gem_alpha: float,
        agem_alpha: float,
        ewc_rwalk_alpha: float,
        damping: float,
        epsilon: float,
        fine_tuned_step: int,
        **kwargs,
    ):
        super().__init__(
            **kwargs
        )
        self._critic_replay_type = critic_replay_type
        self._critic_replay_lambda = critic_replay_lambda
        self._actor_replay_type = actor_replay_type
        self._actor_replay_lambda = actor_replay_lambda

        self._gem_alpha = gem_alpha
        self._agem_alpha = agem_alpha
        self._ewc_rwalk_alpha = ewc_rwalk_alpha
        self._damping = damping
        self._epsilon = epsilon
        self._fine_tuned_step = fine_tuned_step

        self._clone_q_func = None
        self._clone_policy = None

        self._impl_id = 0
        self._learned_id = []
        self._new_task = True
        self._critic_plug = None
        self._actor_plug = None

    def change_task(self, new_id):
        self._impl_id = new_id
        if new_id not in self._learned_id:
            self._learned_id.append(new_id)
            self._new_task = True
        else:
            self._new_task = False

    def save_task(self):
        self._save_impl_id = self._impl_id

    def load_task(self):
        self._impl_id = self._save_impl_id

    def build(self):
        super().build()
        if not hasattr(self, "critic_plug") and not hasattr(self, "actor_plug"):
            self._continual_build()

    def _continual_build(self):

        self._critic_networks = [self._q_func]
        self._actor_networks = [self._policy]
        assert self._q_func is not None
        if self._critic_replay_type == "ewc":
            self.critic_plug = EWC(self, self._critic_networks)
        elif self._critic_replay_type == "rwalk":
            self.critic_plug = RWalk(self, self._critic_networks)
        elif self._critic_replay_type == 'si':
            self.critic_plug = SI(self, self._critic_networks)
        elif self._critic_replay_type == 'gem':
            self.critic_plug = GEM(self, self._critic_networks)
        elif self._critic_replay_type == 'agem':
            self.critic_plug = AGEM(self, self._critic_networks)
        elif self._critic_replay_type == 'piggyback':
            self.critic_plug = Piggyback(self, self._critic_networks)
        else:
            self.critic_plug = None
        if self.critic_plug != None:
            self.critic_plug.build()

        if self._actor_replay_type == "ewc":
            self.actor_plug = EWC(self, self._actor_networks)
        elif self._actor_replay_type == "rwalk":
            self.actor_plug = RWalk(self, self._actor_networks)
        elif self._actor_replay_type == 'si':
            self.actor_plug = SI(self, self._actor_networks)
        elif self._actor_replay_type == 'gem':
            self.actor_plug = GEM(self, self._actor_networks)
        elif self._actor_replay_type == 'agem':
            self.actor_plug = AGEM(self, self._actor_networks)
        elif self._actor_replay_type == 'piggyback':
            self.actor_plug = Piggyback(self, self._actor_networks)
        else:
            self.actor_plug = None
        if self.actor_plug != None:
            self.actor_plug.build()

    @train_api
    @torch_api(reward_scaler_targets=["batch", "replay_batch"])
    def update_critic(self, batch: TransitionMiniBatch, replay_batch: TransitionMiniBatch=None, clone_critic: bool=False, online: bool=False, update: bool = True):
        assert self._critic_optim is not None
        assert self._q_func is not None
        assert self._policy is not None

        unreg_grads = None
        curr_feat_ext = None

        replay_loss = 0
        if self._impl_id != 0 and not online:
            replay_loss = 0
            if self._critic_replay_type == "orl":
                assert replay_batch is not None
                q_tpn = self.compute_target(replay_batch)
                replay_cql_loss = self.compute_critic_loss(replay_batch, q_tpn, clone_critic=clone_critic, replay=True)
                replay_loss = replay_loss + replay_cql_loss
            elif self._critic_replay_type == "lwf":
                clone_q = self._clone_q_func(batch.observations, batch.actions)
                q = self._q_func(batch.observations, batch.actions)
                replay_bc_loss = F.mse_loss(clone_q, q)
                if hasattr(self, "_value_func"):
                    clone_value = self._clone_value_func(batch.observations, batch.actions)
                    value = self._value_func(batch.observations, batch.actions)
                    replay_bc_loss += F.mse_loss(clone_value, value)
                replay_loss = replay_loss + replay_bc_loss
            elif self._critic_replay_type == "ewc":
                replay_ewc_loss = self.critic_plug._add_ewc_loss(self._critic_networks, self.critic_plug.fisher, self._critic_plug.older_params)
                replay_loss = replay_loss + replay_ewc_loss
            elif self._critic_replay_type == 'rwalk':
                curr_feat_ext = [{n: p.clone().detach() for n, p in network.named_parameters() if p.requires_grad} for network in self._critic_networks]
                # store gradients without regularization term
                self._critic_optim.zero_grad()
                q_tpn = self.compute_target(batch)
                loss = self.compute_critic_loss(batch, q_tpn, clone_critic=clone_critic, online=online, first_time=replay_batch==None)
                loss.backward(retain_graph=True)

                unreg_grads, replay_rwalk_loss = self.critic_plug._pre_rwalk_loss(self._critic_networks, self.critic_plug.fisher, self._critic_plug.scorers, self._critic_plug.older_params)
                replay_loss = replay_loss + replay_rwalk_loss
            elif self._critic_replay_type == 'si':
                replay_si_loss = self._add_si_loss(self._critic_networks)
                replay_loss = replay_loss + replay_si_loss
            elif self._critic_replay_type == 'gem':
                q_tpn = self.compute_target(replay_batch)
                replay_loss = self.compute_critic_loss(replay_batch, q_tpn, clone_critic=clone_critic)
                replay_loss.backward()
                self._pre_gem_loss(self._critic_networks)
            elif self._critic_replay_type == "agem":
                self._pre_agem_loss(self._critic_networks)
                q_tpn = self.compute_target(replay_batch)
                replay_agem_loss += self.compute_critic_loss(replay_batch, q_tpn, clone_critic=clone_critic)
                replay_loss = replay_loss + replay_agem_loss

        self._critic_optim.zero_grad()
        q_tpn = self.compute_target(batch)
        loss = self.compute_critic_loss(batch, q_tpn, clone_critic=clone_critic, online=online, first_time=replay_batch==None)
        if self._critic_replay_type in ['orl', 'ewc', 'rwalk', 'si', 'bc', 'generate', 'lwf']:
            loss = loss + self._critic_replay_lambda * replay_loss
        loss.backward()
        if replay_batch is not None:
            if self._critic_replay_type == 'agem':
                self._critic_optim.zero_grad()
                replay_loss.backward()
                self._pos_agem_loss(self._critic_networks)
            elif self._critic_replay_type == 'gem':
                self._pos_gem_loss(self._critic_networks)
        if update:
            self._critic_optim.step()

        if replay_batch is not None and not online:
            if self._critic_replay_type == 'rwalk':
                assert unreg_grads is not None
                assert curr_feat_ext is not None
                with torch.no_grad():
                    self.critic_plug._pos_rwalk_loss(self._critic_networks, unreg_grads, self.critic_plug.W, curr_feat_ext)

        loss = loss.cpu().detach().numpy()
        if not isinstance(replay_loss, int):
            replay_loss = replay_loss.cpu().detach().numpy()

        return loss, replay_loss

    @train_api
    @torch_api(reward_scaler_targets=["batch", "replay_batch"])
    def merge_update_critic(self, batch, replay_batch):
        replay_loss = 0
        if self._impl_id != 0:
            if self._critic_replay_type == 'bc':
                with torch.no_grad():
                    replay_observations = replay_batch.observations.to(self.device)
                    observations = batch.observations
            merge_observations = torch.cat([replay_observations, observations], dim=0)
            merge_batch_actions = self._policy(merge_observations)
            merge_clone_qs = self._clone_q_func(merge_observations, merge_batch_actions).detach()
            merge_batch_qs = self._q_func(merge_observations, merge_batch_actions)
            max_qs = torch.max(merge_clone_qs, merge_batch_qs)
            replay_loss += F.mse_loss(max_qs, merge_clone_qs)
        self._critic_optim.zero_grad()
        replay_loss.backward()
        self._critic_optim.step()
        return replay_loss.cpu().detach().numpy

    @train_api
    @torch_api(scaler_targets=["batch", "replay_batch"])
    def merge_update_actor(self, batch, replay_batch):
        replay_loss = 0
        if self._impl_id != 0:
            if self._critic_replay_type == 'bc':
                with torch.no_grad():
                    replay_observations = replay_batch.observations.to(self.device)
                    observations = batch.observations
                    merge_observations = torch.cat([replay_observations, observations], dim=0)
            merge_clone_actions = self._clone_policy(merge_observations)
            merge_clone_qs = self._clone_q_func(merge_observations, merge_clone_actions)
            merge_batch_actions = self._policy(merge_observations)
            merge_batch_qs = self._q_func(merge_observations, merge_batch_actions)
            max_actions = torch.where(merge_clone_qs > merge_batch_qs, merge_clone_actions, merge_batch_actions)
            replay_loss += F.mse_loss(max_actions, merge_batch_actions)
        self._actor_optim.zero_grad()
        replay_loss.backward()
        self._actor_optim.step()
        return replay_loss.cpu().detach().numpy

    @train_api
    @torch_api(scaler_targets=["batch", "replay_batch"])
    def update_actor(self, batch: TransitionMiniBatch, replay_batch: Optional[List[torch.Tensor]]=None, clone_actor: bool=False, online: bool=False, update: bool = True) -> np.ndarray:
        assert self._q_func is not None
        assert self._policy is not None
        assert self._actor_optim is not None

        # Q function should be inference mode for stability
        self._q_func.eval()
        self._policy.train()

        unreg_grads = None
        curr_feat_ext = None

        loss = self.compute_actor_loss(batch, clone_actor=clone_actor, online=online)
        replay_loss = 0
        if self._impl_id != 0 and not online:
            replay_loss = 0
            if self._actor_replay_type == "orl":
                replay_loss_ = self.compute_actor_loss(replay_batch, clone_actor=clone_actor, replay=True)
                replay_loss = replay_loss + replay_loss_
            elif self._actor_replay_type == 'lwf':
                clone_actions = self._clone_policy(batch.observations)
                actions = self._policy(batch.observations)
                replay_loss_ = F.mse_loss(clone_actions, actions)
                replay_loss = replay_loss + replay_loss_
            elif self._actor_replay_type == 'lwf_orl':
                clone_actions = self._clone_policy(batch.observations)
                actions = self._policy(batch.observations)
                q_t = self._q_func(batch.observations, actions, "none")[0]
                replay_loss_ = -q_t.mean()
                replay_loss = replay_loss + replay_loss_
            elif self._actor_replay_type == "ewc":
                replay_ewc_loss = self.actor_plug._add_ewc_loss(self._actor_networks, self.actor_plug.fisher, self.actor_plug.older_params)
                replay_loss = replay_loss + replay_ewc_loss
            elif self._actor_replay_type == 'rwalk':
                curr_feat_ext = {n: p.clone().detach() for n, p in self._policy.named_parameters() if p.requires_grad}
                self._actor_optim.zero_grad()
                loss = self.compute_actor_loss(batch, clone_actor=clone_actor)
                loss.backward(retain_graph=True)
                # Eq. 3: elastic weight consolidation quadratic penalty
                unreg_grads, replay_rwalk_loss = self._pre_rwalk_loss(self._actor_networks, self.actor_plug.fisher, self.critic_plug.scorers, self._critic_plug.older_params)
                replay_loss = replay_loss + replay_rwalk_loss
            elif self._actor_replay_type == 'si':
                replay_si_loss = self._add_si_loss(self._actor_networks)
                replay_loss = replay_loss + replay_si_loss
            elif self._actor_replay_type == 'gem':
                replay_loss_ = self.compute_actor_loss(replay_batch, clone_actor=clone_actor)
                replay_loss = replay_loss_
                replay_loss.backward()
                self._pre_gem_loss(self._actor_networks)
            elif self._actor_replay_type == "agem":
                self._pre_agem_loss(self._actor_networks)
                replay_loss_ = self.compute_actor_loss(replay_batch, clone_actor=clone_actor)
                replay_loss = replay_loss + replay_loss_

        self._actor_optim.zero_grad()
        if self._actor_replay_type in ['orl', 'ewc', 'rwalk', 'si', 'bc', 'generate', 'generate_orl', 'lwf', 'lwf_orl']:
            loss = loss + self._actor_replay_lambda * replay_loss
        if not isinstance(loss, int):
            loss.backward()

        if replay_batch is not None and not online:
            if self._actor_replay_type == 'agem':
                self._actor_optim.zero_grad()
                replay_loss.backward()
                self._pos_agem_loss(self._actor_networks)
            elif self._actor_replay_type == 'gem':
                self._pos_gem_loss(self._actor_networks)
        if update:
            self._actor_optim.step()

        if replay_batch is not None and not online:
            if self._actor_replay_type == 'rwalk':
                assert unreg_grads is not None
                assert curr_feat_ext is not None
                with torch.no_grad():
                    self.actor_plug._pos_rwalk_loss(self._actor_networks, unreg_grads, self.actor_plug.W, curr_feat_ext)

        if not isinstance(loss, int):
            loss = loss.cpu().detach().numpy()
        if not isinstance(replay_loss, int):
            replay_loss = replay_loss.cpu().detach().numpy()

        return loss, replay_loss

    def critic_ewc_rwalk_post_train_process(self, iterator, batch_size, n_frames, n_steps, gamma, test=False):
        # calculate Fisher information
        @train_api
        @torch_api()
        def update(self, batch):
            q_tpn = self._algo.compute_target(batch)
            loss = self._algo.compute_critic_loss(batch, q_tpn)
            loss.backward()
        self.critic_plug._ewc_rwalk_post_train_process(iterator, self._critic_optim, update, batch_size=batch_size, n_frames=n_frames, n_steps=n_steps, gamma=gamma, test=test)

    def actor_ewc_rwalk_post_train_process(self, iterator, batch_size, n_frames, n_steps, gamma, test=False):
        @train_api
        @torch_api()
        def update(self, batch):
            loss = self._algo.compute_actor_loss(batch)
            loss.backward()
        self.actor_plug._ewc_rwalk_post_train_process(iterator, self._actor_optim, update, batch_size=batch_size, n_frames=n_frames, n_steps=n_steps, gamma=gamma, test=test)
    def fine_tuned_action(self, observation):
        assert self._policy is not None
        assert self._q_func is not None
        self._q_func.eval()
        _temp_policy = copy.deepcopy(self._policy)
        _temp_actor_optim = self._actor_optim_factory.create(
            _temp_policy.parameters(), lr=self._actor_learning_rate
        )
        for i in range(self._fine_tuned_step):
            action = _temp_policy(observation)
            q_t = self._q_func(observation, action, "none")[0]
            loss = (- q_t).mean()
            _temp_actor_optim.zero_grad()
            loss.backward()
            _temp_actor_optim.step()
        action = _temp_policy(observation)
        return action

    def save_clone_data(self):
        assert self._q_func is not None
        assert self._policy is not None
        self._clone_policy = copy.deepcopy(self._policy)

    def save_clone_policy(self):
        assert self._policy is not None
        self._clone_q_func = copy.deepcopy(self._q_func)
        self._clone_policy = copy.deepcopy(self._policy)

    def load_model(self, fname: str) -> None:
        chkpt = torch.load(fname, map_location=self._device)
        BLACK_LIST = [
            "policy",
            "q_function",
            "policy_optim",
            "q_function_optim",
        ]  # special properties


        keys = [key for key in dir(self) if key not in BLACK_LIST]
        for key in keys:
            if 'clone' not in key:
                obj = getattr(self, key)
                if isinstance(obj, (torch.nn.Module, torch.optim.Optimizer)):
                    obj.load_state_dict(chkpt[key])

    #def rebuild(self):
    #    self._build_alpha()
    #    self._build_temperature()
    #    if self._use_gpu:
    #        self.to_gpu(self._use_gpu)
    #    else:
    #        self.to_cpu()
    #    self._build_alpha_optim()
    #    self._build_temperature_optim()
    #    self._build_actor_optim()
    #    self._build_critic_optim()
    #    assert self._q_func is not None
    #    assert self._policy is not None
    #    if self._critic_replay_type in ['ewc', 'rwalk', 'si'] and '_critic_plug.older_params' not in self.__dict__.keys():
    #        # Store current parameters for the next task
    #        self.critic_plug.older_params = {n: p.clone().detach() for n, p in self._q_func.named_parameters() if p.requires_grad}
    #        if self._critic_replay_type in ['ewc', 'rwalk'] and '_critic_plug.fisher' not in self.__dict__.keys():
    #            # Store fisher information weight importance
    #            self.critic_plug.fisher = {n: torch.zeros(p.shape).to(self.device) for n, p in self._q_func.named_parameters() if p.requires_grad}
    #        if self._critic_replay_type == 'rwalk' and '_critic_plug.W' not in self.__dict__.keys():
    #            # Page 7: "task-specific parameter importance over the entire training trajectory."
    #            self.critic_plug.W = {n: torch.zeros(p.shape).to(self.device) for n, p in self._q_func.named_parameters() if p.requires_grad}
    #            self.critic_plug.scorers = {n: torch.zeros(p.shape).to(self.device) for n, p in self._q_func.named_parameters() if p.requires_grad}
    #        elif self._critic_replay_type == 'si' and '_critic_plug.W' not in self.__dict__.keys():
    #            self.critic_plug.W = {n: p.clone().detach().zero_() for n, p in self._q_func.named_parameters() if p.requires_grad}
    #            self._critic_omega = {n: p.clone().detach().zero_() for n, p in self._q_func.named_parameters() if p.requires_grad}
    #    if self._actor_replay_type in ['ewc', 'rwalk', 'si'] and '_actor_plug.older_params' not in self.__dict__.keys():
    #        # Store current parameters for the next task
    #        self.actor_plug.older_params = {n: p.clone().detach() for n, p in self._policy.named_parameters() if p.requires_grad}
    #        if self._actor_replay_type in ['ewc', 'rwalk'] and '_actor_plug.fisher' not in self.__dict__.keys():
    #            # Store fisher information weight importance
    #            self.actor_plug.fisher = {n: torch.zeros(p.shape).to(self.device) for n, p in self._policy.named_parameters() if p.requires_grad}
    #        if self._actor_replay_type == 'rwalk' and '_actor_w' not in self.__dict__.keys():
    #            # Page 7: "task-specific parameter importance over the entire training trajectory."
    #            self._actor_w = {n: torch.zeros(p.shape).to(self.device) for n, p in self._policy.named_parameters() if p.requires_grad}
    #            self.actor_plug.scorers = {n: torch.zeros(p.shape).to(self.device) for n, p in self._policy.named_parameters() if p.requires_grad}
    #        elif self._critic_replay_type == 'si' and '_actor_plug.W' not in self.__dict__.keys():
    #            self.actor_plug.W = {n: p.clone().detach().zero_() for n, p in self._policy.named_parameters() if p.requires_grad}
    #            self._actor_omega = {n: p.clone().detach().zero_() for n, p in self._policy.named_parameters() if p.requires_grad}
    #    elif self._critic_replay_type == 'gem':
    #        # Allocate temporary synaptic memory
    #        self._critic_grad_dims = []
    #        for pp in self._q_func.parameters():
    #            self._critic_grad_dims.append(pp.data.numel())
    #        self._critic_grads_cs = {}
    #        self._critic_grads_da = torch.zeros(np.sum(self._critic_grad_dims)).to(self.device)

    #    elif self._actor_replay_type == 'gem':
    #        self._actor_grad_dims = []
    #        for pp in self._policy.parameters():
    #            self._actor_grad_dims.append(pp.data.numel())
    #        self._actor_grads_cs = {}
    #        self._actor_grads_da = torch.zeros(np.sum(self._actor_grad_dims)).to(self.device)

    #    elif self._critic_replay_type == 'agem':
    #        self._critic_grad_dims = []
    #        for param in self._q_func.parameters():
    #            self._critic_grad_dims.append(param.data.numel())
    #        self._critic_grad_xy = torch.Tensor(np.sum(self._critic_grad_dims)).to(self.device)
    #        self._critic_grad_er = torch.Tensor(np.sum(self._critic_grad_dims)).to(self.device)
    #    elif self._actor_replay_type == 'agem':
    #        self._actor_grad_dims = []
    #        for param in self._policy.parameters():
    #            self._actor_grad_dims.append(param.data.numel())
    #        self._actor_grad_xy = torch.Tensor(np.sum(self._actor_grad_dims)).to(self.device)
    #        self._actor_grad_er = torch.Tensor(np.sum(self._actor_grad_dims)).to(self.device)
