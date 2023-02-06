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
from d3rlpy.algos.base import AlgoImplBase
from d3rlpy.algos.torch.base import TorchImplBase
from d3rlpy.torch_utility import hard_sync

from myd3rlpy.algos.torch.gem import overwrite_grad, store_grad, project2cone2
from myd3rlpy.algos.torch.agem import project
from utils.utils import Struct


replay_name = ['observations', 'actions', 'rewards', 'next_observations', 'terminals', 'policy_actions', 'qs', 'phis', 'psis']
class STImpl():
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
        clone_actor: bool,
        clone_critic: bool,
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

    def build(self):
        self._dynamic = None

        super().build()

        assert self._q_func is not None
        assert self._policy is not None
        if self._critic_replay_type in ['ewc', 'rwalk', 'si']:
            # Store current parameters for the next task
            self._critic_older_params = {n: p.clone().detach() for n, p in self._q_func.named_parameters() if p.requires_grad}
            if self._critic_replay_type in ['ewc', 'rwalk']:
                # Store fisher information weight importance
                self._critic_fisher = {n: torch.zeros(p.shape).to(self.device) for n, p in self._q_func.named_parameters() if p.requires_grad}
            if self._critic_replay_type == 'rwalk':
                # Page 7: "task-specific parameter importance over the entire training trajectory."
                self._critic_W = {n: torch.zeros(p.shape).to(self.device) for n, p in self._q_func.named_parameters() if p.requires_grad}
                self._critic_scores = {n: torch.zeros(p.shape).to(self.device) for n, p in self._q_func.named_parameters() if p.requires_grad}
            elif self._critic_replay_type == 'si':
                self._critic_W = {n: p.clone().detach().zero_() for n, p in self._q_func.named_parameters() if p.requires_grad}
                self._critic_omega = {n: p.clone().detach().zero_() for n, p in self._q_func.named_parameters() if p.requires_grad}
        elif self._critic_replay_type == 'gem':
            # Allocate temporary synaptic memory
            self._critic_grad_dims = []
            for pp in self._q_func.parameters():
                self._critic_grad_dims.append(pp.data.numel())
            self._critic_grads_cs = {}
            self._critic_grads_da = torch.zeros(np.sum(self._critic_grad_dims)).to(self.device)
        elif self._critic_replay_type == 'agem':
            self._critic_grad_dims = []
            for param in self._q_func.parameters():
                self._critic_grad_dims.append(param.data.numel())
            self._critic_grad_xy = torch.Tensor(np.sum(self._critic_grad_dims)).to(self.device)
            self._critic_grad_er = torch.Tensor(np.sum(self._critic_grad_dims)).to(self.device)

        if self._actor_replay_type in ['ewc', 'rwalk', 'si']:
            # Store current parameters for the next task
            self._actor_older_params = {n: p.clone().detach() for n, p in self._policy.named_parameters() if p.requires_grad}
            if self._actor_replay_type in ['ewc', 'rwalk']:
                # Store fisher information weight importance
                self._actor_fisher = {n: torch.zeros(p.shape).to(self.device) for n, p in self._policy.named_parameters() if p.requires_grad}
            if self._actor_replay_type == 'rwalk':
                # Page 7: "task-specific parameter importance over the entire training trajectory."
                self._actor_w = {n: torch.zeros(p.shape).to(self.device) for n, p in self._policy.named_parameters() if p.requires_grad}
                self._actor_scores = {n: torch.zeros(p.shape).to(self.device) for n, p in self._policy.named_parameters() if p.requires_grad}
            elif self._critic_replay_type == 'si':
                self._actor_W = {n: p.clone().detach().zero_() for n, p in self._policy.named_parameters() if p.requires_grad}
                self._actor_omega = {n: p.clone().detach().zero_() for n, p in self._policy.named_parameters() if p.requires_grad}
        elif self._actor_replay_type == 'gem':
            self._actor_grad_dims = []
            for pp in self._policy.parameters():
                self._actor_grad_dims.append(pp.data.numel())
            self._actor_grads_cs = {}
            self._actor_grads_da = torch.zeros(np.sum(self._actor_grad_dims)).to(self.device)
        elif self._actor_replay_type == 'agem':
            self._actor_grad_dims = []
            for param in self._policy.parameters():
                self._actor_grad_dims.append(param.data.numel())
            self._actor_grad_xy = torch.Tensor(np.sum(self._actor_grad_dims)).to(self.device)
            self._actor_grad_er = torch.Tensor(np.sum(self._actor_grad_dims)).to(self.device)

        self._vae_impl = None

    def rebuild(self):
        self._build_alpha()
        self._build_temperature()
        if self._use_gpu:
            self.to_gpu(self._use_gpu)
        else:
            self.to_cpu()
        self._build_alpha_optim()
        self._build_temperature_optim()
        self._build_actor_optim()
        self._build_critic_optim()
        assert self._q_func is not None
        assert self._policy is not None
        if self._critic_replay_type in ['ewc', 'rwalk', 'si'] and '_critic_older_params' not in self.__dict__.keys():
            # Store current parameters for the next task
            self._critic_older_params = {n: p.clone().detach() for n, p in self._q_func.named_parameters() if p.requires_grad}
            if self._critic_replay_type in ['ewc', 'rwalk'] and '_critic_fisher' not in self.__dict__.keys():
                # Store fisher information weight importance
                self._critic_fisher = {n: torch.zeros(p.shape).to(self.device) for n, p in self._q_func.named_parameters() if p.requires_grad}
            if self._critic_replay_type == 'rwalk' and '_critic_W' not in self.__dict__.keys():
                # Page 7: "task-specific parameter importance over the entire training trajectory."
                self._critic_W = {n: torch.zeros(p.shape).to(self.device) for n, p in self._q_func.named_parameters() if p.requires_grad}
                self._critic_scores = {n: torch.zeros(p.shape).to(self.device) for n, p in self._q_func.named_parameters() if p.requires_grad}
            elif self._critic_replay_type == 'si' and '_critic_W' not in self.__dict__.keys():
                self._critic_W = {n: p.clone().detach().zero_() for n, p in self._q_func.named_parameters() if p.requires_grad}
                self._critic_omega = {n: p.clone().detach().zero_() for n, p in self._q_func.named_parameters() if p.requires_grad}
        if self._actor_replay_type in ['ewc', 'rwalk', 'si'] and '_actor_older_params' not in self.__dict__.keys():
            # Store current parameters for the next task
            self._actor_older_params = {n: p.clone().detach() for n, p in self._policy.named_parameters() if p.requires_grad}
            if self._actor_replay_type in ['ewc', 'rwalk'] and '_actor_fisher' not in self.__dict__.keys():
                # Store fisher information weight importance
                self._actor_fisher = {n: torch.zeros(p.shape).to(self.device) for n, p in self._policy.named_parameters() if p.requires_grad}
            if self._actor_replay_type == 'rwalk' and '_actor_w' not in self.__dict__.keys():
                # Page 7: "task-specific parameter importance over the entire training trajectory."
                self._actor_w = {n: torch.zeros(p.shape).to(self.device) for n, p in self._policy.named_parameters() if p.requires_grad}
                self._actor_scores = {n: torch.zeros(p.shape).to(self.device) for n, p in self._policy.named_parameters() if p.requires_grad}
            elif self._critic_replay_type == 'si' and '_actor_W' not in self.__dict__.keys():
                self._actor_W = {n: p.clone().detach().zero_() for n, p in self._policy.named_parameters() if p.requires_grad}
                self._actor_omega = {n: p.clone().detach().zero_() for n, p in self._policy.named_parameters() if p.requires_grad}
        elif self._critic_replay_type == 'gem':
            # Allocate temporary synaptic memory
            self._critic_grad_dims = []
            for pp in self._q_func.parameters():
                self._critic_grad_dims.append(pp.data.numel())
            self._critic_grads_cs = {}
            self._critic_grads_da = torch.zeros(np.sum(self._critic_grad_dims)).to(self.device)

        elif self._actor_replay_type == 'gem':
            self._actor_grad_dims = []
            for pp in self._policy.parameters():
                self._actor_grad_dims.append(pp.data.numel())
            self._actor_grads_cs = {}
            self._actor_grads_da = torch.zeros(np.sum(self._actor_grad_dims)).to(self.device)

        elif self._critic_replay_type == 'agem':
            self._critic_grad_dims = []
            for param in self._q_func.parameters():
                self._critic_grad_dims.append(param.data.numel())
            self._critic_grad_xy = torch.Tensor(np.sum(self._critic_grad_dims)).to(self.device)
            self._critic_grad_er = torch.Tensor(np.sum(self._critic_grad_dims)).to(self.device)
        elif self._actor_replay_type == 'agem':
            self._actor_grad_dims = []
            for param in self._policy.parameters():
                self._actor_grad_dims.append(param.data.numel())
            self._actor_grad_xy = torch.Tensor(np.sum(self._actor_grad_dims)).to(self.device)
            self._actor_grad_er = torch.Tensor(np.sum(self._actor_grad_dims)).to(self.device)

    @train_api
    def update_critic(self, batch_tran: TransitionMiniBatch, replay_batch: Optional[List[torch.Tensor]]=None, clone_critic: bool=False, online: bool=False):
        batch = TorchMiniBatch(
            batch_tran,
            self.device,
            scaler=self.scaler,
            action_scaler=self.action_scaler,
            reward_scaler=self.reward_scaler,
        )
        assert self._critic_optim is not None
        assert self._q_func is not None
        assert self._policy is not None

        unreg_grads = None
        curr_feat_ext = None

        loss = 0
        replay_loss = 0
        if replay_batch is not None and not online:
            replay_loss = 0
            replay_batch = [x.to(self.device) for x in replay_batch]
            replay_batch = dict(zip(replay_name[:-2], replay_batch))
            replay_batch = Struct(**replay_batch)
            if self._critic_replay_type == "orl":
                replay_batch.n_steps = 1
                replay_batch.masks = None
                replay_batch = cast(TorchMiniBatch, replay_batch)
                q_tpn = self.compute_target(replay_batch)
                replay_cql_loss = self.compute_critic_loss(replay_batch, q_tpn)
                replay_loss = replay_loss + replay_cql_loss
            elif self._critic_replay_type == "bc":
                with torch.no_grad():
                    replay_observations = replay_batch.observations.to(self.device)
                    replay_actions = replay_batch.actions.to(self.device)
                    replay_qs = replay_batch.qs.to(self.device)

                q = self._q_func(replay_observations, replay_actions)
                replay_bc_loss = F.mse_loss(replay_qs, q)
                replay_loss = replay_loss + replay_bc_loss
            elif self._critic_replay_type == 'generate':
                assert self._vae_impl is not None
                assert self._clone_policy is not None
                assert self._clone_q_func is not None
                generate_observations = self._clone_vae_impl.generate(batch.observations.shape[0]).detach()
                generate_actions = self._clone_policy(generate_observations).detach()
                generate_qs = self._clone_q_func(generate_observations, generate_actions).detach()
                qs = self._q_func(generate_observations, generate_actions)
                replay_generate_loss = F.mse_loss(generate_qs, qs)
                replay_loss = replay_loss + replay_bc_loss
            elif self._critic_replay_type == "ewc":
                replay_loss_ = 0
                for n, p in self._q_func.named_parameters():
                    if n in self._critic_fisher.keys():
                        replay_loss = replay_loss + torch.mean(self._critic_fisher[n] * (p - self._critic_older_params[n]).pow(2)) / 2
                replay_loss = replay_loss + replay_loss_
            elif self._critic_replay_type == 'rwalk':
                curr_feat_ext = {n: p.clone().detach() for n, p in self._q_func.named_parameters() if p.requires_grad}
                # store gradients without regularization term
                unreg_grads = {n: p.grad.clone().detach() for n, p in self._q_func.named_parameters()
                               if p.grad is not None}

                self._critic_optim.zero_grad()
                # Eq. 3: elastic weight consolidation quadratic penalty
                replay_loss_ = 0
                for n, p in self._q_func.named_parameters():
                    if n in self._critic_fisher.keys():
                        replay_loss_ = replay_loss_ + torch.mean((self._critic_fisher[n] + self._critic_scores[n]) * (p - self._critic_older_params[n]).pow(2)) / 2
                replay_loss = replay_loss + replay_loss_
            elif self._critic_replay_type == 'si':
                for n, p in self._q_func.named_parameters():
                    if p.grad is not None and n in self._critic_fisher.keys():
                        self._critic_W[n].add_(-p.grad * (p.detach() - self._critic_older_params[n]))
                    self._critic_older_params[n] = p.detach().clone()
                replay_loss_ = 0
                for n, p in self.q_func.named_parameters():
                    if p.requires_grad:
                        replay_loss_ = replay_loss_ + torch.mean(self._critic_omega[n] * (p - self._critic_older_params[n]) ** 2)
                replay_loss = replay_loss + replay_loss_
            elif self._critic_replay_type == 'gem':
                replay_batch = cast(TorchMiniBatch, replay_batch)
                q_tpn = self.compute_target(replay_batch)
                replay_loss_ = self.compute_critic_loss(replay_batch, q_tpn)
                replay_loss = replay_loss_
                replay_loss.backward()
                store_grad(self._q_func.parameters, self._critic_grads_cs[i], self._critic_grad_dims)
            elif self._critic_replay_type == "agem":
                store_grad(self._q_func.parameters, self._critic_grad_xy, self._critic_grad_dims)
                replay_batch.n_steps = 1
                replay_batch.masks = None
                replay_batch = cast(TorchMiniBatch, replay_batch)
                q_tpn = self.compute_target(replay_batch)
                replay_loss_ = self.compute_critic_loss(replay_batch, q_tpn)
                replay_loss = replay_loss + replay_loss_

        self._critic_optim.zero_grad()
        q_tpn = self.compute_target(batch)
        loss = self.compute_critic_loss(batch, q_tpn, clone_critic=clone_critic, online=online)
        if self._critic_replay_type in ['orl', 'ewc', 'rwalk', 'si', 'bc']:
            loss = loss + self._critic_replay_lambda * replay_loss
        loss.backward()
        if replay_batch is not None:
            if self._critic_replay_type == 'agem':
                assert self._single_head
                replay_loss.backward()
                store_grad(self._q_func.parameters, self._critic_grad_er, self._critic_grad_dims)
                dot_prod = torch.dot(self._critic_grad_xy, self._critic_grad_er)
                if dot_prod.item() < 0:
                    g_tilde = project(gxy=self._critic_grad_xy, ger=self._critic_grad_er)
                    overwrite_grad(self._q_func.parameters, g_tilde, self._critic_grad_dims)
                else:
                    overwrite_grad(self._q_func.parameters, self._critic_grad_xy, self._critic_grad_dims)
            elif self._critic_replay_type == 'gem':
                # copy gradient
                store_grad(self._q_func.parameters, self._critic_grads_da, self._critic_grad_dims)
                dot_prod = torch.mm(self._critic_grads_da.unsqueeze(0),
                                torch.stack(list(self._critic_grads_cs).values()).T)
                if (dot_prod < 0).sum() != 0:
                    project2cone2(self._critic_grads_da.unsqueeze(1),
                                  torch.stack(list(self._critic_grads_cs).values()).T, margin=self._gem_alpha)
                    # copy gradients back
                    overwrite_grad(self._q_func.parameters, self._critic_grads_da,
                                   self._critic_grad_dims)
        self._critic_optim.step()

        if replay_batch is not None and not online:
            if self._critic_replay_type == 'rwalk':
                assert unreg_grads is not None
                assert curr_feat_ext is not None
                with torch.no_grad():
                    for n, p in self._q_func.named_parameters():
                        if n in unreg_grads.keys():
                            self._critic_W[n] -= unreg_grads[n] * (p.detach() - curr_feat_ext[n])

        loss = loss.cpu().detach().numpy()
        if not isinstance(replay_loss, int):
            replay_loss = replay_loss.cpu().detach().numpy()

        return loss, replay_loss

    @train_api
    def update_actor(self, batch_tran: TransitionMiniBatch, replay_batch: Optional[List[torch.Tensor]]=None, clone_actor=False, online: bool=False) -> np.ndarray:
    # def update_actor(self, batch_tran: TransitionMiniBatch, replay_batch: Optional[List[torch.Tensor]]=None, online: bool=False) -> np.ndarray:
        assert self._q_func is not None
        assert self._policy is not None
        assert self._actor_optim is not None
        batch = TorchMiniBatch(
            batch_tran,
            self.device,
            scaler=self.scaler,
            action_scaler=self.action_scaler,
            reward_scaler=self.reward_scaler,
        )

        # Q function should be inference mode for stability
        self._q_func.eval()
        self._policy.train()

        unreg_grads = None
        curr_feat_ext = None

        loss = 0
        replay_loss = 0
        if replay_batch is not None and not online:
            replay_loss = 0
            replay_batch = [x.to(self.device) for x in replay_batch]
            replay_batch = dict(zip(replay_name[:-2], replay_batch))
            replay_batch = Struct(**replay_batch)
            if self._actor_replay_type == "orl":
                replay_batch = cast(TorchMiniBatch, replay_batch)
                replay_loss_ = self.compute_actor_loss(replay_batch)
                replay_loss = replay_loss + replay_loss_
            elif self._actor_replay_type == "bc":
                with torch.no_grad():
                    replay_observations = replay_batch.observations.to(self.device)
                    replay_policy_actions = replay_batch.policy_actions.to(self.device)
                    replay_qs = replay_batch.qs.to(self.device)
                replay_batch = cast(TorchMiniBatch, replay_batch)
                actions = self._policy(replay_batch.observations)
                q_t = self._q_func(replay_batch.observations, actions, "min")
                replay_q_t = self._q_func(replay_batch.observations, replay_batch.actions, "min")
                replay_loss_ = torch.mean((replay_policy_actions - actions) ** 2, dim=1)
                zero_loss = torch.zeros_like(replay_loss_)
                replay_loss_ = torch.where(replay_q_t > q_t, replay_loss_, zero_loss).mean()
                replay_loss = replay_loss + replay_loss_
            elif self._actor_replay_type == 'generate':
                assert self._vae_impl is not None
                assert self._clone_policy is not None
                generate_observations = self._clone_vae_impl.generate(batch.observations.shape[0]).detach()
                generate_actions = self._clone_policy(generate_observations).detach()
                actions = self._policy(generate_observations)
                qs = self._q_func(generate_observations, generate_actions)
                replay_generate_loss = F.mse_loss(generate_qs, qs)
                replay_loss = replay_loss + replay_bc_loss
            elif self._actor_replay_type == "ewc":
                replay_loss_ = 0
                for n, p in self._policy.named_parameters():
                    if n in self._actor_fisher.keys():
                        replay_loss_ = torch.mean(self._actor_fisher[n] * (p - self._actor_older_params[n]).pow(2)) / 2
                replay_loss = replay_loss + replay_loss_
            elif self._actor_replay_type == 'rwalk':
                curr_feat_ext = {n: p.clone().detach() for n, p in self._policy.named_parameters() if p.requires_grad}
                # store gradients without regularization term
                unreg_grads = {n: p.grad.clone().detach() for n, p in self._policy.named_parameters()
                               if p.grad is not None}

                self._actor_optim.zero_grad()
                # Eq. 3: elastic weight consolidation quadratic penalty
                replay_loss_ = 0
                for n, p in self._policy.named_parameters():
                    if n in self._actor_fisher.keys():
                        replay_loss_ = replay_loss_  + torch.mean((self._actor_fisher[n] + self._actor_scores[n]) * (p - self._actor_older_params[n]).pow(2)) / 2
                replay_loss = replay_loss + replay_loss_
            elif self._actor_replay_type == 'si':
                for n, p in self._policy.named_parameters():
                    if p.grad is not None and n in self._actor_fisher.keys():
                        self._actor_W[n].add_(-p.grad * (p.detach() - self._actor_older_params[n]))
                    self._actor_older_params[n] = p.detach().clone()
                replay_loss_ = 0
                for n, p in self.named_parameters():
                    if p.requires_grad:
                        replay_loss_ = replay_loss_ + torch.mean(self._actor_omega[n] * (p - self._actor_older_params[n]) ** 2)
                replay_loss = replay_loss + replay_loss_
            elif self._actor_replay_type == 'gem':
                replay_batch = cast(TorchMiniBatch, replay_batch)
                replay_loss_ = self.compute_actor_loss(replay_batch)
                replay_loss = replay_loss_
                replay_loss.backward()
                store_grad(self._policy.parameters, self._actor_grads_cs[i], self._actor_grad_dims)
            elif self._actor_replay_type == "agem":
                store_grad(self._policy.parameters, self._actor_grad_xy, self._actor_grad_dims)
                replay_batch = cast(TorchMiniBatch, replay_batch)
                replay_loss_ = self.compute_actor_loss(replay_batch)
                replay_loss = replay_loss + replay_loss_

        self._actor_optim.zero_grad()
        loss += self.compute_actor_loss(batch, clone_actor, online)
        # loss += self.compute_actor_loss(batch, online)
        if self._actor_replay_type in ['orl', 'ewc', 'rwalk', 'si', 'bc']:
            loss = loss + self._actor_replay_lambda * replay_loss
        if not isinstance(loss, int):
            loss.backward()

        if replay_batch is not None and not online:
            if self._actor_replay_type == 'agem':
                assert self._single_head
                replay_loss.backward()
                store_grad(self._policy.parameters, self._actor_grad_er, self._actor_grad_dims)
                dot_prod = torch.dot(self._actor_grad_xy, self._actor_grad_er)
                if dot_prod.item() < 0:
                    g_tilde = project(gxy=self._actor_grad_xy, ger=self._actor_grad_er)
                    overwrite_grad(self._policy.parameters, g_tilde, self._actor_grad_dims)
                else:
                    overwrite_grad(self._policy.parameters, self._actor_grad_xy, self._actor_grad_dims)
            elif self._actor_replay_type == 'gem':
                # copy gradient
                store_grad(self._policy.parameters, self._actor_grads_da, self._actor_grad_dims)
                dot_prod = torch.mm(self._actor_grads_da.unsqueeze(0),
                                torch.stack(list(self._actor_grads_cs).values()).T)
                if (dot_prod < 0).sum() != 0:
                    project2cone2(self._actor_grads_da.unsqueeze(1),
                                  torch.stack(list(self._actor_grads_cs).values()).T, margin=self._gem_alpha)
                    # copy gradients back
                    overwrite_grad(self._policy.parameters, self._actor_grads_da,
                                   self._actor_grad_dims)

        self._actor_optim.step()

        if not isinstance(loss, int):
            loss = loss.cpu().detach().numpy()
        if not isinstance(replay_loss, int):
            replay_loss = replay_loss.cpu().detach().numpy()

        return loss, replay_loss

    def compute_fisher_matrix_diag(self, iterator, network, optim, update):
        # Store Fisher Information
        fisher = {n: torch.zeros(p.shape).to(self.device) for n, p in network.named_parameters()
                  if p.requires_grad}
        # Do forward and backward pass to compute the fisher information
        network.train()
        replay_loss = 0
        iterator.reset()
        for t in range(len(iterator)):
            batch = next(iterator)
            optim.zero_grad()
            update(batch)
            # Accumulate all gradients from loss with regularization
            for n, p in network.named_parameters():
                if p.grad is not None:
                    fisher[n] += p.grad.pow(2)
        # Apply mean across all samples
        fisher = {n: (p / len(iterator)) for n, p in fisher.items()}
        return fisher

    def gem_post_train_process(self):
        self._critic_grads_cs[self._impl_id] = torch.zeros(np.sum(self._critic_grad_dims)).to(self.device)
        self._actor_grads_cs[self._impl_id] = torch.zeros(np.sum(self._actor_grad_dims)).to(self.device)
        if self._use_model:
            self._model_grads_cs[self._impl_id] = torch.zeros(np.sum(self._model_grad_dims)).to(self.device)

    def critic_ewc_rwalk_post_train_process(self, iterator):
        # calculate Fisher information
        def update(batch):
            batch = TorchMiniBatch(
                batch,
                self.device,
                scaler=self.scaler,
                action_scaler=self.action_scaler,
                reward_scaler=self.reward_scaler,
            )
            q_tpn = self.compute_target(batch)
            loss = self.compute_critic_loss(batch, q_tpn)
            loss.backward
        curr_fisher = self.compute_fisher_matrix_diag(iterator, self._q_func, self._critic_optim, update)
        # merge fisher information, we do not want to keep fisher information for each task in memory
        for n in self._critic_fisher.keys():
            # Added option to accumulate fisher over time with a pre-fixed growing self._ewc_rwalk_alpha
            self._critic_fisher[n] = (self._ewc_rwalk_alpha * self._critic_fisher[n] + (1 - self._ewc_rwalk_alpha) * curr_fisher[n])

        if self._critic_replay_type == 'rwalk':
            # Page 7: Optimization Path-based Parameter Importance: importance scores computation
            curr_score = {n: torch.zeros(p.shape).to(self.device) for n, p in self._q_func.named_parameters()
                          if p.requires_grad}
            with torch.no_grad():
                curr_params = {n: p for n, p in self._q_func.named_parameters() if p.requires_grad}
                for n, p in self._critic_scores.items():
                    curr_score[n] = self._critic_W[n] / (
                            self._critic_fisher[n] * ((curr_params[n] - self._critic_older_params[n]) ** 2) + self._damping)
                    self._critic_W[n].zero_()
                    # Page 7: "Since we care about positive influence of the parameters, negative scores are set to zero."
                    curr_score[n] = torch.nn.functional.relu(curr_score[n])
            # Page 8: alleviating regularization getting increasingly rigid by averaging scores
            for n, p in self._critic_scores.items():
                self._critic_scores[n] = (self._critic_scores[n] + curr_score[n]) / 2

    def actor_ewc_rwalk_post_train_process(self, iterator):
        def update(batch):
            batch = TorchMiniBatch(
                batch,
                self.device,
                scaler=self.scaler,
                action_scaler=self.action_scaler,
                reward_scaler=self.reward_scaler,
            )
            loss = self.compute_actor_loss(batch)
            loss.backward()
        # calculate Fisher information
        curr_fisher = self.compute_fisher_matrix_diag(iterator, self._policy, self._actor_optim, update)
        # merge fisher information, we do not want to keep fisher information for each task in memory
        for n in self._actor_fisher.keys():
            # Added option to accumulate fisher over time with a pre-fixed growing self._ewc_rwalk_alpha
            self._actor_fisher[n] = (self._ewc_rwalk_alpha * self._actor_fisher[n] + (1 - self._ewc_rwalk_alpha) * curr_fisher[n])

        if self._actor_replay_type == 'rwalk':
            # Page 7: Optimization Path-based Parameter Importance: importance scores computation
            curr_score = {n: torch.zeros(p.shape).to(self.device) for n, p in self._policy.named_parameters()
                          if p.requires_grad}
            with torch.no_grad():
                curr_params = {n: p for n, p in self._policy.named_parameters() if p.requires_grad}
                for n, p in self._actor_scores.items():
                    curr_score[n] = self._actor_w[n] / (
                            self._actor_fisher[n] * ((curr_params[n] - self._actor_older_params[n]) ** 2) + self._damping)
                    self._actor_w[n].zero_()
                    # Page 7: "Since we care about positive influence of the parameters, negative scores are set to zero."
                    curr_score[n] = torch.nn.functional.relu(curr_score[n])
            # Page 8: alleviating regularization getting increasingly rigid by averaging scores
            for n, p in self._actor_scores.items():
                self._actor_scores[n] = (self._actor_scores[n] + curr_score[n]) / 2

    def critic_si_post_train_process(self):
        for n, p in self._q_func.named_parameters():
            if p.requires_grad:
                p_change = p.detach().clone() - self._critic_older_params[n]
                omega_add = self._critic_W[n] / (p_change ** 2 + self._epsilon)
                omega = self._critic_omega[n]
                omega_new = omega + omega_add
                self._critic_older_params[n] = p.detach().clone()
                self._critic_omega[n] = omega_new
    def actor_si_post_train_process(self):
        for n, p in self._policy.named_parameters():
            if p.requires_grad:
                p_change = p.detach().clone() - self._actor_older_params[n]
                omega_add = self._actor_W[n] / (p_change ** 2 + self._epsilon)
                omega = self._actor_omega[n]
                omega_new = omega + omega_add
                self._actor_older_params[n] = p.detach().clone()
                self._actor_omega[n] = omega_new

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
        self._clone_q_func = copy.deepcopy(self._q_func)
        self._clone_policy = copy.deepcopy(self._policy)
        if self._vae_impl is not None:
            self._clone_vae_impl  = copy.deepcopy(self._vae_impl)