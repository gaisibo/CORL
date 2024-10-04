from copy import deepcopy
import inspect
import types
import time
import math
import copy
from typing import Optional, Sequence, List, Any, Tuple, Dict, Union

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
from myd3rlpy.models.torch.embed import Embed


replay_name = ['observations', 'actions', 'rewards', 'next_observations', 'terminals', 'policy_actions', 'qs', 'phis', 'psis']
class FSImpl():
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

        embed: bool,
        **kwargs,
    ):
        super().__init__(
            **kwargs
        )
        self._critic_replay_type = critic_replay_type
        self._critic_replay_lambda = critic_replay_lambda
        self._actor_replay_type = actor_replay_type
        self._actor_replay_lambda = actor_replay_lambda
        # self._embed_replay_type = embed_replay_type
        # self._embed_replay_lambda = embed_replay_lambda

        self._gem_alpha = gem_alpha
        self._agem_alpha = agem_alpha
        self._ewc_rwalk_alpha = ewc_rwalk_alpha
        self._damping = damping
        self._epsilon = epsilon
        self._fine_tuned_step = fine_tuned_step

        self._clone_q_func = None
        self._clone_policy = None

        self._embed = embed

    def build(self):
        self._dynamic = None

        super().build()

        batch_size = 256
        if self._embed:
            self._critic_embed = Embed(self._observation_shape[0], self._action_size, 10, output_dims=[p.numel() * 2 for n, p in self._q_func.named_parameters() if 'bias' not in n]).to(self.device)
            self._build_critic_embed_optim()
            if hasattr(self, "_value_func"):
                self._value_embed = Embed(self._observation_shape[0], self._action_size, 10, output_dims=[p.numel() * 2 for n, p in self._value_func.named_parameters() if 'bias' not in n]).to(self.device)
                self._build_value_embed_optim()
            self._actor_embed = Embed(self._observation_shape[0], self._action_size, 10, output_dims=[p.numel() * 2 for n, p in self._policy.named_parameters() if 'bias' not in n]).to(self.device)
            self._build_actor_embed_optim()

            # self._critic_embed = Embed(self._observation_shape[0], self._action_size, 10, output_dims=[list(self._q_func.parameters())[0].numel() * 2]).to(self.device)
            # self._build_critic_embed_optim()
            # if hasattr(self, "_value_func"):
            #     self._value_embed = Embed(self._observation_shape[0], self._action_size, 10, output_dims=[list(self._value_func.parameters())[0].numel() * 2]).to(self.device)
            #     self._build_value_embed_optim()
            # self._actor_embed = Embed(self._observation_shape[0], self._action_size, 10, output_dims=[list(self._policy.parameters())[0].numel() * 2]).to(self.device)
            # self._build_actor_embed_optim()

            if self._critic_replay_type in ['ewc', 'rwalk', 'si']:
                # Store current parameters for the next task
                self._critic_embed_older_params = {n: p.clone().detach() for n, p in self._critic_embed.named_parameters() if p.requires_grad}
                if self._critic_replay_type in ['ewc', 'rwalk']:
                    # Store fisher information weight importance
                    self._critic_embed_fisher = {n: torch.zeros(p.shape).to(self.device) for n, p in self._critic_embed.named_parameters() if p.requires_grad}
                if self._critic_replay_type == 'rwalk':
                    # Page 7: "task-specific parameter importance over the entire training trajectory."
                    self._critic_embed_W = {n: torch.zeros(p.shape).to(self.device) for n, p in self._critic_embed.named_parameters() if p.requires_grad}
                    self._critic_embed_scores = {n: torch.zeros(p.shape).to(self.device) for n, p in self._critic_embed.named_parameters() if p.requires_grad}
                elif self._critic_replay_type == 'si':
                    self._critic_embed_W = {n: p.clone().detach().zero_() for n, p in self._critic_embed.named_parameters() if p.requires_grad}
                    self._critic_embed_omega = {n: p.clone().detach().zero_() for n, p in self._critic_embed.named_parameters() if p.requires_grad}
            elif self._critic_replay_type == 'gem':
                # Allocate temporary synaptic memory
                self._critic_embed_grad_dims = []
                for pp in self._critic_embed.parameters():
                    self._critic_embed_grad_dims.append(pp.data.numel())
                self._critic_embed_grads_cs = torch.zeros(np.sum(self._critic_embed_grad_dims)).to(self.device)
                self._critic_embed_grads_da = torch.zeros(np.sum(self._critic_embed_grad_dims)).to(self.device)
            elif self._critic_replay_type == 'agem':
                self._critic_embed_grad_dims = []
                for param in self._critic_embed.parameters():
                    self._critic_embed_grad_dims.append(param.data.numel())
                self._critic_embed_grad_xy = torch.Tensor(np.sum(self._critic_embed_grad_dims)).to(self.device)
                self._critic_embed_grad_er = torch.Tensor(np.sum(self._critic_embed_grad_dims)).to(self.device)

            if self._actor_replay_type in ['ewc', 'rwalk', 'si']:
                # Store current parameters for the next task
                self._actor_embed_older_params = {n: p.clone().detach() for n, p in self._actor_embed.named_parameters() if p.requires_grad}
                if self._actor_replay_type in ['ewc', 'rwalk']:
                    # Store fisher information weight importance
                    self._actor_embed_fisher = {n: torch.zeros(p.shape).to(self.device) for n, p in self._actor_embed.named_parameters() if p.requires_grad}
                if self._actor_replay_type == 'rwalk':
                    # Page 7: "task-specific parameter importance over the entire training trajectory."
                    self._actor_embed_W = {n: torch.zeros(p.shape).to(self.device) for n, p in self._actor_embed.named_parameters() if p.requires_grad}
                    self._actor_embed_scores = {n: torch.zeros(p.shape).to(self.device) for n, p in self._actor_embed.named_parameters() if p.requires_grad}
                elif self._actor_replay_type == 'si':
                    self._actor_embed_W = {n: p.clone().detach().zero_() for n, p in self._actor_embed.named_parameters() if p.requires_grad}
                    self._actor_embed_omega = {n: p.clone().detach().zero_() for n, p in self._actor_embed.named_parameters() if p.requires_grad}
            elif self._actor_replay_type == 'gem':
                # Allocate temporary synaptic memory
                self._actor_embed_grad_dims = []
                for pp in self._actor_embed.parameters():
                    self._actor_embed_grad_dims.append(pp.data.numel())
                self._actor_embed_grads_cs = torch.zeros(np.sum(self._actor_embed_grad_dims)).to(self.device)
                self._actor_embed_grads_da = torch.zeros(np.sum(self._actor_embed_grad_dims)).to(self.device)
            elif self._actor_replay_type == 'agem':
                self._actor_embed_grad_dims = []
                for param in self._actor_embed.parameters():
                    self._actor_embed_grad_dims.append(param.data.numel())
                self._actor_embed_grad_xy = torch.Tensor(np.sum(self._actor_embed_grad_dims)).to(self.device)
                self._actor_embed_grad_er = torch.Tensor(np.sum(self._actor_embed_grad_dims)).to(self.device)

            if hasattr(self, "_value_func"):
                if self._critic_replay_type in ['ewc', 'rwalk', 'si']:
                    # Store current parameters for the next task
                    self._value_embed_older_params = {n: p.clone().detach() for n, p in self._value_embed.named_parameters() if p.requires_grad}
                    if self._critic_replay_type in ['ewc', 'rwalk']:
                        # Store fisher information weight importance
                        self._value_embed_fisher = {n: torch.zeros(p.shape).to(self.device) for n, p in self._value_embed.named_parameters() if p.requires_grad}
                    if self._critic_replay_type == 'rwalk':
                        # Page 7: "task-specific parameter importance over the entire training trajectory."
                        self._value_embed_W = {n: torch.zeros(p.shape).to(self.device) for n, p in self._value_embed.named_parameters() if p.requires_grad}
                        self._value_embed_scores = {n: torch.zeros(p.shape).to(self.device) for n, p in self._value_embed.named_parameters() if p.requires_grad}
                    elif self._critic_replay_type == 'si':
                        self._value_embed_W = {n: p.clone().detach().zero_() for n, p in self._value_embed.named_parameters() if p.requires_grad}
                        self._value_embed_omega = {n: p.clone().detach().zero_() for n, p in self._value_embed.named_parameters() if p.requires_grad}
                elif self._critic_replay_type == 'gem':
                    # Allocate temporary synaptic memory
                    self._value_embed_grad_dims = []
                    for pp in self._value_embed.parameters():
                        self._value_embed_grad_dims.append(pp.data.numel())
                    self._value_embed_grads_cs = torch.zeros(np.sum(self._value_embed_grad_dims)).to(self.device)
                    self._value_embed_grads_da = torch.zeros(np.sum(self._value_embed_grad_dims)).to(self.device)
                elif self._critic_replay_type == 'agem':
                    self._value_embed_grad_dims = []
                    for param in self._value_embed.parameters():
                        self._value_embed_grad_dims.append(param.data.numel())
                    self._value_embed_grad_xy = torch.Tensor(np.sum(self._value_embed_grad_dims)).to(self.device)
                    self._value_embed_grad_er = torch.Tensor(np.sum(self._value_embed_grad_dims)).to(self.device)

        assert self._q_func is not None
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
            self._critic_grads_cs = torch.zeros(np.sum(self._critic_grad_dims)).to(self.device)
            self._critic_grads_da = torch.zeros(np.sum(self._critic_grad_dims)).to(self.device)
        elif self._critic_replay_type == 'agem':
            self._critic_grad_dims = []
            for param in self._q_func.parameters():
                self._critic_grad_dims.append(param.data.numel())
            self._critic_grad_xy = torch.Tensor(np.sum(self._critic_grad_dims)).to(self.device)
            self._critic_grad_er = torch.Tensor(np.sum(self._critic_grad_dims)).to(self.device)

        if hasattr(self, "_value_func"):
            if self._critic_replay_type in ['ewc', 'rwalk', 'si']:
                self._value_older_params = {n: p.clone().detach() for n, p in self._value_func.named_parameters() if p.requires_grad}
                if self._critic_replay_type in ['ewc', 'rwalk']:
                    # Store fisher information weight importance
                    self._value_fisher = {n: torch.zeros(p.shape).to(self.device) for n, p in self._value_func.named_parameters() if p.requires_grad}
                if self._critic_replay_type == 'rwalk':
                    # Page 7: "task-specific parameter importance over the entire training trajectory."
                    self._value_W = {n: torch.zeros(p.shape).to(self.device) for n, p in self._value_func.named_parameters() if p.requires_grad}
                    self._value_scores = {n: torch.zeros(p.shape).to(self.device) for n, p in self._value_func.named_parameters() if p.requires_grad}
                elif self._critic_replay_type == 'si':
                    self._value_W = {n: p.clone().detach().zero_() for n, p in self._value_func.named_parameters() if p.requires_grad}
                    self._value_omega = {n: p.clone().detach().zero_() for n, p in self._value_func.named_parameters() if p.requires_grad}
            elif self._critic_replay_type == 'gem':
                # Allocate temporary synaptic memory
                self._value_grad_dims = []
                for pp in self._value_func.parameters():
                    self._value_grad_dims.append(pp.data.numel())
                self._value_grads_cs = torch.zeros(np.sum(self._value_grad_dims)).to(self.device)
                self._value_grads_da = torch.zeros(np.sum(self._value_grad_dims)).to(self.device)
            elif self._critic_replay_type == 'agem':
                self._value_grad_dims = []
                for param in self._value_func.parameters():
                    self._value_grad_dims.append(param.data.numel())
                self._value_grad_xy = torch.Tensor(np.sum(self._value_grad_dims)).to(self.device)
                self._value_grad_er = torch.Tensor(np.sum(self._value_grad_dims)).to(self.device)

        assert self._policy is not None
        if self._actor_replay_type in ['ewc', 'rwalk', 'si']:
            # Store current parameters for the next task
            self._actor_older_params = {n: torch.zeros(p.shape).to(self.device) for n, p in self._policy.named_parameters() if p.requires_grad}
            for p_old, p in zip(self._actor_older_params.values(), self._policy.parameters()):
                p_old.data = p.data.clone()
            if self._actor_replay_type in ['ewc', 'rwalk']:
                # Store fisher information weight importance
                self._actor_fisher = {n: torch.zeros(p.shape).to(self.device) for n, p in self._policy.named_parameters() if p.requires_grad}
            if self._actor_replay_type == 'rwalk':
                # Page 7: "task-specific parameter importance over the entire training trajectory."
                self._actor_w = {n: torch.zeros(p.shape).to(self.device) for n, p in self._policy.named_parameters() if p.requires_grad}
                self._actor_scores = {n: torch.zeros(p.shape).to(self.device) for n, p in self._policy.named_parameters() if p.requires_grad}
            elif self._actor_replay_type == 'si':
                self._actor_W = {n: p.clone().detach().zero_() for n, p in self._policy.named_parameters() if p.requires_grad}
                self._actor_omega = {n: p.clone().detach().zero_() for n, p in self._policy.named_parameters() if p.requires_grad}
        elif self._actor_replay_type == 'gem':
            self._actor_grad_dims = []
            for pp in self._policy.parameters():
                self._actor_grad_dims.append(pp.data.numel())
            self._actor_grads_cs = torch.zeros(np.sum(self._actor_grad_dims)).to(self.device)
            self._actor_grads_da = torch.zeros(np.sum(self._actor_grad_dims)).to(self.device)
        elif self._actor_replay_type == 'agem':
            self._actor_grad_dims = []
            for param in self._policy.parameters():
                self._actor_grad_dims.append(param.data.numel())
            self._actor_grad_xy = torch.Tensor(np.sum(self._actor_grad_dims)).to(self.device)
            self._actor_grad_er = torch.Tensor(np.sum(self._actor_grad_dims)).to(self.device)

    def _build_actor_embed_optim(self) -> None:
        assert self._actor_embed is not None
        self._actor_embed_optim = self._actor_optim_factory.create(
            self._actor_embed.parameters(), lr=self._actor_learning_rate
        )

    def _build_critic_embed_optim(self) -> None:
        assert self._critic_embed is not None
        self._critic_embed_optim = self._critic_optim_factory.create(
            self._critic_embed.parameters(), lr=self._critic_learning_rate
        )

    def _build_value_embed_optim(self) -> None:
        assert self._critic_embed is not None
        assert self._value_embed is not None
        self._critic_embed_optim = self._critic_optim_factory.create(
            list(self._critic_embed.parameters()) + list(self._value_embed.parameters()), lr=self._critic_learning_rate
        )

    @train_api
    @torch_api(reward_scaler_targets=["batch"])
    def embed(self, batch_random: TransitionMiniBatch):
        self._critic_state_dict, self._targ_critic_state_dict, self._actor_state_dict, self._targ_actor_state_dict = {}, {}, {}, {}
        self._critic_embed_dict, self._actor_embed_dict = {}, {}
        embed_network = list(zip([self._critic_embed, self._actor_embed], [self._q_func, self._policy], [self._critic_state_dict, self._actor_state_dict], [self._critic_embed_dict, self._actor_embed_dict]))
        if hasattr(self, "_value_func"):
            self._value_state_dict, self._value_embed_dict = {}, {}
            embed_network.append((self._value_embed, self._value_func, self._value_state_dict, self._value_embed_dict))
        for network_num, (embed, network, state_dict, embed_dict) in enumerate(embed_network):
            feature, embeds = embed(torch.cat([batch_random.observations[:10], batch_random.actions[:10]], dim=1).view(1, -1))
            i = 0
            for n, p in network.named_parameters():
                # if i <= 0:
                if 'bias' not in n:
                    embed = embeds[i][0]
                    param_length = embed.shape[0] // 2
                    embed_dict[n] = (embed[: param_length].view(p.shape), embed[param_length :].view(p.shape))
                    state_dict[n] = p * embed_dict[n][0] + embed_dict[n][1]
                    i += 1
                else:
                    state_dict[n] = p
        if hasattr(self, "_targ_q_func"):
            i = 0
            for n, p in self._targ_q_func.named_parameters():
                # if i <= 0:
                if 'bias' not in n:
                    self._targ_critic_state_dict[n] = p * self._critic_embed_dict[n][0] + self._critic_embed_dict[n][1]
                    i += 1
                else:
                    self._targ_critic_state_dict[n] = p
        if hasattr(self, "_targ_policy"):
            i = 0
            for n, p in self._targ_policy.named_parameters():
                if i <= 0:
                    self._targ_actor_state_dict[n] = p * self._actor_embed_dict[n][0] + self._actor_embed_dict[n][1]
                    i += 1
                else:
                    self._targ_actor_state_dict[n] = p
        return feature

    @train_api
    @torch_api(reward_scaler_targets=["batch"])
    def inner_update_critic(self, batch: TransitionMiniBatch, clone_critic: bool=False, online: bool=False):
        assert self._critic_optim is not None
        assert self._q_func is not None
        assert self._policy is not None

        unreg_grads = None
        curr_feat_ext = None

        self._critic_optim.zero_grad()
        q_tpn = self.compute_target(batch)
        critic_loss, value_loss, value, y, diff = self.compute_critic_loss(batch, q_tpn)
        loss = critic_loss + value_loss
        loss.backward(create_graph=True)

        # nn.utils.clip_grad_norm_([p for p in self._q_func.parameters() if p.requires_grad], 1.0)
        # if hasattr(self, "_value_func"):
        #     nn.utils.clip_grad_norm_([p for p in self._value_func.parameters() if p.requires_grad], 1.0)
        # if self._embed:
        #     nn.utils.clip_grad_norm_([p for p in self._critic_embed.parameters() if p.requires_grad], 1.0)
        #     if hasattr(self, "_value_func"):
        #         nn.utils.clip_grad_norm_([p for p in self._value_embed.parameters() if p.requires_grad], 1.0)

        if not hasattr(self, "_critic_embed_dict"):
            self._critic_state_dict = {}
            self._targ_critic_state_dict = {}
        i = 0
        for name, param in self._q_func.named_parameters():
            if hasattr(self, "_critic_embed_dict") and name in self._critic_embed_dict.keys():
                self._critic_state_dict[name] = (param - self._critic_learning_rate * 10 * param.grad) * self._critic_embed_dict[name][0] + self._critic_embed_dict[name][1]
            else:
                self._critic_state_dict[name] = (param - self._critic_learning_rate * 10 * param.grad)

        if hasattr(self, "_targ_q_func"):
            with torch.no_grad():
                i = 0
                for (name, targ_param), param in zip(self._targ_q_func.named_parameters(), self._q_func.parameters()):
                    if hasattr(self, "_critic_embed_dict") and name in self._critic_embed_dict.keys():
                        self._targ_critic_state_dict[name] = (targ_param - self._critic_learning_rate * 10 * param.grad) * self._critic_embed_dict[name][0] + self._critic_embed_dict[name][1]
                    else:
                        self._targ_critic_state_dict[name] = (targ_param - self._critic_learning_rate * 10 * param.grad)

        if hasattr(self, "_value_func"):
            if not hasattr(self, "_value_embed_dict"):
                self._value_state_dict = {}
            i = 0
            for name, param in self._value_func.named_parameters():
                if hasattr(self, "_value_embed_dict") and name in self._value_embed_dict.keys():
                    self._value_state_dict[name] = (param - self._critic_learning_rate * 10 * param.grad) * self._value_embed_dict[name][0] + self._value_embed_dict[name][1]
                else:
                    self._value_state_dict[name] = (param - self._critic_learning_rate * 10 * param.grad)

        critic_loss = critic_loss.cpu().detach().numpy()
        value_loss = value_loss.cpu().detach().numpy()
        value = value.cpu().detach().numpy()
        y = y.cpu().detach().numpy()
        diff = diff.cpu().detach().numpy()

        return critic_loss, value_loss, value, y, diff

    @train_api
    @torch_api(reward_scaler_targets=["batch"])
    def outer_update_critic(self, batch: TransitionMiniBatch, clone_critic: bool=False, online: bool=False, score=False):
        assert self._critic_optim is not None
        assert self._q_func is not None
        assert self._policy is not None

        unreg_grads = None
        curr_feat_ext = None

        replay_loss = 0
        if self._impl_id != 0 and not online:
            replay_loss = 0
            if self._critic_replay_type == "lwf":
                clone_q = self._clone_q_func(batch.observations, batch.actions)
                q = self._q_func(batch.observations, batch.actions)
                replay_bc_loss = F.mse_loss(clone_q, q)
                if hasattr(self, "_value_func"):
                    clone_value = self._clone_value_func(batch.observations, batch.actions)
                    value = self._value_func(batch.observations, batch.actions)
                    replay_bc_loss += F.mse_loss(clone_value, value)
                replay_loss = replay_loss + replay_bc_loss
            elif self._critic_replay_type == "ewc":
                replay_ewc_loss = 0
                for n, p in self._q_func.named_parameters():
                    if n in self._critic_fisher.keys():
                        replay_ewc_loss += torch.mean(self._critic_fisher[n] * (p - self._critic_older_params[n]).pow(2)) / 2
                if hasattr(self, "_critic_embed"):
                    for n, p in self._critic_embed.named_parameters():
                        if n in self._critic_embed_fisher.keys():
                            replay_ewc_loss += torch.mean(self._critic_embed_fisher[n] * (p - self._critic_embed_older_params[n]).pow(2)) / 2
                if hasattr(self, "_value_func"):
                    for n, p in self._value_func.named_parameters():
                        if n in self._value_fisher.keys():
                            replay_ewc_loss += torch.mean(self._value_fisher[n] * (p - self._value_older_params[n]).pow(2)) / 2
                    if hasattr(self, "_value_embed"):
                        for n, p in self._value_embed.named_parameters():
                            if n in self._value_embed_fisher.keys():
                                replay_ewc_loss += torch.mean(self._value_embed_fisher[n] * (p - self._value_embed_older_params[n]).pow(2)) / 2
                replay_loss = replay_loss + replay_ewc_loss
            elif self._critic_replay_type == 'rwalk':
                replay_rwalk_loss = 0
                curr_critic_feat_ext = {n: p.clone().detach() for n, p in self._q_func.named_parameters() if p.requires_grad}
                if hasattr(self, "_value_func"):
                    curr_value_feat_ext = {n: p.clone().detach() for n, p in self._value_func.named_parameters() if p.requires_grad}
                # store gradients without regularization term
                self._critic_optim.zero_grad()
                q_tpn = self.compute_target(batch)
                critic_loss, value_loss, value, y, diff = self.compute_critic_loss(batch, q_tpn)
                loss = critic_loss + value_loss
                loss.backward(retain_graph=True)
                unreg_critic_grads = {n: p.grad.clone().detach() for n, p in self._q_func.named_parameters() if p.grad is not None}
                if hasattr(self, "_value_func"):
                    unreg_value_grads = {n: p.grad.clone().detach() for n, p in self._value_func.named_parameters() if p.grad is not None}

                # Eq. 3: elastic weight consolidation quadratic penalty
                for n, p in self._q_func.named_parameters():
                    if n in self._critic_fisher.keys():
                        replay_rwalk_loss += torch.mean((self._critic_fisher[n] + self._critic_scores[n]) * (p - self._critic_older_params[n]).pow(2)) / 2
                if hasattr(self, "_value_func"):
                    for n, p in self._q_func.named_parameters():
                        if n in self._critic_fisher.keys():
                            replay_rwalk_loss += torch.mean((self._critic_fisher[n] + self._critic_scores[n]) * (p - self._critic_older_params[n]).pow(2)) / 2

                replay_loss = replay_loss + replay_rwalk_loss
            elif self._critic_replay_type == 'si':
                for n, p in self._q_func.named_parameters():
                    if p.grad is not None and n in self._critic_W.keys():
                        p_change = p.detach().clone() - self._critic_older_params[n]
                        self._critic_W[n].add_(-p.grad * p_change)
                        omega_add = self._critic_W[n] / (p_change ** 2 + self._epsilon)
                        omega = self._critic_omega[n]
                        omega_new = omega + omega_add
                        self._critic_omega[n] = omega_new
                if hasattr(self, "_value_func"):
                    for n, p in self._value_func.named_parameters():
                        if p.grad is not None and n in self._value_W.keys():
                            p_change = p.detach().clone() - self._value_older_params[n]
                            self._value_W[n].add_(-p.grad * p_change)
                            omega_add = self._value_W[n] / (p_change ** 2 + self._epsilon)
                            omega = self._value_omega[n]
                            omega_new = omega + omega_add
                            self._value_omega[n] = omega_new
                replay_si_loss = 0
                for n, p in self._q_func.named_parameters():
                    if p.requires_grad:
                        replay_si_loss += torch.mean(self._critic_omega[n] * (p - self._critic_older_params[n]) ** 2)
                    self._critic_older_params[n].data = p.data.clone()
                if hasattr(self, "_value_func"):
                    for n, p in self._value_func.named_parameters():
                        if p.requires_grad:
                            replay_si_loss += torch.mean(self._value_omega[n] * (p - self._value_older_params[n]) ** 2)
                        self._value_older_params[n] = p.detach().clone()
                replay_loss = replay_loss + replay_si_loss

        self._critic_optim.zero_grad()
        if self._embed:
            self._critic_embed_optim.zero_grad()
        q_tpn = self.compute_target(batch)
        critic_loss, value_loss, value, y, diff = self.compute_critic_loss(batch, q_tpn)
        loss = critic_loss + value_loss
        if self._critic_replay_type in ['orl', 'ewc', 'rwalk', 'si', 'bc', 'generate', 'lwf']:
            loss = loss + self._critic_replay_lambda * replay_loss
        loss.backward()

        # nn.utils.clip_grad_norm_([p for p in self._q_func.parameters() if p.requires_grad], 1.0)
        # if hasattr(self, "_value_func"):
        #     nn.utils.clip_grad_norm_([p for p in self._value_func.parameters() if p.requires_grad], 1.0)
        # if self._embed:
        #     nn.utils.clip_grad_norm_([p for p in self._critic_embed.parameters() if p.requires_grad], 1.0)
        #     if hasattr(self, "_value_func"):
        #         nn.utils.clip_grad_norm_([p for p in self._value_embed.parameters() if p.requires_grad], 1.0)

        if not score:
            self._critic_optim.step()
            if self._embed:
                self._critic_embed_optim.step()

        critic_loss = critic_loss.cpu().detach().numpy()
        value_loss = value_loss.cpu().detach().numpy()
        value = value.cpu().detach().numpy()
        y = y.cpu().detach().numpy()
        diff = diff.cpu().detach().numpy()
        if not isinstance(replay_loss, int):
            replay_loss = replay_loss.cpu().detach().numpy()

        return critic_loss, value_loss, value, y, diff, replay_loss

    @train_api
    @torch_api(scaler_targets=["batch"])
    def inner_update_actor(self, batch: TransitionMiniBatch, clone_actor: bool=False, online: bool=False) -> np.ndarray:
        assert self._q_func is not None
        assert self._policy is not None
        assert self._actor_optim is not None

        # Q function should be inference mode for stability
        self._q_func.eval()
        self._policy.train()

        unreg_grads = None
        curr_feat_ext = None

        loss, weight, log_prob = self.compute_actor_loss(batch)
        self._actor_optim.zero_grad()
        loss.backward(create_graph=True)

        # nn.utils.clip_grad_norm_([p for p in self._policy.parameters() if p.requires_grad], 1.0)
        # if self._embed:
        #     nn.utils.clip_grad_norm_([p for p in self._actor_embed.parameters() if p.requires_grad], 1.0)

        if not hasattr(self, "_actor_embed_dict"):
            self._actor_state_dict = {}
        i = 0
        for name, param in self._policy.named_parameters():
            if hasattr(self, "_actor_embed_dict") and name in self._actor_embed_dict.keys():
                self._actor_state_dict[name] = (param - self._actor_learning_rate * 10 * param.grad) * self._actor_embed_dict[name][0] + self._actor_embed_dict[name][1]
            else:
                self._actor_state_dict[name] = (param - self._actor_learning_rate * 10 * param.grad)
        if hasattr(self, "_targ_policy"):
            if not hasattr(self, "_targ_actor_state_dict"):
                self._targ_actor_state_dict = {}
            with torch.no_grad():
                for (name, targ_param), param in zip(self._targ_policy.named_parameters(), self._policy.parameters()):
                    if hasattr(self, "_actor_embed_dict") and name in self._actor_embed_dict.keys():
                        self._targ_actor_state_dict[name] = (targ_param - self._actor_learning_rate * 10 * param.grad) * self._actor_embed_dict[name][0] + self._actor_embed_dict[name][1]
                    else:
                        self._targ_actor_state_dict[name] = (targ_param - self._actor_learning_rate * 10 * param.grad)

        return loss.detach().cpu().numpy(), weight.detach().cpu().numpy(), log_prob.detach().cpu().numpy()

    @train_api
    @torch_api(scaler_targets=["batch"])
    def outer_update_actor(self, batch: TransitionMiniBatch, clone_actor: bool=False, online: bool=False, score=False) -> np.ndarray:
        assert self._q_func is not None
        assert self._policy is not None
        assert self._actor_optim is not None

        # Q function should be inference mode for stability
        self._q_func.eval()
        self._policy.train()

        unreg_grads = None
        curr_feat_ext = None

        replay_loss = 0
        if self._impl_id != 0 and not online:
            replay_loss = 0
            if self._actor_replay_type == "ewc":
                replay_loss_ = 0
                for n, p in self._policy.named_parameters():
                    if n in self._actor_fisher.keys():
                        replay_loss_ = torch.mean(self._actor_fisher[n] * (p - self._actor_older_params[n]).pow(2)) / 2
                if hasattr(self, "_actor_embed"):
                    for n, p in self._actor_embed.named_parameters():
                        if n in self._actor_embed_fisher.keys():
                            replay_loss_ += torch.mean(self._actor_embed_fisher[n] * (p - self._actor_embed_older_params[n]).pow(2)) / 2
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
                    if p.grad is not None and n in self._actor_W.keys():
                        p_change = p.detach().clone() - self._actor_older_params[n]
                        self._actor_W[n].add_(-p.grad * p_change)
                        omega_add = self._actor_W[n] / (p_change ** 2 + self._epsilon)
                        omega = self._actor_omega[n]
                        omega_new = omega + omega_add
                        self._actor_omega[n] = omega_new
                replay_loss_ = 0
                for param_id, (n, p) in enumerate(self._policy.named_parameters()):
                    if p.requires_grad:
                        replay_loss_ = replay_loss_ + torch.mean(self._actor_omega[n] * (p - self._actor_older_params[n]) ** 2)
                    self._actor_older_params[n].data = p.data.clone()
                replay_loss = replay_loss + replay_loss_
        loss, weight, log_prob = self.compute_actor_loss(batch)
        self._actor_optim.zero_grad()
        if self._embed:
            self._actor_embed_optim.zero_grad()
        if self._actor_replay_type in ['orl', 'ewc', 'rwalk', 'si', 'bc', 'generate', 'generate_orl', 'lwf', 'lwf_orl']:
            loss = loss + self._actor_replay_lambda * replay_loss
        actor_parameters = [p for p in self._policy.parameters() if p.requires_grad]
        if hasattr(self, "_actor_embed"):
            actor_parameters += [p for p in self._actor_embed.parameters() if p.requires_grad]
        loss.backward(inputs=actor_parameters)

        # nn.utils.clip_grad_norm_([p for p in self._policy.parameters() if p.requires_grad], 1.0)
        # if self._embed:
        #     nn.utils.clip_grad_norm_([p for p in self._actor_embed.parameters() if p.requires_grad], 1.0)

        if not score:
            self._actor_optim.step()
            if self._embed:
                self._actor_embed_optim.step()
        if not isinstance(replay_loss, int):
            replay_loss = replay_loss.cpu().detach().numpy()

        return loss.detach().cpu().numpy(), weight.detach().cpu().numpy(), log_prob.detach().cpu().numpy(), replay_loss

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
            update(self, batch)
            # Accumulate all gradients from loss with regularization
            for n, p in network.named_parameters():
                if p.grad is not None:
                    fisher[n] += p.grad.pow(2)
        # Apply mean across all samples
        fisher = {n: (p / len(iterator)) for n, p in fisher.items()}
        return fisher

    def critic_ewc_rwalk_post_train_process(self, iterator):
        # calculate Fisher information
        @train_api
        @torch_api()
        def update(self, batch):
            q_tpn = self.compute_target(batch)
            loss = self.compute_critic_loss(batch, q_tpn)
            loss.backward()
        curr_fisher = self.compute_fisher_matrix_diag(iterator, self._q_func, self._critic_optim, update)
        # merge fisher information, we do not want to keep fisher information for each task in memory
        for n in self._critic_fisher.keys():
            # Added option to accumulate fisher over time with a pre-fixed growing self._ewc_rwalk_alpha
            self._critic_fisher[n] = (self._ewc_rwalk_alpha * self._critic_fisher[n] + (1 - self._ewc_rwalk_alpha) * curr_fisher[n])

        if self._critic_replay_type == 'rwalk':
            # Page 7: Optimization Path-based Parameter Importance: importance scores computation
            curr_critic_score = {n: torch.zeros(p.shape).to(self.device) for n, p in self._q_func.named_parameters() if p.requires_grad}
            with torch.no_grad():
                curr_critic_params = {n: p for n, p in self._q_func.named_parameters() if p.requires_grad}
                for n, p in self._critic_scores.items():
                    curr_critic_score[n] = self._critic_W[n] / (
                            self._critic_fisher[n] * ((curr_critic_params[n] - self._critic_older_params[n]) ** 2) + self._damping)
                    self._critic_W[n].zero_()
                    # Page 7: "Since we care about positive influence of the parameters, negative scores are set to zero."
                    curr_critic_score[n] = torch.nn.functional.relu(curr_critic_score[n])
                    self._critic_older_params[n].data = curr_critic_params[n].data.clone()
            # Page 8: alleviating regularization getting increasingly rigid by averaging scores
            for n, p in self._critic_scores.items():
                self._critic_scores[n] = (self._critic_scores[n] + curr_critic_score[n]) / 2

        if hasattr(self, "_value_func"):
            @train_api
            @torch_api()
            def update(self, batch):
                q_tpn = self.compute_target(batch)
                loss = self.compute_critic_loss(batch, q_tpn)
                loss.backward()
            curr_fisher = self.compute_fisher_matrix_diag(iterator, self._value_func, self._critic_optim, update)
            for n in self._value_fisher.keys():
                # Added option to accumulate fisher over time with a pre-fixed growing self._ewc_rwalk_alpha
                self._value_fisher[n] = (self._ewc_rwalk_alpha * self._value_fisher[n] + (1 - self._ewc_rwalk_alpha) * curr_fisher[n])
            if self._critic_replay_type == 'rwalk':
                curr_value_score = {n: torch.zeros(p.shape).to(self.device) for n, p in self._value_func.named_parameters() if p.requires_grad}
                with torch.no_grad():
                    curr_value_params = {n: p for n, p in self._value_func.named_parameters() if p.requires_grad}
                    for n, p in self._value_scores.items():
                        curr_value_score[n] = self._value_W[n] / (
                                self._value_fisher[n] * ((curr_value_params[n] - self._value_older_params[n]) ** 2) + self._damping)
                        self._value_W[n].zero_()
                        # Page 7: "Since we care about positive influence of the parameters, negative scores are set to zero."
                        curr_value_score[n] = torch.nn.functional.relu(curr_value_score[n])
                        self._value_older_params[n].data = curr_value_params[n].data.clone()
                # Page 8: alleviating regularization getting increasingly rigid by averaging scores
                for n, p in self._value_scores.items():
                    self._value_scores[n] = (self._value_scores[n] + curr_value_score[n]) / 2
    def actor_ewc_rwalk_post_train_process(self, iterator):
        @train_api
        @torch_api()
        def update(self, batch):
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
                    self._actor_older_params[n].data = curr_params[n].data.clone()
            # Page 8: alleviating regularization getting increasingly rigid by averaging scores
            for n, p in self._actor_scores.items():
                self._actor_scores[n] = (self._actor_scores[n] + curr_score[n]) / 2

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
    #    if self._critic_replay_type in ['ewc', 'rwalk', 'si'] and '_critic_older_params' not in self.__dict__.keys():
    #        # Store current parameters for the next task
    #        self._critic_older_params = {n: p.clone().detach() for n, p in self._q_func.named_parameters() if p.requires_grad}
    #        if self._critic_replay_type in ['ewc', 'rwalk'] and '_critic_fisher' not in self.__dict__.keys():
    #            # Store fisher information weight importance
    #            self._critic_fisher = {n: torch.zeros(p.shape).to(self.device) for n, p in self._q_func.named_parameters() if p.requires_grad}
    #        if self._critic_replay_type == 'rwalk' and '_critic_W' not in self.__dict__.keys():
    #            # Page 7: "task-specific parameter importance over the entire training trajectory."
    #            self._critic_W = {n: torch.zeros(p.shape).to(self.device) for n, p in self._q_func.named_parameters() if p.requires_grad}
    #            self._critic_scores = {n: torch.zeros(p.shape).to(self.device) for n, p in self._q_func.named_parameters() if p.requires_grad}
    #        elif self._critic_replay_type == 'si' and '_critic_W' not in self.__dict__.keys():
    #            self._critic_W = {n: p.clone().detach().zero_() for n, p in self._q_func.named_parameters() if p.requires_grad}
    #            self._critic_omega = {n: p.clone().detach().zero_() for n, p in self._q_func.named_parameters() if p.requires_grad}
    #    if self._actor_replay_type in ['ewc', 'rwalk', 'si'] and '_actor_older_params' not in self.__dict__.keys():
    #        # Store current parameters for the next task
    #        self._actor_older_params = {n: p.clone().detach() for n, p in self._policy.named_parameters() if p.requires_grad}
    #        if self._actor_replay_type in ['ewc', 'rwalk'] and '_actor_fisher' not in self.__dict__.keys():
    #            # Store fisher information weight importance
    #            self._actor_fisher = {n: torch.zeros(p.shape).to(self.device) for n, p in self._policy.named_parameters() if p.requires_grad}
    #        if self._actor_replay_type == 'rwalk' and '_actor_w' not in self.__dict__.keys():
    #            # Page 7: "task-specific parameter importance over the entire training trajectory."
    #            self._actor_w = {n: torch.zeros(p.shape).to(self.device) for n, p in self._policy.named_parameters() if p.requires_grad}
    #            self._actor_scores = {n: torch.zeros(p.shape).to(self.device) for n, p in self._policy.named_parameters() if p.requires_grad}
    #        elif self._critic_replay_type == 'si' and '_actor_W' not in self.__dict__.keys():
    #            self._actor_W = {n: p.clone().detach().zero_() for n, p in self._policy.named_parameters() if p.requires_grad}
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
