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
class COImpl():
    def copy_q_function_from(self, impl: AlgoImplBase) -> None:
        impl = cast("TorchImplBase", impl)
        # 因为parallel改变了q_funcs的结构，这里就没必要检查类型了。
        # q_func = self.q_function.q_funcs[0]
        # if not isinstance(impl.q_function.q_funcs[0], type(q_func)):
        #     raise ValueError(
        #         f"Invalid Q-function type: expected={type(q_func)},"
        #         f"actual={type(impl.q_function.q_funcs[0])}"
        #     )
        hard_sync(self.q_function, impl.q_function)

    def build(self, task_id):
        self._impl_id = task_id

        super().build()

        if self._clone_actor and self._replay_type == 'bc':
            self._clone_policy = copy.deepcopy(self._policy)
            self._clone_actor_optim = self._actor_optim_factory.create(
                self._clone_policy.parameters(), lr=self._actor_learning_rate
            )

        assert self._q_func is not None
        assert self._policy is not None
        self._actor_grad_save_dims = []
        for name, pp in self._policy.named_parameters():
            if '_fc' not in name:
                self._actor_grad_save_dims.append(pp.data.numel())
        self._actor_grad_save = torch.zeros(np.sum(self._actor_grad_save_dims)).to(self.device)
        if self._clone_actor and self._replay_type == 'bc':
            self._clone_actor_grad_save_dims = []
            for name, pp in self._clone_policy.named_parameters():
                if '_fc' not in name:
                    self._clone_actor_grad_save_dims.append(pp.data.numel())
            self._clone_actor_grad_save = torch.zeros(np.sum(self._clone_actor_grad_save_dims)).to(self.device)
        else:
            self._clone_actor_grad_save = None
        # To save the actor grad for DBC and BC
        if self._replay_type in ['ewc', 'rwalk', 'si']:
            if self._replay_critic:
                # Store current parameters for the next task
                self._critic_older_params = {n: p.clone().detach() for n, p in self._q_func.named_parameters() if p.requires_grad and '_fc' not in n}
            # Store current parameters for the next task
            self._actor_older_params = {n: p.clone().detach() for n, p in self._policy.named_parameters() if p.requires_grad and '_fc' not in n}
            if self._replay_type in ['ewc', 'rwalk']:
                if self._replay_critic:
                    # Store fisher information weight importance
                    self._critic_fisher = {n: torch.zeros(p.shape).to(self.device) for n, p in self._q_func.named_parameters() if p.requires_grad and '_fc' not in n}
                # Store fisher information weight importance
                self._actor_fisher = {n: torch.zeros(p.shape).to(self.device) for n, p in self._policy.named_parameters() if p.requires_grad and '_fc' not in n}
                if self._replay_type == 'rwalk':
                    if self._replay_critic:
                        # Page 7: "task-specific parameter importance over the entire training trajectory."
                        self._critic_w = {n: torch.zeros(p.shape).to(self.device) for n, p in self._q_func.named_parameters() if p.requires_grad and '_fc' not in n}
                        self._critic_scores = {n: torch.zeros(p.shape).to(self.device) for n, p in self._q_func.named_parameters() if p.requires_grad and '_fc' not in n}
                    # Page 7: "task-specific parameter importance over the entire training trajectory."
                    self._actor_w = {n: torch.zeros(p.shape).to(self.device) for n, p in self._policy.named_parameters() if p.requires_grad and '_fc' not in n}
                    self._actor_scores = {n: torch.zeros(p.shape).to(self.device) for n, p in self._policy.named_parameters() if p.requires_grad and '_fc' not in n}
                elif self._replay_type == 'si':
                    if self._replay_critic:
                        self._critic_W = {n: p.clone().detach().zero_() for n, p in self._q_func.named_parameters() if p.requires_grad and '_fc' not in n}
                        self._critic_omega = {n: p.clone().detach().zero_() for n, p in self._q_func.named_parameters() if p.requires_grad and '_fc' not in n}
                    self._actor_W = {n: p.clone().detach().zero_() for n, p in self._policy.named_parameters() if p.requires_grad and '_fc' not in n}
                    self._actor_omega = {n: p.clone().detach().zero_() for n, p in self._policy.named_parameters() if p.requires_grad and '_fc' not in n}
        elif self._replay_type == 'gem':
            if self._replay_critic:
                # Allocate temporary synaptic memory
                self._critic_grad_dims = []
                for name, pp in self._q_func.named_parameters():
                    if '_fc' not in name:
                        self._critic_grad_dims.append(pp.data.numel())
                self._critic_grad_cs = {}
                self._critic_grad_da = torch.zeros(np.sum(self._critic_grad_dims)).to(self.device)

            self._actor_grad_dims = []
            for name, pp in self._policy.named_parameters():
                if '_fc' not in name:
                    self._actor_grad_dims.append(pp.data.numel())
            self._actor_grad_cs = {}
            self._actor_grad_da = torch.zeros(np.sum(self._actor_grad_dims)).to(self.device)

        elif self._replay_type == 'agem':
            if self._replay_critic:
                self._critic_grad_dims = []
                for name, param in self._q_func.named_parameters():
                    if '_fc' not in name:
                        self._critic_grad_dims.append(param.data.numel())
                self._critic_grad_xy = torch.Tensor(np.sum(self._critic_grad_dims)).to(self.device)
                self._critic_grad_er = torch.Tensor(np.sum(self._critic_grad_dims)).to(self.device)
            self._actor_grad_dims = []
            for name, param in self._policy.named_parameters():
                if '_fc' not in name:
                    self._actor_grad_dims.append(param.data.numel())
            self._actor_grad_xy = torch.Tensor(np.sum(self._actor_grad_dims)).to(self.device)
            self._actor_grad_er = torch.Tensor(np.sum(self._actor_grad_dims)).to(self.device)

    def rebuild_critic(self):
        if self._replay_critic:
            return
        assert self._q_func is not None
        for q_func in self._q_func._q_funcs:
            for m in q_func.modules():
                if isinstance(m, nn.Linear):
                    nn.init.kaiming_normal_(m.weight.data)
                    if m.bias is not None:
                        nn.init.zeros_(m.bias.data)
        self._targ_q_func = copy.deepcopy(self._q_func)

    @train_api
    def retrain_update_critic(self, batch: TorchMiniBatch):
        assert self._critic_optim is not None
        assert self._q_func is not None
        assert self._policy is not None
        with torch.enable_grad():
            self.change_task(self._impl_id)
            self._q_func.train()
            self._policy.eval()

            self._critic_optim.zero_grad()

            q_tpn = self.compute_target(batch)

            loss = self.compute_critic_loss(batch, q_tpn)

            loss.backward()
            self._critic_optim.step()

            loss = loss.cpu().detach().numpy()
            self.change_task(self._impl_id)

            return loss

    @train_api
    def update_critic(self, batch_tran: TransitionMiniBatch, replay_batches: Optional[Dict[int, List[torch.Tensor]]]=None):
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
        self._q_func.train()
        self._policy.eval()

        unreg_grad = None
        curr_feat_ext = None

        loss = 0
        replay_loss = 0
        replay_losses = []
        save_id = self._impl_id
        if replay_batches is not None and len(replay_batches) != 0:
            for i, replay_batch in replay_batches.items():
                replay_loss = 0
                self.change_task(i)
                replay_batch = [x.to(self.device) for x in replay_batch]
                replay_batch = dict(zip(replay_name[:-2], replay_batch))
                replay_batch = Struct(**replay_batch)
                if self._replay_type == "orl":
                    replay_batch.n_steps = 1
                    replay_batch.masks = None
                    replay_batch = cast(TorchMiniBatch, replay_batch)
                    q_tpn = self.compute_target(replay_batch)
                    replay_cql_loss = self.compute_critic_loss(replay_batch, q_tpn)
                    replay_losses.append(replay_cql_loss.cpu().detach().numpy())
                    replay_loss = replay_loss + replay_cql_loss
                elif self._replay_type == "bc":
                    with torch.no_grad():
                        replay_observations = replay_batch.observations.to(self.device)
                        replay_qs = replay_batch.qs.to(self.device)
                        replay_actions = replay_batch.actions.to(self.device)

                    q = self._q_func(replay_observations, replay_actions)
                    replay_bc_loss = F.mse_loss(replay_qs, q) / len(replay_batches)
                    replay_losses.append(replay_bc_loss.cpu().detach().numpy())
                    replay_loss = replay_loss + replay_bc_loss
                elif self._replay_type == "ewc":
                    replay_loss_ = 0
                    for n, p in self._q_func.named_parameters():
                        if n in self._critic_fisher.keys():
                            replay_loss_ = replay_loss_ + torch.mean(self._critic_fisher[n] * (p - self._critic_older_params[n]).pow(2)) / 2
                    replay_losses.append(replay_loss_.cpu().detach().numpy())
                    replay_loss = replay_loss + replay_loss_
                    break
                elif self._replay_type == 'rwalk':
                    curr_feat_ext = {n: p.clone().detach() for n, p in self._q_func.named_parameters() if p.requires_grad and '_fc' not in n}
                    # store gradients without regularization term
                    unreg_grad = {n: p.grad.clone().detach() for n, p in self._q_func.named_parameters()
                                   if p.grad is not None and '_fc' not in n}

                    self._critic_optim.zero_grad()
                    # Eq. 3: elastic weight consolidation quadratic penalty
                    replay_loss_ = 0
                    for n, p in self._q_func.named_parameters():
                        if n in self._critic_fisher.keys():
                            replay_loss_ = replay_loss_ + torch.mean((self._critic_fisher[n] + self._critic_scores[n]) * (p - self._critic_older_params[n]).pow(2)) / 2
                    replay_losses.append(replay_loss_.cpu().detach().numpy())
                    replay_loss = replay_loss + replay_loss_
                    break
                elif self._replay_type == 'si':
                    for n, p in self._q_func.named_parameters():
                        if p.grad is not None and n in self._critic_fisher.keys():
                            self._critic_W[n].add_(-p.grad * (p.detach() - self._critic_older_params[n]))
                        self._critic_older_params[n] = p.detach().clone()
                    replay_loss_ = 0
                    for n, p in self.q_func.named_parameters():
                        if p.requires_grad and n in self._critic_fisher.keys():
                            replay_loss_ = replay_loss_ + torch.mean(self._critic_omega[n] * (p - self._critic_older_params[n]) ** 2)
                    replay_losses.append(replay_loss_.cpu().detach().numpy())
                    replay_loss = replay_loss + replay_loss_
                    break
                elif self._replay_type == 'gem':
                    replay_batch = cast(TorchMiniBatch, replay_batch)
                    q_tpn = self.compute_target(replay_batch)
                    replay_loss_ = self.compute_critic_loss(replay_batch, q_tpn) / len(replay_batches)
                    replay_loss = replay_loss_
                    replay_losses.append(replay_loss_.cpu().detach().numpy())
                    replay_loss.backward()
                    store_grad([p for n, p in self._q_func.named_parameters() if p.requires_grad and '_fc' not in n], self._critic_grad_cs[i], self._critic_grad_dims)
                elif self._replay_type == "agem":
                    store_grad([p for n, p in self._q_func.named_parameters() if p.requires_grad and '_fc' not in n], self._critic_grad_xy, self._critic_grad_dims)
                    replay_batch.n_steps = 1
                    replay_batch.masks = None
                    replay_batch = cast(TorchMiniBatch, replay_batch)
                    q_tpn = self.compute_target(replay_batch)
                    replay_loss_ = self.compute_critic_loss(replay_batch, q_tpn)
                    replay_losses.append(replay_cql_loss.cpu().detach().numpy())
                    replay_loss = replay_loss + replay_loss_
                    replay_losses.append(replay_loss_.cpu().detach().numpy())
                # if self._replay_type in ['orl', 'bc', 'ewc', 'rwalk', 'si']:
                #     time_replay_loss = self._replay_alpha * replay_loss / len(replay_batches)
                #     self._critic_optim.zero_grad()
                #     time_replay_loss.backward()
                #     self._critic_optim.step()
                #     replay_loss = replay_loss.cpu().detach().numpy()

        self.change_task(save_id)
        self._critic_optim.zero_grad()
        q_tpn = self.compute_target(batch)
        loss = self.compute_critic_loss(batch, q_tpn)
        if self._replay_type in ['orl', 'ewc', 'rwalk', 'si', 'bc']:
            loss = loss + self._replay_alpha * replay_loss
        loss.backward()
        if replay_batches is not None and len(replay_batches) != 0:
            if self._replay_type == 'agem':
                self._critic_optim.zero_grad()
                replay_loss.backward()
                store_grad([p for n, p in self._q_func.named_parameters() if p.requires_grad and '_fc' not in n], self._critic_grad_er, self._critic_grad_dims)
                dot_prod = torch.dot(self._critic_grad_xy, self._critic_grad_er)
                if dot_prod.item() < 0:
                    g_tilde = project(gxy=self._critic_grad_xy, ger=self._critic_grad_er)
                    overwrite_grad([p for n, p in self._q_func.named_parameters() if p.requires_grad and '_fc' not in n], g_tilde, self._critic_grad_dims)
                else:
                    overwrite_grad([p for n, p in self._q_func.named_parameters() if p.requires_grad and '_fc' not in n], self._critic_grad_xy, self._critic_grad_dims)
            elif self._replay_type == 'gem':
                # copy gradient
                store_grad([p for n, p in self._q_func.named_parameters() if p.requires_grad and '_fc' not in n], self._critic_grad_da, self._critic_grad_dims)
                dot_prod = torch.mm(self._critic_grad_da.unsqueeze(0),
                                torch.stack(list(self._critic_grad_cs).values()).T)
                if (dot_prod < 0).sum() != 0:
                    project2cone2(self._critic_grad_da.unsqueeze(1),
                                  torch.stack(list(self._critic_grad_cs).values()).T, margin=self._gem_alpha)
                    # copy gradients back
                    overwrite_grad(self._q_func.parameters, self._critic_grad_da,
                                   self._critic_grad_dims)
        self._critic_optim.step()

        if replay_batches is not None and len(replay_batches) != 0:
            if self._replay_type == 'rwalk':
                assert unreg_grad is not None
                assert curr_feat_ext is not None
                with torch.no_grad():
                    for n, p in self._q_func.named_parameters():
                        if n in unreg_grad.keys():
                            self._critic_w[n] -= unreg_grad[n] * (p.detach() - curr_feat_ext[n])

        loss = loss.cpu().detach().numpy()
        if not isinstance(replay_loss, int):
            replay_loss = replay_loss.cpu().detach().numpy()

        return loss, replay_loss, replay_losses

    @train_api
    def replay_update_actor(self, batch_tran: TransitionMiniBatch, replay_batches: Optional[Dict[int, List[torch.Tensor]]]=None) -> np.ndarray:
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

        unreg_grad = None
        curr_feat_ext = None
        loss = 0
        replay_loss = 0
        replay_losses = []
        save_id = self._impl_id
        if replay_batches is not None and len(replay_batches) != 0:
            for i, replay_batch in replay_batches.items():
                replay_loss = 0
                replay_batch = [x.to(self.device) for x in replay_batch]
                self.change_task(i)
                replay_batch = dict(zip(replay_name[:-2], replay_batch))
                replay_batch = Struct(**replay_batch)

                with torch.no_grad():
                    replay_observations = replay_batch.observations.to(self.device)
                    replay_policy_actions = replay_batch.policy_actions.to(self.device)
                actions = self._clone_policy(replay_observations)
                replay_loss_ = torch.mean((replay_policy_actions - actions) ** 2)
                replay_losses.append(replay_loss_.cpu().detach().numpy())
                replay_loss = replay_loss + replay_loss_
                time_replay_loss = self._replay_alpha * replay_loss / len(replay_batches)
                self._clone_actor_optim.zero_grad()
                time_replay_loss.backward()
                self._clone_actor_optim.step()
                replay_loss = replay_loss.cpu().detach().numpy()

        clone_loss = 0
        if replay_batches is not None and len(replay_batches) != 0:
            with torch.no_grad():
                observations = batch.observations.to(self.device)
                actions = self._policy(observations).detach()
            clone_actions = self._clone_policy(observations)
            clone_loss = torch.mean((actions - clone_actions) ** 2)
            self._clone_actor_optim.zero_grad()
            clone_loss.backward()
            self._clone_actor_optim.step()
            clone_loss = clone_loss.cpu().detach().numpy()

        loss = loss.cpu().detach().numpy()
        print(f'replay_losses: {replay_losses}')

        return loss, replay_loss, clone_loss, replay_losses

    @train_api
    def update_actor(self, batch_tran: TransitionMiniBatch, replay_batches: Optional[Dict[int, List[torch.Tensor]]]=None) -> np.ndarray:
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

        unreg_grad = None
        curr_feat_ext = None
        loss = 0
        replay_loss = 0
        replay_losses = []
        save_id = self._impl_id
        if replay_batches is not None and len(replay_batches) != 0:
            for i, replay_batch in replay_batches.items():
                replay_loss = 0
                replay_batch = [x.to(self.device) for x in replay_batch]
                self.change_task(i)
                replay_batch = dict(zip(replay_name[:-2], replay_batch))
                replay_batch = Struct(**replay_batch)
                if self._replay_type == "orl":
                    replay_batch = cast(TorchMiniBatch, replay_batch)
                    replay_loss_ = self.compute_actor_loss(replay_batch)
                    replay_loss = replay_loss + replay_loss_
                    replay_losses.append(replay_loss_.cpu().detach().numpy())
                elif self._replay_type == "bc":
                    with torch.no_grad():
                        replay_observations = replay_batch.observations.to(self.device)
                        replay_policy_actions = replay_batch.policy_actions.to(self.device)
                    if self._clone_actor:
                        actions = self._clone_policy(replay_observations)
                    else:
                        actions = self._policy(replay_observations)
                    replay_loss_ = torch.mean((replay_policy_actions - actions) ** 2)
                    replay_losses.append(replay_loss_.cpu().detach().numpy())
                    replay_loss = replay_loss + replay_loss_
                elif self._replay_type == "ewc":
                    replay_loss_ = 0
                    for n, p in self._policy.named_parameters():
                        if n in self._actor_fisher.keys():
                            replay_loss_ = torch.mean(self._actor_fisher[n] * (p - self._actor_older_params[n]).pow(2)) / 2
                    replay_losses.append(replay_loss_)
                    replay_loss = replay_loss + replay_loss_
                    break
                elif self._replay_type == 'rwalk':
                    curr_feat_ext = {n: p.clone().detach() for n, p in self._policy.named_parameters() if p.requires_grad and '_fc' not in n}
                    # store gradients without regularization term
                    unreg_grad = {n: p.grad.clone().detach() for n, p in self._policy.named_parameters()
                                   if p.grad is not None and '_fc' not in n}

                    self._actor_optim.zero_grad()
                    # Eq. 3: elastic weight consolidation quadratic penalty
                    replay_loss_ = 0
                    for n, p in self._policy.named_parameters():
                        if n in self._actor_fisher.keys():
                            replay_loss_ = replay_loss_  + torch.mean((self._actor_fisher[n] + self._actor_scores[n]) * (p - self._actor_older_params[n]).pow(2)) / 2
                    replay_losses.append(replay_loss_)
                    replay_loss = replay_loss + replay_loss_
                    break
                elif self._replay_type == 'si':
                    for n, p in self._policy.named_parameters():
                        if p.grad is not None and n in self._actor_fisher.keys():
                            self._actor_W[n].add_(-p.grad * (p.detach() - self._actor_older_params[n]))
                        self._actor_older_params[n] = p.detach().clone()
                    replay_loss_ = 0
                    for n, p in self._policy.named_parameters():
                        if p.requires_grad and n in self._actor_fisher.keys():
                            replay_loss_ = replay_loss_ + torch.mean(self._actor_omega[n] * (p - self._actor_older_params[n]) ** 2)
                    replay_losses.append(replay_loss_)
                    replay_loss = replay_loss + replay_loss_
                    break
                elif self._replay_type == 'gem':
                    replay_batch = cast(TorchMiniBatch, replay_batch)
                    replay_loss_ = self.compute_actor_loss(replay_batch) / len(replay_batches)
                    replay_loss = replay_loss_
                    replay_loss.backward()
                    store_grad([p for n, p in self._policy.named_parameters() if p.requires_grad and '_fc' not in n], self._actor_grad_cs[i], self._actor_grad_dims)
                elif self._replay_type == "agem":
                    store_grad([p for n, p in self._policy.named_parameters() if p.requires_grad and '_fc' not in n], self._actor_grad_xy, self._actor_grad_dims)
                    replay_batch = cast(TorchMiniBatch, replay_batch)
                    replay_loss_ = self.compute_actor_loss(replay_batch) / len(replay_batches)
                    replay_loss = replay_loss + replay_loss_
                    replay_losses.append(replay_loss_)
                if self._replay_type in ['orl', 'ewc', 'rwalk', 'si'] or (not self._clone_actor and self._replay_type == 'bc'):
                    time_replay_loss = self._replay_alpha * replay_loss / len(replay_batches)
                    self._actor_optim.zero_grad()
                    time_replay_loss.backward()
                    self._actor_optim.step()
                    replay_loss = replay_loss
                elif self._clone_actor and self._replay_type == 'bc':
                    time_replay_loss = self._replay_alpha * replay_loss / len(replay_batches)
                    self._clone_actor_optim.zero_grad()
                    time_replay_loss.backward()
                    self._clone_actor_optim.step()
                    replay_loss = replay_loss

        self.change_task(save_id)
        self._actor_optim.zero_grad()
        loss += self.compute_actor_loss(batch)
        if self._replay_type in ['orl', 'ewc', 'rwalk', 'si'] or (not self._clone_actor and self._replay_type == 'bc'):
            loss = loss + self._replay_alpha * replay_loss
        loss.backward()

        if replay_batches is not None and len(replay_batches) != 0:
            if self._replay_type == 'agem':
                self._actor_optim.zero_grad()
                replay_loss.backward()
                store_grad([p for n, p in self._policy.named_parameters() if p.requires_grad and '_fc' not in n], self._actor_grad_er, self._actor_grad_dims)
                replay_loss.backward()
                store_grad([p for n, p in self._policy.named_parameters() if p.requires_grad and '_fc' not in n], self._actor_grad_er, self._actor_grad_dims)
                dot_prod = torch.dot(self._actor_grad_xy, self._actor_grad_er)
                if dot_prod.item() < 0:
                    g_tilde = project(gxy=self._actor_grad_xy, ger=self._actor_grad_er)
                    overwrite_grad(self._policy.parameters, g_tilde, self._actor_grad_dims)
                else:
                    overwrite_grad(self._policy.parameters, self._actor_grad_xy, self._actor_grad_dims)
            elif self._replay_type == 'gem':
                # copy gradient
                store_grad([p for n, p in self._policy.named_parameters() if p.requires_grad and '_fc' not in n], self._actor_grad_da, self._actor_grad_dims)
                dot_prod = torch.mm(self._actor_grad_da.unsqueeze(0),
                                torch.stack(list(self._actor_grad_cs).values()).T)
                if (dot_prod < 0).sum() != 0:
                    project2cone2(self._actor_grad_da.unsqueeze(1),
                                  torch.stack(list(self._actor_grad_cs).values()).T, margin=self._gem_alpha)
                    # copy gradients back
                    overwrite_grad(self._policy.parameters, self._actor_grad_da,
                                   self._actor_grad_dims)

        store_grad([p for n, p in self._policy.named_parameters() if p.requires_grad and '_fc' not in n], self._actor_grad_save, self._actor_grad_save_dims)
        self._actor_optim.step()

        clone_loss = 0
        if replay_batches is not None and len(replay_batches) != 0:
            if self._replay_type == 'rwalk':
                assert unreg_grad is not None
                assert curr_feat_ext is not None
                with torch.no_grad():
                    for n, p in self._policy.named_parameters():
                        if n in unreg_grad.keys():
                            self._actor_w[n] -= unreg_grad[n] * (p.detach() - curr_feat_ext[n])
            if self._clone_actor and self._replay_type == 'bc':
                with torch.no_grad():
                    observations = batch.observations.to(self.device)
                    actions = self._policy(observations).detach()
                clone_actions = self._clone_policy(observations)
                clone_loss = torch.mean((actions - clone_actions) ** 2)
                self._clone_actor_optim.zero_grad()
                clone_loss.backward()
                self._clone_actor_optim.step()
                store_grad([p for n, p in self._policy.named_parameters() if p.requires_grad and '_fc' not in n], self._clone_actor_grad_save, self._clone_actor_grad_save_dims)
                clone_loss = clone_loss.cpu().detach().numpy()

        loss = loss.cpu().detach().numpy()
        if not isinstance(replay_loss, int):
            replay_loss = replay_loss.cpu().detach().numpy()

        return loss, replay_loss, clone_loss, replay_losses, self._actor_grad_save, self._clone_actor_grad_save
 
    @train_api
    def retrain_update_model(self, batch: TransitionMiniBatch, retrain_batch: TransitionMiniBatch):
        assert self._dynamic is not None
        assert self._model_optim is not None
        batch = TorchMiniBatch(
            batch,
            self.device,
            scaler=self.scaler,
            action_scaler=self.action_scaler,
            reward_scaler=self.reward_scaler,
        )
        with torch.enable_grad():

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
            retrain_loss = self._dynamic.compute_error(
                observations=retrain_batch.observations,
                actions=retrain_batch.actions[:, :self._action_size],
                rewards=retrain_batch.rewards,
                next_observations=retrain_batch.next_observations,
            )
            loss = loss - self._retrain_model_alpha * retrain_loss

            self._model_optim.zero_grad()
            loss.backward()
            self._model_optim.step()

            loss = loss.cpu().detach().numpy()

            return loss

    @train_api
    def update_model(self, batch: TransitionMiniBatch, replay_batches: Optional[Dict[int, List[torch.Tensor]]]=None):
        assert self._dynamic is not None
        assert self._model_optim is not None
        batch = TorchMiniBatch(
            batch,
            self.device,
            scaler=self.scaler,
            action_scaler=self.action_scaler,
            reward_scaler=self.reward_scaler,
        )

        self.change_task(self._impl_id)
        self._q_func.eval()
        self._policy.eval()
        self._dynamic.train()

        unreg_grad = None
        curr_feat_ext = None
        loss = 0
        replay_loss = 0
        replay_losses = []
        save_id = self._impl_id
        if replay_batches is not None and len(replay_batches) != 0:
            for i, replay_batch in replay_batches.items():
                replay_loss = 0
                self.change_task(i)
                replay_batch = [x.to(self.device) for x in replay_batch]
                replay_batch = dict(zip(replay_name[:-2], replay_batch))
                replay_batch = Struct(**replay_batch)
                if self._replay_type in ["orl", 'bc']:
                    # 对于model来说没有bc和orl之分。
                    with torch.no_grad():
                        replay_observations = replay_batch.observations.to(self.device)
                        replay_rewards = replay_batch.rewards.to(self.device)
                        replay_actions = replay_batch.actions.to(self.device)
                        replay_next_observations = replay_batch.next_observations.to(self.device)
                    replay_loss_ = self._dynamic.compute_error(
                        observations=replay_observations,
                        actions=replay_actions[:, :self._action_size],
                        rewards=replay_rewards,
                        next_observations=replay_next_observations,
                    )
                    replay_loss = replay_loss + replay_loss_
                    replay_losses.append(replay_loss_.cpu().detach().numpy())
                elif self._replay_type == "ewc":
                    replay_loss_ = 0
                    for n, p in self._model_func.named_parameters():
                        if n in self._model_fisher.keys():
                            replay_loss_ = torch.sum(self._model_fisher[n] * (p - self._model_older_params[n]).pow(2)) / 2
                    replay_losses.append(replay_loss_)
                    replay_loss = replay_loss + replay_loss_
                elif self._replay_type == 'rwalk':
                    curr_feat_ext = {n: p.clone().detach() for n, p in self._dynamic.named_parameters() if p.requires_grad}
                    # store gradients without regularization term
                    unreg_grad = {n: p.grad.clone().detach() for n, p in self._dynamic.named_parameters()
                                   if p.grad is not None}

                    self._model_optim.zero_grad()
                    # Eq. 3: elastic weight consolidation quadratic penalty
                    replay_loss_ = 0
                    for n, p in self._model.named_parameters():
                        if n in self._model_fisher.keys():
                            replay_loss_ = torch.sum((self._model_fisher[n] + self._model_scores[n]) * (p - self.__model_params[n]).pow(2)) / 2
                    replay_losses.append(replay_loss_)
                    replay_loss = replay_loss + replay_loss_
                elif self._replay_type == 'si':
                    for n, p in self._dynamic.named_parameters():
                        if p.grad is not None and n in self._model_fisher.keys():
                            self._model_W[n].add_(-p.grad * (p.detach() - self._model_older_params[n]))
                        self._model_older_params[n] = p.detach().clone()
                    replay_loss_ = 0
                    for n, p in self.named_parameters():
                        if p.requires_grad:
                            replay_loss_ = replay_loss_ + torch.sum(self._model_omega[n] * (p - self._model_older_params[n]) ** 2)
                    replay_losses.append(replay_loss_)
                    replay_loss = replay_loss + replay_loss_
                elif self._replay_type == 'gem':
                    with torch.no_grad():
                        replay_observations = replay_batch.observations.to(self.device)
                        replay_rewards = replay_batch.rewards.to(self.device)
                        replay_actions = replay_batch.actions.to(self.device)
                        replay_next_observations = replay_batch.next_observations.to(self.device)
                    replay_loss_ = self._dynamic.compute_error(
                        observations=replay_observations,
                        actions=replay_actions[:, :self._action_size],
                        rewards=replay_rewards,
                        next_observations=replay_next_observations,
                    )
                    replay_loss_ = replay_loss_ / len(replay_batches)
                    replay_loss = replay_loss_
                    replay_loss.backward()
                    store_grad(self._dynamic.parameters(), self._model_grad_cs[i], self._model_grad_dims)
                elif self._replay_type == "agem":
                    store_grad(self._dynamic.parameters(), self._model_grad_xy, self._model_grad_dims)
                    replay_batch = cast(TorchMiniBatch, replay_batch)
                    replay_loss_ = self._dynamic.compute_error(
                        observations=replay_batch.observations,
                        actions=replay_batch.actions,
                        rewards=replay_batch.rewards,
                        next_observations=replay_batch.next_observations,
                    )
                    replay_loss_ = replay_loss_ / len(replay_batches)
                    replay_loss = replay_loss + replay_loss_
                    replay_losses.append(replay_loss_)
                if self._replay_type in ['orl', 'bc', 'ewc', 'rwalk', 'si']:
                    time_replay_loss = self._replay_alpha * replay_loss / len(replay_batches)
                    self._model_optim.zero_grad()
                    time_replay_loss.backward()
                    self._model_optim.step()
                    replay_loss = replay_loss.cpu().detach().numpy()

        self.change_task(save_id)
        self._model_optim.zero_grad()
        loss = self._dynamic.compute_error(
            observations=batch.observations,
            actions=batch.actions[:, :self._action_size],
            rewards=batch.rewards,
            next_observations=batch.next_observations,
        )
        loss.backward()

        if replay_batches is not None and len(replay_batches) != 0:
            if self._replay_type == 'agem':
                self._model_optim.zero_grad()
                replay_loss.backward()
                store_grad(self._dynamic.parameters(), self._model_grad_cs[i], self._model_grad_dims)
                dot_prod = torch.dot(self._model_grad_xy, self._model_grad_er)
                if dot_prod.item() < 0:
                    g_tilde = project(gxy=self._model_grad_xy, ger=self._model_grad_er)
                    overwrite_grad(self._dynamic.parameters, g_tilde, self._model_grad_dims)
                else:
                    overwrite_grad(self._dynamic.parameters, self._model_grad_xy, self._model_grad_dims)
            elif self._replay_type == 'gem':
                # copy gradient
                store_grad(self._dynamic.parameters(), self._model_grad_da, self._model_grad_dims)
                dot_prod = torch.mm(self._model_grad_da.unsqueeze(0),
                                torch.stack(list(self._model_grad_cs).values()).T)
                if (dot_prod < 0).sum() != 0:
                    project2cone2(self._model_grad_da.unsqueeze(1),
                                  torch.stack(list(self._model_grad_cs).values()).T, margin=self._gem_gamma)
                    # copy gradients back
                    overwrite_grad(self._dynamic.parameters, self._model_grad_da,
                                   self._model_grad_dims)
        self._model_optim.step()

        if replay_batches is not None and len(replay_batches) != 0:
            if self._replay_type == 'rwalk':
                assert unreg_grad is not None
                assert curr_feat_ext is not None
                with torch.no_grad():
                    for n, p in self._dynamic.named_parameters():
                        if n in unreg_grad.keys():
                            self._model_w[n] -= unreg_grad[n] * (p.detach() - curr_feat_ext[n])

        loss = loss.cpu().detach().numpy()
        self.change_task(self._impl_id)

        return loss, replay_loss, replay_losses

    def update_critic_target(self) -> None:
        assert self._q_func is not None
        assert self._targ_q_func is not None
        soft_sync(self._targ_q_func, self._q_func, self._tau)
        if self._replay_type == 'orl' and '_q_funcs' in self._q_func.__dict__ and '_fcs' in self._q_func._q_funcs[0].__dict__:
            if self._replay_critic:
                with torch.no_grad():
                    for q_func, targ_q_func in zip(self._q_func._q_funcs, self._targ_q_func._q_funcs):
                        for key in q_func._fcs.keys():
                            for key_in in q_func._fcs[key].keys():
                                targ_param = targ_q_func._fcs[key][key_in]
                                param = q_func._fcs[key][key_in]
                                targ_param.data.mul_(1 - self._tau)
                                targ_param.data.add_(self._tau * param.data)

    def update_actor_target(self) -> None:
        assert self._policy is not None
        assert self._targ_policy is not None
        soft_sync(self._targ_policy, self._policy, self._tau)
        if self._replay_type == 'orl':
            if '_fcs' in self._policy.__dict__:
                with torch.no_grad():
                    for key in self._policy._fcs.keys():
                        for key_in in self._policy._fcs[key].keys():
                            targ_param = self._targ_policy._fcs[key][key_in]
                            param = self._policy._fcs[key][key_in]
                            targ_param.data.mul_(1 - self._tau)
                            targ_param.data.add_(self._tau * param.data)
            if '_mus' in self._policy.__dict__:
                with torch.no_grad():
                    for key in self._policy._mus.keys():
                        for key_in in self._policy._mus[key].keys():
                            targ_param = self._targ_policy._logstds[key][key_in]
                            param = self._policy._logstds[key][key_in]
                            targ_param.data.mul_(1 - self._tau)
                            targ_param.data.add_(self._tau * param.data)
                if isinstance(self._targ_policy._logstd, torch.nn.parameter.Parameter):
                    for key in self._policy._logstds.keys():
                        self._targ_policy._logstds[key].data.mul_(1 - self._tau)
                        self._targ_policy._logstds[key].data.add_(self._tau * self._policy._logstds[key].data)
                else:
                    for key in self._policy._logstds.keys():
                        for key_in in self._policy._logstds[key].keys():
                            targ_param = self._targ_policy._logstds[key][key_in]
                            param = self._policy._logstds[key][key_in]
                            targ_param.data.mul_(1 - self._tau)
                            targ_param.data.add_(self._tau * param.data)

    def compute_fisher_matrix_diag(self, iterator, network, optim, update):
        # Store Fisher Information
        fisher = {n: torch.zeros(p.shape).to(self.device) for n, p in network.named_parameters()
                  if p.requires_grad and '_fc' not in n}
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
                if p.grad is not None and '_fc' not in n:
                    fisher[n] += p.grad.pow(2)
        # Apply mean across all samples
        fisher = {n: (p / len(iterator)) for n, p in fisher.items()}
        return fisher

    def fix_post_train_process(self):
        for name, param in self._policy.named_parameters():
            if param.requires_grad and '_fc' not in name:
                param.requires_grad_(False)

    def bc_post_train_process(self):
        self._policy.load_state_dict(copy.deepcopy(self._clone_policy.state_dict()))

    def gem_post_train_process(self):
        self._critic_grad_cs[self._impl_id] = torch.zeros(np.sum(self._critic_grad_dims)).to(self.device)
        self._actor_grad_cs[self._impl_id] = torch.zeros(np.sum(self._actor_grad_dims)).to(self.device)

    def ewc_rwalk_post_train_process(self, iterator):
        if self._replay_critic:
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
                loss.backward()
            curr_fisher = self.compute_fisher_matrix_diag(iterator, self._q_func, self._critic_optim, update)
            # merge fisher information, we do not want to keep fisher information for each task in memory
            for n in self._critic_fisher.keys():
                # Added option to accumulate fisher over time with a pre-fixed growing self._ewc_rwalk_alpha
                self._critic_fisher[n] = (self._ewc_rwalk_alpha * self._critic_fisher[n] + (1 - self._ewc_rwalk_alpha) * curr_fisher[n])

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

        if self._replay_type == 'rwalk':
            if self._replay_critic:
                # Page 7: Optimization Path-based Parameter Importance: importance scores computation
                curr_score = {n: torch.zeros(p.shape).to(self.device) for n, p in self._q_func.named_parameters()
                              if p.requires_grad and '_fc' not in n}
                with torch.no_grad():
                    curr_params = {n: p for n, p in self._q_func.named_parameters() if p.requires_grad and '_fc' not in n}
                    for n, p in self._critic_scores.items():
                        curr_score[n] = self._critic_w[n] / (
                                self._critic_fisher[n] * ((curr_params[n] - self._critic_older_params[n]) ** 2) + self._damping)
                        self._critic_w[n].zero_()
                        # Page 7: "Since we care about positive influence of the parameters, negative scores are set to zero."
                        curr_score[n] = torch.nn.functional.relu(curr_score[n])
                # Page 8: alleviating regularization getting increasingly rigid by averaging scores
                for n, p in self._critic_scores.items():
                    self._critic_scores[n] = (self._critic_scores[n] + curr_score[n]) / 2

            # Page 7: Optimization Path-based Parameter Importance: importance scores computation
            curr_score = {n: torch.zeros(p.shape).to(self.device) for n, p in self._policy.named_parameters()
                          if p.requires_grad and '_fc' not in n}
            with torch.no_grad():
                curr_params = {n: p for n, p in self._policy.named_parameters() if p.requires_grad and '_fc' not in n}
                for n, p in self._actor_scores.items():
                    curr_score[n] = self._actor_w[n] / (
                            self._actor_fisher[n] * ((curr_params[n] - self._actor_older_params[n]) ** 2) + self._damping)
                    self._actor_w[n].zero_()
                    # Page 7: "Since we care about positive influence of the parameters, negative scores are set to zero."
                    curr_score[n] = torch.nn.functional.relu(curr_score[n])
            # Page 8: alleviating regularization getting increasingly rigid by averaging scores
            for n, p in self._actor_scores.items():
                self._actor_scores[n] = (self._actor_scores[n] + curr_score[n]) / 2

    def si_post_train_process(self):
        if self._replay_critic:
            for n, p in self._q_func.named_parameters():
                if p.requires_grad and '_fc' not in n:
                    p_change = p.detach().clone() - self._critic_older_params[n]
                    omega_add = self._critic_W[n] / (p_change ** 2 + self._epsilon)
                    omega = self._critic_omega[n]
                    omega_new = omega + omega_add
                    self._critic_older_params[n] = p.detach().clone()
                    self._critic_omega[n] = omega_new
        for n, p in self._policy.named_parameters():
            if p.requires_grad and '_fc' not in n:
                p_change = p.detach().clone() - self._actor_older_params[n]
                omega_add = self._actor_W[n] / (p_change ** 2 + self._epsilon)
                omega = self._actor_omega[n]
                omega_new = omega + omega_add
                self._actor_older_params[n] = p.detach().clone()
                self._actor_omega[n] = omega_new

    def change_task_singlehead(self, task_id):
        self._impl_id = task_id
