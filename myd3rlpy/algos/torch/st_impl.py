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
from myd3rlpy.iterators.base import TransitionIterator

from myd3rlpy.algos.torch.gem import overwrite_grad, store_grad, project2cone2
from myd3rlpy.algos.torch.agem import project


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

    def build(self):
        super().build()

        self._critic_networks = [self._q_func]
        self._actor_networks = [self._policy]

        assert self._q_func is not None
        if self._critic_replay_type in ['ewc', 'rwalk', 'si']:
            # Store current parameters for the next task
            self._critic_older_params = [{n: p.clone().detach() for n, p in network.named_parameters() if p.requires_grad} for network in self._critic_networks]
            if self._critic_replay_type in ['ewc', 'rwalk']:
                # Store fisher information weight importance
                self._critic_fisher = [{n: torch.zeros(p.shape).to(self.device) for n, p in network.named_parameters() if p.requires_grad} for network in self._critic_networks]
            if self._critic_replay_type == 'rwalk':
                # Page 7: "task-specific parameter importance over the entire training trajectory."
                self._critic_W = [{n: torch.zeros(p.shape).to(self.device) for n, p in network.named_parameters() if p.requires_grad} for network in self._critic_networks]
                self._critic_scores = [{n: torch.zeros(p.shape).to(self.device) for n, p in network.named_parameters() if p.requires_grad} for network in self._critic_networks]
            elif self._critic_replay_type == 'si':
                self._critic_W = [{n: p.clone().detach().zero_() for n, p in network.named_parameters() if p.requires_grad} for network in self._critic_networks]
                self._critic_omega = [{n: p.clone().detach().zero_() for n, p in network.named_parameters() if p.requires_grad} for network in self._critic_networks]
        elif self._critic_replay_type == 'gem':
            # Allocate temporary synaptic memory
            self._critic_grad_dims = [[pp.data.numel() for pp in network.parameters()] for network in self._critic_networks]
            self._critic_grads_cs = [torch.zeros(np.sum(critic_grad_dims)).to(self.device) for critic_grad_dims in self._critic_grad_dims]
            self._critic_grads_da = [torch.zeros(np.sum(critic_grad_dims)).to(self.device) for critic_grad_dims in self._critic_grad_dims]
        elif self._critic_replay_type == 'agem':
            self._critic_grad_dims = [[pp.data.numel() for pp in network.parameters()] for network in self._critic_networks]
            self._critic_grad_xy = [torch.zeros(np.sum(critic_grad_dims)).to(self.device) for critic_grad_dims in self._critic_grad_dims]
            self._critic_grad_er = [torch.zeros(np.sum(critic_grad_dims)).to(self.device) for critic_grad_dims in self._critic_grad_dims]

        if self._actor_replay_type in ['ewc', 'rwalk', 'si']:
            # Store current parameters for the next task
            self._actor_older_params = [{n: p.clone().detach() for n, p in network.named_parameters() if p.requires_grad} for network in self._actor_networks]
            if self._actor_replay_type in ['ewc', 'rwalk']:
                # Store fisher information weight importance
                self._actor_fisher = [{n: torch.zeros(p.shape).to(self.device) for n, p in network.named_parameters() if p.requires_grad} for network in self._actor_networks]
            if self._actor_replay_type == 'rwalk':
                # Page 7: "task-specific parameter importance over the entire training trajectory."
                self._actor_W = [{n: torch.zeros(p.shape).to(self.device) for n, p in network.named_parameters() if p.requires_grad} for network in self._actor_networks]
                self._actor_scores = [{n: torch.zeros(p.shape).to(self.device) for n, p in network.named_parameters() if p.requires_grad} for network in self._actor_networks]
            elif self._actor_replay_type == 'si':
                self._actor_W = [{n: p.clone().detach().zero_() for n, p in network.named_parameters() if p.requires_grad} for network in self._actor_networks]
                self._actor_omega = [{n: p.clone().detach().zero_() for n, p in network.named_parameters() if p.requires_grad} for network in self._actor_networks]
        elif self._actor_replay_type == 'gem':
            # Allocate temporary synaptic memory
            self._actor_grad_dims = [[pp.data.numel() for pp in network.parameters()] for network in self._actor_networks]
            self._actor_grads_cs = [torch.zeros(np.sum(actor_grad_dims)).to(self.device) for actor_grad_dims in self._actor_grad_dims]
            self._actor_grads_da = [torch.zeros(np.sum(actor_grad_dims)).to(self.device) for actor_grad_dims in self._actor_grad_dims]
        elif self._actor_replay_type == 'agem':
            self._actor_grad_dims = [[pp.data.numel() for pp in network.parameters()] for network in self._actor_networks]
            self._actor_grad_xy = [torch.zeros(np.sum(actor_grad_dims)).to(self.device) for actor_grad_dims in self._actor_grad_dims]
            self._actor_grad_er = [torch.zeros(np.sum(actor_grad_dims)).to(self.device) for actor_grad_dims in self._actor_grad_dims]

    def _add_ewc_loss(self, networks, fishers, older_params):
        replay_ewc_loss = 0
        for network, fisher, older_param in zip(networks, fishers, older_params):
            for n, p in network.named_parameters():
                if n in fisher.keys():
                    replay_ewc_loss += torch.mean(fisher[n] * (p - older_param[n]).pow(2)) / 2
        return replay_ewc_loss

    def _pre_rwalk_loss(self, networks, fishers, scores, older_params):
        unreg_grads = [{n: p.grad.clone().detach() for n, p in network.named_parameters() if p.grad is not None} for network in networks]
        replay_rwalk_loss = 0
        for network, fisher, score, older_param in zip(networks, fishers, scores, older_params):
            for n, p in network.named_parameters():
                if n in fisher.keys():
                    replay_rwalk_loss += torch.mean((fisher[n] + score[n]) * (p - older_param[n]).pow(2)) / 2
        return unreg_grads, replay_rwalk_loss

    def _add_si_loss(self, networks, older_params, Ws, omegas):
        for network, older_param, W, omega in zip(networks, older_params, Ws, omegas):
            for n, p in network.named_parameters():
                if p.grad is not None and n in W.keys():
                    p_change = p.detach().clone() - older_param[n]
                    W[n].add_(-p.grad * p_change)
                    omega_add = W[n] / (p_change ** 2 + self._epsilon)
                    omega_old = omega[n]
                    omega_new = omega_old + omega_add
                    omega[n] = omega_new
        replay_si_loss = 0
        for network, omega, older_param in zip(networks, omegas, older_params):
            for n, p in network.named_parameters():
                if p.requires_grad:
                    replay_si_loss += torch.mean(omega[n] * (p - older_param[n]) ** 2)
                older_param[n].data = p.data.clone()
        return replay_si_loss

    def _pre_gem_loss(self, networks, grads_cs, grad_dims):
        for network, grads_cs, grad_dim in zip(networks, grads_cs, grad_dims):
            store_grad(network.parameters(), grads_cs, grad_dim)

    def _pre_agem_loss(self, networks, grad_xys, grad_dims):
        for network, grad_xy, grad_dim in zip(networks, grad_xys, grad_dims):
            store_grad(network.parameters(), grad_xy, grad_dim)

    def _pos_agem_loss(self, networks, grad_ers, grad_dims, grad_xys):
        for network, grad_er, grad_dim, grad_xy in zip(networks, grad_ers, grad_dims, grad_xys):
            store_grad(network.parameters(), grad_er, grad_dim)
            dot_prod = torch.dot(grad_xy, grad_er)
            if dot_prod.item() < 0:
                g_tilde = project(gxy=grad_xy, ger=grad_er)
                overwrite_grad(network.parameters, g_tilde, grad_dim)
            else:
                overwrite_grad(network.parameters, grad_xy, grad_dim)

    def _pos_gem_loss(self, networks, grads_das, grad_dims, grads_css):
        for network, grads_da, grad_dim, grads_cs in zip(networks, grads_das, grad_dims, grads_css):
            # copy gradient
            store_grad(network.parameters(), grads_da, grad_dim)
            dot_prod = torch.dot(grads_da, grads_cs)
            if (dot_prod < 0).sum() != 0:
                # project2cone2(self._actor_grads_da.unsqueeze(1),
                #               torch.stack(list(self._actor_grads_cs).values()).T, margin=self._gem_alpha)
                project2cone2(grads_da.unsqueeze(dim=1), grads_cs.unsqueeze(dim=1), margin=self._gem_alpha)
                # copy gradients back
                overwrite_grad(network.parameters, grads_da, grad_dims)

    def _pos_rwalk_loss(self, networks, unreg_grads, Ws, curr_feat_exts):
        for network, grad, W, curr_feat_ext in zip(networks, unreg_grads, Ws, curr_feat_exts):
            for n, p in network.named_parameters():
                if n in grad.keys():
                    W[n] -= grad[n] * (p.detach() - curr_feat_ext[n])

    #@eval_api
    #@torch_api(scaler_targets=["x"])
    #def predict_best_action(self, x: torch.Tensor) -> np.ndarray:
    #    return super().predict_best_action(x)

    @train_api
    @torch_api(reward_scaler_targets=["batch", "replay_batch"])
    def update_critic(self, batch: TransitionMiniBatch, replay_batch: TransitionMiniBatch=None, clone_critic: bool=False, online: bool=False):
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
                replay_ewc_loss = self._add_ewc_loss(self._critic_networks, self._critic_fisher, self._critic_older_params)
                replay_loss = replay_loss + replay_ewc_loss
            elif self._critic_replay_type == 'rwalk':
                curr_feat_ext = [{n: p.clone().detach() for n, p in network.named_parameters() if p.requires_grad} for network in self._critic_networks]
                # store gradients without regularization term
                self._critic_optim.zero_grad()
                q_tpn = self.compute_target(batch)
                loss = self.compute_critic_loss(batch, q_tpn, clone_critic=clone_critic, online=online, first_time=replay_batch==None)
                loss.backward(retain_graph=True)

                unreg_grads, replay_rwalk_loss = self._pre_rwalk_loss(self._critic_networks, self._critic_fisher, self._critic_scores, self._critic_older_params)
                replay_loss = replay_loss + replay_rwalk_loss
            elif self._critic_replay_type == 'si':
                replay_si_loss = self._add_si_loss(self._critic_networks, self._critic_older_params, self._critic_W, self._critic_omega)
                replay_loss = replay_loss + replay_si_loss
            elif self._critic_replay_type == 'gem':
                q_tpn = self.compute_target(replay_batch)
                replay_loss = self.compute_critic_loss(replay_batch, q_tpn, clone_critic=clone_critic)
                replay_loss.backward()
                self._pre_gem_loss(self._critic_networks, self._critic_grads_cs, self._critic_grad_dims)
            elif self._critic_replay_type == "agem":
                self._pre_agem_loss(self._critic_networks, self._critic_grad_xy, self._critic_grad_dims)
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
                self._pos_agem_loss(self._critic_networks, self._critic_grad_er, self._critic_grad_dims, self._critic_grad_xy)
            elif self._critic_replay_type == 'gem':
                self._pos_gem_loss(self._critic_networks, self._critic_grads_da, self._critic_grad_dims, self._critic_grads_cs)
        self._critic_optim.step()

        if replay_batch is not None and not online:
            if self._critic_replay_type == 'rwalk':
                assert unreg_grads is not None
                assert curr_feat_ext is not None
                with torch.no_grad():
                    self._pos_rwalk_loss(self._critic_networks, unreg_grads, self._critic_W, curr_feat_ext)

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
    def update_actor(self, batch: TransitionMiniBatch, replay_batch: Optional[List[torch.Tensor]]=None, clone_actor: bool=False, online: bool=False) -> np.ndarray:
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
                replay_ewc_loss = self._add_ewc_loss(self._actor_networks, self._actor_fisher, self._actor_older_params)
                replay_loss = replay_loss + replay_ewc_loss
            elif self._actor_replay_type == 'rwalk':
                curr_feat_ext = {n: p.clone().detach() for n, p in self._policy.named_parameters() if p.requires_grad}
                self._actor_optim.zero_grad()
                loss = self.compute_actor_loss(batch, clone_actor=clone_actor)
                loss.backward(retain_graph=True)
                # Eq. 3: elastic weight consolidation quadratic penalty
                unreg_grads, replay_rwalk_loss = self._pre_rwalk_loss(self._actor_networks, self._actor_fisher, self._critic_scores, self._critic_older_params)
                replay_loss = replay_loss + replay_rwalk_loss
            elif self._actor_replay_type == 'si':
                replay_si_loss = self._add_si_loss(self._actor_networks, self._actor_older_params, self._actor_W, self._actor_omega)
                replay_loss = replay_loss + replay_si_loss
            elif self._actor_replay_type == 'gem':
                replay_loss_ = self.compute_actor_loss(replay_batch, clone_actor=clone_actor)
                replay_loss = replay_loss_
                replay_loss.backward()
                self._pre_gem_loss(self._actor_networks, self._actor_grads_cs, self._actor_grad_dims)
            elif self._actor_replay_type == "agem":
                self._pre_agem_loss(self._actor_networks, self._actor_grad_xy, self._actor_grad_dims)
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
                self._pos_agem_loss(self._actor_networks, self._actor_grad_er, self._actor_grad_dims, self._actor_grad_xy)
            elif self._actor_replay_type == 'gem':
                self._pos_gem_loss(self._actor_networks, self._actor_grads_da, self._actor_grad_dims, self._actor_grads_cs)

        self._actor_optim.step()

        if replay_batch is not None and not online:
            if self._actor_replay_type == 'rwalk':
                assert unreg_grads is not None
                assert curr_feat_ext is not None
                with torch.no_grad():
                    self._pos_rwalk_loss(self._actor_networks, unreg_grads, self._actor_W, curr_feat_ext)

        if not isinstance(loss, int):
            loss = loss.cpu().detach().numpy()
        if not isinstance(replay_loss, int):
            replay_loss = replay_loss.cpu().detach().numpy()

        return loss, replay_loss

    def compute_fisher_matrix_diag(self, iterator, network, optim, update, batch_size=None, n_frames=None, n_steps=None, gamma=None, test=False):
        # Store Fisher Information
        fisher = {n: torch.zeros(p.shape).to(self.device) for n, p in network.named_parameters()
                  if p.requires_grad}
        # Do forward and backward pass to compute the fisher information
        network.train()
        replay_loss = 0
        if isinstance(iterator, TransitionIterator):
            iterator.reset()
        else:
            pass
        for t in range(len(iterator) if not test else 2):
            if isinstance(iterator, TransitionIterator):
                batch = next(iterator)
            else:
                batch = iterator.sample(batch_size=batch_size,
                        n_frames=n_frames,
                        n_steps=n_steps,
                        gamma=gamma)
            optim.zero_grad()
            update(self, batch)
            # Accumulate all gradients from loss with regularization
            for n, p in network.named_parameters():
                if p.grad is not None:
                    fisher[n] += p.grad.pow(2)
        # Apply mean across all samples
        fisher = {n: (p / len(iterator)) for n, p in fisher.items()}
        return fisher

    def _ewc_rwalk_post_train_process(self, networks, fishers, older_params, iterator, optim, update, scores=None, Ws=None, batch_size=None, n_frames=None, n_steps=None, gamma=None, test=False):
        if self._critic_replay_type == 'rwalk':
            looper = zip(networks, fishers, scores, Ws, older_params)
        else:
            looper = zip(networks, fishers, older_params)
        for i, elems in enumerate(looper):
            if self._critic_replay_type == 'rwalk':
                network, fisher, score, W, older_param = elems
            else:
                network, fisher, older_param = elems
            curr_fisher = self.compute_fisher_matrix_diag(iterator, network, optim, update, batch_size, n_frames=n_frames, n_steps=n_steps, gamma=gamma, test=test)
            # merge fisher information, we do not want to keep fisher information for each task in memory
            for n in fisher.keys():
                # Added option to accumulate fisher over time with a pre-fixed growing self._ewc_rwalk_alpha
                fisher[n] = (self._ewc_rwalk_alpha * fisher[n] + (1 - self._ewc_rwalk_alpha) * curr_fisher[n])

            if self._critic_replay_type == 'rwalk':
                # Page 7: Optimization Path-based Parameter Importance: importance scores computation
                curr_critic_score = {n: torch.zeros(p.shape).to(self.device) for n, p in network.named_parameters() if p.requires_grad}
                with torch.no_grad():
                    curr_critic_params = {n: p for n, p in network.named_parameters() if p.requires_grad}
                    for n, p in score.items():
                        curr_critic_score[n] = W[n] / (
                                fisher[n] * ((curr_critic_params[n] - older_param[n]) ** 2) + self._damping)
                        W[n].zero_()
                        # Page 7: "Since we care about positive influence of the parameters, negative scores are set to zero."
                        curr_critic_score[n] = torch.nn.functional.relu(curr_critic_score[n])
                        older_param[n].data = curr_critic_params[n].data.clone()
                # Page 8: alleviating regularization getting increasingly rigid by averaging scores
                for n, p in score.items():
                    score[n] = (p + curr_critic_score[n]) / 2

    def critic_ewc_rwalk_post_train_process(self, iterator, batch_size, n_frames, n_steps, gamma, test=False):
        # calculate Fisher information
        @train_api
        @torch_api()
        def update(self, batch):
            q_tpn = self.compute_target(batch)
            loss = self.compute_critic_loss(batch, q_tpn)
            loss.backward()
        if self._critic_replay_type == 'rwalk':
            self._ewc_rwalk_post_train_process(self._critic_networks, self._critic_fisher, self._critic_older_params, iterator, self._critic_optim, update, scores=self._critic_scores, Ws=self._critic_W, batch_size=batch_size, n_frames=n_frames, n_steps=n_steps, gamma=gamma, test=test)
        else:
            self._ewc_rwalk_post_train_process(self._critic_networks, self._critic_fisher, self._critic_older_params, iterator, self._critic_optim, update, batch_size=batch_size, n_frames=n_frames, n_steps=n_steps, gamma=gamma, test=test)

    def actor_ewc_rwalk_post_train_process(self, iterator, batch_size, n_frames, n_steps, gamma, test=False):
        @train_api
        @torch_api()
        def update(self, batch):
            loss = self.compute_actor_loss(batch)
            loss.backward()
        if self._actor_replay_type == 'rwalk':
            self._ewc_rwalk_post_train_process(self._actor_networks, self._actor_fisher, self._actor_older_params, iterator, self._actor_optim, update, scores=self._actor_scores, Ws=self._actor_W, batch_size=batch_size, n_frames=n_frames, n_steps=n_steps, gamma=gamma, test=test)
        else:
            self._ewc_rwalk_post_train_process(self._actor_networks, self._actor_fisher, self._actor_older_params, iterator, self._actor_optim, update, batch_size=batch_size, n_frames=n_frames, n_steps=n_steps, gamma=gamma, test=test)
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
