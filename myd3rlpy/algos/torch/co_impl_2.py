import time
import math
import copy
from typing import Optional, Sequence, List, Any, Tuple, Dict, Union, cast

import numpy as np
import torch
from torch import nn
import torch.nn.functional as F

from d3rlpy.gpu import Device
from d3rlpy.models.encoders import EncoderFactory
from d3rlpy.models.optimizers import OptimizerFactory
from d3rlpy.models.q_functions import QFunctionFactory
from d3rlpy.preprocessing import ActionScaler, RewardScaler, Scaler
from d3rlpy.torch_utility import TorchMiniBatch, soft_sync, train_api, torch_api
from d3rlpy.dataset import TransitionMiniBatch
from d3rlpy.algos.torch.cql_impl import CQLImpl

from myd3rlpy.models.builders import create_phi, create_psi
from myd3rlpy.algos.torch.gem import overwrite_grad, store_grad, project2cone2
from myd3rlpy.algos.torch.agem import project
from utils.utils import Struct


class COImpl(CQLImpl):
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
        actor_optim_factory: OptimizerFactory,
        critic_optim_factory: OptimizerFactory,
        temp_optim_factory: OptimizerFactory,
        alpha_optim_factory: OptimizerFactory,
        phi_optim_factory: OptimizerFactory,
        psi_optim_factory: OptimizerFactory,
        actor_encoder_factory: EncoderFactory,
        critic_encoder_factory: EncoderFactory,
        q_func_factory: QFunctionFactory,
        replay_actor_alpha: float,
        replay_critic_alpha: float,
        replay_type: str,
        gamma: float,
        gem_gamma: float,
        agem_alpha: float,
        tau: float,
        n_critics: int,
        initial_alpha: float,
        initial_temperature: float,
        alpha_threshold: float,
        conservative_weight: float,
        n_action_samples: int,
        soft_q_backup: bool,
        use_gpu: Optional[Device],
        scaler: Optional[Scaler],
        action_scaler: Optional[ActionScaler],
        reward_scaler: Optional[RewardScaler],
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
            initial_temperature = initial_temperature,
            initial_alpha = initial_alpha,
            alpha_threshold = alpha_threshold,
            actor_encoder_factory = actor_encoder_factory,
            critic_encoder_factory = critic_encoder_factory,
            q_func_factory = q_func_factory,
            gamma = gamma,
            tau = tau,
            n_critics = n_critics,
            conservative_weight = conservative_weight,
            n_action_samples = n_action_samples,
            soft_q_backup = soft_q_backup,
            use_gpu = use_gpu,
            scaler = scaler,
            action_scaler = action_scaler,
            reward_scaler = reward_scaler,
        )
        self._replay_type = replay_type
        self._replay_actor_alpha = replay_actor_alpha
        self._replay_critic_alpha = replay_critic_alpha

        self._alpha_learning_rate = alpha_learning_rate
        self._alpha_optim_factory = alpha_optim_factory
        self._initial_alpha = initial_alpha
        self._alpha_threshold = alpha_threshold
        self._gem_gamma = gem_gamma
        self._agem_alpha = agem_alpha

        self._phi_learning_rate = phi_learning_rate
        self._psi_learning_rate = psi_learning_rate
        self._phi_optim_factory = phi_optim_factory
        self._psi_optim_factory = psi_optim_factory

        # initialized in build

    def build(self):
        self._phi = create_phi(self._observation_shape, self._action_size, self._critic_encoder_factory)
        self._psi = create_psi(self._observation_shape, self._actor_encoder_factory)
        self._targ_phi = copy.deepcopy(self._phi)
        self._targ_psi = copy.deepcopy(self._psi)
        super().build()
        self._phi_optim = self._phi_optim_factory.create(
            self._phi.parameters(), lr=self._phi_learning_rate
        )
        self._psi_optim = self._psi_optim_factory.create(
            self._psi.parameters(), lr=self._psi_learning_rate
        )
        assert self._q_func is not None
        assert self._policy is not None
        if self._replay_type in ['ewc', 'r_walk']:
            # Store fisher information weight importance
            self._critic_fisher = {n: torch.zeros(p.shape).to(self.device) for n, p in self._q_func.named_parameters() if p.requires_grad}
            # Store current parameters for the next task
            self._critic_older_params = {n: p.clone().detach() for n, p in self._q_func.named_parameters() if p.requires_grad}
            # Store fisher information weight importance
            self._actor_fisher = {n: torch.zeros(p.shape).to(self.device) for n, p in self._policy.named_parameters() if p.requires_grad}
            # Store current parameters for the next task
            self._actor_older_params = {n: p.clone().detach() for n, p in self._policy.named_parameters() if p.requires_grad}
            if self._replay_type == 'r_walk':
                # Page 7: "task-specific parameter importance over the entire training trajectory."
                self._critic_w = {n: torch.zeros(p.shape).to(self.device) for n, p in self._q_func.named_parameters() if p.requires_grad}
                self._critic_scores = {n: torch.zeros(p.shape).to(self.device) for n, p in self._q_func.named_parameters() if p.requires_grad}

                # Page 7: "task-specific parameter importance over the entire training trajectory."
                self._actor_w = {n: torch.zeros(p.shape).to(self.device) for n, p in self._policy.named_parameters() if p.requires_grad}
                self._actor_scores = {n: torch.zeros(p.shape).to(self.device) for n, p in self._policy.named_parameters() if p.requires_grad}
        elif self._replay_type == 'gem':
            # Allocate temporary synaptic memory
            self._critic_grad_dims = []
            for pp in self._q_func.parameters():
                self._critic_grad_dims.append(pp.data.numel())
            self._critic_grads_cs = []
            self._critic_grads_da = torch.zeros(np.sum(self._critic_grad_dims)).to(self.device)

            self._actor_grad_dims = []
            for pp in self._policy.parameters():
                self._actor_grad_dims.append(pp.data.numel())
            self._actor_grads_cs = []
            self._actor_grads_da = torch.zeros(np.sum(self._actor_grad_dims)).to(self.device)
        elif self._replay_type == 'agem':
            self._critic_grad_dims = []
            for param in self._q_func.parameters():
                self._critic_grad_dims.append(param.data.numel())
            self._critic_grad_xy = torch.Tensor(np.sum(self._critic_grad_dims)).to(self.device)
            self._critic_grad_er = torch.Tensor(np.sum(self._critic_grad_dims)).to(self.device)
            self._actor_grad_dims = []
            for param in self._policy.parameters():
                self._actor_grad_dims.append(param.data.numel())
            self._actor_grad_xy = torch.Tensor(np.sum(self._actor_grad_dims)).to(self.device)
            self._actor_grad_er = torch.Tensor(np.sum(self._actor_grad_dims)).to(self.device)

    @train_api
    def update_critic(self, batch_tran: TransitionMiniBatch, replay_batches: Optional[Dict[int, List[torch.Tensor]]]=None, step=True):
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

        if replay_batches is not None and len(replay_batches) != 0 and self._replay_type == 'gem':
            for i, replay_batch in replay_batches.items():
                replay_batch = dict(zip(replay_name, replay_batch))
                replay_batch = Struct(**replay_batch)
                replay_batch = cast(TorchMiniBatch, replay_batch)
                q_tpn = self.compute_target(replay_batch)
                replay_loss_ = self.compute_critic_loss(replay_batch, q_tpn) / len(replay_batches)
                replay_loss = replay_loss_
                replay_loss.backward()
                store_grad(self._q_func.parameters, self._critic_grads_cs[i], self._critic_grad_dims)
        elif replay_batches is not None and len(replay_batches) != 0 and self._replay_type == 'r_walk':
            curr_feat_ext = {n: p.clone().detach() for n, p in self._q_func.named_parameters() if p.requires_grad}

        self._critic_optim.zero_grad()

        q_tpn = self.compute_target(batch)

        loss = self.compute_critic_loss(batch, q_tpn)
        unreg_grads = None
        curr_feat_ext = None
        replay_loss = 0
        replay_losses = []
        if replay_batches is not None and len(replay_batches) != 0:
            for i, replay_batch in replay_batches.items():
                replay_batch = dict(zip(replay_name, replay_batch))
                replay_batch = Struct(**replay_batch)
                replay_loss = 0
                if self._replay_type == "orl":
                    replay_batch.n_steps = 1
                    replay_batch.masks = None
                    replay_batch = cast(TorchMiniBatch, replay_batch)
                    q_tpn = self.compute_target(replay_batch)
                    replay_cql_loss = self.compute_critic_loss(replay_batch, q_tpn)
                    replay_losses.append(replay_cql_loss.cpu().detach().numpy())
                    replay_loss += replay_cql_loss
                elif self._replay_type == "bc":
                    with torch.no_grad():
                        replay_observations = replay_batch.observations.to(self.device)
                        replay_qs = replay_batch.qs.to(self.device)
                    replay_actions = self._policy(replay_observations)
                    q = self._q_func(replay_observations, replay_actions)
                    replay_bc_loss = F.mse_loss(replay_qs, q) / len(replay_batches)
                    replay_losses.append(replay_bc_loss.cpu().detach().numpy())
                    replay_loss += replay_bc_loss
                elif self._replay_type == "ewc":
                    replay_loss = 0
                    for n, p in self._q_func.named_parameters():
                        if n in self._critic_fisher.keys():
                            replay_loss += torch.sum(self._critic_fisher[n] * (p - self._critic_older_params[n]).pow(2)) / 2
                elif self._replay_type == 'r_walk':
                    # store gradients without regularization term
                    unreg_grads = {n: p.grad.clone().detach() for n, p in self._q_func.named_parameters()
                                   if p.grad is not None}

                    self._critic_optim.zero_grad()
                    replay_loss = 0
                    # Eq. 3: elastic weight consolidation quadratic penalty
                    for n, p in self._q_func.named_parameters():
                        if n in self._critic_fisher.keys():
                            replay_loss += torch.sum((self._critic_fisher[n] + self._critic_scores[n]) * (p - self._critic_older_params[n]).pow(2)) / 2
                elif self._replay_type == "agem":
                    store_grad(self._q_func.parameters, self._critic_grad_xy, self._critic_grad_dims)
                    replay_batch.n_steps = 1
                    replay_batch.masks = None
                    replay_batch = cast(TorchMiniBatch, replay_batch)
                    q_tpn = self.compute_target(replay_batch)
                    replay_cql_loss = self.compute_critic_loss(replay_batch, q_tpn)
                    replay_losses.append(replay_cql_loss.cpu().detach().numpy())
                    replay_loss += replay_cql_loss

            if self._replay_type in ['orl', 'bc', 'ewc', 'lamb']:
                loss += self._replay_critic_alpha * replay_loss
                replay_loss = replay_loss.cpu().detach().numpy()

        loss.backward()
        if self._replay_type == 'agem':
            replay_loss.backward()
            store_grad(self._q_func.parameters, self._critic_grad_er, self._critic_grad_dims)
            dot_prod = torch.dot(self._critic_grad_xy, self._critic_grad_er)
            if dot_prod.item() < 0:
                g_tilde = project(gxy=self._critic_grad_xy, ger=self._critic_grad_er)
                overwrite_grad(self._q_func.parameters, g_tilde, self._critic_grad_dims)
            else:
                overwrite_grad(self._q_func.parameters, self._critic_grad_xy, self._critic_grad_dims)
        elif self._replay_type == 'gem':
            # copy gradient
            store_grad(self._q_func.parameters, self._critic_grads_da, self._critic_grad_dims)
            dot_prod = torch.mm(self._critic_grads_da.unsqueeze(0),
                            torch.stack(self._critic_grads_cs).T)
            if (dot_prod < 0).sum() != 0:
                project2cone2(self._critic_grads_da.unsqueeze(1),
                              torch.stack(self._critic_grads_cs).T, margin=self._gem_gamma)
                # copy gradients back
                overwrite_grad(self._q_func.parameters, self._critic_grads_da,
                               self._critic_grad_dims)
        if step:
            self._critic_optim.step()

        if self._replay_type == 'r_walk':
            assert unreg_grads is not None
            assert curr_feat_ext is not None
            with torch.no_grad():
                for n, p in self._q_func.named_parameters():
                    if n in unreg_grads.keys():
                        self._critic_w[n] -= unreg_grads[n] * (p.detach() - curr_feat_ext[n])

        loss = loss.cpu().detach().numpy()

        return loss

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
        print(batch.actions[:, :self._action_size])
        print(batch.actions[:, :self._action_size].shape)
        assert False
        conservative_loss = self._compute_conservative_loss(
            batch.observations, batch.actions[:, :self._action_size], batch.next_observations
        )
        return loss + conservative_loss

    @train_api
    def update_actor(self, batch_tran: TransitionMiniBatch, replay_batches: Optional[Dict[int, List[torch.Tensor]]]=None, step=True) -> np.ndarray:
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

        if replay_batches is not None and len(replay_batches) != 0 and self._replay_type == 'gem':
            for i, replay_batch in replay_batches.items():
                replay_batch = dict(zip(replay_name, replay_batch))
                replay_batch = Struct(**replay_batch)
                replay_batch = cast(TorchMiniBatch, replay_batch)
                replay_loss_ = self.compute_actor_loss(replay_batch) / len(replay_batches)
                replay_loss = replay_loss_
                replay_loss.backward()
                store_grad(self._policy.parameters, self._actor_grads_cs[i], self._actor_grad_dims)
        elif replay_batches is not None and len(replay_batches) != 0 and self._replay_type == 'r_walk':
            curr_feat_ext = {n: p.clone().detach() for n, p in self._policy.named_parameters() if p.requires_grad}

        self._actor_optim.zero_grad()

        loss = self.compute_actor_loss(batch)
        unreg_grads = None
        curr_feat_ext = None
        replay_loss = 0
        replay_losses = []
        if replay_batches is not None and len(replay_batches) != 0:
            for i, replay_batch in replay_batches.items():
                replay_batch = dict(zip(replay_name, replay_batch))
                replay_batch = Struct(**replay_batch)
                if self._replay_type == "orl":
                    replay_batch = cast(TorchMiniBatch, replay_batch)
                    replay_loss_ = self.compute_actor_loss(replay_batch) / len(replay_batches)
                    replay_loss += replay_loss_
                    replay_losses.append(replay_loss_.cpu().detach().numpy())
                if self._replay_type == "bc":
                    with torch.no_grad():
                        replay_observations = replay_batch.observations.to(self.device)
                        replay_means = replay_batch.means.to(self.device)
                        replay_std_logs = replay_batch.std_logs.to(self.device)
                        replay_actions = torch.distributions.normal.Normal(replay_means, torch.exp(replay_std_logs))
                    actions = self._policy.dist(replay_observations)
                    replay_loss_ = torch.distributions.kl.kl_divergence(actions, replay_actions).mean() / len(replay_batches)
                    replay_losses.append(replay_loss.cpu().detach().numpy())
                    replay_loss = replay_loss_
                if self._replay_type == "ewc":
                    replay_loss = 0
                    for n, p in self._q_func.named_parameters():
                        if n in self._critic_fisher.keys():
                            replay_loss += torch.sum(self._critic_fisher[n] * (p - self._critic_older_params[n]).pow(2)) / 2
                elif self._replay_type == 'r_walk':
                    # store gradients without regularization term
                    unreg_grads = {n: p.grad.clone().detach() for n, p in self._policy.named_parameters()
                                   if p.grad is not None}

                    self._actor_optim.zero_grad()
                    replay_loss = 0
                    # Eq. 3: elastic weight consolidation quadratic penalty
                    for n, p in self._policy.named_parameters():
                        if n in self._actor_fisher.keys():
                            replay_loss += torch.sum((self._actor_fisher[n] + self._actor_scores[n]) * (p - self._actor_older_params[n]).pow(2)) / 2
                if self._replay_type == "agem":
                    store_grad(self._policy.parameters, self._actor_grad_xy, self._actor_grad_dims)
                    replay_batch = cast(TorchMiniBatch, replay_batch)
                    replay_loss_ = self.compute_actor_loss(replay_batch) / len(replay_batches)
                    replay_loss += replay_loss_
                    replay_losses.append(replay_loss_.cpu().detach().numpy())

            if self._replay_type in ['orl', 'bc', 'ewc', 'lamb']:
                loss += self._replay_actor_alpha * replay_loss
                replay_loss = replay_loss.cpu().detach().numpy()

        loss.backward()
        if self._replay_type == 'agem':
            replay_loss.backward()
            store_grad(self._policy.parameters, self._actor_grad_er, self._actor_grad_dims)
            dot_prod = torch.dot(self._actor_grad_xy, self._actor_grad_er)
            if dot_prod.item() < 0:
                g_tilde = project(gxy=self._actor_grad_xy, ger=self._actor_grad_er)
                overwrite_grad(self._policy.parameters, g_tilde, self._actor_grad_dims)
            else:
                overwrite_grad(self._policy.parameters, self._actor_grad_xy, self._actor_grad_dims)
        elif self._replay_type == 'gem':
            # copy gradient
            store_grad(self._policy.parameters, self._actor_grads_da, self._actor_grad_dims)
            dot_prod = torch.mm(self._actor_grads_da.unsqueeze(0),
                            torch.stack(self._actor_grads_cs).T)
            if (dot_prod < 0).sum() != 0:
                project2cone2(self._actor_grads_da.unsqueeze(1),
                              torch.stack(self._actor_grads_cs).T, margin=self._gem_gamma)
                # copy gradients back
                overwrite_grad(self._policy.parameters, self._actor_grads_da,
                               self._actor_grad_dims)
        if step:
            self._actor_optim.step()

        if self._replay_type == 'r_walk':
            assert unreg_grads is not None
            assert curr_feat_ext is not None
            with torch.no_grad():
                for n, p in self._policy.named_parameters():
                    if n in unreg_grads.keys():
                        self._actor_w[n] -= unreg_grads[n] * (p.detach() - curr_feat_ext[n])

        loss = loss.cpu().detach().numpy()

        return loss

    @train_api
    @torch_api()
    def update_alpha(self, batch: TorchMiniBatch) -> np.ndarray:
        assert self._alpha_optim is not None
        assert self._q_func is not None
        assert self._log_alpha is not None

        # Q function should be inference mode for stability
        self._q_func.eval()

        self._alpha_optim.zero_grad()

        # the original implementation does scale the loss value
        loss = -self._compute_conservative_loss(
            batch.observations, batch.actions[:, :self._action_size], batch.next_observations
        )

        loss.backward()
        self._alpha_optim.step()

        cur_alpha = self._log_alpha().exp().cpu().detach().numpy()[0][0]

        return loss.cpu().detach().numpy(), cur_alpha

    @train_api
    @torch_api()
    def update_phi(self, batch: TorchMiniBatch):
        assert self._phi_optim is not None
        self._phi.train()
        self._psi.eval()
        self._q_func.eval()
        self._policy.eval()

        self._phi_optim.zero_grad()

        loss, diff_phi, diff_r, diff_kl, diff_psi = self.compute_phi_loss(batch)

        loss.backward()
        self._phi_optim.step()

        return loss.cpu().detach().numpy()

    def compute_phi_loss(self, batch: TorchMiniBatch):
        assert self._phi is not None
        assert self._psi is not None
        assert self._policy is not None
        assert self._q_func is not None
        s, a, r, sp = batch.observations.to(self.device), batch.actions[:, :self.action_size].to(self.device), batch.rewards.to(self.device), batch.next_observations.to(self.device)
        ap = self._policy(sp)
        half_size = batch.observations.shape[0] // 2
        end_size = 2 * half_size
        phi = self._phi(s, a[:, :end_size])
        psi = self._psi(sp)
        diff_phi = torch.linalg.vector_norm(phi[:half_size] - phi[half_size:end_size], dim=1).mean()

        q = self._q_func(s, a)
        qp = self._q_func(sp, ap)
        r = q - self._gamma * qp
        diff_r = torch.abs(r[:half_size] - r[half_size:end_size]).mean()
        diff_kl = torch.distributions.kl.kl_divergence(self._policy.dist(s[:half_size]), self._policy.dist(s[half_size:end_size])).mean()
        diff_psi = self._gamma * torch.linalg.vector_norm(psi[:half_size] - psi[half_size:end_size], dim=1).mean()
        loss_phi = diff_phi + diff_r + diff_kl + diff_psi
        return loss_phi

    @train_api
    @torch_api()
    def update_psi(self, batch: TorchMiniBatch, pretrain=False):
        assert self._psi_optim is not None
        self._phi.eval()
        self._psi.train()
        self._q_func.eval()
        self._policy.eval()

        self._psi_optim.zero_grad()

        loss, loss_psi_diff, loss_psi_u = self.compute_psi_loss(batch, pretrain)
        loss.backward()
        self._psi_optim.step()

        return loss.cpu().detach().numpy()

    def compute_psi_loss(self, batch: TorchMiniBatch, pretrain: bool = False):
        assert self._phi is not None
        assert self._psi is not None
        assert self._policy is not None
        s, a = batch.observations.to(self.device), batch.actions.to(self.device)
        half_size = batch.observations.shape[0] // 2
        end_size = 2 * half_size
        psi = self._psi(s[:, :end_size])
        loss_psi_diff = torch.linalg.vector_norm(psi[:half_size] - psi[half_size:end_size], dim=1).mean()
        action = self._policy.dist(s)
        action1 = torch.distributions.normal.Normal(action.mean[:half_size], action.stddev[:half_size])
        action2 = torch.distributions.normal.Normal(action.mean[half_size:end_size], action.stddev[half_size:end_size])
        with torch.no_grad():
            u, _ = self._policy.sample_with_log_prob(s)
            phi = self._phi(s, u)
            loss_psi_u = torch.linalg.vector_norm(phi[:half_size] - phi[half_size:end_size], dim=1).mean()
        loss_psi = loss_psi_diff - loss_psi_u
        loss_psi_diff = loss_psi_diff.cpu().detach().numpy()
        loss_psi_u = loss_psi_u.cpu().detach().numpy()
        return loss_psi

    def update_phi_target(self) -> None:
        assert self._phi is not None
        assert self._targ_phi is not None
        soft_sync(self._targ_phi, self._phi, self._tau)

    def update_psi_target(self) -> None:
        assert self._psi is not None
        assert self._targ_psi is not None
        soft_sync(self._targ_psi, self._psi, self._tau)

    def compute_fisher_matrix_diag(self, iterator, network, optimizer, update):
        # Store Fisher Information
        fisher = {n: torch.zeros(p.shape).to(self.device) for n, p in network.named_parameters()
                  if p.requires_grad}
        # Do forward and backward pass to compute the fisher information
        network.train()
        replay_loss = 0
        iterator.reset()
        for t in range(len(iterator)):
            batch = next(iterator)
            update(batch)
            # Accumulate all gradients from loss with regularization
            for n, p in network.named_parameters():
                if p.grad is not None:
                    fisher[n] += p.grad.pow(2)
        # Apply mean across all samples
        fisher = {n: (p / len(iterator)) for n, p in fisher.items()}
        return fisher

    def agem_post_train_process(self, iterator):

        # calculate Fisher information
        curr_fisher = self.compute_fisher_matrix_diag(iterator, self._q_func, self._critic_optim, update=self.update_critic)
        # merge fisher information, we do not want to keep fisher information for each task in memory
        for n in self._critic_fisher.keys():
            # Added option to accumulate fisher over time with a pre-fixed growing alpha
            self._critic_fisher[n] = (self._agem_alpha * self._critic_fisher[n] + (1 - self._agem_alpha) * curr_fisher[n])

        # calculate Fisher information
        curr_fisher = self.compute_fisher_matrix_diag(iterator, self._policy, self._actor_optim, update=self.update_actor)
        # merge fisher information, we do not want to keep fisher information for each task in memory
        for n in self._actor_fisher.keys():
            # Added option to accumulate fisher over time with a pre-fixed growing alpha
            self._actor_fisher[n] = (self._agem_alpha * self._actor_fisher[n] + (1 - self._agem_alpha) * curr_fisher[n])

    def gem_post_train_process(self):
        assert self._q_func is not None
        assert self._policy is not None
        # copy gradient
        store_grad(self._q_func.parameters, self._critic_grads_da, self._critic_grad_dims)
        dot_prod = torch.mm(self._critic_grads_da.unsqueeze(0),
                        torch.stack(self._critic_grads_cs).T)
        if (dot_prod < 0).sum() != 0:
            project2cone2(self._critic_grads_da.unsqueeze(1),
                          torch.stack(self._critic_grads_cs).T, margin=self._gem_gamma)
            # copy gradients back
            overwrite_grad(self._q_func.parameters, self._critic_grads_da,
                           self._critic_grad_dims)
        # copy gradient
        store_grad(self._policy.parameters, self._actor_grads_da, self._actor_grad_dims)
        dot_prod = torch.mm(self._actor_grads_da.unsqueeze(0),
                        torch.stack(self._actor_grads_cs).T)
        if (dot_prod < 0).sum() != 0:
            project2cone2(self._actor_grads_da.unsqueeze(1),
                          torch.stack(self._actor_grads_cs).T, margin=self._gem_gamma)
            # copy gradients back
            overwrite_grad(self._policy.parameters, self._actor_grads_da,
                           self._actor_grad_dims)

    def copy_weight(self):
        state_dicts = [self._q_func.state_dict(), self._policy.state_dict(), self._log_alpha.state_dict(), self._log_temp.state_dict(), self._phi.state_dict(), self._psi.state_dict(), self._critic_optim.state_dict(), self._actor_optim.state_dict(), self._alpha_optim.state_dict(), self._temp_optim.state_dict(), self._phi_optim.state_dict(), self._psi_optim.state_dict()]
        return state_dicts

    def reload_weight(self, state_dicts):
        self._q_func.load_state_dict(state_dicts[0])
        self._policy.load_state_dict(state_dicts[1])
        self._log_alpha.load_state_dict(state_dicts[2])
        self._log_temp.load_state_dict(state_dicts[3])
        self._phi.load_state_dict(state_dicts[4])
        self._psi.load_state_dict(state_dicts[5])
        self._critic_optim.load_state_dict(state_dicts[6])
        self._actor_optim.load_state_dict(state_dicts[7])
        self._alpha_optim.load_state_dict(state_dicts[8])
        self._temp_optim.load_state_dict(state_dicts[9])
        self._phi_optim.load_state_dict(state_dicts[10])
        self._psi_optim.load_state_dict(state_dicts[11])
