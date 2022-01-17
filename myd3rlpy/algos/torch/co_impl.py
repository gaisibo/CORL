import time
import math
import copy
from typing import Optional, Sequence, List, Any, Tuple, Dict, Union, cast

import numpy as np
import torch
from torch import nn
from torch.optim import Optimizer
import torch.nn.functional as F

from d3rlpy.gpu import Device
from d3rlpy.models.encoders import EncoderFactory
from d3rlpy.models.optimizers import OptimizerFactory
from d3rlpy.models.torch import Policy
from d3rlpy.models.q_functions import QFunctionFactory
from d3rlpy.preprocessing import ActionScaler, RewardScaler, Scaler
from d3rlpy.torch_utility import TorchMiniBatch, soft_sync, train_api, eval_api, torch_api, torch_api
from d3rlpy.dataset import MDPDataset, TransitionMiniBatch
from d3rlpy.algos.base import AlgoBase
from d3rlpy.algos.torch.base import TorchImplBase
from d3rlpy.algos.torch.cql_impl import CQLImpl

from d3rlpy.models.builders import create_squashed_normal_policy, create_continuous_q_function, create_parameter
from myd3rlpy.models.builders import create_phi, create_psi
from myd3rlpy.siamese_similar import similar_euclid, similar_psi, similar_phi
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
        cql_loss: bool,
        q_bc_loss: bool,
        td3_loss: bool,
        policy_bc_loss: bool,
        gamma: float,
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
        self._cql_loss = cql_loss
        self._q_bc_loss = q_bc_loss
        self._td3_loss = td3_loss
        self._policy_bc_loss = policy_bc_loss
        self._replay_actor_alpha = replay_actor_alpha
        self._replay_critic_alpha = replay_critic_alpha

        self._alpha_learning_rate = alpha_learning_rate
        self._alpha_optim_factory = alpha_optim_factory
        self._initial_alpha = initial_alpha
        self._alpha_threshold = alpha_threshold

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

    @train_api
    def update_critic(self, batch: TransitionMiniBatch, replay_batches: Optional[Dict[int, List[torch.Tensor]]]=None):
        batch = TorchMiniBatch(
            batch,
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

        self._critic_optim.zero_grad()

        q_tpn = self.compute_target(batch)

        loss = self.compute_critic_loss(batch, q_tpn)
        replay_loss = 0
        replay_losses = []
        if replay_batches is not None and len(replay_batches) != 0:
            for i, replay_batch in replay_batches.items():
                replay_batch = dict(zip(replay_name, replay_batch))
                replay_batch = Struct(**replay_batch)
                replay_loss = 0
                if self._cql_loss:
                    replay_batch.n_steps = 1
                    replay_batch.masks = None
                    q_tpn = self.compute_target(batch)
                    replay_cql_loss = self.compute_critic_loss(batch, q_tpn)
                    replay_losses.append(replay_cql_loss.cpu().detach().numpy())
                    replay_loss += replay_cql_loss
                if self._q_bc_loss:
                    with torch.no_grad():
                        replay_observations = replay_batch.observations.to(self.device)
                        replay_actions = replay_batch.policy_actions.to(self.device)
                        replay_qs = batch.qs.to(self.device)
                    q = self._q_func(replay_observations, replay_actions)
                    replay_bc_loss = F.mse_loss(replay_qs, q) / len(replay_batches)
                    replay_losses.append(replay_bc_loss.cpu().detach().numpy())
                    replay_loss += replay_bc_loss
            loss += self._replay_critic_alpha * replay_loss
            if self._cql_loss or self._q_bc_loss:
                replay_loss = replay_loss.cpu().detach().numpy()

        loss.backward()
        self._critic_optim.step()

        loss = loss.cpu().detach().numpy()

        return loss, replay_loss, replay_losses

    def compute_critic_loss(
        self, batch: TorchMiniBatch, q_tpn: torch.Tensor
    ) -> torch.Tensor:
        assert self._q_func is not None
        loss = self._q_func.compute_error(
            obs_t=batch.observations,
            act_t=batch.actions[:, :self._action_size],
            rew_tp1=batch.next_rewards,
            q_tp1=q_tpn,
            ter_tp1=batch.terminals,
            gamma=self._gamma ** batch.n_steps,
        )
        conservative_loss = self._compute_conservative_loss(
            batch.observations, batch.actions[:, :self._action_size], batch.next_observations
        )
        return loss + conservative_loss

    @train_api
    def update_actor(self, batch: TransitionMiniBatch, replay_batches: Optional[Dict[int, List[torch.Tensor]]]=None) -> np.ndarray:
        assert self._q_func is not None
        assert self._policy is not None
        assert self._actor_optim is not None
        batch = TorchMiniBatch(
            batch,
            self.device,
            scaler=self.scaler,
            action_scaler=self.action_scaler,
            reward_scaler=self.reward_scaler,
        )

        # Q function should be inference mode for stability
        self._q_func.eval()
        self._policy.train()

        self._actor_optim.zero_grad()

        loss = self.compute_actor_loss(batch)
        replay_loss = 0
        replay_losses = []
        if replay_batches is not None and len(replay_batches) != 0:
            for i, replay_batch in replay_batches.items():
                replay_batch_s = dict(zip(replay_name, replay_batch))
                replay_batch_s = Struct(**replay_batch_s)
                if self._td3_loss:
                    replay_loss_ = self.compute_actor_loss(replay_batch_s, all_data=None) / len(replay_batches)
                    replay_loss += replay_loss_
                    replay_losses.append(replay_loss_.cpu().detach().numpy())
                if self._policy_bc_loss:
                    with torch.no_grad():
                        replay_observations = replay_batch.observations.to(self.device)
                        replay_means = replay_batch.means.to(self.device)
                        replay_std_logs = replay_batch.std_logs.to(self.device)
                        replay_actions = torch.distributions.normal.Normal(replay_means, torch.exp(replay_std_logs))
                    actions = self._policy.dist(replay_observations)
                    replay_loss_ = torch.distributions.kl.kl_divergence(actions, replay_actions).mean() / len(replay_batches)
                    replay_losses.append(replay_loss.cpu().detach().numpy())
                    replay_loss = replay_loss_
            loss += self._replay_actor_alpha * replay_loss
            if self._td3_loss or self._policy_bc_loss:
                replay_loss = replay_loss.cpu().detach().numpy()

        loss.backward()
        self._actor_optim.step()

        loss = loss.cpu().detach().numpy()

        return loss, replay_loss, replay_losses

    @train_api
    def update_phi(self, batch: TransitionMiniBatch, replay_batches: Optional[Dict[int, List[torch.Tensor]]]=None):
        assert self._phi_optim is not None
        self._phi.train()
        self._psi.eval()
        self._q_func.eval()
        self._policy.eval()
        batch = TorchMiniBatch(
            batch,
            self.device,
            scaler=self.scaler,
            action_scaler=self.action_scaler,
            reward_scaler=self.reward_scaler,
        )

        self._phi_optim.zero_grad()

        loss, diff_phi, diff_r, diff_psi = self.compute_phi_loss(batch)
        policy_loss = loss.cpu().detach().numpy()
        replay_loss = 0
        if replay_batches is not None and len(replay_batches) != 0 and self._phi_bc_loss:
            for i, replay_batch in replay_batches.items():
                replay_batch_s = Struct(**dict(zip(replay_name, replay_batch)))
                # replay_loss = self.compute_phi_loss(replay_batch_s)
                with torch.no_grad():
                    replay_observations = replay_batch_s.observations.to(self.device)
                    replay_actions = replay_batch_s.policy_actions.to(self.device)
                    replay_phis = replay_batch_s.phis.to(self.device)
                rebuild_phis = self._phi(replay_observations, replay_actions)
                replay_bc_loss = F.mse_loss(replay_phis, rebuild_phis) / len(replay_batches)
                replay_loss += replay_bc_loss
            loss += self._replay_phi_alpha * replay_loss
            replay_loss = replay_loss.cpu().detach().numpy()

        loss.backward()
        self._phi_optim.step()

        return loss.cpu().detach().numpy(), policy_loss, diff_phi, diff_r, diff_psi, replay_loss

    def compute_phi_loss(self, batch: TorchMiniBatch):
        assert self._phi is not None
        assert self._psi is not None
        s, a, r, sp = batch.observations.to(self.device), batch.actions[:, :self.action_size].to(self.device), batch.rewards.to(self.device), batch.next_observations.to(self.device)
        half_size = batch.observations.shape[0] // 2
        phi = self._phi(s, a[:, :self._action_size])
        psi = self._psi(sp)
        diff_phi = torch.linalg.vector_norm(phi[:half_size] - phi[half_size:], dim=1).mean()
        diff_r = torch.abs(r[:half_size] - r[half_size:]).mean()
        diff_psi = self._gamma * torch.linalg.vector_norm(psi[:half_size] - psi[half_size:], dim=1).mean()
        loss_phi = diff_phi + diff_r + diff_psi
        return loss_phi, diff_phi.cpu().detach().numpy(), diff_r.cpu().detach().numpy(), diff_psi.cpu().detach().numpy()

    @train_api
    def update_psi(self, batch: TransitionMiniBatch, replay_batches: Optional[Dict[int, List[torch.Tensor]]]=None, pretrain=False):
        assert self._psi_optim is not None
        self._phi.eval()
        self._psi.train()
        self._q_func.eval()
        self._policy.eval()
        batch = TorchMiniBatch(
            batch,
            self.device,
            scaler=self.scaler,
            action_scaler=self.action_scaler,
            reward_scaler=self.reward_scaler,
        )

        self._psi_optim.zero_grad()

        loss, loss_psi_diff, loss_psi_kl, loss_psi_u = self.compute_psi_loss(batch, pretrain)
        policy_loss = loss.cpu().detach().numpy()
        replay_loss = 0
        if replay_batches is not None and len(replay_batches) != 0 and self._psi_bc_loss:
            for i, replay_batch in replay_batches.items():
                replay_batch_s = Struct(**dict(zip(replay_name, replay_batch)))
                with torch.no_grad():
                    replay_observations = replay_batch_s.observations.to(self.device)
                    replay_psis = replay_batch_s.psis.to(self.device)
                replay_loss += self._replay_psi_alpha * F.mse_loss(self._psi(replay_observations), replay_psis)
            loss += replay_loss
            replay_loss = replay_loss.cpu().detach().numpy()

        loss.backward()
        self._psi_optim.step()

        return loss.cpu().detach().numpy(), policy_loss, loss_psi_diff, loss_psi_kl, loss_psi_u, replay_loss

    def compute_psi_loss(self, batch: TorchMiniBatch, pretrain: bool = False):
        assert self._phi is not None
        assert self._psi is not None
        assert self._policy is not None
        s, a = batch.observations.to(self.device), batch.actions.to(self.device)
        half_size = batch.observations.shape[0] // 2
        psi = self._psi(s)
        loss_psi_diff = torch.linalg.vector_norm(psi[:half_size] - psi[half_size:], dim=1).mean()
        action = self._policy.dist(s)
        action1 = torch.distributions.normal.Normal(action.mean[:half_size], action.stddev[:half_size])
        action2 = torch.distributions.normal.Normal(action.mean[half_size:], action.stddev[half_size:])
        loss_psi_kl = torch.distributions.kl.kl_divergence(action1, action2).mean()
        with torch.no_grad():
            if not pretrain:
                u, _ = self._policy.sample_with_log_prob(s)
            else:
                u = torch.randn(a.shape[0], self._action_size).to(self.device)
            phi = self._phi(s, u)
            loss_psi_u = torch.linalg.vector_norm(phi[:half_size] - phi[half_size], dim=1).mean()
        loss_psi = loss_psi_diff + loss_psi_kl - loss_psi_u
        loss_psi_diff = loss_psi_diff.cpu().detach().numpy()
        loss_psi_kl = loss_psi_kl.cpu().detach().numpy()
        loss_psi_u = loss_psi_u.cpu().detach().numpy()
        return loss_psi, loss_psi_diff, loss_psi_kl, loss_psi_u

    def update_phi_target(self) -> None:
        assert self._phi is not None
        assert self._targ_phi is not None
        soft_sync(self._targ_phi, self._phi, self._tau)

    def update_psi_target(self) -> None:
        assert self._psi is not None
        assert self._targ_psi is not None
        soft_sync(self._targ_psi, self._psi, self._tau)

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
