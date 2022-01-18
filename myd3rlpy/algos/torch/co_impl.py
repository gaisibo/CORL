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

    def build_double(self):
        self._phi = create_phi(self._observation_shape, self._action_size, self._critic_encoder_factory)
        self._psi = create_psi(self._observation_shape, self._actor_encoder_factory)
        self._targ_phi = copy.deepcopy(self._phi)
        self._targ_psi = copy.deepcopy(self._psi)
        super().build()
        self._q_func_2 = self._q_func
        self._policy_2 = self._policy
        self._targ_q_func_2 = self._targ_q_func
        self._targ_policy_2 = self._targ_policy
        self._critic_optim_2 = self._critic_optim
        self._actor_optim_2 = self._actor_optim
        self._log_alpha_2 = self._log_alpha
        self._alpha_optim_2 = self._alpha_optim
        self._log_temp_2 = self._log_temp
        self._temp_optim_2 = self._temp_optim
        # _1和原名表示一种东西，方便其他文件使用。
        super().build()
        self._q_func_1 = self._q_func
        self._policy_1 = self._policy
        self._targ_q_func_1 = self._targ_q_func
        self._targ_policy_1 = self._targ_policy
        self._critic_optim_1 = self._critic_optim
        self._actor_optim_1 = self._actor_optim
        self._log_alpha_1 = self._log_alpha
        self._alpha_optim_1 = self._alpha_optim
        self._log_temp_1 = self._log_temp
        self._temp_optim_1 = self._temp_optim
        self._phi_optim = self._phi_optim_factory.create(
            self._phi.parameters(), lr=self._phi_learning_rate
        )
        self._psi_optim = self._psi_optim_factory.create(
            self._psi.parameters(), lr=self._psi_learning_rate
        )

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
    def update_critic_double(self, batch_1: TransitionMiniBatch, batch_2: TransitionMiniBatch, replay_batches_1: Optional[Dict[int, List[torch.Tensor]]]=None):
        self._q_func = self._q_func_2
        self._critic_optim = self._critic_optim_2
        self._policy = self._policy_2
        self._actor_optim = self._actor_optim_2
        self._log_temp = self._log_temp_2
        self._temp_optim = self._temp_optim_2
        self._log_alpha = self._log_alpha_2
        self._alpha_optim = self._alpha_optim_2
        loss_2, replay_loss_2, replay_losses_2 = self.update_critic(batch_2, None)

        self._q_func = self._q_func_1
        self._critic_optim = self._critic_optim_1
        self._policy = self._policy_1
        self._actor_optim = self._actor_optim_1
        self._log_temp = self._log_temp_1
        self._temp_optim = self._temp_optim_1
        self._log_alpha = self._log_alpha_1
        self._alpha_optim = self._alpha_optim_1
        loss_1, replay_loss_1, replay_losses_1 = self.update_critic(batch_1, replay_batches_1)
        return loss_1, replay_loss_1, replay_loss_1, loss_2, replay_loss_2, replay_losses_2

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
                    q_tpn = self.compute_target(replay_batch)
                    replay_cql_loss = self.compute_critic_loss(replay_batch, q_tpn)
                    replay_losses.append(replay_cql_loss.cpu().detach().numpy())
                    replay_loss += replay_cql_loss
                if self._q_bc_loss:
                    with torch.no_grad():
                        replay_observations = replay_batch.observations.to(self.device)
                        replay_actions = replay_batch.policy_actions.to(self.device)
                        replay_qs = replay_batch.qs.to(self.device)
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
        loss =  self._q_func.compute_error(
            observations=batch.observations,
            actions=batch.actions[:, :self._action_size],
            rewards=batch.rewards,
            target=q_tpn,
            terminals=batch.terminals,
            gamma=self._gamma ** batch.n_steps,
        )
        conservative_loss = self._compute_conservative_loss(
            batch.observations, batch.actions[:, :self._action_size], batch.next_observations
        )
        return loss + conservative_loss

    @train_api
    def update_actor_double(self, batch_1: TransitionMiniBatch, batch_2: TransitionMiniBatch, replay_batches_1: Optional[Dict[int, List[torch.Tensor]]]=None):
        self._q_func = self._q_func_2
        self._critic_optim = self._critic_optim_2
        self._policy = self._policy_2
        self._actor_optim = self._actor_optim_2
        self._log_temp = self._log_temp_2
        self._temp_optim = self._temp_optim_2
        self._log_alpha = self._log_alpha_2
        self._alpha_optim = self._alpha_optim_2
        loss_2, replay_loss_2, replay_losses_2 = self.update_actor(batch_2, None)

        self._q_func = self._q_func_1
        self._critic_optim = self._critic_optim_1
        self._policy = self._policy_1
        self._actor_optim = self._actor_optim_1
        self._log_temp = self._log_temp_1
        self._temp_optim = self._temp_optim_1
        self._log_alpha = self._log_alpha_1
        self._alpha_optim = self._alpha_optim_1
        loss_1, replay_loss_1, replay_losses_1 = self.update_actor(batch_1, replay_batches_1)
        return loss_1, replay_loss_1, replay_loss_1, loss_2, replay_loss_2, replay_losses_2

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
    def update_alpha_double(self, batch_1: TransitionMiniBatch, batch_2: TransitionMiniBatch):
        self._q_func = self._q_func_2
        self._critic_optim = self._critic_optim_2
        self._policy = self._policy_2
        self._actor_optim = self._actor_optim_2
        self._log_temp = self._log_temp_2
        self._temp_optim = self._temp_optim_2
        self._log_alpha = self._log_alpha_2
        self._alpha_optim = self._alpha_optim_2
        loss_2, curr_alpha_2 = self.update_alpha(batch_2)

        self._q_func = self._q_func_1
        self._critic_optim = self._critic_optim_1
        self._policy = self._policy_1
        self._actor_optim = self._actor_optim_1
        self._log_temp = self._log_temp_1
        self._temp_optim = self._temp_optim_1
        self._log_alpha = self._log_alpha_1
        self._alpha_optim = self._alpha_optim_1
        loss_1, curr_alpha_1 = self.update_alpha(batch_1)
        return loss_1, curr_alpha_1, loss_2, curr_alpha_2

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
    def update_temp_double(self, batch_1: TransitionMiniBatch, batch_2: TransitionMiniBatch):
        self._q_func = self._q_func_2
        self._critic_optim = self._critic_optim_2
        self._policy = self._policy_2
        self._actor_optim = self._actor_optim_2
        self._log_temp = self._log_temp_2
        self._temp_optim = self._temp_optim_2
        self._log_alpha = self._log_alpha_2
        self._alpha_optim = self._alpha_optim_2
        loss_2, curr_temp_2 = self.update_temp(batch_2)

        self._q_func = self._q_func_1
        self._critic_optim = self._critic_optim_1
        self._policy = self._policy_1
        self._actor_optim = self._actor_optim_1
        self._log_temp = self._log_temp_1
        self._temp_optim = self._temp_optim_1
        self._log_alpha = self._log_alpha_1
        self._alpha_optim = self._alpha_optim_1
        loss_1, curr_temp_1 = self.update_temp(batch_1)
        return loss_1, curr_temp_1, loss_2, curr_temp_2

    @train_api
    def update_phi_double(self, batch_1: TransitionMiniBatch, batch_2: TransitionMiniBatch):
        self._phi.train()
        self._psi.eval()
        self._q_func.eval()
        self._policy.eval()
        batch_1 = TorchMiniBatch(
            batch_1,
            self.device,
            scaler=self.scaler,
            action_scaler=self.action_scaler,
            reward_scaler=self.reward_scaler,
        )
        batch_2 = TorchMiniBatch(
            batch_2,
            self.device,
            scaler=self.scaler,
            action_scaler=self.action_scaler,
            reward_scaler=self.reward_scaler,
        )

        self._phi_optim.zero_grad()

        loss, diff_phi_1, diff_r_1, diff_psi_1, diff_phi_2, diff_r_2, diff_psi_2 = self.compute_phi_loss_double(batch_1, batch_2)

        loss.backward()
        self._phi_optim.step()

        return loss.cpu().detach().numpy(), diff_phi_1, diff_r_1, diff_psi_1, diff_phi_2, diff_r_2, diff_psi_2

    def compute_phi_loss_double(self, batch_1: TorchMiniBatch, batch_2: TorchMiniBatch):
        assert self._phi is not None
        assert self._psi is not None
        loss_phi_1, diff_phi_1, diff_r_1, diff_psi_1 = self.compute_phi_loss(batch_1)
        loss_phi_2, diff_phi_2, diff_r_2, diff_psi_2 = self.compute_phi_loss(batch_2)
        loss_phi = loss_phi_1 + loss_phi_2
        return loss_phi, diff_phi_1.cpu().detach().numpy(), diff_r_1.cpu().detach().numpy(), diff_psi_1.cpu().detach().numpy(), diff_phi_2.cpu().detach().numpy(), diff_r_2.cpu().detach().numpy(), diff_psi_2.cpu().detach().numpy()

    @train_api
    def update_phi(self, batch: TransitionMiniBatch):
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

        loss.backward()
        self._phi_optim.step()

        return loss.cpu().detach().numpy(), diff_phi, diff_r, diff_psi

    def compute_phi_loss(self, batch: TorchMiniBatch):
        assert self._phi is not None
        assert self._psi is not None
        s, a, r, sp = batch.observations.to(self.device), batch.actions[:, :self.action_size].to(self.device), batch.rewards.to(self.device), batch.next_observations.to(self.device)
        half_size = batch.observations.shape[0] // 2
        end_size = 2 * half_size
        phi = self._phi(s, a[:, :end_size])
        psi = self._psi(sp)
        diff_phi = torch.linalg.vector_norm(phi[:half_size] - phi[half_size:end_size], dim=1).mean()
        diff_r = torch.abs(r[:half_size] - r[half_size:end_size]).mean()
        diff_kl = torch.distributions.kl.kl_divergence(self._policy.dist(s[:half_size]), self._policy.dist(s[half_size:end_size]))
        diff_psi = self._gamma * torch.linalg.vector_norm(psi[:half_size] - psi[half_size:end_size], dim=1).mean()
        loss_phi = diff_phi + diff_r + diff_psi
        return loss_phi, diff_phi.cpu().detach().numpy(), diff_r.cpu().detach().numpy(), diff_psi.cpu().detach().numpy()

    @train_api
    def update_psi_double(self, batch_1: TransitionMiniBatch, batch_2: TransitionMiniBatch):
        assert self._psi_optim is not None
        self._phi.eval()
        self._psi.train()
        self._q_func.eval()
        self._policy.eval()
        batch_1 = TorchMiniBatch(
            batch_1,
            self.device,
            scaler=self.scaler,
            action_scaler=self.action_scaler,
            reward_scaler=self.reward_scaler,
        )
        batch_2 = TorchMiniBatch(
            batch_2,
            self.device,
            scaler=self.scaler,
            action_scaler=self.action_scaler,
            reward_scaler=self.reward_scaler,
        )

        self._psi_optim.zero_grad()

        loss_1, loss_psi_diff_1, loss_psi_u_1 = self.compute_psi_loss_double(batch_1)
        loss_2, loss_psi_diff_2, loss_psi_u_2 = self.compute_psi_loss_double(batch_2)
        loss = loss_1 + loss_2
        loss.backward()
        self._psi_optim.step()

        return loss.cpu().detach().numpy(), loss_psi_diff_1, loss_psi_u_1, loss_psi_diff_2, loss_psi_u_2

    def compute_psi_loss_double(self, batch: TorchMiniBatch, pretrain: bool = False):
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
        return loss_psi, loss_psi_diff, loss_psi_u

    @train_api
    def update_psi(self, batch: TransitionMiniBatch, pretrain=False):
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
        loss.backward()
        self._psi_optim.step()

        return loss.cpu().detach().numpy(), loss_psi_diff, loss_psi_kl, loss_psi_u

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
        loss_psi_kl = torch.distributions.kl.kl_divergence(action1, action2).mean()
        with torch.no_grad():
            u, _ = self._policy.sample_with_log_prob(s)
            phi = self._phi(s, u)
            loss_psi_u = torch.linalg.vector_norm(phi[:half_size] - phi[half_size:end_size], dim=1).mean()
        loss_psi = loss_psi_diff + loss_psi_kl - loss_psi_u
        loss_psi_diff = loss_psi_diff.cpu().detach().numpy()
        loss_psi_kl = loss_psi_kl.cpu().detach().numpy()
        loss_psi_u = loss_psi_u.cpu().detach().numpy()
        return loss_psi, loss_psi_diff, loss_psi_kl, loss_psi_u

    def update_critic_target_double(self) -> None:
        assert self._q_func_1 is not None
        assert self._targ_q_func_1 is not None
        assert self._q_func_2 is not None
        assert self._targ_q_func_2 is not None
        soft_sync(self._targ_q_func_1, self._q_func_1, self._tau)
        soft_sync(self._targ_q_func_2, self._q_func_2, self._tau)

    def update_actor_target_double(self) -> None:
        assert self._policy_1 is not None
        assert self._targ_policy_1 is not None
        assert self._policy_2 is not None
        assert self._targ_policy_2 is not None
        soft_sync(self._targ_policy_1, self._policy_1, self._tau)
        soft_sync(self._targ_policy_2, self._policy_2, self._tau)

    def update_phi_target(self) -> None:
        assert self._phi is not None
        assert self._targ_phi is not None
        soft_sync(self._targ_phi, self._phi, self._tau)

    def update_psi_target(self) -> None:
        assert self._psi is not None
        assert self._targ_psi is not None
        soft_sync(self._targ_psi, self._psi, self._tau)

    def copy_weight_double(self):
        state_dicts = [self._q_func_1.state_dict(), self._q_func_2.state_dict(), self._policy_1.state_dict(), self._policy_2.state_dict(), self._log_alpha_1.state_dict(), self._log_alpha_2.state_dict(), self._log_temp_1.state_dict(), self._log_temp_2.state_dict(), self._phi.state_dict(), self._psi.state_dict(), self._critic_optim_1.state_dict(), self._critic_optim_2.state_dict(), self._actor_optim_1.state_dict(), self._actor_optim_2.state_dict(), self._alpha_optim_1.state_dict(), self._alpha_optim_2.state_dict(), self._temp_optim_1.state_dict(), self._temp_optim_2.state_dict(), self._phi_optim.state_dict(), self._psi_optim.state_dict()]
        return state_dicts

    def reload_weight_double(self, state_dicts):
        self._q_func_1.load_state_dict(state_dicts[0])
        self._q_func_2.load_state_dict(state_dicts[1])
        self._policy_1.load_state_dict(state_dicts[2])
        self._policy_2.load_state_dict(state_dicts[3])
        self._log_alpha_1.load_state_dict(state_dicts[4])
        self._log_alpha_2.load_state_dict(state_dicts[5])
        self._log_temp_1.load_state_dict(state_dicts[6])
        self._log_temp_2.load_state_dict(state_dicts[7])
        self._phi.load_state_dict(state_dicts[8])
        self._psi.load_state_dict(state_dicts[9])
        self._critic_optim_1.load_state_dict(state_dicts[10])
        self._critic_optim_2.load_state_dict(state_dicts[11])
        self._actor_optim_1.load_state_dict(state_dicts[12])
        self._actor_optim_2.load_state_dict(state_dicts[13])
        self._alpha_optim_1.load_state_dict(state_dicts[14])
        self._alpha_optim_2.load_state_dict(state_dicts[15])
        self._temp_optim_1.load_state_dict(state_dicts[16])
        self._temp_optim_2.load_state_dict(state_dicts[17])
        self._phi_optim.load_state_dict(state_dicts[18])
        self._psi_optim.load_state_dict(state_dicts[19])

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
