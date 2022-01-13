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
from d3rlpy.algos.torch.combo_impl import COMBOImpl

from d3rlpy.models.builders import create_squashed_normal_policy, create_continuous_q_function, create_parameter
from myd3rlpy.siamese_similar import similar_euclid, similar_psi, similar_phi
from utils.utils import Struct


class COMBImpl(COMBOImpl):
    def __init__(
        self,
        observation_shape: Sequence[int],
        action_size: int,
        actor_learning_rate: float,
        critic_learning_rate: float,
        temp_learning_rate: float,
        alpha_learning_rate: float,
        actor_optim_factory: OptimizerFactory,
        critic_optim_factory: OptimizerFactory,
        temp_optim_factory: OptimizerFactory,
        alpha_optim_factory: OptimizerFactory,
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
        initial_temperature: float,
        initial_alpha: float,
        alpha_threshold: float,
        conservative_weight: float,
        n_action_samples: int,
        real_ratio: float,
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

        # initialized in build

    def print_q_func(self, batch: TransitionMiniBatch):

        batch = TorchMiniBatch(
            batch,
            self.device,
            scaler=self.scaler,
            action_scaler=self.action_scaler,
            reward_scaler=self.reward_scaler,
        )

        print('end update')
        fake_obs_t = batch.observations[int(batch.observations.shape[0] * self._real_ratio) :]
        fake_act_t = batch.actions[int(batch.actions.shape[0] * self._real_ratio):, :self._action_size]
        real_obs_t = batch.observations[: int(batch.observations.shape[0] * self._real_ratio)]
        real_act_t = batch.actions[: int(batch.actions.shape[0] * self._real_ratio), :self._action_size]
        print(f'q_fake: {self._q_func(fake_obs_t, fake_act_t)}')
        print(f'q_real: {self._q_func(real_obs_t, real_act_t)}')

    @train_api
    def update_critic(self, batch: TransitionMiniBatch, replay_batches: Optional[Dict[int, List[torch.Tensor]]]=None) -> np.ndarray:
        batch = TorchMiniBatch(
            batch,
            self.device,
            scaler=self.scaler,
            action_scaler=self.action_scaler,
            reward_scaler=self.reward_scaler,
        )
        assert self._critic_optim is not None
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

    def _combo_compute_conservative_loss(
        self, obs_t: torch.Tensor, act_t: torch.Tensor, obs_tp1: torch.Tensor
    ) -> torch.Tensor:
        assert self._policy is not None
        assert self._q_func is not None
        assert self._log_alpha is not None

        # split batch
        fake_obs_t = obs_t[int(obs_t.shape[0] * self._real_ratio) :]
        fake_obs_tp1 = obs_tp1[int(obs_tp1.shape[0] * self._real_ratio) :]
        real_obs_t = obs_t[: int(obs_t.shape[0] * self._real_ratio)]
        real_act_t = act_t[: int(act_t.shape[0] * self._real_ratio), :self._action_size]

        # compute conservative loss only with generated transitions
        random_values = self._compute_random_is_values(fake_obs_t)
        policy_values_t = self._compute_policy_is_values(fake_obs_t, fake_obs_t)
        policy_values_tp1 = self._compute_policy_is_values(
            fake_obs_tp1, fake_obs_t
        )

        # compute logsumexp
        # (n critics, batch, 3 * n samples) -> (n critics, batch, 1)
        target_values = torch.cat(
            [policy_values_t, policy_values_tp1, random_values], dim=2
        )
        logsumexp = torch.logsumexp(target_values, dim=2, keepdim=True)

        # estimate action-values for real data actions
        data_values = self._q_func(real_obs_t, real_act_t, "none")

        loss = logsumexp.sum(dim=0).mean() - data_values.sum(dim=0).mean()
        scaled_loss = self._conservative_weight * loss

        clipped_alpha = self._log_alpha().exp().clamp(0, 1e6)[0][0]

        return clipped_alpha * (scaled_loss - self._alpha_threshold)

    def _compute_conservative_loss(
        self, obs_t: torch.Tensor, act_t: torch.Tensor, obs_tp1: torch.Tensor
    ) -> torch.Tensor:
        assert self._policy is not None
        assert self._q_func is not None
        assert self._log_alpha is not None

        policy_values_t = self._compute_policy_is_values(obs_t, obs_t)
        policy_values_tp1 = self._compute_policy_is_values(obs_tp1, obs_t)
        random_values = self._compute_random_is_values(obs_t)

        # compute logsumexp
        # (n critics, batch, 3 * n samples) -> (n critics, batch, 1)
        target_values = torch.cat(
            [policy_values_t, policy_values_tp1, random_values], dim=2
        )
        logsumexp = torch.logsumexp(target_values, dim=2, keepdim=True)

        # estimate action-values for data actions
        data_values = self._q_func(obs_t, act_t[:, :self._action_size], "none")

        loss = logsumexp.mean(dim=0).mean() - data_values.mean(dim=0).mean()
        scaled_loss = self._conservative_weight * loss

        # clip for stability
        clipped_alpha = self._log_alpha().exp().clamp(0, 1e6)[0][0]

        return clipped_alpha * (scaled_loss - self._alpha_threshold)

    @train_api
    def update_actor(self, batch: TransitionMiniBatch, replay_batches: Optional[Dict[int, List[torch.Tensor]]]=None) -> np.ndarray:
        assert self._q_func is not None
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
