
import time
import math
import copy
from typing import Optional, Sequence, List, Any, Tuple, Dict, Union

import numpy as np
import torch
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
from d3rlpy.algos.torch.td3_impl import TD3Impl

from myd3rlpy.models.torch.siamese import Phi, Psi
from d3rlpy.models.builders import create_deterministic_policy, create_continuous_q_function
from myd3rlpy.models.builders import create_phi, create_psi
from myd3rlpy.siamese_similar import similar_euclid, similar_psi, similar_phi
from utils.utils import Struct


class COImpl(TD3Impl):
    _phi_learning_rate: float
    _psi_learning_rate: float
    _phi_optim_factory: OptimizerFactory
    _psi_optim_factory: OptimizerFactory
    _phi_encoder_factory: EncoderFactory
    _psi_encoder_factory: EncoderFactory
    _phi_optim: Optional[Optimizer]
    _psi_optim: Optional[Optimizer]
    def __init__(
        self,
        observation_shape: Sequence[int],
        action_size: int,
        actor_learning_rate: float,
        critic_learning_rate: float,
        alpha: float,
        phi_learning_rate: float,
        psi_learning_rate: float,
        actor_optim_factory: OptimizerFactory,
        critic_optim_factory: OptimizerFactory,
        phi_optim_factory: OptimizerFactory,
        psi_optim_factory: OptimizerFactory,
        actor_encoder_factory: EncoderFactory,
        critic_encoder_factory: EncoderFactory,
        q_func_factory: QFunctionFactory,
        replay_actor_alpha: float,
        replay_critic_alpha: float,
        siamese_actor_alpha: float,
        siamese_critic_alpha: float,
        replay_phi_alpha: float,
        replay_psi_alpha: float,
        replay_critic: bool,
        use_phi_update: bool,
        use_same_encoder:bool,
        gamma: float,
        tau: float,
        n_critics: int,
        target_reduction_type: str,
        target_smoothing_sigma: float,
        target_smoothing_clip: float,
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
            actor_optim_factory = actor_optim_factory,
            critic_optim_factory = critic_optim_factory,
            actor_encoder_factory = actor_encoder_factory,
            critic_encoder_factory = critic_encoder_factory,
            q_func_factory = q_func_factory,
            gamma = gamma,
            tau = tau,
            n_critics = n_critics,
            target_reduction_type = target_reduction_type,
            target_smoothing_sigma = target_smoothing_sigma,
            target_smoothing_clip = target_smoothing_clip,
            use_gpu = use_gpu,
            scaler = scaler,
            action_scaler = action_scaler,
            reward_scaler = reward_scaler,
        )
        self._alpha = alpha
        self._phi_learning_rate = phi_learning_rate
        self._psi_learning_rate = psi_learning_rate
        self._phi_optim_factory = phi_optim_factory
        self._psi_optim_factory = psi_optim_factory
        self._replay_actor_alpha = replay_actor_alpha
        self._replay_critic_alpha = replay_critic_alpha
        self._siamese_actor_alpha = siamese_actor_alpha
        self._siamese_critic_alpha = siamese_critic_alpha
        self._temp_siamese_actor_alpha = siamese_actor_alpha
        self._temp_siamese_critic_alpha = siamese_critic_alpha
        self._replay_phi_alpha = replay_phi_alpha
        self._replay_psi_alpha = replay_psi_alpha
        self._replay_critic = replay_critic

        # initialized in build
        self._phi_optim = None
        self._psi_optim = None
        self._use_phi_update = use_phi_update
        self._use_same_encoder = use_same_encoder

        self.replay_name = ['observations', 'normal_actions', 'rewards', 'next_observations', 'next_actions', 'next_rewards', 'terminals', 'policy_actions', 'qs', 'phis', 'psis']
    def build(self) -> None:

        # 共用encoder
        # 在共用encoder的情况下replay_不起作用。
        self._q_func = create_continuous_q_function(
            observation_shape=self._observation_shape,
            action_size=self._action_size,
            encoder_factory=self._critic_encoder_factory,
            q_func_factory=self._q_func_factory,
            n_ensembles=self._n_critics,
        )
        self._policy = create_deterministic_policy(
            observation_shape=self._observation_shape,
            action_size=self._action_size,
            encoder_factory=self._actor_encoder_factory,
        )
        self._targ_q_func = copy.deepcopy(self._q_func)
        self._targ_policy = copy.deepcopy(self._policy)
        if self._use_same_encoder:
            self._phi = Phi([q_func.encoder for q_func in self._q_func.q_funcs])
            self._psi = Psi(self._policy._encoder)
        else:
            self._phi = create_phi(self._observation_shape, self._action_size, self._critic_encoder_factory)
            self._psi = create_psi(self._observation_shape, self._actor_encoder_factory)
        self._targ_phi = copy.deepcopy(self._phi)
        self._targ_psi = copy.deepcopy(self._psi)

        if self._use_gpu:
            self.to_gpu(self._use_gpu)
        else:
            self.to_cpu()

        # setup optimizer after the parameters move to GPU
        self._critic_optim = self._critic_optim_factory.create(
            self._q_func.parameters(), lr=self._critic_learning_rate
        )
        self._actor_optim = self._actor_optim_factory.create(
            self._policy.parameters(), lr=self._actor_learning_rate
        )
        self._phi_optim = self._phi_optim_factory.create(
            self._phi.parameters(), lr=self._phi_learning_rate
        )
        self._psi_optim = self._psi_optim_factory.create(
            self._psi.parameters(), lr=self._psi_learning_rate
        )

        self.update_alpha()

    def update_alpha(self):
        if self._use_phi_update:
            self._temp_siamese_critic_alpha = self._siamese_critic_alpha
            self._temp_siamese_actor_alpha = self._siamese_actor_alpha

    # def increase_siamese_alpha(self, epoch, itr):
    #     if self._use_phi_update:
    #         self._temp_siamese_critic_alpha = self._siamese_critic_alpha * math.exp(epoch) * (1 + itr)
    #         self._temp_siamese_actor_alpha = self._siamese_actor_alpha * math.exp(epoch) * (1 + itr)

    @train_api
    def update_critic(self, batch: TransitionMiniBatch, replay_batches: Optional[Dict[int, List[torch.Tensor]]]=None, all_data: MDPDataset=None) -> np.ndarray:
        batch = TorchMiniBatch(
            batch,
            self.device,
            scaler=self.scaler,
            action_scaler=self.action_scaler,
            reward_scaler=self.reward_scaler,
        )
        assert self._critic_optim is not None
        self._phi.eval()
        self._psi.eval()
        self._q_func.train()
        self._policy.eval()

        self._critic_optim.zero_grad()

        q_tpn, action = self.compute_target(batch)

        loss, q_func_loss, up_loss, down_loss, siamese_loss, smallest_distance = self.compute_critic_loss(batch, q_tpn, action, all_data=all_data)
        replay_loss = 0
        replay_losses = []
        replay_q_func_losses = []
        replay_up_losses = []
        replay_down_losses = []
        replay_siamese_losses = []
        replay_smallest_distances = []
        replay_bc_losses = []
        if replay_batches is not None and len(replay_batches) != 0:
            for i, replay_batch in replay_batches.items():
                replay_batch = dict(zip(self.replay_name, replay_batch))
                replay_batch = Struct(**replay_batch)
                replay_batch.n_steps = 1
                replay_batch.masks = None
                q_tpn, action = self.compute_target(batch)
                replay_loss, replay_q_func_loss, replay_up_loss, replay_down_loss, replay_siamese_loss, replay_smallest_distance = self.compute_critic_loss(batch, q_tpn, action, all_data=None)
                replay_losses.append(replay_loss)
                replay_q_func_losses.append(replay_q_func_loss)
                replay_up_losses.append(replay_up_loss)
                replay_down_losses.append(replay_down_loss)
                replay_siamese_losses.append(replay_siamese_loss)
                replay_smallest_distances.append(replay_smallest_distance)
                with torch.no_grad():
                    replay_observations = replay_batch.observations.to(self.device)
                    replay_actions = replay_batch.policy_actions.to(self.device)
                    replay_qs = batch.qs.to(self.device)
                q = self._q_func(replay_observations, replay_actions)
                replay_bc_loss = F.mse_loss(replay_qs, q) / len(replay_batches)
                replay_bc_losses.append(replay_bc_loss.cpu().detach().numpy())
                replay_loss += replay_bc_loss
            loss += self._replay_critic_alpha * replay_loss
            replay_loss = replay_loss.cpu().detach().numpy()

        loss.backward()
        self._critic_optim.step()

        loss = loss.cpu().detach().numpy()

        return loss, q_func_loss, up_loss, down_loss, siamese_loss, replay_loss, smallest_distance, replay_q_func_losses, replay_up_losses, replay_down_losses, replay_siamese_losses, replay_smallest_distances, replay_bc_losses

    def compute_critic_loss(self, batch: TransitionMiniBatch, q_tpn: torch.Tensor, action: torch.Tensor, all_data: Optional[MDPDataset]=None, beta=0.5) -> torch.Tensor:
        assert self._q_func is not None
        if self._use_phi_update and all_data is not None:
            with torch.no_grad():
                near_observations = all_data._observations[batch.next_actions[:, self._action_size:].to(torch.int64).cpu().numpy()]
                near_actions = all_data._actions[batch.next_actions[:, self._action_size:].to(torch.int64).cpu().numpy()]
                near_actions = near_actions[:, :, :self._action_size]
                _, _, smallest_distance = similar_phi(batch.next_observations, action, near_observations, near_actions, self._targ_phi)
                # b = torch.mean(q_tpn * torch.exp(- beta * smallest_distance))
                b = torch.mean(torch.exp(- beta * smallest_distance))
                # siamese_loss = self._temp_siamese_critic_alpha * b
                siamese_loss = self._siamese_critic_alpha * b
        else:
            siamese_loss = 0
        q_func_loss = self._q_func.compute_error(
            obs_t=batch.observations,
            act_t=batch.actions[:, :self.action_size],
            # input_indexes=batch.actions[:, self.action_size:],
            # 把siamese_loss插到这里了。
            rew_tp1=batch.next_rewards + siamese_loss,
            q_tp1=q_tpn,
            ter_tp1=batch.terminals,
            gamma=self._gamma ** batch.n_steps,
            use_independent_target=self._target_reduction_type == "none",
            masks=batch.masks,
        )
        up_loss = torch.mean(self._targ_q_func.compute_target(batch.observations, self._targ_policy(batch.observations), reduction=self._target_reduction_type), dim=0)
        down_loss = torch.mean(self._targ_q_func.compute_target(batch.observations, batch.actions[:, :self.action_size]), dim=0)
        loss = q_func_loss + up_loss - down_loss
        if self._use_phi_update and all_data is not None:
            return loss, q_func_loss.cpu().detach().numpy(), up_loss.cpu().detach().numpy(), down_loss.cpu().detach().numpy(), siamese_loss.cpu().detach().numpy(), smallest_distance.mean().cpu().detach().numpy()
        else:
            return loss, q_func_loss.cpu().detach().numpy(), up_loss.cpu().detach().numpy(), down_loss.cpu().detach().numpy(), 0, 0

    def compute_target(self, batch: TorchMiniBatch) -> torch.Tensor:
        action = self._targ_policy(batch.next_observations)
        # smoothing target
        noise = torch.randn(action.shape, device=batch.device)
        scaled_noise = self._target_smoothing_sigma * noise
        clipped_noise = scaled_noise.clamp(
            -self._target_smoothing_clip, self._target_smoothing_clip
        )
        smoothed_action = action + clipped_noise
        clipped_action = smoothed_action.clamp(-1.0, 1.0)
        return self._targ_q_func.compute_target(
            batch.next_observations,
            clipped_action,
            reduction=self._target_reduction_type,
        ), action
    @train_api
    def update_actor(self, batch: TransitionMiniBatch, replay_batches: Optional[Dict[int, List[torch.Tensor]]]=None, all_data: MDPDataset=None) -> np.ndarray:
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
        self._phi.eval()
        self._psi.eval()
        self._q_func.eval()
        self._policy.train()

        self._actor_optim.zero_grad()

        loss, policy_loss, bc_loss, siamese_loss, smallest_distance = self.compute_actor_loss(batch, all_data)
        replay_loss = 0
        replay_losses = []
        replay_policy_losses = []
        replay_siamese_losses = []
        if replay_batches is not None and len(replay_batches) != 0:
            for i, replay_batch in replay_batches.items():
                replay_batch_s = dict(zip(self.replay_name, replay_batch))
                replay_batch_s = Struct(**replay_batch_s)
                replay_loss, replay_policy_loss, replay_bc_loss, replay_siamese_loss, replay_smallest_distance = self.compute_actor_loss(replay_batch_s, all_data=None)
                replay_losses.append(replay_loss)
                replay_policy_losses.append(replay_policy_loss)
                replay_siamese_losses.append(replay_siamese_loss)
                with torch.no_grad():
                    replay_observations = replay_batch.observations.to(self.device)
                    replay_actions = replay_batch.policy_actions.to(self.device)
                actions = self._policy(replay_observations)
                replay_bc_loss = F.mse_loss(replay_actions, actions).mean() / len(replay_batches)
                replay_bc_losses.append(replay_bc_loss.cpu().detach().numpy())
                replay_loss += replay_bc_loss
            loss += self._replay_actor_alpha * replay_loss
            replay_loss = replay_loss.cpu().detach().numpy()

        loss.backward()
        self._actor_optim.step()

        loss = loss.cpu().detach().numpy()

        return loss, policy_loss, siamese_loss, replay_loss, smallest_distance, replay_policy_losses, replay_siamese_losses

    def compute_actor_loss(self, batch: TorchMiniBatch, all_data: MDPDataset=None, beta=0.5) -> torch.Tensor:
        assert self._policy is not None
        assert self._q_func is not None
        action = self._policy(batch.observations)
        q_t = self._q_func(batch.observations, action)[0]
        if self._use_phi_update and all_data is not None:
            with torch.no_grad():
                near_observations = all_data._observations[batch.actions[:, self._action_size:].to(torch.int64).cpu().numpy()]
                near_actions = all_data._actions[batch.actions[:, self._action_size:].to(torch.int64).cpu().numpy()]
                near_actions = near_actions[:, :, :self._action_size]
                _, _, smallest_distance = similar_phi(batch.observations, action, near_observations, near_actions, self._targ_phi)
                b = torch.mean(torch.exp(- beta * smallest_distance))
                # b = torch.mean(q_t * torch.exp(- beta * smallest_distance))
            lam = self._alpha / (q_t.abs().mean()).detach()
            lam = self._alpha / (q_t.abs().mean()).detach()
            bc_loss = ((batch.actions[:, :self._action_size] - action) ** 2).mean()
            # siamese_loss = - b * self._temp_siamese_actor_alpha
            siamese_loss = - b * self._siamese_actor_alpha
            loss = lam * -q_t.mean() + bc_loss
            return loss, -q_t.mean().cpu().detach().numpy(), bc_loss.cpu().detach().numpy(), siamese_loss.cpu().detach().numpy(), smallest_distance.mean().cpu().detach().numpy()
        else:
            lam = self._alpha / (q_t.abs().mean()).detach()
            bc_loss = ((batch.actions[:, :self._action_size] - action) ** 2).mean()
            loss = lam * -q_t.mean() + bc_loss
            return loss, -q_t.mean().cpu().detach().numpy(), bc_loss.cpu().detach().numpy(), 0, 0

    @train_api
    def update_phi(self, batch: TransitionMiniBatch, replay_batches: Optional[Dict[int, List[torch.Tensor]]]=None) -> np.ndarray:
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
        if replay_batches is not None and len(replay_batches) != 0:
            for i, replay_batch in replay_batches.items():
                replay_batch_s = Struct(**dict(zip(self.replay_name, replay_batch)))
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

    def compute_phi_loss(self, batch: TorchMiniBatch) -> torch.Tensor:
        assert self._phi is not None
        assert self._psi is not None
        s, a, r, sp = batch.observations, batch.actions[:, :self.action_size], batch.rewards, batch.next_observations
        half_size = batch.observations.shape[0] // 2
        phi = self._phi(s, a[:, :self._action_size])
        psi = self._psi(sp)
        diff_phi = torch.linalg.vector_norm(phi[:half_size] - phi[half_size:], dim=1)
        diff_r = torch.abs(r[:half_size] - r[half_size:])
        diff_psi = self._gamma * torch.linalg.vector_norm(psi[:half_size] - psi[half_size:], dim=1)
        loss_phi = diff_phi + diff_r + diff_psi
        loss_phi = torch.mean(loss_phi)
        return loss_phi, diff_phi.mean().cpu().detach().numpy(), diff_r.mean().cpu().detach().numpy(), diff_psi.mean().cpu().detach().numpy()

    @train_api
    def update_psi(self, batch: TransitionMiniBatch, replay_batches: Optional[Dict[int, List[torch.Tensor]]]=None, pretrain=False) -> np.ndarray:
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

        loss = self.compute_psi_loss(batch, pretrain)
        policy_loss = loss.cpu().detach().numpy()
        replay_loss = 0
        if replay_batches is not None and len(replay_batches) != 0:
            for i, replay_batch in replay_batches.items():
                replay_batch_s = Struct(**dict(zip(self.replay_name, replay_batch)))
                with torch.no_grad():
                    replay_observations = replay_batch_s.observations.to(self.device)
                    replay_psis = replay_batch_s.psis.to(self.device)
                replay_loss += self._replay_psi_alpha * F.mse_loss(self._psi(replay_observations), replay_psis)
            loss += replay_loss
            replay_loss = replay_loss.cpu().detach().numpy()

        loss.backward()
        self._psi_optim.step()

        return loss.cpu().detach().numpy(), policy_loss, replay_loss

    def compute_psi_loss(self, batch: TransitionMiniBatch, pretrain: bool = False) -> torch.Tensor:
        assert self._phi is not None
        assert self._psi is not None
        assert self._policy is not None
        s, a = batch.observations, batch.actions
        half_size = batch.observations.shape[0] // 2
        psi = self._psi(s)
        loss_psi = torch.linalg.vector_norm(psi[:half_size] - psi[half_size:], dim=1)
        with torch.no_grad():
            if not pretrain:
                u = self._policy(s)
            else:
                u = torch.randn(a.shape[0], self._action_size).to(self.device)
            loss_psi_u = 0
            phi = self._phi(s, u)
            loss_psi_u = torch.linalg.vector_norm(phi[:half_size] - phi[half_size:], dim=1)
        loss_psi = loss_psi - loss_psi_u
        return torch.mean(loss_psi)

    def update_phi_target(self) -> None:
        assert self._phi is not None
        assert self._targ_phi is not None
        soft_sync(self._targ_phi, self._phi, self._tau)

    def update_psi_target(self) -> None:
        assert self._psi is not None
        assert self._targ_psi is not None
        soft_sync(self._targ_psi, self._psi, self._tau)
