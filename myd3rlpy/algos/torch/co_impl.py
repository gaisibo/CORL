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
from d3rlpy.algos.torch.td3_plus_bc_impl import TD3Impl

from myd3rlpy.models.torch.siamese import Phi, Psi
from d3rlpy.models.builders import create_squashed_normal_policy, create_continuous_q_function
from utils.siamese_similar import similar_euclid, similar_psi, similar_phi

class COImpl(TD3Impl):
    _phi_learning_rate: float
    _psi_learning_rate: float
    _phi_optim_factory: OptimizerFactory
    _psi_optim_factory: OptimizerFactory
    _phi_encoder_factory: EncoderFactory
    _psi_encoder_factory: EncoderFactory
    _n_sample_actions: int
    _phi_optim: Optional[Optimizer]
    _psi_optim: Optional[Optimizer]
    def __init__(
        self,
        observation_shape: Sequence[int],
        action_size: int,
        actor_learning_rate: float,
        critic_learning_rate: float,
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
        replay_critic: bool,
        replay_phi: bool,
        replay_psi: bool,
        gamma: float,
        tau: float,
        n_critics: int,
        target_reduction_type: str,
        target_smoothing_sigma: float,
        target_smoothing_clip: float,
        n_sample_actions: int,
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
        self._phi_learning_rate = phi_learning_rate
        self._psi_learning_rate = psi_learning_rate
        self._phi_optim_factory = phi_optim_factory
        self._psi_optim_factory = psi_optim_factory
        self._replay_actor_alpha = replay_actor_alpha
        self._replay_critic_alpha = replay_critic_alpha
        self._replay_critic = replay_critic
        self._replay_phi = replay_phi
        self._replay_psi = replay_psi
        self._n_sample_actions = n_sample_actions

        # initialized in build
        self._phi_optim = None
        self._psi_optim = None
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
        self._policy = create_squashed_normal_policy(
            observation_shape=self._observation_shape,
            action_size=self._action_size,
            encoder_factory=self._actor_encoder_factory,
        )
        self._targ_q_func = copy.deepcopy(self._q_func)
        self._targ_policy = copy.deepcopy(self._policy)
        self._phi = Phi([q_func.encoder for q_func in self._q_func.q_funcs])
        self._psi = Psi(self._policy._encoder)


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

    @train_api
    def update_critic(self, batch: TransitionMiniBatch, replay_batches: Optional[Dict[int, List[torch.Tensor]]]=None, all_data: MDPDataset=None) -> np.ndarray:
        assert all_data is not None
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

        loss = self.compute_critic_loss(batch, q_tpn, all_data, action)
        if replay_batches is not None:
            for i, replay_batch in replay_batches.items():
                with torch.no_grad():
                    replay_observations, replay_actionss, _, _, replay_qss = replay_batch
                    replay_observations = replay_observations.to(self.device)
                    replay_actionss = replay_actionss.to(self.device)
                    replay_qss = replay_qss.to(self.device)
                for action_sample_num in range(replay_actionss.shape[1]):
                    q = self._q_func(replay_observations, replay_actionss[:, action_sample_num, :])
                    loss += self._replay_critic_alpha * F.mse_loss(replay_qss[:, action_sample_num], q) / len(replay_batches)

        loss.backward()
        self._critic_optim.step()

        return loss.cpu().detach().numpy()

    def compute_critic_loss(self, batch: TransitionMiniBatch, q_tpn: torch.Tensor, all_data: MDPDataset, action: torch.Tensor, alpha_c=10, beta=0.5) -> torch.Tensor:
        assert self._q_func is not None
        with torch.no_grad():
            near_observations = all_data._observations[batch.actions[:, self._action_size:].to(torch.int64).cpu().numpy()]
            near_actions = all_data._actions[batch.actions[:, self._action_size:].to(torch.int64).cpu().numpy()]
            near_actions = near_actions[:, :, :self._action_size]
            _, smallest_index, smallest_distance = similar_phi(batch.next_observations, action, near_observations, near_actions, self._phi, input_indexes=batch.next_actions[:, self._action_size:])
        b = torch.mean(torch.mul(q_tpn.squeeze(), torch.exp(- beta * smallest_distance)))
        return self._q_func.compute_error(
            obs_t=batch.observations,
            act_t=batch.actions[:, :self.action_size],
            # input_indexes=batch.actions[:, self.action_size:],
            rew_tp1=batch.next_rewards,
            q_tp1=q_tpn,
            ter_tp1=batch.terminals,
            gamma=self._gamma ** batch.n_steps,
            use_independent_target=self._target_reduction_type == "none",
            masks=batch.masks,
        ) + alpha_c * b

    def compute_target(self, batch: TorchMiniBatch) -> torch.Tensor:
        assert self._targ_q_func is not None
        assert self._targ_policy is not None
        with torch.no_grad():
            action = self._targ_policy(batch.next_observations)
            return self._targ_q_func.compute_target(
                batch.next_observations,
                action.clamp(-1.0, 1.0),
                reduction=self._target_reduction_type,
            ), action

    @train_api
    def update_actor(self, batch: TransitionMiniBatch, replay_batches: Optional[Dict[int, List[torch.Tensor]]]=None, all_data: MDPDataset=None) -> np.ndarray:
        assert all_data is not None
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

        loss = self.compute_actor_loss(batch, all_data)
        if replay_batches is not None:
            for i, replay_batch in replay_batches.items():
                with torch.no_grad():
                    replay_observations, _, replay_means, replay_stddevs, _ = replay_batch
                    replay_observations = replay_observations.to(self.device)
                    replay_means = replay_means.to(self.device)
                    replay_stddevs = replay_stddevs.to(self.device)
                dist = self._policy.dist(replay_observations)
                dist_ = torch.distributions.normal.Normal(replay_means, replay_stddevs)
                loss += self._replay_actor_alpha * torch.distributions.kl.kl_divergence(dist_, dist).mean() / len(replay_batches)

        loss.backward()
        self._actor_optim.step()

        return loss.cpu().detach().numpy()

    def compute_actor_loss(self, batch: TorchMiniBatch, all_data: MDPDataset, alpha_a=10, beta=0.5) -> torch.Tensor:
        assert self._policy is not None
        assert self._q_func is not None
        with torch.no_grad():
            action, _ = self._policy.sample_with_log_prob(batch.observations)
            near_observations = all_data._observations[batch.actions[:, self._action_size:].to(torch.int64).cpu().numpy()]
            near_actions = all_data._actions[batch.actions[:, self._action_size:].to(torch.int64).cpu().numpy()]
            near_actions = near_actions[:, :, :self._action_size]
            _, smallest_index, smallest_distance = similar_phi(batch.observations, action, near_observations, near_actions, self._phi, input_indexes=batch.actions[:, self._action_size:])
        q_tpn, _ = self.compute_target(batch)
        b = torch.mean(torch.mul(q_tpn.squeeze(), torch.exp(- beta * smallest_distance)))
        q_t = self._q_func(batch.observations, action)[0]
        lam = self._replay_actor_alpha / (q_t.abs().mean()).detach()
        return lam * -q_t.mean() + b * alpha_a

    @train_api
    @torch_api()
    def update_phi(self, batch1: TransitionMiniBatch, batch2: TransitionMiniBatch) -> np.ndarray:
        assert self._phi_optim is not None
        self._phi.train()
        self._psi.eval()
        self._q_func.eval()
        self._policy.eval()

        self._phi_optim.zero_grad()

        loss = self.compute_phi_loss(batch1, batch2)

        loss.backward()
        self._phi_optim.step()

        self._psi.train()

        return loss.cpu().detach().numpy()

    def compute_phi_loss(self, batch1: TransitionMiniBatch, batch2: TransitionMiniBatch) -> torch.Tensor:
        assert self._phi is not None
        assert self._psi is not None
        s1, a1, r1, sp1 = batch1.observations, batch1.actions[:, :self.action_size], batch1.rewards, batch1.next_observations
        s2, a2, r2, sp2 = batch2.observations, batch2.actions[:, :self.action_size], batch2.rewards, batch2.next_observations
        loss_phi = torch.norm(self._phi(s1, a1[:, :self._action_size]) - self._phi(s2, a2[:, :self._action_size]), dim=1) + torch.abs(r1 - r2) + self._gamma * torch.norm(self._psi(sp1) - self._psi(sp2), dim=1)
        loss_phi = torch.mean(loss_phi)
        return loss_phi

    @train_api
    @torch_api()
    def update_psi(self, batch1: TransitionMiniBatch, batch2: TransitionMiniBatch) -> np.ndarray:
        assert self._psi_optim is not None
        self._phi.eval()
        self._psi.train()
        self._q_func.eval()
        self._policy.eval()

        self._phi.eval()
        self._psi.train()
        self._psi_optim.zero_grad()

        loss = self.compute_psi_loss(batch1, batch2)

        loss.backward()
        self._psi_optim.step()

        self._phi.train()

        return loss.cpu().detach().numpy()

    def compute_psi_loss(self, batch1: TransitionMiniBatch, batch2: TransitionMiniBatch) -> torch.Tensor:
        assert self._phi is not None
        assert self._psi is not None
        assert self._policy is not None
        s1 = batch1.observations
        s2 = batch2.observations
        loss_psi = torch.norm(self._psi(s1) - self._psi(s2), dim=1)
        u1, _ = self._policy.sample_n_with_log_prob(s1, self._n_sample_actions)
        u2, _ = self._policy.sample_n_with_log_prob(s2, self._n_sample_actions)
        loss_psi_u = 0
        for u_num in range(u1.shape[1]):
            loss_psi_u += torch.norm(self._phi(s1, u1[:, u_num, :]) - self._phi(s2, u2[:, u_num, :]), dim=1)
        loss_psi_u /= self._n_sample_actions
        loss_psi -= loss_psi_u
        return torch.mean(loss_psi)

    def _phi_diff(self, state1: torch.Tensor, action1: torch.Tensor, state2: torch.Tensor, action2: torch.Tensor) -> torch.Tensor:
        assert self._phi is not None
        return torch.norm(self._phi(state1, action1) - self._phi(state2, action2, dim=1))
