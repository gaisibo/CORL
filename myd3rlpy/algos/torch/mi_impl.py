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

from d3rlpy.models.builders import create_squashed_normal_policy, create_continuous_q_function

class MIImpl(TD3Impl):
    _n_sample_actions: int
    def __init__(
        self,
        observation_shape: Sequence[int],
        action_size: int,
        actor_learning_rate: float,
        critic_learning_rate: float,
        actor_optim_factory: OptimizerFactory,
        critic_optim_factory: OptimizerFactory,
        actor_encoder_factory: EncoderFactory,
        critic_encoder_factory: EncoderFactory,
        q_func_factory: QFunctionFactory,
        replay_actor_alpha: float,
        replay_critic_alpha: float,
        replay_critic: bool,
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
        self._replay_actor_alpha = replay_actor_alpha
        self._replay_critic_alpha = replay_critic_alpha
        self._replay_critic = replay_critic
        self._n_sample_actions = n_sample_actions
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

        q_tpn, action = self.compute_target(batch)

        loss = self.compute_critic_loss(batch, q_tpn, action)
        policy_loss = loss.cpu().detach().numpy()
        replay_loss = 0
        if replay_batches is not None:
            for i, replay_batch in replay_batches.items():
                with torch.no_grad():
                    replay_observations, replay_actionss, _, _, replay_qss = replay_batch
                    replay_observations = replay_observations.to(self.device)
                    replay_actionss = replay_actionss.to(self.device)
                    replay_qss = replay_qss.to(self.device)
                for action_sample_num in range(replay_actionss.shape[1]):
                    q = self._q_func(replay_observations, replay_actionss[:, action_sample_num, :])
                    replay_loss += self._replay_critic_alpha * F.mse_loss(replay_qss[:, action_sample_num], q) / len(replay_batches)
            loss += replay_loss
            replay_loss = replay_loss.cpu().detach().numpy()

        loss.backward()
        self._critic_optim.step()
        loss = loss.cpu().detach().numpy()

        return loss, policy_loss, replay_loss

    def compute_critic_loss(self, batch: TransitionMiniBatch, q_tpn: torch.Tensor, action: torch.Tensor) -> torch.Tensor:
        assert self._q_func is not None
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
        )

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
        policy_loss = loss.cpu().detach().numpy()
        replay_loss = 0
        if replay_batches is not None:
            for i, replay_batch in replay_batches.items():
                with torch.no_grad():
                    replay_observations, _, replay_means, replay_stddevs, _ = replay_batch
                    replay_observations = replay_observations.to(self.device)
                    replay_means = replay_means.to(self.device)
                    replay_stddevs = replay_stddevs.to(self.device)
                    dist_ = torch.distributions.normal.Normal(replay_means, replay_stddevs)
                dist = self._policy.dist(replay_observations)
                replay_loss += self._replay_actor_alpha * torch.distributions.kl.kl_divergence(dist_, dist).mean() / len(replay_batches)
            loss += replay_loss
            replay_loss = replay_loss.cpu().detach().numpy()

        loss.backward()
        self._actor_optim.step()
        loss = loss.cpu().detach().numpy()

        return loss, policy_loss, replay_loss

    def compute_actor_loss(self, batch: TorchMiniBatch) -> torch.Tensor:
        assert self._policy is not None
        assert self._q_func is not None
        action, _ = self._policy.sample_with_log_prob(batch.observations)
        q_t = self._q_func(batch.observations, action)[0]
        return -q_t.mean()
