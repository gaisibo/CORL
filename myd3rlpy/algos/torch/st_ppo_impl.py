import copy
from typing import Optional, Sequence, Tuple

import numpy as np
import torch
import torch.nn.functional as F
from d3rlpy.gpu import Device
from d3rlpy.models.builders import create_squashed_normal_policy
from d3rlpy.torch_utility import TorchMiniBatch, train_api, torch_api
from d3rlpy.models.encoders import EncoderFactory
from d3rlpy.models.optimizers import OptimizerFactory
from d3rlpy.models.torch import SquashedNormalPolicy
from d3rlpy.algos.torch.base import TorchImplBase
from d3rlpy.preprocessing import ActionScaler, RewardScaler, Scaler

from myd3rlpy.algos.torch.st_impl import STImpl
from myd3rlpy.models.builders import create_parallel_continuous_q_function
from utils.networks import ParallelizedEnsembleFlattenMLP


class STPPOImpl(STImpl, TorchImplBase):

    _policy: Optional[SquashedNormalPolicy]

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
        value_encoder_factory: EncoderFactory,
        gamma: float,
        tau: float,
        n_critics: int,
        use_gpu: Optional[Device],
        scaler: Optional[Scaler],
        action_scaler: Optional[ActionScaler],
        reward_scaler: Optional[RewardScaler],

        clip_ratio: float,
        decay: float,
        entropy_weight: float,
        is_clip_decay: bool,
        omega: float,
        **kwargs
    ):
        super().__init__(
            observation_shape=observation_shape,
            action_size=action_size,
            scaler=scaler,
            action_scaler=action_scaler,
            reward_scaler=reward_scaler,
            **kwargs,
        )
        self._actor_learning_rate = actor_learning_rate
        self._critic_learning_rate = critic_learning_rate
        self._actor_optim_factory = actor_optim_factory
        self._critic_optim_factory = critic_optim_factory
        self._actor_encoder_factory = actor_encoder_factory
        self._critic_encoder_factory = critic_encoder_factory
        self._value_encoder_factory = value_encoder_factory
        self._gamma = gamma
        self._tau = tau
        self._n_critics = n_critics
        self._use_gpu = use_gpu

        # initialized in build
        self._q_func = None
        self._policy = None
        self._targ_q_func = None
        self._actor_optim = None
        self._critic_optim = None

        self._clip_ratio = clip_ratio
        self._decay = decay
        self._entropy_weight = entropy_weight
        self._is_clip_decay = is_clip_decay
        self._omega = omega

    def _build_critic(self) -> None:
        self._value_func = ParallelizedEnsembleFlattenMLP(self._n_ensemble, [256, 256], self._observation_shape[0], 1, device=self.device)
        self._q_func = create_parallel_continuous_q_function(
            self._observation_shape,
            self._action_size,
            n_ensembles=self._n_critics,
            reduction='min',
        )
        self._targ_q_func = copy.deepcopy(self._q_func)

    def _build_actor(self) -> None:
        self._policy = create_squashed_normal_policy(
            self._observation_shape,
            self._action_size,
            self._actor_encoder_factory,
        )
        self._old_policy = copy.deepcopy(self._policy)

    def compute_target(self, batch: TorchMiniBatch):
        next_actions = self._policy(batch.next_observations)
        q_tpn = batch.rewards + batch.terminals * self._gamma * self._targ_q_func(batch.next_observations, next_actions)
        return q_tpn

    def compute_critic_loss(self, batch: TorchMiniBatch, q_tpn: torch.Tensor, clone_critic: bool=False, online: bool=False, replay=False, first_time=False) -> torch.Tensor:
        assert self._policy is not None
        assert self._q_func is not None
        assert self._targ_q_func is not None
        critic_loss = F.mse_loss(self._q_func(batch.observations, batch.actions), q_tpn)
        return critic_loss
    #def compute_critic_loss(self, batch, q_tpn, clone_critic: bool=False, online: bool=False, replay=False, first_time=False):
    #    value = self._q_func(batch.observations, batch.actions)
    #    y = batch.rewards + self._gamma * q_tpn * (1 - batch.terminals)
    #    loss = F.mse_loss(value, y, reduction="mean")
    #    return loss

    #def compute_target(self, batch: TorchMiniBatch) -> torch.Tensor:
    #    with torch.no_grad():
    #        # v_t = []
    #        # for value_func in self._value_func:
    #        #     v_t.append(value_func(batch.next_observations))
    #        # v_t, _ = torch.min(torch.stack(v_t, dim=0), dim=0)
    #        action, log_prob = self._policy.sample_with_log_prob(batch.next_observations)
    #        q_t = torch.mean(self._targ_q_func(batch.next_observations, action), dim=0)
    #        return q_t

    def weighted_advantage(self, advantage: torch.Tensor) -> torch.Tensor:
        if self._omega == 0.5:
            return advantage
        else:
            weight = torch.zeros_like(advantage)
            index = torch.where(advantage > 0)[0]
            weight[index] = self._omage
            weight[torch.where(weight == 0)[0]] = 1 - self._omage
            weight.to(self._device)
            return weight * advantage

    def compute_actor_loss(self, batch: TorchMiniBatch, clone_actor=False, online: bool=False, replay=False) -> torch.Tensor:
        assert self._policy is not None
        assert self._old_policy is not None
        assert self._q_func is not None
        new_dist = self._policy.dist(batch.observations)
        new_log_prob = new_dist.log_prob()
        old_dist = self._old_policy.dist(batch.observations)
        old_log_prob = old_dist.log_prob()
        ratio = (new_log_prob - old_log_prob).exp()
        advantage = self.weighted_advantage(batch.rtgs)
        loss1 = ratio * advantage
        if self._is_clip_decay:
            self._clip_ratio = self._clip_ratio * self._decay
        else:
            self._clip_ratio = self._clip_ratio
        loss2 = torch.clamp(ratio, 1 - self._clip_ratio, 1 + self._clip_ratio) * advantage
        entropy_loss = new_dist.entropy().sum(-1, keepdim=True) * self._entropy_weight
        loss = -(torch.min(loss1, loss2) + entropy_loss).mean()
        return loss

    @train_api
    @torch_api()
    def update_critic_clone(self, batch: TorchMiniBatch) -> Tuple[np.ndarray, np.ndarray]:
        assert self._q_func is not None
        assert self._targ_q_func is not None
        assert self._critic_optim is not None

        self._critic_optim.zero_grad()

        q_tpn = batch.rewards + batch.terminals * self._gamma * self._targ_q_func(batch.next_observations, batch.next_actions)
        critic_loss, value_loss = self.compute_critic_loss_clone(batch, q_tpn)

        critic_loss.backward()
        self._critic_optim.step()

        return critic_loss.cpu().detach().numpy(), value_loss.cpu().detach().numpy()

    def compute_critic_loss_clone(self, batch, q_tpn, clone_critic: bool=False, online: bool=False, replay=False, first_time=False):
        assert self._q_func is not None
        critic_loss = F.mse_loss(self._q_func(batch.observations, batch.actions), q_tpn)
        value_loss = F.mse_loss(self._value_func(batch.observations), batch.rtgs)
        return critic_loss, value_loss
    #def compute_critic_loss(self, batch, q_tpn, clone_critic: bool=False, online: bool=False, replay=False, first_time=False):
    #    value = self._q_func(batch.observations, batch.actions)
    #    y = batch.rewards + self._gamma * q_tpn * (1 - batch.terminals)
    #    loss = F.mse_loss(value, y, reduction="mean")
    #    return loss

    #def compute_target(self, batch: TorchMiniBatch) -> torch.Tensor:
    #    with torch.no_grad():
    #        # v_t = []
    #        # for value_func in self._value_func:
    #        #     v_t.append(value_func(batch.next_observations))
    #        # v_t, _ = torch.min(torch.stack(v_t, dim=0), dim=0)
    #        action, log_prob = self._policy.sample_with_log_prob(batch.next_observations)
    #        q_t = torch.mean(self._targ_q_func(batch.next_observations, action), dim=0)
    #        return q_t
