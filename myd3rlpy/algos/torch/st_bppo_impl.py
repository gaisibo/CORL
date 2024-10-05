from typing import Optional

import numpy as np
import torch
from d3rlpy.models.builders import create_squashed_normal_policy
from d3rlpy.torch_utility import TorchMiniBatch, train_api, torch_api
from d3rlpy.models.encoders import EncoderFactory
from d3rlpy.models.optimizers import OptimizerFactory
from d3rlpy.algos.torch.base import TorchImplBase
from d3rlpy.models.torch import SquashedNormalPolicy

from myd3rlpy.algos.torch.st_ppo_impl import STPPOImpl


class STBPPOImpl(STPPOImpl):

    _policy: Optional[SquashedNormalPolicy]

    def __init__(
        self,
        const_eps,
        **kwargs
    ):
        super().__init__(
            **kwargs,
        )
        self._const_eps = const_eps

    def compute_actor_loss(self, batch: TorchMiniBatch, clone_actor=False, online: bool=False, replay=False) -> torch.Tensor:
        assert self._policy is not None
        assert self._old_policy is not None
        assert self._q_func is not None

        new_dist = self._policy.dist(batch.observations)
        new_log_prob = new_dist.log_prob()
        old_dist = self._old_policy.dist(batch.observations)

        # new
        action = old_dist.rsample()
        advantage = self._q_func(batch.observations, action) - self._value_func(batch.observations)
        advantage = (advantage - advantage.mean()) / (advantage.std() + self._const_eps)

        old_log_prob = old_dist.log_prob()
        ratio = (new_log_prob - old_log_prob).exp()

        advantage = self.weighted_advantage(advantage)
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
    def update_bc(self, batch: TorchMiniBatch) -> np.ndarray:
        assert self._policy is not None
        assert self._actor_optim is not None
        dist = self._policy(batch.observations)
        log_prob = dist.log_prob(batch.actions)
        loss = (-log_prob).mean()
        return loss.detech().cpu().numpy()
