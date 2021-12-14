import math
from abc import ABCMeta, abstractmethod
from typing import Tuple, Union, cast

import torch
import torch.nn.functional as F
from torch import nn
from torch.distributions import Categorical, Normal

from d3rlpy.models.torch.policies import squash_action, Policy, DeterministicPolicy, DeterministicResidualPolicy, SquashedNormalPolicy, CategoricalPolicy
from myd3rlpy.models.torch.encoders import EncoderWithTaskID, EncoderWithActionWithTaskID


class DeterministicPolicyWithTaskID(DeterministicPolicy):

    def forward(self, x: torch.Tensor, task_id: torch.Tensor) -> torch.Tensor:
        h = self._encoder(x, task_id)
        return torch.tanh(self._fc(h))

    def __call__(self, x: torch.Tensor, task_id: torch.Tensor) -> torch.Tensor:
        return cast(torch.Tensor, super().__call__(x, task_id))

    def sample_with_log_prob(
        self, x: torch.Tensor, task_id: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        raise NotImplementedError(
            "deterministic policy does not support sample"
        )

    def sample_n_with_log_prob(
        self, x: torch.Tensor, task_id: torch.Tensor, n: int
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        raise NotImplementedError(
            "deterministic policy does not support sample_n"
        )

    def best_action(self, x: torch.Tensor, task_id: torch.Tensor) -> torch.Tensor:
        return self.forward(x, task_id)


class DeterministicResidualPolicyWithTaskID(DeterministicResidualPolicy):

    def forward(self, x: torch.Tensor, action: torch.Tensor, task_id: torch.Tensor) -> torch.Tensor:
        h = self._encoder(x, action, task_id)
        residual_action = self._scale * torch.tanh(self._fc(h))
        return (action + cast(torch.Tensor, residual_action)).clamp(-1.0, 1.0)

    def __call__(self, x: torch.Tensor, action: torch.Tensor, task_id: torch.Tensor) -> torch.Tensor:
        return cast(torch.Tensor, super().__call__(x, action, task_id))

    def best_residual_action(
        self, x: torch.Tensor, action: torch.Tensor, task_id: torch.Tensor
    ) -> torch.Tensor:
        return self.forward(x, action, task_id)

    def best_action(self, x: torch.Tensor, task_id: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError(
            "residual policy does not support best_action"
        )

    def sample_with_log_prob(
        self, x: torch.Tensor, task_id: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        raise NotImplementedError(
            "deterministic policy does not support sample"
        )

    def sample_n_with_log_prob(
        self, x: torch.Tensor, task_id: torch.Tensor, n: int
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        raise NotImplementedError(
            "deterministic policy does not support sample_n"
        )


class SquashedNormalPolicyWithTaskID(SquashedNormalPolicy):

    def dist(self, x: torch.Tensor, task_id: torch.Tensor) -> Normal:
        h = self._encoder(x, task_id)
        mu = self._mu(h)
        clipped_logstd = self._compute_logstd(h)
        return Normal(mu, clipped_logstd.exp())

    def forward(
        self,
        x: torch.Tensor,
        task_id: torch.Tensor,
        deterministic: bool = False,
        with_log_prob: bool = False,
    ) -> Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        if deterministic:
            # to avoid errors at ONNX export because broadcast_tensors in
            # Normal distribution is not supported by ONNX
            action = self._mu(self._encoder(x, task_id))
        else:
            dist = self.dist(x, task_id)
            action = dist.rsample()

        if with_log_prob:
            return squash_action(dist, action)

        return torch.tanh(action)

    def sample_with_log_prob(
        self, x: torch.Tensor, task_id: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        out = self.forward(x, task_id, with_log_prob=True)
        return cast(Tuple[torch.Tensor, torch.Tensor], out)

    def sample_n_with_log_prob(
        self,
        x: torch.Tensor,
        task_id: torch.Tensor,
        n: int,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        dist = self.dist(x, task_id)

        action = dist.rsample((n,))

        squashed_action_T, log_prob_T = squash_action(dist, action)

        # (n, batch, action) -> (batch, n, action)
        squashed_action = squashed_action_T.transpose(0, 1)
        # (n, batch, 1) -> (batch, n, 1)
        log_prob = log_prob_T.transpose(0, 1)

        return squashed_action, log_prob

    def sample_n_without_squash(self, x: torch.Tensor, task_id: torch.Tensor, n: int) -> torch.Tensor:
        dist = self.dist(x, task_id)
        action = dist.rsample((n,))
        return action.transpose(0, 1)

    def onnx_safe_sample_n(self, x: torch.Tensor, task_id: torch.Tensor, n: int) -> torch.Tensor:
        h = self._encoder(x, task_id)
        mean = self._mu(h)
        std = self._compute_logstd(h).exp()

        # expand shape
        # (batch_size, action_size) -> (batch_size, N, action_size)
        expanded_mean = mean.view(-1, 1, self._action_size).repeat((1, n, 1))
        expanded_std = std.view(-1, 1, self._action_size).repeat((1, n, 1))

        # sample noise from Gaussian distribution
        noise = torch.randn(x.shape[0], n, self._action_size, device=x.device)

        return torch.tanh(expanded_mean + noise * expanded_std)

    def best_action(self, x: torch.Tensor, task_id: torch.Tensor) -> torch.Tensor:
        action = self.forward(x, task_id, deterministic=True, with_log_prob=False)
        return cast(torch.Tensor, action)


class CategoricalPolicyWithTaskID(CategoricalPolicy):

    def dist(self, x: torch.Tensor, task_id: torch.Tensor) -> Categorical:
        h = self._encoder(x, task_id)
        h = self._fc(h)
        return Categorical(torch.softmax(h, dim=1))

    def forward(
        self,
        x: torch.Tensor,
        task_id: torch.Tensor,
        deterministic: bool = False,
        with_log_prob: bool = False,
    ) -> Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        dist = self.dist(x, task_id)

        if deterministic:
            action = cast(torch.Tensor, dist.probs.argmax(dim=1))
        else:
            action = cast(torch.Tensor, dist.sample())

        if with_log_prob:
            return action, dist.log_prob(action)

        return action

    def sample_with_log_prob(
        self, x: torch.Tensor, task_id: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        out = self.forward(x, task_id, with_log_prob=True)
        return cast(Tuple[torch.Tensor, torch.Tensor], out)

    def sample_n_with_log_prob(
        self, x: torch.Tensor, task_id: torch.Tensor, n: int
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        dist = self.dist(x, task_id)

        action_T = cast(torch.Tensor, dist.sample((n,)))
        log_prob_T = dist.log_prob(action_T)

        # (n, batch) -> (batch, n)
        action = action_T.transpose(0, 1)
        # (n, batch) -> (batch, n)
        log_prob = log_prob_T.transpose(0, 1)

        return action, log_prob

    def best_action(self, x: torch.Tensor, task_id: torch.Tensor) -> torch.Tensor:
        return cast(torch.Tensor, self.forward(x, task_id, deterministic=True))

    def log_probs(self, x: torch.Tensor, task_id: torch.Tensor) -> torch.Tensor:
        dist = self.dist(x, task_id)
        return cast(torch.Tensor, dist.logits)
