from typing import List, Optional, Union, cast

import torch
from torch import nn
import torch.nn.functional as F

from myd3rlpy.models.torch import pytorch_util as ptu
from myd3rlpy.models.torch.modules import LayerNorm
from myd3rlpy.models.torch.parallel_ensemble import ParallelizedEnsembleFlattenMLP
from d3rlpy.models.torch.q_functions.base import ContinuousQFunction, DiscreteQFunction
from d3rlpy.models.torch.q_functions.utility import compute_reduce


def _reduce_ensemble(
    y: torch.Tensor, reduction: str = "min", dim: int = 0, lam: float = 0.75
) -> torch.Tensor:
    if reduction == "min":
        return y.min(dim=dim).values
    elif reduction == "max":
        return y.max(dim=dim).values
    elif reduction == "mean":
        return y.mean(dim=dim)
    elif reduction == "none":
        return y
    elif reduction == "mix":
        max_values = y.max(dim=dim).values
        min_values = y.min(dim=dim).values
        return lam * min_values + (1.0 - lam) * max_values
    raise ValueError


def _gather_quantiles_by_indices(
    y: torch.Tensor, indices: torch.Tensor
) -> torch.Tensor:
    # TODO: implement this in general case
    if y.dim() == 3:
        # (N, batch, n_quantiles) -> (batch, n_quantiles)
        return y.transpose(0, 1)[torch.arange(y.shape[1]), indices]
    elif y.dim() == 4:
        # (N, batch, action, n_quantiles) -> (batch, action, N, n_quantiles)
        transposed_y = y.transpose(0, 1).transpose(1, 2)
        # (batch, action, N, n_quantiles) -> (batch * action, N, n_quantiles)
        flat_y = transposed_y.reshape(-1, y.shape[0], y.shape[3])
        head_indices = torch.arange(y.shape[1] * y.shape[2])
        # (batch * action, N, n_quantiles) -> (batch * action, n_quantiles)
        gathered_y = flat_y[head_indices, indices.view(-1)]
        # (batch * action, n_quantiles) -> (batch, action, n_quantiles)
        return gathered_y.view(y.shape[1], y.shape[2], -1)
    raise ValueError


def _reduce_quantile_ensemble(
    y: torch.Tensor, reduction: str = "min", dim: int = 0, lam: float = 0.75
) -> torch.Tensor:
    # reduction beased on expectation
    mean = y.mean(dim=-1)
    if reduction == "min":
        indices = mean.min(dim=dim).indices
        return _gather_quantiles_by_indices(y, indices)
    elif reduction == "max":
        indices = mean.max(dim=dim).indices
        return _gather_quantiles_by_indices(y, indices)
    elif reduction == "none":
        return y
    elif reduction == "mix":
        min_indices = mean.min(dim=dim).indices
        max_indices = mean.max(dim=dim).indices
        min_values = _gather_quantiles_by_indices(y, min_indices)
        max_values = _gather_quantiles_by_indices(y, max_indices)
        return lam * min_values + (1.0 - lam) * max_values
    raise ValueError


class ParallelEnsembleQFunction(ParallelizedEnsembleFlattenMLP):  # type: ignore
    _action_size: int

    def __init__(
        self,
        ensemble_size,
        hidden_sizes,
        input_size,
        output_size,
        init_w=3e-3,
        hidden_init=ptu.fanin_init,
        w_scale=1,
        b_init_value=0.1,
        layer_norm=None,
        batch_norm=False,
        final_init_scale=None,
        reduction='mean',
    ):
        super().__init__(
            ensemble_size,
            hidden_sizes,
            input_size,
            output_size,
            init_w,
            hidden_init,
            w_scale,
            b_init_value,
            layer_norm,
            batch_norm,
            final_init_scale,
        )
        self.reduction = reduction

    def compute_error(
        self,
        observations: torch.Tensor,
        actions: torch.Tensor,
        rewards: torch.Tensor,
        target: torch.Tensor,
        terminals: torch.Tensor,
        gamma: float = 0.99,
        reduction: str = "mean",
    ) -> torch.Tensor:
        assert target.ndim == 2
        value = self.forward(observations, actions)
        y = rewards + gamma * target * (1 - terminals)
        loss = F.mse_loss(value, y, reduction="none")
        return compute_reduce(loss, reduction)

    def _compute_target(
        self,
        x: torch.Tensor,
        action: Optional[torch.Tensor] = None,
        lam: float = 0.75,
    ) -> torch.Tensor:
        values = super().forward(x, action)

        if action is None:
            # mean Q function
            if values.shape[2] == self._action_size:
                return _reduce_ensemble(values, self.reduction)
            # distributional Q function
            n_q_funcs = values.shape[0]
            values = values.view(n_q_funcs, x.shape[0], self._action_size, -1)
            return _reduce_quantile_ensemble(values, self.reduction)

        if values.shape[2] == 1:
            return _reduce_ensemble(values, self.reduction, lam=lam)
        return _reduce_quantile_ensemble(values, self.reduction, lam=lam)

    # 这里主要是为了给compute_max_with_n_actions_and_indices调用用的。
    @property
    def q_funcs(self):
        return [0 for _ in range(self.ensemble_size)]

class ParallelEnsembleDiscreteQFunction(ParallelEnsembleQFunction):
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        values = super().forward(x)
        return _reduce_ensemble(values, self.reduction)

    def __call__(
        self, x: torch.Tensor
    ) -> torch.Tensor:
        return cast(torch.Tensor, super().__call__(x))

    def compute_target(
        self,
        x: torch.Tensor,
        action: Optional[torch.Tensor] = None,
        lam: float = 0.75,
    ) -> torch.Tensor:
        return self._compute_target(x, action, lam)


class ParallelEnsembleContinuousQFunction(ParallelEnsembleQFunction):
    def forward(
        self, x: torch.Tensor, action: torch.Tensor
    ) -> torch.Tensor:
        values = super().forward(x, action)
        return _reduce_ensemble(values, reduction=self.reduction)

    def __call__(
        self, x: torch.Tensor, action: torch.Tensor
    ) -> torch.Tensor:
        return cast(torch.Tensor, super().__call__(x, action))

    def compute_target(
        self,
        x: torch.Tensor,
        action: torch.Tensor,
        lam: float = 0.75,
    ) -> torch.Tensor:
        return self._compute_target(x, action, lam)
