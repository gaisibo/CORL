from typing import Optional, cast

import torch
import torch.nn.functional as F
from torch import nn

from d3rlpy.models.torch.q_functions.base import ContinuousQFunction, DiscreteQFunction
from d3rlpy.models.torch.q_functions.iqn_q_function import compute_iqn_feature
from d3rlpy.models.torch.q_functions.utility import (
    compute_quantile_loss,
    compute_reduce,
    pick_quantile_value_by_action,
)

from d3rlpy.models.torch.q_functions.qr_q_function import _make_taus, DiscreteQRQFunction, ContinuousQRQFunction
from myd3rlpy.models.encoders import EncoderWithTaskID, EncoderWithActionWithTaskID


class DiscreteQRQFunctionWithTaskID(DiscreteQRQFunction, nn.Module):  # type: ignore
    _task_id_size: int

    def __init__(self, encoder: EncoderWithTaskID, action_size: int, n_quantiles: int, task_id_size: int):
        super().__init__(
            encoder = encoder,
            action_size = action_size,
            n_quantiles = n_quantiles
        )
        self._task_id_size = task_id_size

    def forward(self, x: torch.Tensor, task_id: torch.Tensor) -> torch.Tensor:
        h = self._encoder(x, task_id)
        taus = _make_taus(h, self._n_quantiles)
        quantiles = self._compute_quantiles(h, taus)
        return quantiles.mean(dim=2)

    def compute_error(
        self,
        obs_t: torch.Tensor,
        act_t: torch.Tensor,
        tid_t: torch.Tensor,
        rew_tp1: torch.Tensor,
        q_tp1: torch.Tensor,
        ter_tp1: torch.Tensor,
        gamma: float = 0.99,
        reduction: str = "mean",
    ) -> torch.Tensor:
        assert q_tp1.shape == (obs_t.shape[0], self._n_quantiles)

        # extraect quantiles corresponding to act_t
        h = self._encoder(obs_t, tid_t)
        taus = _make_taus(h, self._n_quantiles)
        quantiles = self._compute_quantiles(h, taus)
        quantiles_t = pick_quantile_value_by_action(quantiles, act_t)

        loss = compute_quantile_loss(
            quantiles_t=quantiles_t,
            rewards_tp1=rew_tp1,
            quantiles_tp1=q_tp1,
            terminals_tp1=ter_tp1,
            taus=taus,
            gamma=gamma,
        )

        return compute_reduce(loss, reduction)

    def compute_target(
        self, x: torch.Tensor, task_id: torch.Tensor, action: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        h = self._encoder(x, task_id)
        taus = _make_taus(h, self._n_quantiles)
        quantiles = self._compute_quantiles(h, taus)
        if action is None:
            return quantiles
        return pick_quantile_value_by_action(quantiles, action)

    @property
    def task_id_size(self) -> int:
        return self._task_id_size


class ContinuousQRQFunctionWithTaskID(ContinuousQRQFunction, nn.Module):  # type: ignore
    _task_id_size: int

    def __init__(self, encoder: EncoderWithActionWithTaskID, n_quantiles: int):
        super().__init__(
            encoder = encoder,
            n_quantiles = n_quantiles
        )
        self._task_id_size = encoder.task_id_size

    def forward(self, x: torch.Tensor, action: torch.Tensor, task_id: torch.Tensor) -> torch.Tensor:
        h = self._encoder(x, action, task_id)
        taus = _make_taus(h, self._n_quantiles)
        quantiles = self._compute_quantiles(h, taus)
        return quantiles.mean(dim=1, keepdim=True)

    def compute_error(
        self,
        obs_t: torch.Tensor,
        act_t: torch.Tensor,
        tid_t: torch.Tensor,
        rew_tp1: torch.Tensor,
        q_tp1: torch.Tensor,
        ter_tp1: torch.Tensor,
        gamma: float = 0.99,
        reduction: str = "mean",
    ) -> torch.Tensor:
        assert q_tp1.shape == (obs_t.shape[0], self._n_quantiles)

        h = self._encoder(obs_t, act_t, tid_t)
        taus = _make_taus(h, self._n_quantiles)
        quantiles_t = self._compute_quantiles(h, taus)

        loss = compute_quantile_loss(
            quantiles_t=quantiles_t,
            rewards_tp1=rew_tp1,
            quantiles_tp1=q_tp1,
            terminals_tp1=ter_tp1,
            taus=taus,
            gamma=gamma,
        )

        return compute_reduce(loss, reduction)

    def compute_target(
        self, x: torch.Tensor, action: torch.Tensor, task_id: torch.Tensor
    ) -> torch.Tensor:
        h = self._encoder(x, action, task_id)
        taus = _make_taus(h, self._n_quantiles)
        return self._compute_quantiles(h, taus)

    @property
    def task_id_size(self) -> int:
        return self._task_id_size
