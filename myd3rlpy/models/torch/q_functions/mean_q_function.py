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

from d3rlpy.models.torch.q_functions.mean_q_function import DiscreteMeanQFunction, ContinuousMeanQFunction
from myd3rlpy.models.encoders import EncoderWithTaskID, EncoderWithActionWithTaskID


class DiscreteMeanQFunctionWithTaskID(DiscreteMeanQFunction, nn.Module):  # type: ignore
    _task_id_size: int

    def __init__(self, encoder: EncoderWithTaskID, action_size: int, task_id_size: torch.Tensor):
        super().__init__(encoder=encoder, action_size=action_size)
        self._task_id_size = task_id_size

    def forward(self, x: torch.Tensor, task_id: torch.Tensor) -> torch.Tensor:
        return cast(torch.Tensor, self._fc(self._encoder(x, task_id)))

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
        one_hot = F.one_hot(act_t.view(-1), num_classes=self.action_size)
        q_t = (self.forward(obs_t, tid_t) * one_hot.float()).sum(dim=1, keepdim=True)
        y = rew_tp1 + gamma * q_tp1 * (1 - ter_tp1)
        loss = compute_huber_loss(q_t, y)
        return compute_reduce(loss, reduction)

    def compute_target(
        self, x: torch.Tensor, task_id: torch.Tensor, action: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        if action is None:
            return self.forward(x, task_id)
        return pick_value_by_action(self.forward(x, task_id), action, keepdim=True)

    @property
    def task_id_size(self) -> int:
        return self._task_id_size


class ContinuousMeanQFunctionWithTaskID(ContinuousMeanQFunction, nn.Module):  # type: ignore
    _task_id_size: int

    def __init__(self, encoder: EncoderWithActionWithTaskID):
        super().__init__(encoder=encoder)
        self._task_id_size = encoder.task_id_size

    def forward(self, x: torch.Tensor, action: torch.Tensor, task_id: torch.Tensor) -> torch.Tensor:
        print(self._encoder)
        return cast(torch.Tensor, self._fc(self._encoder(x, action, task_id)))

    def __call__(self, x: torch.Tensor, action: torch.Tensor, task_id: torch.Tensor) -> torch.Tensor:
        return self.forward(x, action, task_id)

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
        q_t = self.forward(obs_t, act_t, tid_t)
        y = rew_tp1 + gamma * q_tp1 * (1 - ter_tp1)
        loss = F.mse_loss(q_t, y, reduction="none")
        return compute_reduce(loss, reduction)

    def compute_target(
        self, x: torch.Tensor, action: torch.Tensor, task_id: torch.Tensor
    ) -> torch.Tensor:
        return self.forward(x, action, task_id)

    @property
    def task_id_size(self) -> int:
        return self._task_id_size
