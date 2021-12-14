from typing import List, Optional, Union, cast

import torch
from torch import nn

from d3rlpy.model.torch.q_functions.ensembel_q_function import _reduce_ensemble, _gather_quantiles_by_indices, _reduce_quantile_ensemble, EnsembleQFunction, EnsembleDiscreteQFunction, EnsembleContinuousQFunction


class EnsembleQFunctionWithTaskID(EnsembleQFunction):  # type: ignore
    _task_id_size: int

    def __init__(
        self,
        q_funcs: Union[List[DiscreteQFunction], List[ContinuousQFunction]],
        bootstrap: bool = False,
    ):
        super().__init__(
            q_funcs=q_funcs,
            bootstrap=bootstrap,
        )
        self._task_id_size = q_funcs[0].task_id_size

    def compute_error(
        self,
        obs_t: torch.Tensor,
        act_t: torch.Tensor,
        tid_t: torch.Tensor,
        rew_tp1: torch.Tensor,
        q_tp1: torch.Tensor,
        ter_tp1: torch.Tensor,
        gamma: float = 0.99,
        use_independent_target: bool = False,
        masks: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        if use_independent_target:
            assert q_tp1.ndim == 3
        else:
            assert q_tp1.ndim == 2

        if self._bootstrap and masks is not None:
            assert masks.shape == (len(self._q_funcs), obs_t.shape[0], 1,), (
                "Invalid mask shape is detected. "
                f"mask_size must be {len(self._q_funcs)}."
            )

        td_sum = torch.tensor(0.0, dtype=torch.float32, device=obs_t.device)
        for i, q_func in enumerate(self._q_funcs):
            if use_independent_target:
                target = q_tp1[i]
            else:
                target = q_tp1

            loss = q_func.compute_error(
                obs_t, act_t, tid_t, rew_tp1, target, ter_tp1, gamma, reduction="none"
            )

            if self._bootstrap:
                if masks is None:
                    mask = torch.randint(0, 2, loss.shape, device=obs_t.device)
                else:
                    mask = masks[i]
                loss *= mask.float()
                td_sum += loss.sum() / (mask.sum().float() + 1e-10)
            else:
                td_sum += loss.mean()

        return td_sum

    def _compute_target(
        self,
        x: torch.Tensor,
        task_id: torch.Tensor,
        action: Optional[torch.Tensor] = None,
        reduction: str = "min",
        lam: float = 0.75,
    ) -> torch.Tensor:
        values_list: List[torch.Tensor] = []
        for q_func in self._q_funcs:
            target = q_func.compute_target(x, task_id, action)
            values_list.append(target.reshape(1, x.shape[0], -1))

        values = torch.cat(values_list, dim=0)

        if action is None:
            # mean Q function
            if values.shape[2] == self._action_size:
                return _reduce_ensemble(values, reduction)
            # distributional Q function
            n_q_funcs = values.shape[0]
            values = values.view(n_q_funcs, x.shape[0], self._action_size, -1)
            return _reduce_quantile_ensemble(values, reduction)

        if values.shape[2] == 1:
            return _reduce_ensemble(values, reduction, lam=lam)

        return _reduce_quantile_ensemble(values, reduction, lam=lam)


class EnsembleDiscreteQFunctionWithTaskID(EnsembleQFunctionWithTaskID):
    def forward(self, x: torch.Tensor, task_id: torch.Tensor, reduction: str = "mean") -> torch.Tensor:
        values = []
        for q_func in self._q_funcs:
            values.append(q_func(x, task_id).view(1, x.shape[0], self._action_size))
        return _reduce_ensemble(torch.cat(values, dim=0), reduction)

    def __call__(
        self, x: torch.Tensor, task_id: torch.Tensor, reduction: str = "mean"
    ) -> torch.Tensor:
        return cast(torch.Tensor, super().__call__(x, task_id, reduction))

    def compute_target(
        self,
        x: torch.Tensor,
        task_id: torch.Tensor,
        action: Optional[torch.Tensor] = None,
        reduction: str = "min",
        lam: float = 0.75,
    ) -> torch.Tensor:
        return self._compute_target(x, task_id, action, reduction, lam)


class EnsembleContinuousQFunctionWithTaskID(EnsembleQFunctionWithTaskID):
    def forward(
        self, x: torch.Tensor, task_id: torch.Tensor, action: torch.Tensor, reduction: str = "mean"
    ) -> torch.Tensor:
        values = []
        for q_func in self._q_funcs:
            values.append(q_func(x, task_id, action).view(1, x.shape[0], 1))
        return _reduce_ensemble(torch.cat(values, dim=0), reduction)

    def __call__(
        self, x: torch.Tensor, task_id: torch.Tensor, action: torch.Tensor, reduction: str = "mean"
    ) -> torch.Tensor:
        return cast(torch.Tensor, super().__call__(x, action, task_id, reduction))

    def compute_target(
        self,
        x: torch.Tensor,
        action: torch.Tensor,
        task_id: torch.Tensor,
        reduction: str = "min",
        lam: float = 0.75,
    ) -> torch.Tensor:
        return self._compute_target(x, task_id, action, reduction, lam)
