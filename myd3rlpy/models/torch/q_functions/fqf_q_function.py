from typing import Optional, Tuple, cast

import torch
from torch import nn

from d3rlpy.models.torch.base import ContinuousQFunction, DiscreteQFunction
from d3rlpy.models.torch.iqn_q_function import compute_iqn_feature
from d3rlpy.models.torch.utility import (
    compute_quantile_loss,
    compute_reduce,
    pick_quantile_value_by_action,
)
from d3rlpy.models.torch.q_functions.fqf_q_function import _make_taus, DiscreteFQFQFunction, ContinuousFQFQFunction

from myd3rlpy.models.encoders import EncoderWithTaskID, EncoderWithActionWithTaskID


class DiscreteFQFQFunctionWithTaskID(DiscreteFQFQFunction, nn.Module):  # type: ignore
    _task_id_size: int

    def __init__(
        self,
        encoder: EncoderWithTaskID,
        action_size: int,
        task_id_size: int,
        n_quantiles: int,
        embed_size: int,
        entropy_coeff: float = 0.0,
    ):
        super().__init__(
            encoder = encoder,
            action_size = action_size,
            task_id_size = task_id_size,
            n_quantiles = n_quantiles,
            embed_size = embed_size,
            entropy_coeff = entropy_coeff,
        )
        self._task_id_size = task_id_size

    def forward(self, x: torch.Tensor, task_id: torch.Tensor) -> torch.Tensor:
        h = self._encoder(x, task_id)
        taus, taus_minus, taus_prime, _ = _make_taus(h, self._proposal)
        quantiles = self._compute_quantiles(h, taus_prime.detach())
        weight = (taus - taus_minus).view(-1, 1, self._n_quantiles).detach()
        return (weight * quantiles).sum(dim=2)

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

        # compute quantiles
        h = self._encoder(obs_t, tid_t)
        taus, _, taus_prime, entropies = _make_taus(h, self._proposal)
        quantiles = self._compute_quantiles(h, taus_prime.detach())
        quantiles_t = pick_quantile_value_by_action(quantiles, act_t)

        quantile_loss = compute_quantile_loss(
            quantiles_t=quantiles_t,
            rewards_tp1=rew_tp1,
            quantiles_tp1=q_tp1,
            terminals_tp1=ter_tp1,
            taus=taus_prime.detach(),
            gamma=gamma,
        )

        # compute proposal network loss
        # original paper explicitly separates the optimization process
        # but, it's combined here
        proposal_loss = self._compute_proposal_loss(h, act_t, taus, taus_prime)
        proposal_params = list(self._proposal.parameters())
        proposal_grads = torch.autograd.grad(
            outputs=proposal_loss.mean(),
            inputs=proposal_params,
            retain_graph=True,
        )
        # directly apply gradients
        for param, grad in zip(list(proposal_params), proposal_grads):
            param.grad = 1e-4 * grad

        loss = quantile_loss - self._entropy_coeff * entropies

        return compute_reduce(loss, reduction)

    def compute_target(
        self, x: torch.Tensor, task_id: torch.Tensor, action: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        h = self._encoder(x, task_id)
        _, _, taus_prime, _ = _make_taus(h, self._proposal)
        quantiles = self._compute_quantiles(h, taus_prime.detach())
        if action is None:
            return quantiles
        return pick_quantile_value_by_action(quantiles, action)

    @property
    def task_id_size(self) -> int:
        return self._task_id_size


class ContinuousFQFQFunctionWithTaskID(ContinuousFQFQFunctionWithTaskID, nn.Module):  # type: ignore

    def __init__(
        self,
        encoder: EncoderWithActionWithTaskID,
        n_quantiles: int,
        embed_size: int,
        entropy_coeff: float = 0.0,
    ):
        super().__init__(
            encoder = encoder,
            n_quantiles = n_quantiles,
            embed_size = embed_size,
            entropy_coeff = entropy_coeff,
        )
        self._task_id_size = encoder.task_id_size

    def forward(self, x: torch.Tensor, task_id: torch.Tensor, action: torch.Tensor) -> torch.Tensor:
        h = self._encoder(x, action, task_id)
        taus, taus_minus, taus_prime, _ = _make_taus(h, self._proposal)
        quantiles = self._compute_quantiles(h, taus_prime.detach())
        weight = (taus - taus_minus).detach()
        return (weight * quantiles).sum(dim=1, keepdim=True)

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
        taus, _, taus_prime, entropies = _make_taus(h, self._proposal)
        quantiles_t = self._compute_quantiles(h, taus_prime.detach())

        quantile_loss = compute_quantile_loss(
            quantiles_t=quantiles_t,
            rewards_tp1=rew_tp1,
            quantiles_tp1=q_tp1,
            terminals_tp1=ter_tp1,
            taus=taus_prime.detach(),
            gamma=gamma,
        )

        # compute proposal network loss
        # original paper explicitly separates the optimization process
        # but, it's combined here
        proposal_loss = self._compute_proposal_loss(h, taus, taus_prime)
        proposal_params = list(self._proposal.parameters())
        proposal_grads = torch.autograd.grad(
            outputs=proposal_loss.mean(),
            inputs=proposal_params,
            retain_graph=True,
        )
        # directly apply gradients
        for param, grad in zip(list(proposal_params), proposal_grads):
            param.grad = 1e-4 * grad

        loss = quantile_loss - self._entropy_coeff * entropies

        return compute_reduce(loss, reduction)

    def compute_target(
        self, x: torch.Tensor, task_id: torch.Tensor, action: torch.Tensor
    ) -> torch.Tensor:
        h = self._encoder(x, action, task_id)
        _, _, taus_prime, _ = _make_taus(h, self._proposal)
        return self._compute_quantiles(h, taus_prime.detach())

    @property
    def task_id_size(self) -> int:
        return self._task_id_size
