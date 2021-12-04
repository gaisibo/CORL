import copy
from typing import Optional, Sequence, List, Any

import numpy as np
import torch
from torch.optim import Optimizer

from d3rlpy.gpu import Device
from d3rlpy.models.encoders import EncoderFactory
from d3rlpy.models.optimizers import OptimizerFactory
from d3rlpy.models.torch import Policy
from d3rlpy.preprocessing import ActionScaler, RewardScaler, Scaler
from d3rlpy.torch_utility import TorchMiniBatch, soft_sync, torch_api, train_api
from d3rlpy.algos.base import AlgoBase
from d3rlpy.algos.torch.base import TorchImplBase
from myd3rlpy.models.builder import create_phi, create_psi
from myd3rlpy.models.torch.siamese import Phi, Psi


class SiameseImpl(TorchImplBase):

    _phi_learning_rate: float
    _psi_learning_rate: float
    _phi_optim_factory: OptimizerFactory
    _psi_optim_factory: OptimizerFactory
    _phi_encoder_factory: EncoderFactory
    _psi_encoder_factory: EncoderFactory
    _gamma: float
    _n_sample_actions: int
    _use_gpu: Optional[Device]
    _policy: Optional[Policy]
    _phi_optim: Optional[Optimizer]
    _psi_optim: Optional[Optimizer]

    def __init__(
        self,
        policy: AlgoBase,
        *,
        observation_shape: Sequence[int],
        action_size: int,
        phi_learning_rate: float,
        psi_learning_rate: float,
        phi_optim_factory: OptimizerFactory,
        psi_optim_factory: OptimizerFactory,
        phi_encoder_factory: EncoderFactory,
        psi_encoder_factory: EncoderFactory,
        gamma: float,
        n_sample_actions: int,
        use_gpu: Optional[Device],
        scaler: Optional[Scaler],
        action_scaler: Optional[ActionScaler],
        reward_scaler: Optional[RewardScaler],
    ):
        super().__init__(
            observation_shape=observation_shape,
            action_size=action_size,
            scaler=scaler,
            action_scaler=action_scaler,
            reward_scaler=reward_scaler,
        )
        self._policy = policy
        self._phi_learning_rate = phi_learning_rate
        self._psi_learning_rate = psi_learning_rate
        self._phi_optim_factory = phi_optim_factory
        self._psi_optim_factory = psi_optim_factory
        self._phi_encoder_factory = phi_encoder_factory
        self._psi_encoder_factory = psi_encoder_factory
        self._gamma = gamma
        self._n_sample_actions = n_sample_actions
        self._use_gpu = use_gpu

        # initialized in build
        self._phi_optim = None
        self._psi_optim = None

    def build(self) -> None:
        # setup torch models
        self._build_phi()
        self._build_psi()

        # setup optimizer after the parameters move to GPU
        self._build_phi_optim()
        self._build_psi_optim()

        if self._use_gpu:
            self.to_gpu(self._use_gpu)
        else:
            self.to_cpu()

    def _build_phi(self) -> None:
        self._phi = create_phi(
            self._observation_shape,
            self._action_size,
            self._phi_encoder_factory,
        )

    def _build_phi_optim(self) -> None:
        assert self._phi is not None
        self._phi_optim = self._phi_optim_factory.create(
            self._phi.parameters(), lr=self._phi_learning_rate
        )

    def _build_psi(self) -> None:
        self._psi = create_psi(
            self._observation_shape,
            self._phi_encoder_factory,
        )

    def _build_psi_optim(self) -> None:
        assert self._psi is not None
        self._psi_optim = self._psi_optim_factory.create(
            self._psi.parameters(), lr=self._psi_learning_rate
        )

    @train_api
    @torch_api()
    def update_phi(self, batch1: TorchMiniBatch, batch2: TorchMiniBatch) -> np.ndarray:
        assert self._phi_optim is not None

        self._phi.train()
        self._psi.eval()
        self._phi_optim.zero_grad()

        loss = self.compute_phi_loss(batch1, batch2)

        loss.backward()
        self._phi_optim.step()

        self._psi.train()

        return loss.cpu().detach().numpy()

    def compute_phi_loss(
        self, batch1: TorchMiniBatch, batch2: TorchMiniBatch
    ) -> torch.Tensor:
        assert self._phi is not None
        assert self._psi is not None
        s1, a1, r1, sp1 = batch1.observations, batch1.actions, batch1.rewards, batch1.next_observations
        s2, a2, r2, sp2 = batch2.observations, batch2.actions, batch2.rewards, batch2.next_observations
        loss_phi = torch.norm(self._phi(s1, a1) - self._phi(s2, a2), dim=1) + torch.abs(r1 - r2) + self._gamma * torch.norm(self._psi(sp1) - self._psi(sp2), dim=1)
        loss_phi = torch.mean(loss_phi)
        return loss_phi

    @train_api
    @torch_api()
    def update_psi(self, batch1: TorchMiniBatch, batch2: TorchMiniBatch) -> np.ndarray:
        assert self._psi_optim is not None

        self._phi.eval()
        self._psi.train()
        self._psi_optim.zero_grad()

        loss = self.compute_psi_loss(batch1, batch2)

        loss.backward()
        self._psi_optim.step()

        self._phi.train()

        return loss.cpu().detach().numpy()

    def compute_psi_loss(self, batch1: TorchMiniBatch, batch2: TorchMiniBatch) -> torch.Tensor:
        assert self._phi is not None
        assert self._psi is not None
        assert self._policy is not None
        s1 = batch1.observations
        s2 = batch2.observations
        loss_psi = torch.norm(self._psi(s1) - self._psi(s2), dim=1)
        loss_psi_u = 0
        for _ in range(self._n_sample_actions):
            u1 = self._policy._impl._sample_action(s1)
            u2 = self._policy._impl._sample_action(s2)
            loss_psi_u += torch.norm(self._phi(s1, u1) - self._phi(s2, u2), dim=1)
        loss_psi_u /= self._n_sample_actions
        loss_psi -= loss_psi_u
        return torch.mean(loss_psi)

    def _predict_best_action(self, x: List[Any]) -> torch.Tensor:
        state1, action1, state2, action2 = x[0], x[1], x[2], x[3]
        assert self._phi is not None
        return torch.norm(self._phi(state1, action1) - self._phi(state2, action2), dim=1)

    def _sample_action(self, x: torch.Tensor) -> torch.Tensor:
        assert self._policy is not None
        return self._policy.sample(x)

    @property
    def phi(self) -> Phi:
        assert self._phi
        return self._phi

    @property
    def psi(self) -> Psi:
        assert self._psi
        return self._psi

    @property
    def policy(self) -> Policy:
        assert self._policy
        return self._policy

    @property
    def phi_optim(self) -> Optimizer:
        assert self._phi_optim
        return self._phi_optim

    @property
    def psi_optim(self) -> Optimizer:
        assert self._psi_optim
        return self._psi_optim

    def predict_value(
        self, x: np.ndarray, action: np.ndarray, with_std: bool
    ) -> np.ndarray:
        raise NotImplementedError("BC does not support value estimation")
