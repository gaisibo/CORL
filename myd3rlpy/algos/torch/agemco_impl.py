import copy
from typing import Optional, Sequence, List, Any, Tuple, Dict, Union
import random

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
from d3rlpy.models.builders import create_squashed_normal_policy, create_continuous_q_function

from myd3rlpy.models.torch.siamese import Phi, Psi
from myd3rlpy.algos.torch.co_impl import COImpl
from myd3rlpy.siamese_similar import similar_euclid, similar_psi, similar_phi
from myd3rlpy.algos.torch.gemco_impl import overwrite_grad, store_grad

def project(gxy: torch.Tensor, ger: torch.Tensor) -> torch.Tensor:
    corr = torch.dot(gxy, ger) / torch.dot(ger, ger)
    return gxy - corr * ger

class AGEMCOImpl(COImpl):
    @train_api
    def update_critic(self, batch: TransitionMiniBatch, replay_batches: Optional[Dict[int, List[torch.Tensor]]]=None, all_data: MDPDataset=None) -> np.ndarray:
        assert all_data is not None
        batch = TorchMiniBatch(
            batch,
            self.device,
            scaler=self.scaler,
            action_scaler=self.action_scaler,
            reward_scaler=self.reward_scaler,
        )
        assert self._critic_optim is not None
        self._phi.eval()
        self._psi.eval()
        self._q_func.train()
        self._policy.eval()

        self._critic_optim.zero_grad()

        q_tpn, action = self.compute_target(batch)

        loss = self.compute_critic_loss(batch, q_tpn, all_data, action)
        loss.backward()
        if replay_batches is not None:
            store_grad(self._q_func.parameters, self._critic_grad_xy, self._critic_grad_dims)
            key = random.sample(replay_batches.keys(), 1)
            with torch.no_grad():
                replay_observations, replay_actionss, _, _, replay_qss = replay_batches[key]
                replay_observations = replay_observations.to(self.device)
                replay_actionss = replay_actionss.to(self.device)
                replay_qss = replay_qss.to(self.device)
            loss = 0
            for action_sample_num in range(replay_actionss.shape[1]):
                q = self._q_func(replay_observations, replay_actionss[:, action_sample_num, :])
                loss += self._replay_critic_alpha * F.mse_loss(replay_qss[:, action_sample_num], q) / len(replay_batches)
            loss.backward()
            store_grad(self._q_func.parameters, self._critic_grad_er, self._critic_grad_dims)
            dot_prod = torch.dot(self.grad_xy, self.grad_er)
            if dot_prod.item() < 0:
                g_tilde = project(gxy=self.grad_xy, ger=self.grad_er)
                overwrite_grad(self._q_func.parameters, g_tilde, self.grad_dims)
            else:
                overwrite_grad(self._q_func.parameters, self.grad_xy, self.grad_dims)

        self._critic_optim.step()

        return loss.cpu().detach().numpy()
    @train_api
    def update_actor(self, batch: TransitionMiniBatch, replay_batches: Optional[Dict[int, List[torch.Tensor]]]=None, all_data: MDPDataset=None) -> np.ndarray:
        assert all_data is not None
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
        self._phi.eval()
        self._psi.eval()
        self._q_func.eval()
        self._policy.train()

        self._actor_optim.zero_grad()

        loss = self.compute_actor_loss(batch, all_data)
        loss.backward()
        if replay_batches is not None:
            store_grad(self._policy.parameters, self._actor_grad_xy, self._actor_grad_dims)
            i = random.sample(replay_batches.keys(), 1)
            with torch.no_grad():
                replay_observations, _, replay_means, replay_stddevs, _ = replay_batches[i]
                replay_observations = replay_observations.to(self.device)
                replay_means = replay_means.to(self.device)
                replay_stddevs = replay_stddevs.to(self.device)
            dist = self._policy.dist(replay_observations)
            dist_ = torch.distributions.normal.Normal(replay_means, replay_stddevs)
            loss = self._replay_actor_alpha * torch.distributions.kl.kl_divergence(dist_, dist).mean() / len(replay_batches)
            store_grad(self._policy.parameters, self._actor_grad_er, self._actor_grad_dims)
            dot_prod = torch.dot(self._actor_grad_xy, self._actor_grad_er)
            if dot_prod.item() < 0:
                g_tilde = project(gxy=self._actor_grad_xy, ger=self._actor_grad_er)
                overwrite_grad(self._policy.parameters, g_tilde, self._actor_grad_dims)
            else:
                overwrite_grad(self._policy.parameters, self._actor_grad_xy, self._actor_grad_dims)

        self._actor_optim.step()

        return loss.cpu().detach().numpy()
