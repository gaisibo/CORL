import copy
from typing import Optional, Sequence, List, Any, Tuple, Dict, Union
import quadprog

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

from myd3rlpy.algos.torch.mi_impl import MIImpl
from myd3rlpy.algos.torch.gemco_impl import store_grad, overwrite_grad, project2cone2


class GEMMIImpl(MIImpl):
    @train_api
    def update_critic(self, batch: TransitionMiniBatch, replay_batches: Optional[Dict[int, List[torch.Tensor]]]=None) -> np.ndarray:
        batch = TorchMiniBatch(
            batch,
            self.device,
            scaler=self.scaler,
            action_scaler=self.action_scaler,
            reward_scaler=self.reward_scaler,
        )
        assert self._critic_optim is not None
        self._q_func.train()
        self._policy.eval()

        if replay_batches is not None:
            for i, replay_batch in replay_batches.items():
                self._critic_optim.zero_grad()
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
                store_grad(self._q_func.parameters, self._critic_grads_cs[i], self._critic_grad_dims)

        self._critic_optim.zero_grad()
        q_tpn, action = self.compute_target(batch)

        loss = self.compute_critic_loss(batch, q_tpn, action)
        # check if gradient violates buffer constraints
        if replay_batches is not None:
            # copy gradient
            store_grad(self._q_func.parameters, self._critic_grads_da, self._critic_grad_dims)
            dot_prod = torch.mm(self._critic_grads_da.unsqueeze(0),
                            torch.stack(self._critic_grads_cs).T)
            if (dot_prod < 0).sum() != 0:
                project2cone2(self._critic_grads_da.unsqueeze(1),
                              torch.stack(self._critic_grads_cs).T, margin=self.args.gamma)
                # copy gradients back
                overwrite_grad(self._q_func.parameters, self._critic_grads_da,
                               self._critic_grad_dims)
        loss.backward()
        self._critic_optim.step()

        return loss.cpu().detach().numpy()

    @train_api
    def update_actor(self, batch: TransitionMiniBatch, replay_batches: Optional[Dict[int, List[torch.Tensor]]]=None) -> np.ndarray:
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
        self._q_func.eval()
        self._policy.train()

        if replay_batches is not None:
            for i, replay_batch in replay_batches.items():
                with torch.no_grad():
                    replay_observations, _, replay_means, replay_stddevs, _ = replay_batches[i]
                    replay_observations = replay_observations.to(self.device)
                    replay_means = replay_means.to(self.device)
                    replay_stddevs = replay_stddevs.to(self.device)
                dist = self._policy.dist(replay_observations)
                dist_ = torch.distributions.normal.Normal(replay_means, replay_stddevs)
                loss = self._replay_actor_alpha * torch.distributions.kl.kl_divergence(dist_, dist) / len(replay_batches)
                loss.backward()
                store_grad(self._policy.parameters, self._actor_grads_cs[i], self._actor_grad_dims)

        self._actor_optim.zero_grad()

        loss = self.compute_actor_loss(batch)
        loss.backward()

        # check if gradient violates buffer constraints
        if replay_batches is not None():
            # copy gradient
            store_grad(self._policy.parameters, self._actor_grads_da, self._actor_grad_dims)
            dot_prod = torch.mm(self._actor_grads_da.unsqueeze(0),
                            torch.stack(self._actor_grads_cs).T)
            if (dot_prod < 0).sum() != 0:
                project2cone2(self._actor_grads_da.unsqueeze(1),
                              torch.stack(self._actor_grads_cs).T, margin=self.args.gamma)
                # copy gradients back
                overwrite_grad(self._policy.parameters, self._actor_grads_da,
                               self._actor_grad_dims)
        self._actor_optim.step()

        return loss.cpu().detach().numpy()
