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

from myd3rlpy.models.torch.siamese import Phi, Psi
from myd3rlpy.algos.torch.co_impl import COImpl
from myd3rlpy.siamese_similar import similar_euclid, similar_psi, similar_phi


def store_grad(params, grads, grad_dims):
    """
        This stores parameter gradients of past tasks.
        pp: parameters
        grads: gradients
        grad_dims: list with number of parameters per layers
    """
    # store the gradients
    grads.fill_(0.0)
    count = 0
    for param in params():
        if param.grad is not None:
            begin = 0 if count == 0 else sum(grad_dims[:count])
            end = np.sum(grad_dims[:count + 1])
            grads[begin: end].copy_(param.grad.data.view(-1))
        count += 1


def overwrite_grad(params, newgrad, grad_dims):
    """
        This is used to overwrite the gradients with a new gradient
        vector, whenever violations occur.
        pp: parameters
        newgrad: corrected gradient
        grad_dims: list storing number of parameters at each layer
    """
    count = 0
    for param in params():
        if param.grad is not None:
            begin = 0 if count == 0 else sum(grad_dims[:count])
            end = sum(grad_dims[:count + 1])
            this_grad = newgrad[begin: end].contiguous().view(
                param.grad.data.size())
            param.grad.data.copy_(this_grad)
        count += 1


def project2cone2(gradient, memories, margin=0.5, eps=1e-3):
    """
        Solves the GEM dual QP described in the paper given a proposed
        gradient "gradient", and a memory of task gradients "memories".
        Overwrites "gradient" with the final projected update.

        input:  gradient, p-vector
        input:  memories, (t * p)-vector
        output: x, p-vector
    """
    memories_np = memories.cpu().t().double().numpy()
    gradient_np = gradient.cpu().contiguous().view(-1).double().numpy()
    n_rows = memories_np.shape[0]
    self_prod = np.dot(memories_np, memories_np.transpose())
    self_prod = 0.5 * (self_prod + self_prod.transpose()) + np.eye(n_rows) * eps
    grad_prod = np.dot(memories_np, gradient_np) * -1
    G = np.eye(n_rows)
    h = np.zeros(n_rows) + margin
    v = quadprog.solve_qp(self_prod, grad_prod, G, h)[0]
    x = np.dot(v, memories_np) + gradient_np
    gradient.copy_(torch.from_numpy(x).view(-1, 1))

class GEMCOImpl(COImpl):

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

        loss = self.compute_critic_loss(batch, q_tpn, all_data, action)
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

        if replay_batches is not None:
            for i, replay_batch in replay_batches.items():
                with torch.no_grad():
                    replay_observations, _, replay_means, replay_stddevs, _ = replay_batches[i]
                    replay_observations = replay_observations.to(self.device)
                    replay_means = replay_means.to(self.device)
                    replay_stddevs = replay_stddevs.to(self.device)
                dist = self._policy.dist(replay_observations)
                dist_ = torch.distributions.normal.Normal(replay_means, replay_stddevs)
                loss = self._replay_actor_alpha * torch.distributions.kl.kl_divergence(dist_, dist).mean() / len(replay_batches)
                loss.backward()
                store_grad(self._policy.parameters, self._actor_grads_cs[i], self._actor_grad_dims)

        self._actor_optim.zero_grad()

        loss = self.compute_actor_loss(batch, all_data)
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
