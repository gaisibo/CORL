import copy
from typing import Optional, Sequence, List, Any, Tuple, Dict, Union

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

class EWCMIImpl(MIImpl):
    _n_sample_actions: int
    def __init__(
        self,
        observation_shape: Sequence[int],
        action_size: int,
        actor_learning_rate: float,
        critic_learning_rate: float,
        actor_optim_factory: OptimizerFactory,
        critic_optim_factory: OptimizerFactory,
        actor_encoder_factory: EncoderFactory,
        critic_encoder_factory: EncoderFactory,
        critic_lamb: float,
        actor_lamb: float,
        q_func_factory: QFunctionFactory,
        replay_actor_alpha: float,
        replay_critic_alpha: float,
        replay_critic: bool,
        gamma: float,
        tau: float,
        n_critics: int,
        target_reduction_type: str,
        target_smoothing_sigma: float,
        target_smoothing_clip: float,
        n_sample_actions: int,
        use_gpu: Optional[Device],
        scaler: Optional[Scaler],
        action_scaler: Optional[ActionScaler],
        reward_scaler: Optional[RewardScaler],
    ):
        super().__init__(
            observation_shape = observation_shape,
            action_size = action_size,
            actor_learning_rate = actor_learning_rate,
            critic_learning_rate = critic_learning_rate,
            actor_optim_factory = actor_optim_factory,
            critic_optim_factory = critic_optim_factory,
            actor_encoder_factory = actor_encoder_factory,
            critic_encoder_factory = critic_encoder_factory,
            q_func_factory = q_func_factory,
            replay_actor_alpha = replay_actor_alpha,
            replay_critic_alpha = replay_critic_alpha,
            replay_critic = replay_critic,
            gamma = gamma,
            tau = tau,
            n_critics = n_critics,
            target_reduction_type = target_reduction_type,
            target_smoothing_sigma = target_smoothing_sigma,
            target_smoothing_clip = target_smoothing_clip,
            n_sample_actions = n_sample_actions,
            use_gpu = use_gpu,
            scaler = scaler,
            action_scaler = action_scaler,
            reward_scaler = reward_scaler,
        )

        self._critic_lamb = critic_lamb
        self._actor_lamb = actor_lamb

    def build(self):
        super().build()

        # Store fisher information weight importance
        self._critic_fisher = {n: torch.zeros(p.shape).to(self.device) for n, p in self._q_func.named_parameters() if p.requires_grad}
        # Store current parameters for the next task
        self._critic_older_params = {n: p.clone().detach() for n, p in self._q_func.named_parameters() if p.requires_grad}

        # Store fisher information weight importance
        self._actor_fisher = {n: torch.zeros(p.shape).to(self.device) for n, p in self._policy.named_parameters() if p.requires_grad}
        # Store current parameters for the next task
        self._actor_older_params = {n: p.clone().detach() for n, p in self._policy.named_parameters() if p.requires_grad}

    @train_api
    @torch_api()
    def update_critic(self, batch: TransitionMiniBatch) -> np.ndarray:
        assert self._critic_optim is not None
        self._q_func.train()
        self._policy.eval()

        self._critic_optim.zero_grad()

        q_tpn, action = self.compute_target(batch)

        loss = self.compute_critic_loss(batch, q_tpn, action)
        loss_reg = 0
        # Eq. 3: elastic weight consolidation quadratic penalty
        for n, p in self._q_func.named_parameters():
            if n in self._critic_fisher.keys():
                loss_reg += torch.sum(self._critic_fisher[n] * (p - self._critic_older_params[n]).pow(2)) / 2
        loss += self._critic_lamb * loss_reg

        loss.backward()
        self._critic_optim.step()

        return loss.cpu().detach().numpy()

    @train_api
    @torch_api()
    def update_actor(self, batch: TransitionMiniBatch) -> np.ndarray:
        assert self._q_func is not None
        assert self._actor_optim is not None

        # Q function should be inference mode for stability
        self._q_func.eval()
        self._policy.train()

        self._actor_optim.zero_grad()

        loss = self.compute_actor_loss(batch)
        loss_reg = 0
        # Eq. 3: elastic weight consolidation quadratic penalty
        for n, p in self._policy.named_parameters():
            if n in self._actor_fisher.keys():
                loss_reg += torch.sum(self._actor_fisher[n] * (p - self._actor_older_params[n]).pow(2)) / 2
        loss += self._actor_lamb * loss_reg

        loss.backward()
        self._actor_optim.step()

        return loss.cpu().detach().numpy()

    def compute_critic_fisher_matrix_diag(self, replay_dataloader):
        # Store Fisher Information
        fisher = {n: torch.zeros(p.shape).to(self.device) for n, p in self._q_func.named_parameters()
                  if p.requires_grad}
        # Do forward and backward pass to compute the fisher information
        self._q_func.train()
        replay_dataloader.reset()
        batch_size = replay_dataloader.batch_size
        for itr in range(len(replay_dataloader)):
            batch = next(replay_dataloader)
            name_list = ['observations', 'actions', 'next_observations', 'next_rewards', 'terminals', 'n_steps', 'masks']
            batch = dict(zip(name_list, batch))
            q_tpn, action = self.compute_target(batch)

            loss = self.compute_critic_loss(batch, q_tpn, action)

            self._critic_optimizer.zero_grad()
            loss.backward()
            # Accumulate all gradients from loss with regularization
            for n, p in self._q_func.named_parameters():
                if p.grad is not None:
                    fisher[n] += p.grad.pow(2) * action.shape[0]
        # Apply mean across all samples
        fisher = {n: (p / (len(replay_dataloader) * batch_size)) for n, p in fisher.items()}
        return fisher

    def compute_actor_fisher_matrix_diag(self, replay_dataloader):
        # Store Fisher Information
        fisher = {n: torch.zeros(p.shape).to(self.device) for n, p in self._policy.named_parameters()
                  if p.requires_grad}
        # Do forward and backward pass to compute the fisher information
        self._policy.train()
        replay_dataloader.reset()
        batch_size = replay_dataloader.batch_size
        for itr in range(len(replay_dataloader)):
            batch = next(replay_dataloader)
            name_list = ['observations', 'actions', 'next_observations', 'next_rewards', 'terminals', 'n_steps', 'masks']
            batch = dict(zip(name_list, batch))

            loss = self.compute_actor_loss(batch)

            self._policy_optimizer.zero_grad()
            loss.backward()
            # Accumulate all gradients from loss with regularization
            for n, p in self._policy.named_parameters():
                if p.grad is not None:
                    fisher[n] += p.grad.pow(2) * batch_size
        # Apply mean across all samples
        fisher = {n: (p / (len(replay_dataloader) * batch_size)) for n, p in fisher.items()}
        return fisher

    def post_train_process(self, replay_dataloader, alpha=0.5):

        # calculate Fisher information
        critic_curr_fisher = self.compute_fisher_matrix_diag(replay_dataloader)
        # merge fisher information, we do not want to keep fisher information for each task in memory
        for n in self._critic_fisher.keys():
            # Added option to accumulate fisher over time with a pre-fixed growing alpha
            self._critic_fisher[n] = (alpha * self._critic_fisher[n] + (1 - alpha) * critic_curr_fisher[n])

        # calculate Fisher information
        actor_curr_fisher = self.compute_fisher_matrix_diag(replay_dataloader)
        # merge fisher information, we do not want to keep fisher information for each task in memory
        for n in self._actor_fisher.keys():
            # Added option to accumulate fisher over time with a pre-fixed growing alpha
            self._actor_fisher[n] = (alpha * self._actor_fisher[n] + (1 - alpha) * actor_curr_fisher[n])
