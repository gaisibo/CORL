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

from myd3rlpy.algos.torch.co_impl import COImpl

class EWCCOImpl(COImpl):
    _n_sample_actions: int
    def __init__(
        self,
        observation_shape: Sequence[int],
        action_size: int,
        actor_learning_rate: float,
        critic_learning_rate: float,
        phi_learning_rate: float,
        psi_learning_rate: float,
        actor_optim_factory: OptimizerFactory,
        critic_optim_factory: OptimizerFactory,
        phi_optim_factory: OptimizerFactory,
        psi_optim_factory: OptimizerFactory,
        actor_encoder_factory: EncoderFactory,
        critic_encoder_factory: EncoderFactory,
        critic_lamb: float,
        actor_lamb: float,
        q_func_factory: QFunctionFactory,
        replay_actor_alpha: float,
        replay_critic_alpha: float,
        replay_critic: bool,
        replay_phi: bool,
        replay_psi: bool,
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
            phi_learning_rate = phi_learning_rate,
            psi_learning_rate = psi_learning_rate,
            actor_optim_factory = actor_optim_factory,
            critic_optim_factory = critic_optim_factory,
            phi_optim_factory = phi_optim_factory,
            psi_optim_factory = psi_optim_factory,
            actor_encoder_factory = actor_encoder_factory,
            critic_encoder_factory = critic_encoder_factory,
            q_func_factory = q_func_factory,
            replay_actor_alpha = replay_actor_alpha,
            replay_critic_alpha = replay_critic_alpha,
            replay_critic = replay_critic,
            replay_phi = replay_phi,
            replay_psi = replay_psi,
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

    def build(self) -> None:

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
        if replay_batches is not None:
            loss_reg = 0
            # Eq. 3: elastic weight consolidation quadratic penalty
            for n, p in self._policy.named_parameters():
                if n in self._actor_fisher.keys():
                    loss_reg += torch.sum(self._actor_fisher[n] * (p - self._actor_older_params[n]).pow(2)) / 2
            loss += self._actor_lamb * loss_reg

        loss.backward()
        self._actor_optim.step()

        return loss.cpu().detach().numpy()

    def compute_fisher_matrix_diag(self, replay_dataloader, network, optimizer, sampling_type='max_pred'):
        # Store Fisher Information
        fisher = {n: torch.zeros(p.shape).to(self.device) for n, p in network.named_parameters()
                  if p.requires_grad}
        # Do forward and backward pass to compute the fisher information
        network.train()
        for images, targets in replay_dataloader:
            outputs = network.forward(images.to(self.device))

            if sampling_type == 'true':
                # Use the labels to compute the gradients based on the CE-loss with the ground truth
                preds = targets.to(self.device)
            elif sampling_type == 'max_pred':
                # Not use labels and compute the gradients related to the prediction the model has learned
                preds = torch.cat(outputs, dim=1).argmax(1).flatten()
            elif sampling_type == 'multinomial':
                # Use a multinomial sampling to compute the gradients
                probs = torch.nn.functional.softmax(torch.cat(outputs, dim=1), dim=1)
                preds = torch.multinomial(probs, len(targets)).flatten()

            loss = torch.nn.functional.cross_entropy(torch.cat(outputs, dim=1), preds)
            optimizer.zero_grad()
            loss.backward()
            # Accumulate all gradients from loss with regularization
            for n, p in network.named_parameters():
                if p.grad is not None:
                    fisher[n] += p.grad.pow(2) * len(targets)
        # Apply mean across all samples
        fisher = {n: (p / len(replay_dataloader)) for n, p in fisher.items()}
        return fisher

    def post_train_process(self, replay_dataloader, alpha=0.5):

        # calculate Fisher information
        curr_fisher = self.compute_fisher_matrix_diag(replay_dataloader, self._q_func, self._critic_optim)
        # merge fisher information, we do not want to keep fisher information for each task in memory
        for n in self._critic_fisher.keys():
            # Added option to accumulate fisher over time with a pre-fixed growing alpha
            self._critic_fisher[n] = (alpha * self._critic_fisher[n] + (1 - alpha) * curr_fisher[n])

        # calculate Fisher information
        curr_fisher = self.compute_fisher_matrix_diag(replay_dataloader, self._policy, self._actor_optim)
        # merge fisher information, we do not want to keep fisher information for each task in memory
        for n in self._actor_fisher.keys():
            # Added option to accumulate fisher over time with a pre-fixed growing alpha
            self._actor_fisher[n] = (alpha * self._actor_fisher[n] + (1 - alpha) * curr_fisher[n])
