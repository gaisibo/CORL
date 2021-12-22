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

from myd3rlpy.algos.torch.ewcmi_impl import EWCMIImpl

class RWalkMIImpl(EWCMIImpl):
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
        damping=0.1,
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
            critic_lamb = critic_lamb,
            actor_lamb = actor_lamb,
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
        # Page 7: "task-specific parameter importance over the entire training trajectory."
        self._critic_w = {n: torch.zeros(p.shape).to(self.device) for n, p in self._q_func.named_parameters() if p.requires_grad}
        self._critic_scores = {n: torch.zeros(p.shape).to(self.device) for n, p in self._q_func.named_parameters() if p.requires_grad}

        # Page 7: "task-specific parameter importance over the entire training trajectory."
        self._actor_w = {n: torch.zeros(p.shape).to(self.device) for n, p in self._policy.named_parameters() if p.requires_grad}
        self._actor_scores = {n: torch.zeros(p.shape).to(self.device) for n, p in self._policy.named_parameters() if p.requires_grad}

    @train_api
    def update_critic(self, batch: TransitionMiniBatch, replay_batches: Optional[Dict[int, List[torch.Tensor]]]=None) -> np.ndarray:
        assert all_data is not None
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

        curr_feat_ext = {n: p.clone().detach() for n, p in self._q_func.named_parameters() if p.requires_grad}

        q_tpn, action = self.compute_target(batch)

        loss = self.compute_critic_loss(batch, q_tpn, action)

        self._critic_optim.zero_grad()
        loss.backward(retain_graph=True)
        # store gradients without regularization term
        unreg_grads = {n: p.grad.clone().detach() for n, p in self._q_func.named_parameters()
                       if p.grad is not None}

        self._critic_optim.zero_grad()
        loss_reg = 0
        # Eq. 3: elastic weight consolidation quadratic penalty
        for n, p in self._q_func.named_parameters():
            if n in self._critic_fisher.keys():
                loss_reg += torch.sum((self._critic_fisher[n] + self._critic_scores[n]) * (p - self._critic_older_params[n]).pow(2)) / 2
        loss += self._critic_lamb * loss_reg

        self._critic_optim.zero_grad()
        loss.backward()
        self._critic_optim.step()

        with torch.no_grad():
            for n, p in self._q_func.named_parameters():
                if n in unreg_grads.keys():
                    self._critic_w[n] -= unreg_grads[n] * (p.detach() - curr_feat_ext[n])

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

        curr_feat_ext = {n: p.clone().detach() for n, p in self._policy.named_parameters() if p.requires_grad}

        q_tpn, action = self.compute_target(batch)

        loss = self.compute_actor_loss(batch)

        self._actor_optim.zero_grad()
        loss.backward(retain_graph=True)
        # store gradients without regularization term
        unreg_grads = {n: p.grad.clone().detach() for n, p in self._policy.named_parameters()
                       if p.grad is not None}

        self._actor_optim.zero_grad()
        loss_reg = 0
        # Eq. 3: elastic weight consolidation quadratic penalty
        for n, p in self._policy.named_parameters():
            if n in self._actor_fisher.keys():
                loss_reg += torch.sum((self._actor_fisher[n] + self._actor_scores[n]) * (p - self._actor_older_params[n]).pow(2)) / 2
        loss += self._actor_lamb * loss_reg

        self._actor_optim.zero_grad()
        loss.backward()
        self._actor_optim.step()

        with torch.no_grad():
            for n, p in self._policy.named_parameters():
                if n in unreg_grads.keys():
                    self._actor_w[n] -= unreg_grads[n] * (p.detach() - curr_feat_ext[n])

        return loss.cpu().detach().numpy()

    def post_train_process(self, replay_dataloader, alpha=0.5):

        # calculate Fisher information
        curr_fisher = self.compute_fisher_matrix_diag(replay_dataloader, self._q_func, self._critic_optim)
        # merge fisher information, we do not want to keep fisher information for each task in memory
        for n in self._critic_fisher.keys():
            # Added option to accumulate fisher over time with a pre-fixed growing alpha
            self._critic_fisher[n] = (alpha * self._critic_fisher[n] + (1 - alpha) * curr_fisher[n])
        # Page 7: Optimization Path-based Parameter Importance: importance scores computation
        curr_score = {n: torch.zeros(p.shape).to(self.device) for n, p in self._q_func.named_parameters()
                      if p.requires_grad}
        with torch.no_grad():
            curr_params = {n: p for n, p in self._q_func.named_parameters() if p.requires_grad}
            for n, p in self._critic_scores.items():
                curr_score[n] = self._critic_w[n] / (
                        self._critic_fisher[n] * ((curr_params[n] - self._critic_older_params[n]) ** 2) + self.damping)
                self._critic_w[n].zero_()
                # Page 7: "Since we care about positive influence of the parameters, negative scores are set to zero."
                curr_score[n] = torch.nn.functional.relu(curr_score[n])
        # Page 8: alleviating regularization getting increasingly rigid by averaging scores
        for n, p in self._critic_scores.items():
            self._critic_scores[n] = (self._critic_scores[n] + curr_score[n]) / 2

        # calculate Fisher information
        curr_fisher = self.compute_fisher_matrix_diag(replay_dataloader, self._policy, self._actor_optim)
        # merge fisher information, we do not want to keep fisher information for each task in memory
        for n in self._actor_fisher.keys():
            # Added option to accumulate fisher over time with a pre-fixed growing alpha
            self._actor_fisher[n] = (alpha * self._actor_fisher[n] + (1 - alpha) * curr_fisher[n])
        # Page 7: Optimization Path-based Parameter Importance: importance scores computation
        curr_score = {n: torch.zeros(p.shape).to(self.device) for n, p in self._policy.named_parameters()
                      if p.requires_grad}
        with torch.no_grad():
            curr_params = {n: p for n, p in self._policy.named_parameters() if p.requires_grad}
            for n, p in self._actor_scores.items():
                curr_score[n] = self._actor_w[n] / (
                        self._actor_fisher[n] * ((curr_params[n] - self._actor_older_params[n]) ** 2) + self.damping)
                self._actor_w[n].zero_()
                # Page 7: "Since we care about positive influence of the parameters, negative scores are set to zero."
                curr_score[n] = torch.nn.functional.relu(curr_score[n])
        # Page 8: alleviating regularization getting increasingly rigid by averaging scores
        for n, p in self._actor_scores.items():
            self._actor_scores[n] = (self._actor_scores[n] + curr_score[n]) / 2
