import copy
from typing import Optional, Sequence, List, Any, Tuple, Dict

import numpy as np
import torch
from torch.optim import Optimizer
import torch.nn.functional as F
import quadprog

from d3rlpy.gpu import Device
from d3rlpy.models.encoders import EncoderFactory
from d3rlpy.models.optimizers import OptimizerFactory
from d3rlpy.models.torch import Policy
from d3rlpy.models.q_functions import QFunctionFactory
from d3rlpy.preprocessing import ActionScaler, RewardScaler, Scaler
from d3rlpy.torch_utility import TorchMiniBatch, soft_sync, torch_api, train_api
from d3rlpy.dataset import MDPDataset
from d3rlpy.algos.base import AlgoBase
from d3rlpy.algos.torch.base import TorchImplBase
from d3rlpy.algos.torch.td3_plus_bc_impl import TD3Impl

from myd3rlpy.models.torch.siamese import Phi, Psi
from myd3rlpy.models.builder import create_squashed_normal_policy, create_continuous_q_function
from utils.siamese_similar import similar_euclid, similar_psi, similar_phi

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

class GEMMIImpl(TD3Impl):
    _n_sample_actions: int
    _task_id_size: int
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
        task_id_size: int,
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
            gamma = gamma,
            tau = tau,
            n_critics = n_critics,
            target_reduction_type = target_reduction_type,
            target_smoothing_sigma = target_smoothing_sigma,
            target_smoothing_clip = target_smoothing_clip,
            use_gpu = use_gpu,
            scaler = scaler,
            action_scaler = action_scaler,
            reward_scaler = reward_scaler,
        )
        self._replay_actor_alpha = replay_actor_alpha
        self._replay_critic_alpha = replay_critic_alpha
        self._replay_critic = replay_critic
        self._n_sample_actions = n_sample_actions
        self._task_id_size = task_id_size

        self._actor_grad_dims = []
        for pp in self._policy.parameters():
            self._actor_dims.append(pp.data.numel())
        self._actor_grads_cs = []
        self._actor_grads_da = torch.zeros(np.sum(self._actor_grad_dims)).to(self.device)

        self._critic_grad_dims = []
        for pp in self._q_func.parameters():
            self._critic_dims.append(pp.data.numel())
        self._critic_grads_cs = []
        self._critic_grads_da = torch.zeros(np.sum(self._critic_grad_dims)).to(self.device)

    def build(self) -> None:

        # 共用encoder
        # 在共用encoder的情况下replay_不起作用。
        self._q_func = create_continuous_q_function(
            observation_shape=self._observation_shape,
            action_size=self._action_size,
            encoder_factory=self._critic_encoder_factory,
            q_func_factory=self._q_func_factory,
            n_ensembles=self._n_critics,
            task_id_size=self._task_id_size,
        )
        self._policy = create_squashed_normal_policy(
            observation_shape=self._observation_shape,
            action_size=self._action_size,
            task_id_size=self._task_id_size,
            encoder_factory=self._actor_encoder_factory,
        )

        if self._use_gpu:
            self.to_gpu(self._use_gpu)
        else:
            self.to_cpu()

        # setup optimizer after the parameters move to GPU
        self._critic_optim = self._critic_optim_factory.create(
            self._q_func.parameters(), lr=self._critic_learning_rate
        )
        self._actor_optim = self._actor_optim_factory.create(
            self._policy.parameters(), lr=self._actor_learning_rate
        )

    def get_task_id_tensor(self, batch: TorchMiniBatch, task_id: int):
        task_id_tensor = F.one_hot(torch.full([batch._observations.shape[0]], task_id, dtype=torch.int64), num_classes=self._task_id_size).to(batch._observations.dtype).to(batch._observations.device)
        return task_id_tensor

    def get_task_id_tensor_replay(self, observations: torch.Tensor, task_id: int):
        task_id_tensor = F.one_hot(torch.full([observations.shape[0]], task_id, dtype=torch.int64), num_classes = self._task_id_size).to(observations.dtype).to(observations.device)
        return task_id_tensor

    @train_api
    def update_critic(self, batch: TransitionMiniBatch, task_id: int, replay_batches: Optional[Dict[int, List[torch.Tensor]]]) -> np.ndarray:
        assert self._critic_optim is not None
        batch = TorchMiniBatch(
            batch,
            self.device,
            scaler=self.scaler,
            action_scaler=self.action_scaler,
            reward_scaler=self.reward_scaler,
        )

        task_id = self.get_task_id_tensor(batch, task_id)
        if replay_batches is not None:
            for i, replay_batch in replay_batches.items():
                replay_observations, replay_actionss, replay_means, replay_stddevs, replay_qss = replay_batch
                i_tensor = self.get_task_id_tensor_replay(replay_observations, i)
                loss = 0
                for action_sample_num in range(replay_actionss.shape[1]):
                    q = self._q_func(replay_observations, replay_actionss[:, action_sample_num, :], i_tensor)
                    loss += self._replay_critic_alpha * F.mse_loss(replay_qss[:, action_sample_num], q) / len(replay_batches)
                loss.backward()
                store_grad(self._q_func.parameters, self._critic_grads_cs[i], self._critic_grad_dims)


        self._critic_optim.zero_grad()

        q_tpn, clip_action = self.compute_target(batch, task_id)

        loss = self.compute_critic_loss(batch, task_id, q_tpn, clip_action)

        loss.backward()


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

        self._critic_optim.step()

        return loss.cpu().detach().numpy()

    def compute_target(self, batch: TorchMiniBatch, task_id: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        assert self._targ_policy is not None
        assert self._targ_q_func is not None
        with torch.no_grad():
            action, _ = self._targ_policy.sample_with_log_prob(batch.next_observations, task_id)
            # smoothing target
            noise = torch.randn(action.shape, device=batch.device)
            scaled_noise = self._target_smoothing_sigma * noise
            clipped_noise = scaled_noise.clamp(
                -self._target_smoothing_clip, self._target_smoothing_clip
            )
            smoothed_action = action + clipped_noise
            clipped_action = smoothed_action.clamp(-1.0, 1.0)
            return self._targ_q_func.compute_target(
                x = batch.next_observations,
                task_id = task_id,
                action = clipped_action,
                reduction=self._target_reduction_type,
            ), clipped_action

    def compute_critic_loss(self, batch: TorchMiniBatch, task_id: torch.Tensor, q_tpn: torch.Tensor, action: torch.Tensor, alpha_c=10, beta=0.5) -> torch.Tensor:
        assert self._q_func is not None
        return self._q_func.compute_error(
            obs_t=batch.observations,
            act_t=batch.actions[:, :self.action_size],
            tid_t=task_id,
            input_indexes=batch.actions[:, self.action_size:],
            rew_tp1=batch.next_rewards,
            q_tp1=q_tpn,
            ter_tp1=batch.terminals,
            gamma=self._gamma ** batch.n_steps,
            use_independent_target=self._target_reduction_type == "none",
            masks=batch.masks,
        )

    @train_api
    def update_actor(self, batch: TransitionMiniBatch, task_id: int, replay_batches: Optional[Dict[str, List[torch.Tensor]]]) -> np.ndarray:
        assert self._q_func is not None
        assert self._actor_optim is not None
        batch = TorchMiniBatch(
            batch,
            self.device,
            scaler=self.scaler,
            action_scaler=self.action_scaler,
            reward_scaler=self.reward_scaler,
        )
        task_id = self.get_task_id_tensor(batch, task_id)

        # Q function should be inference mode for stability
        self._q_func.eval()

        if replay_batches is not None:
            for i, replay_batch in replay_batches.items():
                self._actor_optim.zero_grad()
                replay_observations, replay_actionss, replay_means, replay_stddevs, replay_qss = replay_batch
                i_tensor = self.get_task_id_tensor_replay(replay_observations, i)
                dist = self._policy.dist(replay_observations, i_tensor)
                dist_ = torch.distributions.normal.Normal(replay_means, replay_stddevs)
                loss = self._replay_actor_alpha * torch.distributions.kl.kl_divergence(dist_, dist) / len(replay_batches)
                loss.backward()
                store_grad(self._actor_parameters, self.grads_cs[i], self.grad_dims)

        self._actor_optim.zero_grad()

        loss = self.compute_actor_loss(batch, task_id)
        loss.backward()

        # check if gradient violates buffer constraints
        if replay_batches is not None:
            # copy gradient
            store_grad(self._policy.parameters, self._actor_grads_da, self._actor_grad_dims)


            dot_prod = torch.mm(self._actor_grads_da.unsqueeze(0),
                            torch.stack(self._actor_grads_cs).T)
            if (dot_prod < 0).sum() != 0:
                project2cone2(self._actor_grads_da.unsqueeze(1),
                              torch.stack(self._actor_grads_cs).T, margin=self.args.gamma)
                # copy gradients back
                overwrite_grad(self.parameters, self._actor_grads_da,
                               self._actor_grad_dims)

        self._actor_optim.step()

        return loss.cpu().detach().numpy()

    def compute_actor_loss(self, batch: TorchMiniBatch, task_id: torch.Tensor, alpha_a=10, beta=0.5) -> torch.Tensor:
        assert self._policy is not None
        assert self._q_func is not None
        q_tpn, _ = self.compute_target(batch, task_id)
        action, _ = self._policy.sample_with_log_prob(batch.observations, task_id)
        q_t = self._q_func(batch.observations, action, task_id, "none")[0]
        lam = self._replay_actor_alpha / (q_t.abs().mean()).detach()
        return lam * -q_t.mean()
