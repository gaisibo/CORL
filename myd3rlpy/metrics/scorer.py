from typing import List, cast
import gym
import numpy as np
from typing_extensions import Protocol
import torch
import torch.nn.functional as F

from d3rlpy.dataset import Episode, TransitionMiniBatch
from d3rlpy.preprocessing.reward_scalers import RewardScaler
from d3rlpy.preprocessing.stack import StackedObservation
from d3rlpy.metrics.scorer import AlgoProtocol, _make_batches

WINDOW_SIZE = 1024


def get_task_id_tensor(observations: torch.Tensor, task_id_int: int, task_id_size: int):
    task_id_tensor = F.one_hot(torch.full([observations.shape[0]], task_id_int, dtype=torch.int64), num_classes=task_id_size).to(observations.dtype).to(observations.device)
    return task_id_tensor

def bc_error_scorer(algo, replay_iterator, real_action_size: int) -> float:
    total_errors = []
    for batch in replay_iterator:
        observations, actionss, means, stddevs, qss = batch
        observations = observations.to(algo._impl.device)
        actionss = actionss.to(algo._impl.device)
        means = means.to(algo._impl.device)
        stddevs = stddevs.to(algo._impl.device)
        qss = qss.to(algo._impl.device)
        dist = torch.distributions.normal.Normal(means, stddevs)
        rebuild_means = algo._impl._policy.dist(observations).mean
        rebuild_stddevs = algo._impl._policy.dist(observations).stddev
        rebuild_dist = torch.distributions.normal.Normal(rebuild_means, rebuild_stddevs)
        rebuild_qss = []
        for sample_time in range(qss.shape[1]):
            print(f'observations: {observations.shape}')
            print(f'actionss: {actionss.shape}')
            print(f'actionss choose: {actionss[:, sample_time, :real_action_size].shape}')
            rebuild_qs = algo._impl._q_func.forward(observations, actionss[:, sample_time, :real_action_size])
            rebuild_qss.append(rebuild_qs)
        replay_qss = torch.stack(rebuild_qss, dim=1)
        loss = F.mse_loss(replay_qss, qss) + torch.distributions.kl.kl_divergence(rebuild_dist, dist)
        total_errors.append(loss)
    total_errors = torch.cat(total_errors, dim=0)
    return float(torch.mean(total_errors).detach().cpu().numpy())

def td_error_scorer(algo: AlgoProtocol, episodes: List[Episode], real_action_size: int) -> float:
    r"""Returns average TD error.
    This metics suggests how Q functions overfit to training sets.
    If the TD error is large, the Q functions are overfitting.
    .. math::
        \mathbb{E}_{s_t, a_t, r_{t+1}, s_{t+1} \sim D}
            [(Q_\theta (s_t, a_t)
             - r_{t+1} - \gamma \max_a Q_\theta (s_{t+1}, a))^2]
    Args:
        algo: algorithm.
        episodes: list of episodes.
    Returns:
        average TD error.
    """
    total_errors = []
    for episode in episodes:
        for batch in _make_batches(episode, WINDOW_SIZE, algo.n_frames):
            # estimate values for current observations
            values = algo.predict_value(batch.observations, batch.actions[:, :real_action_size])

            # estimate values for next observations
            next_actions = algo.predict(batch.next_observations)
            next_values = algo.predict_value(
                batch.next_observations, next_actions
            )

            # calculate td errors
            mask = (1.0 - np.asarray(batch.terminals)).reshape(-1)
            rewards = np.asarray(batch.next_rewards).reshape(-1)
            if algo.reward_scaler:
                rewards = algo.reward_scaler.transform_numpy(rewards)
            y = rewards + algo.gamma * cast(np.ndarray, next_values) * mask
            total_errors += ((values - y) ** 2).tolist()

    return float(np.mean(total_errors))
