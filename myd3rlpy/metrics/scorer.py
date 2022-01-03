from typing import List, cast, Callable, Any, Dict, Tuple
import matplotlib.pyplot as plt
from typing_extensions import Protocol
import gym
import numpy as np
import torch
import torch.nn.functional as F

from d3rlpy.dataset import Episode, TransitionMiniBatch
from d3rlpy.preprocessing.reward_scalers import RewardScaler
from d3rlpy.preprocessing.stack import StackedObservation
from d3rlpy.metrics.scorer import AlgoProtocol, _make_batches

from utils.utils import Struct

WINDOW_SIZE = 1024


replay_name = ['observations', 'actions', 'rewards', 'next_observations', 'next_actions', 'next_rewards', 'replay_terminals', 'policy_actions', 'means', 'std_logs', 'qs', 'phis', 'psis']
def get_task_id_tensor(observations: torch.Tensor, task_id_int: int, task_id_size: int):
    task_id_tensor = F.one_hot(torch.full([observations.shape[0]], task_id_int, dtype=torch.int64), num_classes=task_id_size).to(observations.dtype).to(observations.device)
    return task_id_tensor

def bc_error_scorer(real_action_size: int) -> Callable[..., float]:
    def scorer(algo, replay_iterator):
        total_errors = []
        for batch in replay_iterator:
            batch = dict(zip(replay_name, batch))
            batch = Struct(**batch)
            observations = batch.observations.to(algo._impl.device)
            actions = batch.policy_actions.to(algo._impl.device)
            qs = batch.qs.to(algo._impl.device)
            rebuild_actions = algo._impl._policy(observations)
            rebuild_qs = algo._impl._q_func.forward(observations, actions[:, :real_action_size])
            loss = F.mse_loss(rebuild_qs, qs) + F.mse_loss(rebuild_actions, actions[:, :real_action_size])
            total_errors.append(loss)
        total_errors = torch.stack(total_errors, dim=0)
        return float(torch.mean(total_errors).detach().cpu().numpy())
    return scorer

def td_error_scorer(real_action_size: int) -> Callable[..., Callable[...,float]]:
    def id_scorer(id: int) -> Callable[..., float]:
        def scorer(algo: AlgoProtocol, episodes: List[Episode]) -> float:
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
        return scorer
    return id_scorer

def evaluate_on_environment(
        envs: Dict[int, gym.Env], end_points: List[Tuple[float, float]], task_nums: int, draw_path: str, n_trials: int = 10, epsilon: float = 0.0, render: bool = False,
) -> Callable[..., Callable[..., float]]:

    # for image observation

    def id_scorer(id: int):
        env = envs[id]
        end_point = end_points[id]
        observation_shape = env.observation_space.shape
        is_image = len(observation_shape) == 3
        def scorer(algo: AlgoProtocol, *args: Any) -> float:

            def draw(trajectories: List[List[Tuple[float, float]]]):
                fig = plt.figure()
                for i, trajectory in enumerate(trajectories):
                    x, y = list(map(list, zip(*trajectory)))
                    plt.plot(x, y)
                plt.plot(end_point[0], end_point[1], 'o', markersize=4)
                plt.savefig(draw_path + str(id) + '.png')
                plt.close('all')
            if is_image:
                stacked_observation = StackedObservation(
                    observation_shape, algo.n_frames
                )

            trajectories = []

            episode_rewards = []
            for _ in range(n_trials):
                trajectory = []
                observation = env.reset()
                task_id_numpy = np.eye(task_nums)[id].squeeze()
                observation = np.concatenate([observation, task_id_numpy], axis=0)
                episode_reward = 0.0

                # frame stacking
                if is_image:
                    stacked_observation.clear()
                    stacked_observation.append(observation)

                while True:
                    # take action
                    if np.random.random() < epsilon:
                        action = env.action_space.sample()
                    else:
                        if is_image:
                            action = algo.predict([stacked_observation.eval()])[0]
                        else:
                            action = algo.predict([observation])[0]

                    observation, reward, done, _ = env.step(action)
                    trajectory.append(observation[:2])
                    task_id_numpy = np.eye(task_nums)[id].squeeze()
                    observation = np.concatenate([observation, task_id_numpy], axis=0)
                    episode_reward += reward

                    if is_image:
                        stacked_observation.append(observation)

                    if render:
                        env.render()

                    if done:
                        break
                trajectories.append(trajectory)
                episode_rewards.append(episode_reward)
            draw(trajectories)
            return float(np.mean(episode_rewards))
        return scorer

    return id_scorer

