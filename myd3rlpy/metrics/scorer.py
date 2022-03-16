import pdb
import os
from typing import List, cast, Callable, Any, Dict, Tuple, Optional
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
            means = batch.means.to(algo._impl.device)
            std_logs = batch.std_logs.to(algo._impl.device)
            dists = torch.distributions.normal.Normal(means, torch.exp(std_logs))
            actions = batch.policy_actions.to(algo._impl.device)
            qs = batch.qs.to(algo._impl.device)
            rebuild_dists = algo._impl._policy.dist(observations)
            rebuild_qs = algo._impl._q_func.forward(observations, actions)
            loss = F.mse_loss(rebuild_qs, qs) + torch.distributions.kl.kl_divergence(rebuild_dists, dists)
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

# def evaluate_on_environment(
#          envs: Dict[int, gym.Env], end_points: Optional[List[Tuple[float, float]]], task_nums: int, draw_path: str, n_trials: int = 100, epsilon: float = 0.0, render: bool = False, dense: bool=False
# ) -> Callable[..., Callable[..., float]]:
# 
#     # for image observation
# 
#     def id_scorer(id: int, epoch: int):
#         env = envs[id]
#         if end_points is not None:
#             end_point = end_points[id]
#         else:
#             end_point = None
#         observation_shape = env.observation_space.shape
#         is_image = len(observation_shape) == 3
#         def scorer(algo: AlgoProtocol, *args: Any) -> float:
# 
#             def draw(trajectories: List[List[Tuple[float, float]]]):
#                 fig = plt.figure()
#                 for i, trajectory in enumerate(trajectories):
#                     x, y = list(map(list, zip(*trajectory)))
#                     plt.plot(x, y)
#                 if end_point is not None:
#                     plt.plot(end_point[0], end_point[1], 'o', markersize=4)
#                 plt.savefig(draw_path + '_' + str(id) + '_' + str(epoch) + '.png')
#                 plt.close('all')
#             if is_image:
#                 stacked_observation = StackedObservation(
#                     observation_shape, algo.n_frames
#                 )
# 
#             trajectories = []
# 
#             rewards = []
#             episode_rewards = []
#             for _ in range(n_trials):
#                 trajectory = []
#                 episode_reward = 0
#                 observation = env.reset()
#                 task_id_numpy = np.eye(task_nums)[id].squeeze()
#                 observation = np.concatenate([observation, task_id_numpy], axis=0)
# 
#                 # frame stacking
#                 if is_image:
#                     stacked_observation.clear()
#                     stacked_observation.append(observation)
# 
#                 time = 0
#                 while True:
#                     # take action
#                     if np.random.random() < epsilon:
#                         action = env.action_space.sample()
#                     else:
#                         if is_image:
#                             action = algo.predict([stacked_observation.eval()])[0]
#                         else:
#                             action = algo.predict([observation])[0]
# 
#                     # pdb.set_trace()
#                     observation, reward, done, _ = env.step(action)
#                     # finish = (np.linalg.norm(np.array(observation[:2]) - np.array(env.target_goal)) < 0.1)
#                     # done = done or finish
#                     # if dense:
#                     #     reward = finish
#                     # else:
#                     #     reward = - np.linalg.norm(np.array(observation[:2]) - np.array(env.target_goal))
#                     time += 1
#                     trajectory.append(observation[:2])
#                     task_id_numpy = np.eye(task_nums)[id].squeeze()
#                     observation = np.concatenate([observation, task_id_numpy], axis=0)
#                     episode_reward += reward
# 
#                     if is_image:
#                         stacked_observation.append(observation)
# 
#                     if render:
#                         env.render()
# 
#                     if done:
#                         break
#                 # trajectories.append(trajectory)
#                 episode_rewards.append(episode_reward)
#                 rewards.append(reward)
#             # draw(trajectories)
#             return float(np.mean(rewards))
#         return scorer
# 
#     return id_scorer
# 

def evaluate_on_environment(
    env: gym.Env, test_id: int, n_trials: int = 10, epsilon: float = 0.0, render: bool = False
) -> Callable[..., float]:
    """Returns scorer function of evaluation on environment.
    This function returns scorer function, which is suitable to the standard
    scikit-learn scorer function style.
    The metrics of the scorer function is ideal metrics to evaluate the
    resulted policies.
    .. code-block:: python
        import gym
        from d3rlpy.algos import DQN
        from d3rlpy.metrics.scorer import evaluate_on_environment
        env = gym.make('CartPole-v0')
        scorer = evaluate_on_environment(env)
        cql = CQL()
        mean_episode_return = scorer(cql)
    Args:
        env: gym-styled environment.
        n_trials: the number of trials.
        epsilon: noise factor for epsilon-greedy policy.
        render: flag to render environment.
    Returns:
        scoerer function.
    """

    # for image observation
    observation_shape = env.observation_space.shape
    is_image = len(observation_shape) == 3

    def scorer(algo: AlgoProtocol, *args: Any) -> float:
        print(f"test_id: {test_id}")
        if is_image:
            stacked_observation = StackedObservation(
                observation_shape, algo.n_frames
            )
        save_id = algo._impl._impl_id
        algo._impl.change_task(test_id)

        episode_rewards = []
        for _ in range(n_trials):
            observation = env.reset()
            episode_reward = 0.0

            # frame stacking
            if is_image:
                stacked_observation.clear()
                stacked_observation.append(observation)

            i = 0
            while True:
                # take action
                if np.random.random() < epsilon:
                    action = env.action_space.sample()
                else:
                    if is_image:
                        action = algo.predict([stacked_observation.eval()])[0]
                    else:
                        action = algo.predict([observation])[0]

                observation, reward, done, pos = env.step(action)
                episode_reward += reward

                if is_image:
                    stacked_observation.append(observation)

                if render:
                    env.render()

                if done:
                    break
                if i > 1000:
                    break

                i += 1
            episode_rewards.append(episode_reward)
        algo._impl.change_task(save_id)
        return float(np.mean(episode_rewards))

    return scorer
