import copy
import pdb
import os
from typing import List, cast, Callable, Any, Dict, Tuple, Optional
import matplotlib.pyplot as plt
from typing_extensions import Protocol
from tqdm import trange
import gym
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import TensorDataset, DataLoader

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

def match_on_environment(
    env: gym.Env, replay_dataset, test_id: str, clone_actor: bool = False, n_trials: int = 1, epsilon: float = 0.0, render: bool = False, mix: bool = False, task_id_dim: int = 0,
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
        # print(f"test_id: {test_id}")
        try:
            env.reset_task(int(test_id))
        except:
            pass
        if is_image:
            stacked_observation = StackedObservation(
                observation_shape, algo.n_frames
            )
        save_id = algo._impl._impl_id
        algo._impl.change_task(test_id)

        replay_observations = []
        for observation, _, _, _, _, _, _ in replay_dataset:
            replay_observations.append(observation.to(algo._impl.device))
        observations = []

        for _ in range(n_trials):
            observation = env.reset()
            if mix:
                observation = np.concatenate([observation, np.zeros([observation.shape[0], 6], dtype=np.float32)], axis=1)
                observation = np.pad(observation, ((0, 0), (0, 6)), 'constant', constant_values=(0, 0))
            observation = torch.from_numpy(observation).to(algo._impl.device).unsqueeze(dim=0).to(torch.float32)
            episode_reward = 0.0
            # frame stacking
            if is_image:
                stacked_observation.clear()
                stacked_observation.append(observation)

            i = 0
            for _ in range(1000):
                if task_id_dim != 0:
                    task_id_tensor = torch.zeros(observation.shape[0], task_id_dim).to(observation.device).to(torch.float32)
                    task_id_tensor[:, test_id] = 1
                    observation = torch.cat([observation, task_id_tensor])
                observations.append(observation)
                # take action
                if np.random.random() < epsilon:
                    action = env.action_space.sample()
                else:
                    if clone_actor and int(save_id) != 0:
                        action = algo._impl._clone_policy(observation)
                        action = action.squeeze().cpu().detach().numpy()
                    else:
                        action = algo._impl._policy(observation)
                        action = action.squeeze().cpu().detach().numpy()

                observation, reward, done, pos = env.step(action)
                episode_reward += reward
                observation = torch.from_numpy(observation).to(algo._impl.device).unsqueeze(dim=0).to(torch.float32)

                if render:
                    env.render()

                if done:
                    break
                if i > 1000:
                    break

                i += 1
        algo._impl.change_task(save_id)

        total_match = 0
        for i, observation in enumerate(observations):
            min_match = []
            for j, replay_observation in enumerate(replay_observations):
                match_ = torch.mean((replay_observation - observation) ** 2).item()
                min_match.append(match_)
            min_match.sort()
            total_match += sum(min_match[:5]) / 5
        total_match /= len(observations)
        return total_match

    return scorer

def evaluate_on_environment(
    env: gym.Env, test_id: str=None, clone_actor: bool = False, n_trials: int = 100, epsilon: float = 0.0, render: bool = False, mix: bool = False, add_on: bool = False, task_id_dim: int = 0,
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
        if test_id is not None:
            # print(f"test_id: {test_id}")
            try:
                env.reset_task(int(test_id))
            except:
                pass
            save_id = algo._impl._impl_id
            algo._impl.change_task(test_id)
        if is_image:
            stacked_observation = StackedObservation(
                observation_shape, algo.n_frames
            )

        episode_rewards = []
        for _ in range(n_trials):
            observation = env.reset()
            if mix:
                observation = np.concatenate([observation, np.zeros([observation.shape[0], 6], dtype=np.float32)], axis=1)
                observation = np.pad(observation, ((0, 0), (0, 6)), 'constant', constant_values=(0, 0))
            observation = torch.from_numpy(observation).to(algo._impl.device).unsqueeze(dim=0).to(torch.float32)
            episode_reward = 0.0

            # frame stacking
            if is_image:
                stacked_observation.clear()
                stacked_observation.append(observation)

            i = 0
            while True:
                if test_id is not None and task_id_dim != 0:
                    task_id_tensor = torch.zeros(observation.shape[0], task_id_dim).to(observation.device).to(torch.float32)
                    task_id_tensor[:, test_id] = 1
                    observation = torch.cat([observation, task_id_tensor])
                # take action
                if np.random.random() < epsilon:
                    action = env.action_space.sample()
                else:
                    if test_id is not None and clone_actor and int(save_id) != 0:
                        action = algo._impl._clone_policy(observation)
                        action = action.squeeze().cpu().detach().numpy()
                    else:
                        action = algo._impl._policy(observation)
                        action = action.squeeze().cpu().detach().numpy()

                observation, reward, done, pos = env.step(action)
                episode_reward += reward
                observation = torch.from_numpy(observation).to(algo._impl.device).unsqueeze(dim=0).to(torch.float32)

                if render:
                    env.render()

                if done:
                    break
                if i > 1000:
                    break

                i += 1
            episode_rewards.append(episode_reward)
        if test_id is not None:
            algo._impl.change_task(save_id)
        if add_on:
            return float(np.mean(episode_rewards))
        else:
            return float(np.max(episode_rewards))

    return scorer

def merge_evaluate_on_environment(
    env: gym.Env, test_id: str=None, clone_actor: bool = False, n_trials: int = 10, epsilon: float = 0.0, render: bool = False, mix: bool = False, add_on: bool = False, task_id_dim: int = 0,
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

    def scorer(algo1: AlgoProtocol, algo2: AlgoProtocol, *args: Any) -> float:
        device = algo1._impl.device
        episode_rewards = []
        for _ in range(n_trials):
            observation = env.reset()
            observation = torch.from_numpy(observation).to(device).unsqueeze(dim=0).to(torch.float32)
            episode_reward = 0.0

            i = 0
            j1 = 0; j2 = 0
            while True:
                action1 = algo1._impl._policy(observation)
                q1 = algo1._impl._q_func(observation, action1)
                action2 = algo2._impl._policy(observation)
                q2 = algo2._impl._q_func(observation, action2)
                if q1 > q2:
                    j1 += 1
                    action = action1.squeeze().cpu().detach().numpy()
                else:
                    j2 += 1
                    action = action2.squeeze().cpu().detach().numpy()

                observation, reward, done, pos = env.step(action)
                episode_reward += reward
                observation = torch.from_numpy(observation).to(device).unsqueeze(dim=0).to(torch.float32)

                if render:
                    env.render()

                if done:
                    break
                if i > 1000:
                    break

                i += 1
            episode_rewards.append(episode_reward)
            print(f'total q1 max time: {j1}; total q2 max time: {j2}')
        if add_on:
            return float(np.mean(episode_rewards))
        else:
            return float(np.max(episode_rewards))

    return scorer

def single_evaluate_on_environment(
    env: gym.Env, n_trials: int = 5, epsilon: float = 0.0, render: bool = False,
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

    def scorer(algo: AlgoProtocol, *args: Any) -> float:
        episode_rewards = []
        for _ in trange(n_trials):
            observation = env.reset()
            observation = torch.from_numpy(observation).to(algo._impl.device).unsqueeze(dim=0).to(torch.float32)
            episode_reward = 0.0

            i = 0
            while True:
                # take action
                action = algo._impl.fine_tuned_action(observation)
                # action = algo._impl._policy(observation)
                action = action.squeeze().cpu().detach().numpy()

                observation, reward, done, pos = env.step(action)
                episode_reward += reward
                observation = torch.from_numpy(observation).to(algo._impl.device).unsqueeze(dim=0).to(torch.float32)

                if render:
                    env.render()

                if done:
                    break
                if i > 1000:
                    break

                i += 1
            episode_rewards.append(episode_reward)
        return float(np.max(episode_rewards))

    return scorer

def dis_on_environment(
    env: gym.Env, test_id: str, replay_dataset, clone_actor: bool = False, n_trials: int = 10, render: bool = False, mix: bool = False, task_id_dim: int = 0
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
        render: flag to render environment.
    Returns:
        scoerer function.
    """

    # for image observation
    observation_shape = env.observation_space.shape
    is_image = len(observation_shape) == 3

    def scorer(algo: AlgoProtocol, *args: Any) -> float:
        # print(f"test_id: {test_id}")
        replay_observations = replay_dataset.tensors[0]
        replay_observations = replay_observations.to(algo._impl.device)
        try:
            env.reset_task(int(test_id))
        except:
            pass
        if is_image:
            stacked_observation = StackedObservation(
                observation_shape, algo.n_frames
            )
        save_id = algo._impl._impl_id
        algo._impl.change_task(test_id)

        episode_rewards = []
        for _ in range(n_trials):
            observation = env.reset()
            if mix:
                observation = np.concatenate([observation, np.zeros([observation.shape[0], 6], dtype=np.float32)], axis=1)
                observation = np.pad(observation, ((0, 0), (0, 6)), 'constant', constant_values=(0, 0))
            observation = torch.from_numpy(observation).to(algo._impl.device).unsqueeze(dim=0).to(torch.float32)
            episode_reward = 0.0

            # frame stacking
            if is_image:
                stacked_observation.clear()
                stacked_observation.append(observation)

            i = 0
            while True:
                if task_id_dim != 0:
                    task_id_tensor = torch.zeros(observation.shape[0], task_id_dim).to(observation.device).to(torch.float32)
                    task_id_tensor[:, test_id] = 1
                    observation = torch.cat([observation, task_id_tensor])
                # take action
                action = algo._impl._policy(observation)
                action = action.squeeze().cpu().detach().numpy()

                observation, reward, done, pos = env.step(action)
                observation = torch.from_numpy(observation).to(algo._impl.device)
                min_dis = 10000
                for replay_observation in replay_observations:
                    dis = F.mse_loss(replay_observation, observation)
                    if dis < min_dis:
                        min_dis = dis
                episode_reward += min_dis.cpu().item()
                observation = observation.unsqueeze(dim=0).to(torch.float32)

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

def q_error_scorer(real_action_size: Optional[int] = None, test_id: Optional[str] = None) -> Callable[..., float]:
    def scorer(algo, replay_iterator):
        with torch.no_grad():
            save_id = algo._impl._impl_id
            if test_id is not None:
                algo._impl.change_task(test_id)
            total_errors = []
            for batch in replay_iterator:
                batch = dict(zip(replay_name, batch))
                batch = Struct(**batch)
                observations = batch.observations.to(algo._impl.device)
                if real_action_size is not None:
                    actions = batch.policy_actions.to(algo._impl.device)[:, :real_action_size]
                else:
                    actions = batch.policy_actions.to(algo._impl.device)
                qs = batch.qs.to(algo._impl.device)
                rebuild_qs = algo._impl._q_func.forward(observations, actions)
                loss = F.mse_loss(rebuild_qs, qs)
                total_errors.append(loss)
            total_errors = torch.stack(total_errors, dim=0)
            algo._impl.change_task(save_id)
        return float(torch.mean(total_errors).detach().cpu().numpy())
    return scorer

def q_mean_scorer(real_action_size: Optional[int] = None, test_id: Optional[str] = None, batch_size: int = 1024) -> Callable[..., float]:
    def scorer(algo, origin_dataset):
        with torch.no_grad():
            save_id = algo._impl._impl_id
            if test_id is not None:
                algo._impl.change_task(test_id)
            inner_qs = []
            outer_qs = []
            dataloader = DataLoader(TensorDataset(torch.from_numpy(origin_dataset.observations), torch.from_numpy(origin_dataset.actions)), batch_size=batch_size, shuffle=False)
            for batch in dataloader:
                observations, actions = batch
                observations = observations.to(algo._impl.device)
                if real_action_size is not None:
                    actions = actions.to(algo._impl.device)[:, :real_action_size]
                else:
                    actions = actions.to(algo._impl.device)
                q = algo._impl._q_func.forward(observations, actions)
                inner_qs.append(q)
                q = algo._impl._q_func.forward(observations, algo._impl._policy(observations))
                outer_qs.append(q)
            inner_qs = torch.cat(inner_qs, dim=0)
            outer_qs = torch.cat(outer_qs, dim=0)
            algo._impl.change_task(save_id)
        return float(torch.mean(inner_qs).detach().cpu().numpy()), float(torch.mean(outer_qs).detach().cpu().numpy())
    return scorer

def q_replay_scorer(real_action_size: Optional[int] = None, test_id: Optional[str] = None, batch_size: int = 1024) -> Callable[..., float]:
    def scorer(algo, replay_dataset):
        with torch.no_grad():
            save_id = algo._impl._impl_id
            if test_id is not None:
                algo._impl.change_task(test_id)
            inner_qs = []
            outer_qs = []
            dataloader = DataLoader(replay_dataset, batch_size=batch_size, shuffle=False)
            for batch in dataloader:
                observations, actions = batch[:2]
                observations = observations.to(algo._impl.device)
                if real_action_size is not None:
                    actions = actions.to(algo._impl.device)[:, :real_action_size]
                else:
                    actions = actions.to(algo._impl.device)
                q = algo._impl._q_func.forward(observations, actions)
                inner_qs.append(q)
                q = algo._impl._q_func.forward(observations, algo._impl._policy(observations))
                outer_qs.append(q)
            inner_qs = torch.cat(inner_qs, dim=0)
            outer_qs = torch.cat(outer_qs, dim=0)
            algo._impl.change_task(save_id)
        return float(torch.mean(inner_qs).detach().cpu().numpy()), float(torch.mean(outer_qs).detach().cpu().numpy())
    return scorer

def action_scorer(real_action_size: Optional[int] = None, test_id: Optional[str] = None, batch_size: int = 1024) -> Callable[..., float]:
    def scorer(algo, replay_dataset):
        with torch.no_grad():
            save_id = algo._impl._impl_id
            if test_id is not None:
                algo._impl.change_task(test_id)
            inner_qs = []
            outer_qs = []
            dataloader = DataLoader(replay_dataset, batch_size=batch_size, shuffle=False)
            for batch in dataloader:
                observations, actions = batch[:2]
                observations = observations.to(algo._impl.device)
                if real_action_size is not None:
                    actions = actions.to(algo._impl.device)[:, :real_action_size]
                else:
                    actions = actions.to(algo._impl.device)
                q = algo._impl._q_func.forward(observations, actions)
                inner_qs.append(q)
                q = algo._impl._q_func.forward(observations, algo._impl._policy(observations))
                outer_qs.append(q)
            inner_qs = torch.cat(inner_qs, dim=0)
            outer_qs = torch.cat(outer_qs, dim=0)
            algo._impl.change_task(save_id)
        return float(torch.mean(inner_qs).detach().cpu().numpy()), float(torch.mean(outer_qs).detach().cpu().numpy())
    return scorer

def dataset_value_scorer(
    algo: AlgoProtocol, episodes: List[Episode]
) -> float:
    r"""Returns average value estimation.
    This metrics suggests the scale for estimation of Q functions.
    If average value estimation is too large, the Q functions overestimate
    action-values, which possibly makes training failed.
    .. math::
        \mathbb{E}_{s_t \sim D} [ \max_a Q_\theta (s_t, a)]
    Args:
        algo: algorithm.
        episodes: list of episodes.
    Returns:
        average value estimation.
    """
    total_values = []
    for episode in episodes:
        for batch in _make_batches(episode, WINDOW_SIZE, algo.n_frames):
            values = algo.predict_value(batch.observations, batch.actions)
            total_values += cast(np.ndarray, values).tolist()
    return float(np.mean(total_values))




# For single task
def q_dataset_scorer(algo, episodes: List[Episode]) -> Callable[..., float]:
    total_values = []
    for episode in episodes:
        for batch in _make_batches(episode, WINDOW_SIZE, algo.n_frames):
            observations = torch.from_numpy(batch.observations).to(algo._impl.device).to(torch.float32)
            actions = torch.from_numpy(batch.actions).to(algo._impl.device).to(torch.float32)
            q_old = algo._impl._clone_q_func(observations, actions)
            q_new = algo._impl._q_func(observations, actions)
            total_values.append(F.mse_loss(q_old, q_new).item())
    return sum(total_values) / len(total_values)

def q_play_scorer(algo, episodes: List[Episode]) -> Callable[..., float]:
    total_values = []
    for episode in episodes:
        for batch in _make_batches(episode, WINDOW_SIZE, algo.n_frames):
            observations = torch.from_numpy(batch.observations).to(algo._impl.device).to(torch.float32)
            actions = algo._impl._policy(observations)
            q_old = algo._impl._clone_q_func(observations, actions)
            q_new = algo._impl._q_func(observations, actions)
            total_values.append(F.mse_loss(q_old, q_new).item())
    return sum(total_values) / len(total_values)

def q_online_diff_scorer(online_network):
    q_network = online_network['trainer/qf1']
    policy_network = online_network['trainer/policy']
    def scorer(algo, episodes: List[Episode]):
        total_values = []
        for episode in episodes:
            for batch in _make_batches(episode, WINDOW_SIZE, algo.n_frames):
                observations = torch.from_numpy(batch.observations).to(algo._impl.device).to(torch.float32)
                dist = policy_network(observations)
                actions = dist.sample()
                q_old = q_network(observations, actions)
                observations = observations + torch.randn_like(observations) * 0.1
                dist = policy_network(observations)
                actions = dist.sample()
                q_new = q_network(observations, actions)
                total_values.append(F.mse_loss(q_old, q_new).item())
        return sum(total_values) / len(total_values)
    return scorer

def q_offline_diff_scorer(algo, episodes: List[Episode]):
    total_values = []
    for episode in episodes:
        for batch in _make_batches(episode, WINDOW_SIZE, algo.n_frames):
            observations = algo._impl._vae.generate(batch.observations.shape[0]).detach()
            actions = algo._impl._policy(observations)
            q_old = algo._impl._q_func(observations, actions)
            observations = observations + torch.randn_like(observations) * 0.1
            actions = algo._impl._policy(observations)
            q_new = algo._impl._q_func(observations, actions)
            total_values.append(F.mse_loss(q_old, q_new).item())
    return sum(total_values) / len(total_values)

def q_id_diff_scorer(online_network):
    q_network = online_network['trainer/qf1']
    def scorer(algo, episodes: List[Episode]):
        total_values = []
        for episode in episodes:
            for batch in _make_batches(episode, WINDOW_SIZE, algo.n_frames):
                observations = torch.from_numpy(batch.observations).to(algo._impl.device).to(torch.float32)
                actions = torch.from_numpy(batch.actions).to(algo._impl.device).to(torch.float32)
                q_old = q_network(observations, actions)
                q_new = algo._impl._q_func(observations, actions)
                total_values.append(F.mse_loss(q_old, q_new).item())
        return sum(total_values) / len(total_values)
    return scorer

def q_ood_diff_scorer(online_network):
    q_network = online_network['trainer/qf1']
    def scorer(algo, episodes: List[Episode]):
        total_values = []
        for episode in episodes:
            for batch in _make_batches(episode, WINDOW_SIZE, algo.n_frames):
                observations = torch.from_numpy(batch.observations).to(algo._impl.device).to(torch.float32)
                observations += torch.randn_like(observations) * 0.1
                actions = algo._impl._policy(observations)
                q_old = q_network(observations, actions)
                q_new = algo._impl._q_func(observations, actions)
                total_values.append(F.mse_loss(q_old, q_new).item())
        return sum(total_values) / len(total_values)
    return scorer

def policy_replay_scorer(algo, episodes: List[Episode]) -> Callable[..., float]:
    total_values = []
    for episode in episodes:
        for batch in _make_batches(episode, WINDOW_SIZE, algo.n_frames):
            observations = torch.from_numpy(batch.observations).to(algo._impl.device).to(torch.float32)
            actions_new = algo._impl._policy(observations)
            actions_old = algo._impl._clone_policy(observations)
            total_values.append(F.mse_loss(actions_old, actions_new).item())
    return sum(total_values) / len(total_values)

def policy_dataset_scorer(algo, episodes: List[Episode]) -> Callable[..., float]:
    total_values = []
    for episode in episodes:
        for batch in _make_batches(episode, WINDOW_SIZE, algo.n_frames):
            observations = torch.from_numpy(batch.observations).to(algo._impl.device).to(torch.float32)
            actions = torch.from_numpy(batch.actions).to(algo._impl.device).to(torch.float32)
            actions_new = algo._impl._policy(observations)
            total_values.append(F.mse_loss(actions, actions_new).item())
    return sum(total_values) / len(total_values)

def policy_online_diff_scorer(online_network):
    policy_network = online_network['trainer/policy']
    def scorer(algo, episodes: List[Episode]):
        total_values = []
        for episode in episodes:
            for batch in _make_batches(episode, WINDOW_SIZE, algo.n_frames):
                observations = torch.from_numpy(batch.observations).to(algo._impl.device).to(torch.float32)
                dist_old = policy_network(observations)
                actions_old = dist_old.sample()
                observations = observations + torch.randn_like(observations) * 0.1
                dist_new = policy_network(observations)
                actions_new = dist_old.sample()
                total_values.append(F.mse_loss(actions_old.to(algo._impl.device), actions_new).item())
        return sum(total_values) / len(total_values)
    return scorer

def policy_offline_diff_scorer(algo, episodes: List[Episode]):
    total_values = []
    for episode in episodes:
        for batch in _make_batches(episode, WINDOW_SIZE, algo.n_frames):
            observations = algo._impl._vae.generate(batch.observations.shape[0]).detach()
            actions_old = algo._impl._policy(observations)
            observations = observations + torch.randn_like(observations) * 0.1
            actions_new = algo._impl._policy(observations)
            total_values.append(F.mse_loss(actions_old, actions_new).item())
    return sum(total_values) / len(total_values)

def policy_id_diff_scorer(online_network):
    policy_network = online_network['trainer/policy']
    def scorer(algo, episodes: List[Episode]):
        total_values = []
        for episode in episodes:
            for batch in _make_batches(episode, WINDOW_SIZE, algo.n_frames):
                observations = torch.from_numpy(batch.observations).to(algo._impl.device).to(torch.float32)
                dist_old = policy_network(observations)
                actions_old = dist_old.sample()
                actions_new = algo._impl._policy(observations)
                total_values.append(F.mse_loss(actions_old.to(algo._impl.device), actions_new).item())
        return sum(total_values) / len(total_values)
    return scorer

def policy_ood_diff_scorer(online_network):
    policy_network = online_network['trainer/policy']
    def scorer(algo, episodes: List[Episode]):
        total_values = []
        for episode in episodes:
            for batch in _make_batches(episode, WINDOW_SIZE, algo.n_frames):
                observations = algo._impl._vae.generate(batch.observations.shape[0]).detach()
                observations = observations + torch.randn_like(observations) * 0.1
                dist_old = policy_network(observations)
                actions_old = dist_old.sample()
                actions_new = algo._impl._policy(observations)
                total_values.append(F.mse_loss(actions_old, actions_new).item())
        return sum(total_values) / len(total_values)
    return scorer
