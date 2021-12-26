from typing import List, cast, Callable, Any, Dict
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

def bc_error_scorer(real_action_size: int) -> Callable[..., float]:
    def scorer(algo, replay_iterator):
        total_errors = []
        for batch in replay_iterator:
            observations, actionss, means, stddevs, qss, _, _ = batch
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
                rebuild_qs = algo._impl._q_func.forward(observations, actionss[:, sample_time, :real_action_size])
                rebuild_qss.append(rebuild_qs)
            replay_qss = torch.stack(rebuild_qss, dim=1)
            loss = F.mse_loss(replay_qss, qss) + torch.distributions.kl.kl_divergence(rebuild_dist, dist)
            total_errors.append(loss)
        total_errors = torch.cat(total_errors, dim=0)
        return float(torch.mean(total_errors).detach().cpu().numpy())
    return scorer

def td_error_scorer(real_action_size: int) -> Callable[...,Dict[int, float]]:
    def scorer(algo: AlgoProtocol, episodess: Dict[int, List[Episode]]) -> Dict[int, float]:
        total_errorss = dict()
        for id, episodes in episodess.items():
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
            total_errorss[id] = float(np.mean(total_errors))

        return total_errorss
    return scorer

def evaluate_on_environment(
        env: gym.Env, task_id: int, task_nums: int, n_trials: int = 10, epsilon: float = 0.0, render: bool = False,
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
        if is_image:
            stacked_observation = StackedObservation(
                observation_shape, algo.n_frames
            )

        episode_rewards = []
        for _ in range(n_trials):
            observation = env.reset()
            task_id_numpy = np.eye(task_nums)[task_id].squeeze()
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
                task_id_numpy = np.eye(task_nums)[task_id].squeeze()
                observation = np.concatenate([observation, task_id_numpy], axis=0)
                episode_reward += reward

                if is_image:
                    stacked_observation.append(observation)

                if render:
                    env.render()

                if done:
                    break
            episode_rewards.append(episode_reward)
        return float(np.mean(episode_rewards))

    return scorer
