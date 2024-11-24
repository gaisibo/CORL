import os
import random
import re
from typing import List, Tuple, Optional
from urllib import request
import enum

import gym
import numpy as np

from myd3rlpy.dataset import Episode, MDPDataset, Transition
from d3rlpy.envs import ChannelFirst


def get_d4rl(
    env_name: str, create_mask: bool = False, mask_size: int = 1, h5path=None,
) -> Tuple[MDPDataset, gym.Env]:
    """Returns d4rl dataset and envrironment.
    The dataset is provided through d4rl.
    .. code-block:: python
        from d3rlpy.datasets import get_d4rl
        dataset, env = get_d4rl('hopper-medium-v0')
    References:
        * `Fu et al., D4RL: Datasets for Deep Data-Driven Reinforcement
          Learning. <https://arxiv.org/abs/2004.07219>`_
        * https://github.com/rail-berkeley/d4rl
    Args:
        env_name: environment id of d4rl dataset.
        create_mask: flag to create binary mask for bootstrapping.
        mask_size: ensemble size for binary mask.
    Returns:
        tuple of :class:`d3rlpy.dataset.MDPDataset` and gym environment.
    """
    try:
        import d4rl  # type: ignore

        env = gym.make(env_name)
        dataset = env.get_dataset(h5path=h5path)

        observations = dataset['observations']
        actions = dataset['actions']
        rewards = dataset['rewards']
        terminals = dataset['terminals']
        timeouts = dataset['timeouts']
        episode_terminals = np.logical_or(terminals, timeouts)

        mdp_dataset = MDPDataset(
            observations=np.array(observations, dtype=np.float32),
            actions=np.array(actions, dtype=np.float32),
            rewards=np.array(rewards, dtype=np.float32),
            terminals=np.array(terminals, dtype=np.float32),
            episode_terminals=np.array(episode_terminals, dtype=np.float32),
        )

        return mdp_dataset, env
    except ImportError as e:
        raise ImportError(
            "d4rl is not installed.\n"
            "pip install git+https://github.com/rail-berkeley/d4rl"
        ) from e

class _MinariEnvType(enum.Enum):
    BOX = 0
    GOAL_CONDITIONED = 1


def get_minari(
    env_name: str,
    tuple_observation: bool = False,
) -> Tuple[MDPDataset, gym.Env]:
    """Returns minari dataset and envrironment.

    The dataset is provided through minari.

    .. code-block:: python
        from d3rlpy.datasets import get_minari
        dataset, env = get_minari('door-cloned-v1')

    Args:
        env_name: environment id of minari dataset.
        tuple_observation: Flag to include goals as tuple element.

    Returns:
        tuple of :class:`d3rlpy.dataset.ReplayBuffer` and gym environment.
    """
    try:
        import minari

    except ImportError as e:
        raise ImportError(
            "minari is not installed.\n" "$ d3rlpy install minari"
        ) from e

    _dataset = minari.load_dataset(env_name, download=True)
    env = _dataset.recover_environment()
    unwrapped_env = env.unwrapped

    if isinstance(env.observation_space, GymnasiumBox):
        env_type = _MinariEnvType.BOX
    elif (
        isinstance(env.observation_space, GymnasiumDictSpace)
        and "observation" in env.observation_space.spaces
        and "desired_goal" in env.observation_space.spaces
    ):
        env_type = _MinariEnvType.GOAL_CONDITIONED
        unwrapped_env = GoalConcatWrapper(
            unwrapped_env, tuple_observation=tuple_observation
        )
    else:
        raise ValueError(
            f"Unsupported observation space: {env.observation_space}"
        )

    observations = []
    actions = []
    rewards = []
    terminals = []
    timeouts = []

    for ep in _dataset:
        if env_type == _MinariEnvType.BOX:
            _observations = ep.observations
        elif env_type == _MinariEnvType.GOAL_CONDITIONED:
            assert isinstance(ep.observations, dict)
            if isinstance(ep.observations["desired_goal"], dict):
                sorted_keys = sorted(
                    list(ep.observations["desired_goal"].keys())
                )
                goal_obs = np.concatenate(
                    [
                        ep.observations["desired_goal"][key]
                        for key in sorted_keys
                    ],
                    axis=-1,
                )
            else:
                goal_obs = ep.observations["desired_goal"]
            if tuple_observation:
                _observations = (ep.observations["observation"], goal_obs)
            else:
                _observations = np.concatenate(
                    [
                        ep.observations["observation"],
                        goal_obs,
                    ],
                    axis=-1,
                )
        else:
            raise ValueError("Unsupported observation format.")
        observations.append(_observations)
        actions.append(ep.actions)
        rewards.append(ep.rewards)
        terminals.append(ep.terminations)
        timeouts.append(ep.truncations)

    if tuple_observation:
        stacked_observations = tuple(
            np.concatenate([observation[i] for observation in observations])
            for i in range(2)
        )
    else:
        stacked_observations = np.concatenate(observations)

    mdp_dataset = MDPDataset(
        observations=np.array(observations, dtype=np.float32),
        actions=np.array(actions, dtype=np.float32),
        rewards=np.array(rewards, dtype=np.float32),
        terminals=np.array(terminals, dtype=np.float32),
        episode_terminals=np.array(episode_terminals, dtype=np.float32),
    )

    return mdp_dataset, env
