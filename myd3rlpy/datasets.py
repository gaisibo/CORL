import os
import random
import re
from typing import List, Tuple
from urllib import request

import gym
import numpy as np

from d3rlpy.dataset import Episode, MDPDataset, Transition
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

        observations = []
        actions = []
        rewards = []
        terminals = []
        episode_terminals = []
        episode_step = 0
        cursor = 0
        dataset_size = dataset["observations"].shape[0]
        while cursor < dataset_size:
            # collect data for step=t
            observation = dataset["observations"][cursor]
            action = dataset["actions"][cursor]
            if episode_step == 0:
                reward = 0.0
            else:
                reward = dataset["rewards"][cursor - 1]

            observations.append(observation)
            actions.append(action)
            rewards.append(reward)
            terminals.append(0.0)

            # skip adding the last step when timeout
            if dataset["timeouts"][cursor]:
                episode_terminals.append(1.0)
                episode_step = 0
                cursor += 1
                continue

            episode_terminals.append(0.0)
            episode_step += 1

            if dataset["terminals"][cursor]:
                # collect data for step=t+1
                dummy_observation = observation.copy()
                dummy_action = action.copy()
                next_reward = dataset["rewards"][cursor]

                # the last observation is rarely used
                observations.append(dummy_observation)
                actions.append(dummy_action)
                rewards.append(next_reward)
                terminals.append(1.0)
                episode_terminals.append(1.0)
                episode_step = 0

            cursor += 1

        mdp_dataset = MDPDataset(
            observations=np.array(observations, dtype=np.float32),
            actions=np.array(actions, dtype=np.float32),
            rewards=np.array(rewards, dtype=np.float32),
            terminals=np.array(terminals, dtype=np.float32),
            episode_terminals=np.array(episode_terminals, dtype=np.float32),
            create_mask=create_mask,
            mask_size=mask_size,
        )

        return mdp_dataset, env
    except ImportError as e:
        raise ImportError(
            "d4rl is not installed.\n"
            "pip install git+https://github.com/rail-berkeley/d4rl"
        ) from e
