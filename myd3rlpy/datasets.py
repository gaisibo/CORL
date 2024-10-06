import os
import random
import re
from typing import List, Tuple
from urllib import request

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
