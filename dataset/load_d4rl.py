import os
import matplotlib.pyplot as plt
import sys
from copy import deepcopy
import random
import numpy as np
import torch
import h5py
from tqdm import tqdm
import pickle

from d4rl.offline_env import get_keys
from d3rlpy.datasets import get_d4rl
from d3rlpy.dataset import MDPDataset
import gym
# from myd3rlpy.siamese_similar import similar_euclid
# from mygym.envs.hopper_back import HopperBackEnv
# from mygym.envs.walker2d_back import Walker2dBackEnv
# from mygym.envs.halfcheetah_back import HalfCheetahBackEnv
# from mygym.envs.hopper_light import HopperLightEnv
# from mygym.envs.walker2d_light import Walker2dLightEnv
# from mygym.envs.halfcheetah_light import HalfCheetahLightEnv
# from mygym.envs.hopper_wind import HopperWindEnv
# from mygym.envs.walker2d_wind import Walker2dWindEnv
# from mygym.envs.halfcheetah_wind import HalfCheetahWindEnv
# from mygym.envs.envs import HalfCheetahDirEnv, HalfCheetahVelEnv, AntDirEnv, AntGoalEnv, HumanoidDirEnv, WalkerRandParamsWrappedEnv, ML45Env


def get_dataset(h5path, expert=False, env=None):
    data_dict = {}
    with h5py.File(h5path, 'r') as dataset_file:
        for k in tqdm(get_keys(dataset_file), desc="load datafile"):
            try:  # first try loading as an array
                data_dict[k] = dataset_file[k][:]
            except ValueError as e:  # try loading as a scalar
                data_dict[k] = dataset_file[k][()]

    # Run a few quick sanity checks
    for key in ['observations', 'actions', 'rewards', 'terminals']:
        assert key in data_dict, 'Dataset is missing key %s' % key
    N_samples = data_dict['observations'].shape[0]
    if data_dict['rewards'].shape == (N_samples, 1):
        data_dict['rewards'] = data_dict['rewards'][:, 0]
    assert data_dict['rewards'].shape == (N_samples,), 'Reward has wrong shape: %s' % (
        str(data_dict['rewards'].shape))
    if data_dict['terminals'].shape == (N_samples, 1):
        data_dict['terminals'] = data_dict['terminals'][:, 0]
    assert data_dict['terminals'].shape == (N_samples,), 'Terminals has wrong shape: %s' % (
        str(data_dict['rewards'].shape))
    return data_dict

def get_d4rl_local(dataset, timeout=1000) -> MDPDataset:

    observations = dataset["observations"]
    actions = dataset["actions"]
    rewards = dataset["rewards"]
    terminals = np.array(dataset["terminals"], dtype=np.float32)
    if 'timeouts' in dataset.keys() and np.sum(dataset["timeouts"]) > 1:
        episode_terminals = np.logical_or(np.array(dataset["timeouts"], dtype=np.float32), terminals)
    else:
        episode_terminals = np.zeros_like(terminals)
    #     i = timeout - 1
    #     while i < terminals.shape[0]:
    #         episode_terminals[i] = 1
    #         i += timeout
    print(sum(terminals))
    print(sum(episode_terminals))

    mdp_dataset = MDPDataset(
        observations=np.array(observations, dtype=np.float32),
        actions=np.array(actions, dtype=np.float32),
        rewards=np.array(rewards, dtype=np.float32),
        terminals=terminals,
        episode_terminals=episode_terminals,
    )

    return mdp_dataset
