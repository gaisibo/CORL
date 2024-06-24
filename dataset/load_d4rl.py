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
from gym.envs.registration import register
from d4rl.locomotion import maze_env, ant, swimmer
from d4rl.locomotion.wrappers import NormalizedBoxEnv
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

def get_macaw_local(dataset_path, timeout=1000):
    with h5py.File(dataset_path, 'r') as f:
        observations = f['obs'][:][::-1]
        actions = f['actions'][:][::-1]
        rewards = f['rewards'][:][::-1]
        terminals = f['terminals'][:][::-1]
        episode_terminals = terminals.copy()

    observations_part = np.array(observations, dtype=np.float32)
    observations = np.zeros((observations_part.shape[0], 27))
    observations[:, :observations_part.shape[1]] = observations_part
    actions_part = np.array(actions, dtype=np.float32)
    actions = np.zeros((actions_part.shape[0], 10))
    actions[:, :actions_part.shape[1]] = actions_part

    mdp_dataset = MDPDataset(
        observations=observations,
        actions=np.array(actions, dtype=np.float32),
        rewards=np.array(rewards, dtype=np.float32),
        terminals=terminals,
        episode_terminals=episode_terminals,
    )
    return mdp_dataset, 27, actions_part.shape[1]

def get_d4rl_local(dataset, timeout=1000, epoch_num=None, epoch_sum=3):

    observations = dataset["observations"]
    actions = dataset["actions"]
    rewards = dataset["rewards"]
    terminals = np.array(dataset["terminals"], dtype=np.float32)
    if 'timeouts' in dataset.keys() and np.sum(dataset["timeouts"]) > 1:
        episode_terminals = np.logical_or(np.array(dataset["timeouts"], dtype=np.float32), terminals)
    else:
        episode_terminals = terminals.copy()
    episode_terminals[-1] = 1

    mdp_dataset = MDPDataset(
        observations=np.array(observations, dtype=np.float32),
        actions=np.array(actions, dtype=np.float32),
        rewards=np.array(rewards, dtype=np.float32),
        terminals=terminals,
        episode_terminals=episode_terminals,
    )

    if epoch_num is not None:
        episodes = mdp_dataset.episodes
        sub_episodes = episodes[epoch_num * len(episodes) // epoch_sum: (epoch_num + 1) * len(episodes) // epoch_sum]
        sub_observations = np.concatenate([sub_episode.observations for sub_episode in sub_episodes], axis=0)
        sub_actions = np.concatenate([sub_episode.actions for sub_episode in sub_episodes], axis=0)
        sub_rewards = np.concatenate([sub_episode.rewards for sub_episode in sub_episodes], axis=0)
        sub_terminals = []
        sub_episode_terminals = []
        for sub_episode in sub_episodes:
            terminals = np.zeros_like(sub_episode.rewards, dtype=np.bool_)
            terminals[-1] = sub_episode.terminal
            episode_terminals = np.zeros_like(sub_episode.rewards, dtype=np.bool_)
            episode_terminals[-1] = True
            sub_terminals.append(terminals)
            sub_episode_terminals.append(episode_terminals)
        sub_terminals = np.concatenate(sub_terminals, axis=0)
        sub_episode_terminals = np.concatenate(sub_episode_terminals, axis=0)
        mdp_dataset = MDPDataset(sub_observations, sub_actions, sub_rewards, sub_terminals, sub_episode_terminals)
        # episodes = mdp_dataset.episodes
        # mean_len, mean_num = 0, 0
        # for episode in episodes:
        #     mean_len += episode.observations.shape[0]
        #     mean_num += 1
        # print(f"{(mean_len / mean_num)=}")
    return mdp_dataset

def get_antmaze_local(dataset, timeout=1000, epoch_num=None, epoch_sum=10):

    observations = dataset["observations"]
    actions = dataset["actions"]
    print(f"observations.shape: {observations.shape}")
    rewards = dataset["rewards"]
    terminals = np.array(dataset["terminals"], dtype=np.float32)
    if 'timeouts' in dataset.keys() and np.sum(dataset["timeouts"]) > 1:
        episode_terminals = np.logical_or(np.array(dataset["timeouts"], dtype=np.float32), terminals)
    else:
        episode_terminals = terminals.copy()
    episode_terminals[-1] = 1

    mdp_dataset = MDPDataset(
        observations=np.array(observations, dtype=np.float32),
        actions=np.array(actions, dtype=np.float32),
        rewards=np.array(rewards, dtype=np.float32),
        terminals=terminals,
        episode_terminals=episode_terminals,
    )

    if epoch_num is not None:
        episodes = mdp_dataset.episodes
        observations = []
        actions = []
        rewards = []
        for episode in episodes:
            if episode.observations.shape[0] < epoch_sum:
                continue
            episode_length, episode_append = divmod(episode.observations.shape[0], epoch_sum)
            episode_start = max(epoch_num * episode_length + min(epoch_num, episode_append), 0)
            episode_end = min((epoch_num + 1) * episode_length + min(epoch_num + 1, episode_append), episode.observations.shape[0])
            observations.append(episode.observations[episode_start : episode_end])
            actions.append(episode.actions[episode_start : episode_end])
            rewards.append(episode.rewards[episode_start : episode_end])
        observations = np.concatenate(observations, axis=0)
        actions = np.concatenate(actions, axis=0)
        rewards = np.concatenate(rewards, axis=0)
        terminals = []
        episode_terminals = []
        for episode in episodes:
            terminal = np.zeros_like(episode.rewards, dtype=np.bool_)
            terminal[-1] = (episode.terminal) and epoch_num + 1 == epoch_sum
            episode_terminal = np.zeros_like(episode.rewards, dtype=np.bool_)
            episode_terminal[-1] = True
            terminals.append(terminal)
            episode_terminals.append(episode_terminal)
        terminals = np.concatenate(terminals, axis=0)
        episode_terminals = np.concatenate(episode_terminals, axis=0)
        mdp_dataset = MDPDataset(observations, actions, rewards, terminals, episode_terminals)
    return mdp_dataset

    # if maze is not None:
    #     register(
    #         id=dataset_name,
    #         entry_point='d4rl.locomotion.ant:make_ant_maze_env',
    #         max_episode_steps=timeout,
    #         kwargs={
    #             'maze_map': maze,
    #             'reward_type':'sparse',
    #             'non_zero_reset':False,
    #             'eval':True,
    #             'maze_size_scaling': 4.0,
    #             'ref_min_score': 0.0,
    #             'ref_max_score': 1.0,
    #             'v2_resets': True,
    #         }
    #     )
    #     eval_env = gym.make(dataset_name)

    #     return mdp_dataset, eval_env
    # else:
