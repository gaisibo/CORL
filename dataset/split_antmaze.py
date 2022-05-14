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
from myd3rlpy.siamese_similar import similar_euclid


def get_dataset(h5path, observation_space, action_space):
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
    if observation_space.shape is not None:
        assert data_dict['observations'].shape[1:] == observation_space.shape, \
            'Observation shape does not match env: %s vs %s' % (
                str(data_dict['observations'].shape[1:]), str(observation_space.shape))
    assert data_dict['actions'].shape[1:] == action_space.shape, \
        'Action shape does not match env: %s vs %s' % (
            str(data_dict['actions'].shape[1:]), str(action_space.shape))
    if data_dict['rewards'].shape == (N_samples, 1):
        data_dict['rewards'] = data_dict['rewards'][:, 0]
    assert data_dict['rewards'].shape == (N_samples,), 'Reward has wrong shape: %s' % (
        str(data_dict['rewards'].shape))
    if data_dict['terminals'].shape == (N_samples, 1):
        data_dict['terminals'] = data_dict['terminals'][:, 0]
    assert data_dict['terminals'].shape == (N_samples,), 'Terminals has wrong shape: %s' % (
        str(data_dict['rewards'].shape))
    return data_dict

def get_d4rl_local(dataset, timeout=300) -> MDPDataset:

    observations = dataset["observations"]
    actions = dataset["actions"]
    rewards = dataset["rewards"]
    terminals = np.array(dataset["terminals"], dtype=np.float32)
    episode_terminals = np.zeros_like(terminals)
    i = timeout - 1
    while i < terminals.shape[0]:
        episode_terminals[i] = 1
        i += timeout

    mdp_dataset = MDPDataset(
        observations=np.array(observations, dtype=np.float32),
        actions=np.array(actions, dtype=np.float32),
        rewards=np.array(rewards, dtype=np.float32),
        terminals=terminals,
        episode_terminals=episode_terminals,
    )

    return mdp_dataset

def split_navigate_antmaze_umaze_v2(top_euclid, device, level):
    _, env = get_d4rl('antmaze-umaze-v2')
    dataset_name = 'antmaze-umaze-v2'
    task_nums = 3
    datasets = dict()
    nearest_indexes = dict()
    for i in range(task_nums):
        dataset_path = './dataset/antmaze/umaze/' + level + '_' + str(task_nums - i - 1) + '.hdf5'
        datasets[str(i)] = get_d4rl_local(get_dataset(dataset_path, env.observation_space, env.action_space))
        dataset_terminals = datasets[str(i)].terminals
        dataset_starts = np.concatenate([np.ones(1), dataset_terminals[:-1]], axis=0).astype(np.int64)
        nearest_indexes[str(i)] = dataset_starts
    if level == 'random':
        end_points = {'0': np.array([-0.3, 9.3]), '1': np.array([9.3, 9.0]), '2': np.array([9.3, 0])}
    elif level == 'medium':
        end_points = {'0': np.array([0.3, 8.8]), '1': np.array([8.5, 9.2]), '2': np.array([3.3, -0.2])}
    elif level == 'expert':
        end_points = {'0': np.array([2.5, 8.5]), '1': np.array([8.5, 5.2]), '2': np.array([5.3, 0.7])}
    else:
        raise NotImplementedError
    return split_antmaze(datasets, env, task_nums, end_points, top_euclid, device)

def split_navigate_antmaze_medium_v2(top_euclid, device):
    origin_dataset, env = get_d4rl('antmaze-medium-play-v2')
    dataset_name = 'antmaze-medium-play-v2'
    task_nums = 3
    end_points = [np.array([16, 20]), np.array([0, 16]), np.array([20, 4])]
    return split_antmaze(origin_dataset, env, dataset_name, task_nums, end_points, top_euclid, device)

def split_navigate_antmaze_large_v2(top_euclid, device):
    origin_dataset, env = get_d4rl('antmaze-large-play-v2')
    dataset_name = 'antmaze-large-play-v2'
    task_nums = 3
    end_points = [np.array([20.0, 0.0]), np.array([16.0, 8.0]), np.array([36.0, 8.0]), np.array([4.0, 16.0]), np.array([12.0, 24.0]), np.array([20.0, 24.0]), np.array([32.0, 24.0]), ]
    return split_antmaze(origin_dataset, env, dataset_name, task_nums, end_points, top_euclid, device)

def split_antmaze(task_datasets, env, task_nums, end_points, top_euclid, device):
    envs = {}
    for index, episodes in task_datasets.items():
        envs[index] = deepcopy(env)
        envs[index]._goal = end_points[index]
        envs[index].target_goal = end_points[index]
    nearest_indexes = {0: np.array([0]), 1: np.array([0]), 2: np.array([0])}

    taskid_task_datasets = dict()
    origin_task_datasets = dict()
    for dataset_num, dataset in task_datasets.items():
        # transitions = [transition for episode in dataset.episodes for transition in episode]
        # observations = np.stack([transition.observation for transition in transitions], axis=0)
        # print(f"observations.shape: {observations.shape}")
        # indexes_euclid = similar_euclid(torch.from_numpy(dataset.observations).to(device), torch.from_numpy(observations).to(device), dataset_name, dataset_num, compare_dim=compare_dim)[:dataset.actions.shape[0], :top_euclid]
        observations = dataset.observations
        # indexes_euclid = np.zeros_like(dataset.actions)
        real_action_size = dataset.actions.shape[1]
        task_id_numpy = np.eye(task_nums)[int(dataset_num)].squeeze()
        task_id_numpy = np.broadcast_to(task_id_numpy, (dataset.observations.shape[0], task_nums))
        real_observation_size = dataset.observations.shape[1]
        # 用action保存一下indexes_euclid，用state保存一下task_id
        taskid_task_datasets[dataset_num] = MDPDataset(np.concatenate([observations, task_id_numpy], axis=1), dataset.actions, dataset.rewards, dataset.terminals, dataset.episode_terminals)
        # action_task_datasets[dataset_num] = MDPDataset(dataset.observations, np.concatenate([dataset.actions, indexes_euclid], axis=1), dataset.rewards, dataset.terminals, dataset.episode_terminals)
        origin_task_datasets[dataset_num] = MDPDataset(observations, dataset.actions, dataset.rewards, dataset.terminals, dataset.episode_terminals)
        # changed_task_datasets[dataset_num] = MDPDataset(np.concatenate([dataset.observations, task_id_numpy], axis=1), np.concatenate([dataset.actions, indexes_euclid], axis=1), dataset.rewards, dataset.terminals, dataset.episode_terminals)
        # indexes_euclids[dataset_num] = indexes_euclid
    # torch.save(task_datasets, dataset_name + '_' + '.pt')

    return origin_task_datasets, taskid_task_datasets, envs, end_points, nearest_indexes, real_action_size, real_observation_size, task_nums
