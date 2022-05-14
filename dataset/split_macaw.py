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
import gym
from mygym.envs.hopper_back import HopperBackEnv
from mygym.envs.walker2d_back import Walker2dBackEnv
from mygym.envs.halfcheetah_back import HalfCheetahBackEnv
from mygym.envs.hopper_light import HopperLightEnv
from mygym.envs.walker2d_light import Walker2dLightEnv
from mygym.envs.halfcheetah_light import HalfCheetahLightEnv
from mygym.envs.hopper_wind import HopperWindEnv
from mygym.envs.walker2d_wind import Walker2dWindEnv
from mygym.envs.halfcheetah_wind import HalfCheetahWindEnv
from mygym.envs.envs import HalfCheetahDirEnv, HalfCheetahVelEnv, AntDirEnv, AntGoalEnv, HumanoidDirEnv, WalkerRandParamsWrappedEnv, ML45Env


def get_dataset(h5path):
    data_dict = {}
    with h5py.File(h5path, 'r') as dataset_file:
        for k in tqdm(get_keys(dataset_file), desc="load datafile"):
            try:  # first try loading as an array
                data_dict[k] = dataset_file[k][:]
            except ValueError as e:  # try loading as a scalar
                data_dict[k] = dataset_file[k][()]

    # Run a few quick sanity checks
    if 'obs' in data_dict.keys():
        data_dict['observations'] = data_dict['obs']
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

def get_d4rl_local(dataset, timeout=300) -> MDPDataset:

    observations = np.flip((dataset["observations"]), axis=0)
    actions = np.flip((dataset["actions"]), axis=0)
    rewards = np.flip((dataset["rewards"]), axis=0)
    terminals = np.flip((np.array(dataset["terminals"], dtype=np.float32)), axis=0)
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

def split_macaw(top_euclid, dataset_name, inner_paths, envs, include_goal=False, multitask=False, one_hot_goal=False, device='cuda:0'):
    task_datasets = dict()
    nearest_indexes = dict()
    tasks = []
    for i, (inner_path, env) in enumerate(zip(inner_paths, envs)):
        with open(env, 'rb') as f:
            task_info = pickle.load(f)
            assert len(task_info) == 1, f'Unexpected task info: {task_info}'
            tasks.append(task_info[0])
    if dataset_name == 'ant-dir':
        env = AntDirEnv(tasks, len(envs), include_goal = include_goal or multitask)
    elif dataset_name == 'cheetah-dir':
        env = HalfCheetahDirEnv(tasks, include_goal = include_goal or multitask)
    elif dataset_name == 'cheetah-vel':
        env = HalfCheetahVelEnv(tasks, include_goal = include_goal or multitask, one_hot_goal=one_hot_goal or multitask)
    elif dataset_name in ['walker-params', 'walker-dir']:
        env = WalkerRandParamsWrappedEnv(tasks, len(envs), include_goal = include_goal or multitask)
    else:
        raise RuntimeError(f'Invalid env name {dataset_name}')
    for i, inner_path in enumerate(inner_paths):
        task_datasets[str(i)] = get_d4rl_local(get_dataset(inner_path))
        task_dataset_terminals = task_datasets[str(i)].terminals
        task_dataset_starts = np.concatenate([np.ones(1), task_dataset_terminals[:-1]], axis=0).astype(np.int64)
        task_dataset_starts = np.where(task_dataset_terminals[:-1] == 1)[0] + 1
        np.insert(task_dataset_starts, 0, 0)
        nearest_indexes[str(i)] = task_dataset_starts
    return split_gym(top_euclid, dataset_name, task_datasets, env, nearest_indexes, compare_dim=3, device=device)

def split_gym(top_euclid, dataset_name, task_datasets, env, nearest_indexes, compare_dim=3, device='cuda:0'):

    # fig = plt.figure()
    # obs = env.reset()
    # for episode in origin_dataset.episodes:
    #     x = episode.observations[:, 0]
    #     y = episode.observations[:, 1]
    #     plt.plot(x, y)
    # plt.plot(obs[0], obs[1], 'o', markersize=4)
    # plt.savefig('try_' + dataset_name + '.png')
    # plt.close('all')
    print(f'task_datasets.keys(): {task_datasets.keys()}')
    task_nums = len(task_datasets.keys())

    filename = 'near_indexes/near_indexes_' + dataset_name + '/nearest_indexes.npy'
    # if os.path.exists(filename):
    #     nearest_indexes = np.load(filename, allow_pickle=True)
    # else:
    #     nearest_indexes = {}
    #     for dataset_num, dataset in task_datasets.items():
    #         env = envs[dataset_num]
    #         nearest_dist = 1000000000
    #         nearest_index = -1
    #         nearest_indexes_ = []
    #         for times in range(100):
    #             origin = env.reset()[:2]
    #             for i in range(dataset.observations.shape[0]):
    #                 dist = np.linalg.norm(origin - dataset.observations[i, :2])
    #                 if dist < nearest_dist:
    #                     nearest_index = i
    #                     nearest_dist = dist
    #             nearest_indexes_.append(nearest_index)
    #         nearest_indexes_ = np.unique(np.array(nearest_indexes_))
    #         nearest_indexes[dataset_num] = nearest_indexes_
    #         print(f'nearest_indexes_: {nearest_indexes_}')
    #     np.save(filename, nearest_indexes)

    changed_task_datasets = dict()
    taskid_task_datasets = dict()
    action_task_datasets = dict()
    origin_task_datasets = dict()
    # indexes_euclids = dict()
    real_action_size = 0
    real_observation_size = 0
    for dataset_num, dataset in task_datasets.items():
        # transitions = [transition for episode in dataset.episodes for transition in episode]
        # observations = np.stack([transition.observation for transition in transitions], axis=0)
        # print(f"observations.shape: {observations.shape}")
        # indexes_euclid = similar_euclid(torch.from_numpy(dataset.observations).to(device), torch.from_numpy(observations).to(device), dataset_name, dataset_num, compare_dim=compare_dim)[:dataset.actions.shape[0], :top_euclid]
        if 'mix' in dataset_name and dataset_num == '0': #  hopper
            observations = np.concatenate([dataset.observations, np.zeros([dataset.observations.shape[0], 6], dtype=np.float32)], axis=1)
        else:
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
    # torch.save(task_datasets, dataset_name  + '.pt')

    # changed_task_datasets = {'0': changed_task_datasets['0'], '3': changed_task_datasets['3'], '2': changed_task_datasets['2'], '1': changed_task_datasets['1']}
    # origin_task_datasets = {'0': origin_task_datasets['0'], '3': origin_task_datasets['3'], '2': origin_task_datasets['2'], '1': origin_task_datasets['1']}
    # taskid_task_datasets = {'0': taskid_task_datasets['0'], '3': taskid_task_datasets['3'], '2': taskid_task_datasets['2'], '1': taskid_task_datasets['1']}
    # action_task_datasets = {'0': action_task_datasets['0'], '3': action_task_datasets['3'], '2': action_task_datasets['2'], '1': action_task_datasets['1']}
    # envs = {'0': envs['0'], '3': envs['3'], '2': envs['2'], '1': envs['1']}
    # nearest_indexes = {'0': nearest_indexes['0'], '3': nearest_indexes['3'], '2': nearest_indexes['2'], '1': nearest_indexes['1']}
    # indexes_euclids = {'0': indexes_euclids['0'], '3': indexes_euclids['3'], '2': indexes_euclids['2'], '1': indexes_euclids['1']}

    # return changed_task_datasets, origin_task_datasets, taskid_task_datasets, action_task_datasets, envs, [None for _ in range(task_nums)], nearest_indexes, real_action_size, real_observation_size, indexes_euclids, task_nums
    return origin_task_datasets, taskid_task_datasets, env, [None for _ in range(task_nums)], nearest_indexes, real_action_size, real_observation_size
