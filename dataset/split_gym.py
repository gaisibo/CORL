import os
import matplotlib.pyplot as plt
import sys
from copy import deepcopy
import random
import numpy as np
import torch
import h5py
from tqdm import tqdm

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
    print(terminals[0])
    assert False
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

def split_mix(top_euclid, dataset_name, device='cuda:0'):
    hopper_name = 'hopper' + dataset_name[3:]
    hopper_dataset, hopper_env = get_d4rl(hopper_name)
    hopper_dataset_terminals = hopper_dataset.terminals
    hopper_dataset_starts = np.concatenate([np.ones(1), hopper_dataset_terminals[:-1]], axis=0).astype(np.int64)
    halfcheetah_name = 'halfcheetah' + dataset_name[3:]
    halfcheetah_dataset, halfcheetah_env = get_d4rl(halfcheetah_name)
    halfcheetah_dataset_terminals = halfcheetah_dataset.terminals
    halfcheetah_dataset_starts = np.concatenate([np.ones(1), halfcheetah_dataset_terminals[:-1]], axis=0).astype(np.int64)
    walker2d_name = 'walker2d' + dataset_name[3:]
    walker2d_dataset, walker2d_env = get_d4rl(walker2d_name)
    walker2d_dataset_terminals = walker2d_dataset.terminals
    walker2d_dataset_starts = np.concatenate([np.ones(1), walker2d_dataset_terminals[:-1]], axis=0).astype(np.int64)
    task_datasets = {'0': hopper_dataset, '1': halfcheetah_dataset, '2': walker2d_dataset}
    envs = {'0': hopper_env, '1': halfcheetah_env, '2': walker2d_env}
    nearest_indexes = {'0': hopper_dataset_starts, '1': halfcheetah_dataset_starts, '2': walker2d_dataset_starts}
    return split_gym(top_euclid, dataset_name, task_datasets, envs, nearest_indexes, compare_dim=3, device=device)

def split_cheetah(top_euclid, dataset_name, device='cuda:0'):
    origin_dataset, env = get_d4rl(dataset_name)
    origin_dataset_terminals = origin_dataset.terminals
    origin_dataset_starts = np.concatenate([np.ones(1), origin_dataset_terminals[:-1]], axis=0).astype(np.int64)
    names = dataset_name.split('-', 1)
    # dataset_path = './dataset/gym_back/halfcheetah_wind_' + names[1][:-3].replace('-', '_') + '.h5df'
    # origin_dataset_wind = get_d4rl_local(get_dataset(dataset_path, env.observation_space, env.action_space))
    # env_wind = HalfCheetahWindEnv()
    # dataset_path = './dataset/gym_back/halfcheetah_light_' + names[1][:-3].replace('-', '_') + '.h5df'
    # origin_dataset_light = get_d4rl_local(get_dataset(dataset_path, env.observation_space, env.action_space))
    # env_light = HalfCheetahLightEnv()
    dataset_path = './dataset/gym_back/halfcheetah_back_' + names[1][:-3].replace('-', '_') + '.h5df'
    origin_dataset_back = get_d4rl_local(get_dataset(dataset_path, env.observation_space, env.action_space))
    origin_dataset_back_terminals = origin_dataset_back.terminals
    origin_dataset_back_starts = np.concatenate([np.ones(1), origin_dataset_terminals[:-1]], axis=0).astype(np.int64)
    env_back = HalfCheetahBackEnv()
    # task_datasets = {'0': origin_dataset, '1': origin_dataset_wind, '2': origin_dataset_light, '3':origin_dataset_back}
    # envs = {'0': env, '1': env_wind, '2': env_light, '3':env_back}
    task_datasets = {'0': origin_dataset, '1':origin_dataset_back}
    envs = {'0': env, '1':env_back}
    nearest_indexes = {'0': origin_dataset_starts, '1': origin_dataset_back_starts}
    return split_gym(top_euclid, dataset_name, task_datasets, envs, nearest_indexes, compare_dim=3, device=device)

def split_hopper(top_euclid, dataset_name, device='cuda:0'):
    origin_dataset, env = get_d4rl(dataset_name)
    origin_dataset_terminals = origin_dataset.terminals
    origin_dataset_starts = np.concatenate([np.ones(1), origin_dataset_terminals[:-1]], axis=0).astype(np.int64)
    names = dataset_name.split('-', 1)
    # dataset_path = './dataset/gym_back/hopper_wind_' + names[1][:-3].replace('-', '_') + '.h5df'
    # origin_dataset_wind = get_d4rl_local(get_dataset(dataset_path, env.observation_space, env.action_space))
    # env_wind = HopperWindEnv()
    # dataset_path = './dataset/gym_back/hopper_light_' + names[1][:-3].replace('-', '_') + '.h5df'
    # origin_dataset_light = get_d4rl_local(get_dataset(dataset_path, env.observation_space, env.action_space))
    # env_light = HopperLightEnv()
    dataset_path = './dataset/gym_back/hopper_back_' + names[1][:-3].replace('-', '_') + '.h5df'
    origin_dataset_back = get_d4rl_local(get_dataset(dataset_path, env.observation_space, env.action_space))
    origin_dataset_back_terminals = origin_dataset_back.terminals
    origin_dataset_back_starts = np.concatenate([np.ones(1), origin_dataset_terminals[:-1]], axis=0).astype(np.int64)
    env_back = HopperBackEnv()
    # task_datasets = {'0': origin_dataset, '1': origin_dataset_wind, '2': origin_dataset_light, '3':origin_dataset_back}
    # envs = {'0': env, '1': env_wind, '2': env_light, '3':env_back}
    task_datasets = {'0': origin_dataset, '1':origin_dataset_back}
    envs = {'0': env, '1':env_back}
    nearest_indexes = {'0': origin_dataset_starts, '1': origin_dataset_back_starts}
    return split_gym(top_euclid, dataset_name, task_datasets, envs, nearest_indexes, compare_dim=3, device=device)

def split_walker(top_euclid, dataset_name, device='cuda:0'):
    origin_dataset, env = get_d4rl(dataset_name)
    origin_dataset_terminals = origin_dataset.terminals
    origin_dataset_starts = np.concatenate([np.ones(1), origin_dataset_terminals[:-1]], axis=0).astype(np.int64)
    names = dataset_name.split('-', 1)
    # dataset_path = './dataset/gym_back/walker2d_wind_' + names[1][:-3].replace('-', '_') + '.h5df'
    # origin_dataset_wind = get_d4rl_local(get_dataset(dataset_path, env.observation_space, env.action_space))
    # env_wind = Walker2dWindEnv()
    # dataset_path = './dataset/gym_back/walker2d_light_' + names[1][:-3].replace('-', '_') + '.h5df'
    # origin_dataset_light = get_d4rl_local(get_dataset(dataset_path, env.observation_space, env.action_space))
    # env_light = Walker2dLightEnv()
    dataset_path = './dataset/gym_back/walker2d_back_' + names[1][:-3].replace('-', '_') + '.h5df'
    origin_dataset_back = get_d4rl_local(get_dataset(dataset_path, env.observation_space, env.action_space))
    origin_dataset_back_terminals = origin_dataset_back.terminals
    origin_dataset_back_starts = np.concatenate([np.ones(1), origin_dataset_terminals[:-1]], axis=0).astype(np.int64)
    env_back = Walker2dBackEnv()
    # task_datasets = {'0': origin_dataset, '1': origin_dataset_wind, '2': origin_dataset_light, '3':origin_dataset_back}
    # envs = {'0': env, '1': env_wind, '2': env_light, '3':env_back}
    task_datasets = {'0': origin_dataset, '1':origin_dataset_back}
    envs = {'0': env, '1':env_back}
    nearest_indexes = {'0': origin_dataset_starts, '1': origin_dataset_back_starts}
    return split_gym(top_euclid, dataset_name, task_datasets, envs, nearest_indexes, compare_dim=3, device=device)

def split_gym(top_euclid, dataset_name, task_datasets, envs, nearest_indexes, compare_dim=3, device='cuda:0'):

    # fig = plt.figure()
    # obs = env.reset()
    # for episode in origin_dataset.episodes:
    #     x = episode.observations[:, 0]
    #     y = episode.observations[:, 1]
    #     plt.plot(x, y)
    # plt.plot(obs[0], obs[1], 'o', markersize=4)
    # plt.savefig('try_' + dataset_name + '.png')
    # plt.close('all')
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
    return origin_task_datasets, taskid_task_datasets, envs, [None for _ in range(task_nums)], nearest_indexes, real_action_size, real_observation_size, task_nums
