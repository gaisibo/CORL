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

def split_cheetah(top_euclid, dataset_name):
    origin_dataset, env = get_d4rl(dataset_name)
    names = dataset_name.split('-', 1)
    dataset_path = './dataset/gym_back/cheetah_bach_' + names[1][:-3].replace('-', '_') + '.hdf5'
    origin_dataset_back = get_d4rl_local(get_dataset(dataset_path, env.observation_space, env.action_space))
    from mygym.envs.halfcheetah_back import HalfCheetahEnvBack
    env_back = HalfCheetahEnvBack()
    return split_gym(top_euclid, dataset_name, origin_dataset, env, origin_dataset_back, env_back, compare_dim=1)

def split_hopper(top_euclid, dataset_name):
    origin_dataset, env = get_d4rl(dataset_name)
    names = dataset_name.split('-', 1)
    dataset_path = './dataset/gym_back/hopper_bach_' + names[1][:-3].replace('-', '_') + '.hdf5'
    origin_dataset_back = get_d4rl_local(get_dataset(dataset_path, env.observation_space, env.action_space))
    dataset_name = dataset_name + '-v0'
    from mygym.envs.hopper_back import HopperEnvBack
    env_back = HopperEnvBack()
    return split_gym(top_euclid, dataset_name, origin_dataset, env, origin_dataset_back, env_back, compare_dim=3)

def split_walker(top_euclid, dataset_name):
    origin_dataset, env = get_d4rl(dataset_name)
    names = dataset_name.split('-', 1)
    dataset_path = './dataset/gym_back/walker_bach_' + names[1][:-3].replace('-', '_') + '.hdf5'
    origin_dataset_back = get_d4rl_local(get_dataset(dataset_path, env.observation_space, env.action_space))
    dataset_name = dataset_name + '-v0'
    from mygym.envs.walker2d_back import Walker2dEnvBack
    env_back = Walker2dEnvBack()
    return split_gym(top_euclid, dataset_name, origin_dataset, env, origin_dataset_back, env_back, compare_dim=3)

def split_gym(top_euclid, dataset_name, origin_dataset, env, origin_dataset_back, env_back, compare_dim=3):

    # fig = plt.figure()
    # obs = env.reset()
    # for episode in origin_dataset.episodes:
    #     x = episode.observations[:, 0]
    #     y = episode.observations[:, 1]
    #     plt.plot(x, y)
    # plt.plot(obs[0], obs[1], 'o', markersize=4)
    # plt.savefig('try_' + dataset_name + '.png')
    # plt.close('all')
    task_datasets = {0: origin_dataset, 1: origin_dataset_back}
    envs = {0: env, 1: env_back}
    task_nums = 2

    changed_task_datasets = dict()
    taskid_task_datasets = dict()
    origin_task_datasets = dict()
    indexes_euclids = dict()
    real_action_size = 0
    real_observation_size = 0
    for dataset_num, dataset in task_datasets.items():
        transitions = [transition for episode in dataset.episodes for transition in episode]
        observations = np.stack([transition.observation for transition in transitions], axis=0)
        indexes_euclid = similar_euclid(torch.from_numpy(dataset.observations).cuda(), torch.from_numpy(observations).cuda(), dataset_name, dataset_num, compare_dim=compare_dim)[:, :top_euclid]
        real_action_size = dataset.actions.shape[1]
        task_id_numpy = np.eye(task_nums)[dataset_num].squeeze()
        task_id_numpy = np.broadcast_to(task_id_numpy, (dataset.observations.shape[0], task_nums))
        real_observation_size = dataset.observations.shape[1]
        # 用action保存一下indexes_euclid，用state保存一下task_id
        changed_task_datasets[dataset_num] = MDPDataset(np.concatenate([dataset.observations, task_id_numpy], axis=1), np.concatenate([dataset.actions, indexes_euclid.cpu().numpy()], axis=1), dataset.rewards, dataset.terminals, dataset.episode_terminals)
        taskid_task_datasets[dataset_num] = MDPDataset(np.concatenate([dataset.observations, task_id_numpy], axis=1), dataset.actions, dataset.rewards, dataset.terminals, dataset.episode_terminals)
        origin_task_datasets[dataset_num] = MDPDataset(dataset.observations, dataset.actions, dataset.rewards, dataset.terminals, dataset.episode_terminals)
        indexes_euclids[dataset_num] = indexes_euclid
    # torch.save(task_datasets, dataset_name  + '.pt')

    original = 0
    return changed_task_datasets, envs, [None for _ in range(task_nums)], original, real_action_size, real_observation_size, indexes_euclids, task_nums