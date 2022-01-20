import matplotlib.pyplot as plt
import sys
from copy import deepcopy
import random
from typing_extensions import final
import numpy as np
from numpy.lib.utils import who
import torch

from d3rlpy.datasets import get_d4rl
from d3rlpy.dataset import MDPDataset
from myd3rlpy.siamese_similar import similar_euclid


def split_navigate_antmaze_umaze_v2(task_split_type, top_euclid, device, dense):
    origin_dataset, env = get_d4rl('antmaze-umaze-v2')
    dataset_name = 'antmaze-umaze-v2'
    task_nums = 3
    dense = dense == 'dense'
    end_points = [np.array([8, 2]), np.array([0, 8]), np.array([8, 8])]
    return split_antmaze(origin_dataset, env, dataset_name, task_nums, end_points, task_split_type, top_euclid, device, dense)

def split_navigate_antmaze_medium_v2(task_split_type, top_euclid, device, dense):
    origin_dataset, env = get_d4rl('antmaze-medium-play-v2')
    dataset_name = 'antmaze-medium-play-v2'
    task_nums = 3
    dense = dense == 'dense'
    end_points = [np.array([5, 0]), np.array([0, 20]), np.array([20, 5])]
    return split_antmaze(origin_dataset, env, dataset_name, task_nums, end_points, task_split_type, top_euclid, device, dense)

def split_navigate_antmaze_large_v2(task_split_type, top_euclid, device, dense):
    origin_dataset, env = get_d4rl('antmaze-large-play-v2')
    dataset_name = 'antmaze-large-play-v2'
    task_nums = 3
    dense = dense == 'dense'
    end_points = [np.array([0, 25]), np.array([20, 15]), np.array([35, 0])]
    return split_antmaze(origin_dataset, env, dataset_name, task_nums, end_points, task_split_type, top_euclid, device, dense)

def split_antmaze(origin_dataset, env, dataset_name, task_nums, end_points, task_split_type, top_euclid, device, dense):
    fig = plt.figure()
    obs = env.reset()
    for episode in origin_dataset.episodes:
        x = episode.observations[:, 0]
        y = episode.observations[:, 1]
        plt.plot(x, y)
    plt.plot(obs[0], obs[1], 'o', markersize=4)
    plt.savefig('try_' + dataset_name + '.png')
    plt.close('all')
    task_datasets = {k: [] for k in range(task_nums)}
    if task_split_type == 'directed':
        for episode in origin_dataset.episodes:
            if len(episode) == 1:
                continue
            position = episode[-1].observation[:2]
            end_point_distances = []
            for end_point in end_points:
                end_point_distance = np.linalg.norm(position, end_point)
                end_point_distances.append(end_point_distance)
            min_index = np.argmin(end_point_distance)
            task_datasets[min_index].append(episode)
    else:
        all_episodes = origin_dataset.episodes
        episodes = []
        for episode in all_episodes:
            if len(episode) > 1:
                episodes.append(episode)
        # episodes = all_episodes
        random.shuffle(episodes)
        task_length = len(episodes) // task_nums
        task_mod = len(episodes) - task_nums * task_length
        i = 0
        for task_num in range(task_nums):
            if task_num < task_mod:
                task_datasets[task_num] = episodes[i: i + task_length + 1]
                i += task_length + 1
            else:
                task_datasets[task_num] = episodes[i: i + task_length]
                i += task_length
    task_datasets_ = {}
    envs = {}
    for index, episodes in task_datasets.items():
        envs[index] = deepcopy(env)
        envs[index]._goal = end_points[index]
        envs[index].target_goal = end_points[index]
        observations = np.concatenate([episode.observations for episode in episodes], axis=0)
        actions = np.concatenate([episode.actions for episode in episodes], axis=0)
        # rewards = np.concatenate([episode.rewards for episode in episodes], axis=0)
        rewards = np.where(np.linalg.norm(observations[:, :2] - end_points[index], axis=1) < 0.5, 1, 0)
        terminals = [np.zeros(episode.observations.shape[0]) for episode in episodes]
        for terminal in terminals:
            terminal[-1] = 1
        terminals = np.concatenate(terminals, axis=0)
        task_datasets_[index] = MDPDataset(observations, actions, rewards, terminals)
    task_datasets = task_datasets_

    changed_task_datasets = dict()
    taskid_task_datasets = dict()
    origin_task_datasets = dict()
    indexes_euclids = dict()
    real_action_size = 0
    real_observation_size = 0
    for dataset_num, dataset in task_datasets.items():
        transitions = [transition for episode in dataset.episodes for transition in episode]
        observations = np.stack([transition.observation for transition in transitions], axis=0)
        indexes_euclid = similar_euclid(torch.from_numpy(dataset.observations).cuda(), torch.from_numpy(observations).cuda(), dataset_name, dataset_num)[:, :top_euclid]
        real_action_size = dataset.actions.shape[1]
        task_id_numpy = np.eye(task_nums)[dataset_num].squeeze()
        task_id_numpy = np.broadcast_to(task_id_numpy, (dataset.observations.shape[0], task_nums))
        real_observation_size = dataset.observations.shape[1]
        # 用action保存一下indexes_euclid，用state保存一下task_id
        changed_task_datasets[dataset_num] = MDPDataset(np.concatenate([dataset.observations, task_id_numpy], axis=1), np.concatenate([dataset.actions, indexes_euclid.cpu().numpy()], axis=1), dataset.rewards, dataset.terminals, dataset.episode_terminals)
        taskid_task_datasets[dataset_num] = MDPDataset(np.concatenate([dataset.observations, task_id_numpy], axis=1), dataset.actions, dataset.rewards, dataset.terminals, dataset.episode_terminals)
        origin_task_datasets[dataset_num] = MDPDataset(dataset.observations, dataset.actions, dataset.rewards, dataset.terminals, dataset.episode_terminals)
        indexes_euclids[dataset_num] = indexes_euclid
    torch.save(dataset_name + '_' + task_split_type + '.pt', task_datasets)

    original = 0
    return origin_dataset, changed_task_datasets, taskid_task_datasets, origin_task_datasets, envs, end_points, original, real_action_size, real_observation_size, indexes_euclids, task_nums
