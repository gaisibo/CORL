import sys
from copy import deepcopy
import random
import numpy as np
import torch

from d3rlpy.datasets import get_d4rl
from d3rlpy.dataset import MDPDataset
from myd3rlpy.siamese_similar import similar_euclid
from dataset.split_antmaze import split_antmaze


def split_navigate_maze_open_v0(task_split_type, top_euclid, device, dense):
    origin_dataset, env = get_d4rl('maze-open-v0')
    dataset_name = 'maze-open-v0'
    task_nums = 7
    dense = dense == 'dense'
    end_points = [np.array([32.41604, 24.43354]), np.array([21.3771, 17.4113]), np.array([20.8545, 25.0958]), np.array([4.5582, 17.7067]), np.array([18.1493, 8.9290]), np.array([0.1346, 13.3144]), np.array([37.0817, 12.0133])]
    return split_antmaze(origin_dataset, env, dataset_name, task_nums, end_points, task_split_type, top_euclid, device, dense)

def split_navigate_maze_umaze_v1(task_split_type, top_euclid, device, dense):
    origin_dataset, env = get_d4rl('maze-umaze-v1')
    dataset_name = 'maze-umaze-v1'
    task_nums = 7
    dense = dense == 'dense'
    end_points = [np.array([32.41604, 24.43354]), np.array([21.3771, 17.4113]), np.array([20.8545, 25.0958]), np.array([4.5582, 17.7067]), np.array([18.1493, 8.9290]), np.array([0.1346, 13.3144]), np.array([37.0817, 12.0133])]
    return split_antmaze(origin_dataset, env, dataset_name, task_nums, end_points, task_split_type, top_euclid, device, dense)

def split_navigate_maze_medium_v1(task_split_type, top_euclid, device, dense):
    origin_dataset, env = get_d4rl('maze-medium-v1')
    dataset_name = 'maze-medium-v1'
    task_nums = 7
    dense = dense == 'dense'
    end_points = [np.array([32.41604, 24.43354]), np.array([21.3771, 17.4113]), np.array([20.8545, 25.0958]), np.array([4.5582, 17.7067]), np.array([18.1493, 8.9290]), np.array([0.1346, 13.3144]), np.array([37.0817, 12.0133])]
    return split_antmaze(origin_dataset, env, dataset_name, task_nums, end_points, task_split_type, top_euclid, device, dense)

def split_navigate_maze_large_v1(task_split_type, top_euclid, device, dense):
    origin_dataset, env = get_d4rl('maze-large-v1')
    dataset_name = 'maze-large-v1'
    task_nums = 7
    dense = dense == 'dense'
    end_points = [np.array([32.41604, 24.43354]), np.array([21.3771, 17.4113]), np.array([20.8545, 25.0958]), np.array([4.5582, 17.7067]), np.array([18.1493, 8.9290]), np.array([0.1346, 13.3144]), np.array([37.0817, 12.0133])]
    return split_antmaze(origin_dataset, env, dataset_name, task_nums, end_points, task_split_type, top_euclid, device, dense)
