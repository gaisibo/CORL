import sys
from copy import deepcopy
import random
import numpy as np
import torch

from d3rlpy.datasets import get_d4rl
from d3rlpy.dataset import MDPDataset
from myd3rlpy.siamese_similar import similar_euclid
from dataset.split_antmaze import split_antmaze


def split_navigate_maze2d_open_v0(task_split_type, top_euclid, device, dense):
    origin_dataset, env = get_d4rl('maze2d-open-dense-v0')
    dataset_name = 'maze2d-open-dense-v0'
    dense = dense == 'dense'
    task_nums = 3
    end_points = [np.array(x) for x in [[1, 4], [3, 1], [3, 4]]]
    return split_antmaze(origin_dataset, env, dataset_name, task_nums, end_points, task_split_type, top_euclid, device, dense)

def split_navigate_maze2d_umaze_v1(task_split_type, top_euclid, device, dense):
    origin_dataset, env = get_d4rl('maze2d-umaze-dense-v1')
    dataset_name = 'maze2d-umaze-dense-v1'
    dense = dense == 'dense'
    task_nums = 3
    end_points = [np.array(x) for x in [[1, 3], [3, 1], [3, 3]]]
    return split_antmaze(origin_dataset, env, dataset_name, task_nums, end_points, task_split_type, top_euclid, device, dense)

def split_navigate_maze2d_medium_v1(task_split_type, top_euclid, device, dense):
    origin_dataset, env = get_d4rl('maze2d-medium-dense-v1')
    dataset_name = 'maze2d-medium-dense-v1'
    dense = dense == 'dense'
    task_nums = 3
    end_points = [np.array(x) for x in [[6, 1], [1, 6], [6, 6]]]
    return split_antmaze(origin_dataset, env, dataset_name, task_nums, end_points, task_split_type, top_euclid, device, dense)

def split_navigate_maze2d_large_v1(task_split_type, top_euclid, device, dense):
    origin_dataset, env = get_d4rl('maze2d-large-dense-v1')
    dataset_name = 'maze2d-large-dense-v1'
    dense = dense == 'dense'
    task_nums = 3
    end_points = [np.array(x) for x in [[7, 10], [1.1, 1], [7, 1.1]]]
    return split_antmaze(origin_dataset, env, dataset_name, task_nums, end_points, task_split_type, top_euclid, device, dense)
