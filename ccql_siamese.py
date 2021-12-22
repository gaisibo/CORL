import argparse
import json
import random
from collections import namedtuple
import pickle
import time
from functools import partial
from dataset.d4rl import cut_antmaze
from envs import HalfCheetahDirEnv
from utils.utils import ReplayBuffer
import numpy as np
import torch

import d3rlpy
from d3rlpy.ope import FQE
from d3rlpy.metrics.scorer import initial_state_value_estimation_scorer
from d3rlpy.metrics.scorer import soft_opc_scorer
from d3rlpy.dataset import MDPDataset
# from myd3rlpy.datasets import get_d4rl
from utils.siamese_similar import similar_psi, similar_phi
from utils.k_means import kmeans
from dataset.split_navigate import split_navigate_antmaze_large_play_v0
from myd3rlpy.metrics.scorer import bc_error_scorer, td_error_scorer


def main(args, device):
#     with open(f"./{args.task_config}", "r") as f:
#         task_config = json.load(
#             f, object_hook=lambda d: namedtuple("X", d.keys())(*d.values())
#         )
#     env = get_env(task_config)
#     buffers = build_networks_and_buffers(args, env, task_config)
#     task_num = len(buffers)
#     datasets = []
#     for task_id, buffer_ in enumerate(buffers):
#         task_id_np = np.zeros((buffer_._obs.shape[0], task_num), dtype=np.float32)
#         task_id_np[:, task_id] = 1
#         buffer_._obs = np.hstack((buffer_._obs, task_id_np))
#         datasets.append(MDPDataset(buffer_._obs, buffer_._actions, buffer_._rewards, buffer_._terminals))
#         break
    origin_dataset, env, task_datasets, end_points, original, real_action_size, real_observation_size, indexes_euclids = split_navigate_antmaze_large_play_v0(args.task_split_type, device)
    np.set_printoptions(precision=1, suppress=True)

    # prepare algorithm
    if args.algos == 'co':
        from myd3rlpy.algos.co import CO
        from myd3rlpy.finish_task.finish_task_co import finish_task_co
        co = CO(use_gpu=True, batch_size=args.batch_size)
    else:
        raise NotImplementedError

    replay_datasets = dict()
    for dataset_num, dataset in task_datasets.items():
        episodes = dataset.episodes
        # train
        co.fit(
            dataset,
            replay_datasets,
            dataset,
            real_action_size = real_action_size,
            real_observation_size = real_observation_size,
            eval_episodes=dataset,
            replay_eval_episodess = replay_datasets,
            n_epochs=1,
            scorers={
                # 'environment': evaluate_on_environment(env),
                'td_error': partial(td_error_scorer, real_action_size=real_action_size)
            },
            replay_scorers={
                'bc_error': partial(bc_error_scorer, real_action_size=real_action_size)
            }
        )
        assert co._impl is not None
        assert co._impl._q_func is not None
        assert co._impl._policy is not None
        if args.algos == 'co':
            replay_datasets[dataset_num] = finish_task_co(dataset_num, dataset, original, co, indexes_euclids[dataset_num], real_action_size, args, device)

    for dataset_num, dataset in task_datasets.items():
        # off-policy evaluation algorithm
        fqe = FQE(algo=co)

        # metrics to evaluate with

        # train estimators to evaluate the trained policy
        fqe.fit(dataset.episodes,
                eval_episodes=dataset.episodes,
                n_epochs=1,
                scorers={
                   'init_value': initial_state_value_estimation_scorer,
                   'soft_opc': soft_opc_scorer(return_threshold=600)
                })

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Experimental evaluation of lifelong PG learning')
    parser.add_argument('--inner_buffer_size', default=-1, type=int)
    parser.add_argument('--task_config', default='task_config/cheetah_dir.json', type=str)
    parser.add_argument('--siamese_hidden_size', default=100, type=int)
    parser.add_argument('--near_threshold', default=1, type=float)
    parser.add_argument('--siamese_threshold', default=1, type=float)
    parser.add_argument('--eval_batch_size', default=256, type=int)
    parser.add_argument('--batch_size', default=256, type=int)
    parser.add_argument('--sample_times', default=4, type=int)
    parser.add_argument('--task_split_type', default='undirected', type=str)
    parser.add_argument('--task_nums', default=7, type=int)
    parser.add_argument('--dataset_name', default='antmaze-large-play-v0', type=str)
    parser.add_argument('--algos', default='co', type=str)
    args = parser.parse_args()
    global DATASET_PATH
    DATASET_PATH = './.d4rl/datasets/'
    device = torch.device('cuda:0')
    random.seed(12345)
    np.random.seed(12345)
    main(args, device)
