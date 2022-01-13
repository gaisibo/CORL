import sys
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
from d3rlpy.metrics.scorer import soft_opc_scorer, initial_state_value_estimation_scorer
from d3rlpy.dataset import MDPDataset
# from myd3rlpy.datasets import get_d4rl
from utils.k_means import kmeans
from myd3rlpy.metrics.scorer import bc_error_scorer, td_error_scorer, evaluate_on_environment
from myd3rlpy.siamese_similar import similar_psi, similar_phi
from myd3rlpy.dynamics.probabilistic_ensemble_dynamics import ProbabilisticEnsembleDynamics


replay_name = ['observations', 'actions', 'rewards', 'next_observations', 'next_actions', 'next_rewards', 'terminals', 'means', 'std_logs', 'qs', 'phis', 'psis']
# 暂时只练出来一个。
# dynamics_path = ['d3rlpy_logs/ProbabilisticEnsembleDynamics_20220110095933/model_1407000.pt' for _ in range(7)]
dynamics_path = [None for _ in range(4)]
def main(args, device):
    np.set_printoptions(precision=1, suppress=True)
    if args.dataset == 'ant_maze':
        from dataset.split_navigate import split_navigate_antmaze_large_play_v0
        origin_dataset, task_datasets, taskid_task_datasets, origin_task_datasets, envs, end_points, original, real_action_size, real_observation_size, indexes_euclids, task_nums = split_navigate_antmaze_large_play_v0(args.task_split_type, args.top_euclid, device)
    elif args.dataset == 'maze':
        from dataset.split_maze import split_navigate_maze_large_dense_v1
        origin_dataset, task_datasets, taskid_task_datasets, origin_task_datasets, envs, end_points, original, real_action_size, real_observation_size, indexes_euclids, task_nums = split_navigate_maze_large_dense_v1(args.task_split_type, args.top_euclid, device)
    else:
        assert False
    transitions = [transition for episodes in task_datasets[0].episodes for transition in episodes]
    indexes_euclids = task_datasets[0].actions[:, real_action_size:]
    max_indexes_euclids = np.max(indexes_euclids)

    # prepare algorithm
    if args.algos == 'co':
        from myd3rlpy.algos.comb import COMB
        co = COMB(use_gpu=True, batch_size=args.batch_size, n_action_samples=args.n_action_samples, cql_loss=args.cql_loss, q_bc_loss=args.q_bc_loss, td3_loss=args.td3_loss, policy_bc_loss=args.policy_bc_loss, mb_generate=args.mb_generate)
    else:
        raise NotImplementedError
    experiment_name = "COMB"
    algos_name = "_orl" if args.orl else "_noorl"
    algos_name += "_mb_generate" if args.mb_generate else "_no_mb_generate"
    algos_name += "_mb_replay" if args.mb_replay else "_no_mb_replay"
    algos_name += '_' + args.dataset_name

    if not args.eval:
        replay_datasets = dict()
        save_datasets = dict()
        eval_datasets = dict()
        for task_id, dataset in task_datasets.items():
            eval_datasets[task_id] = dataset
            draw_path = args.model_path + algos_name + '_trajectories_' + str(task_id) + '_'

            dynamics = ProbabilisticEnsembleDynamics(task_id=task_id, original=original, learning_rate=1e-4, use_gpu=True, id_size=task_nums)
            dynamics.create_impl([real_observation_size], real_action_size)
            if dynamics_path[task_id] is not None:
                dynamics.load_model(dynamics_path[task_id])
# same as algorithms
            co._dynamics = dynamics
            co._origin = original
            # train
            co.fit(
                task_id,
                dataset,
                origin_task_datasets[task_id],
                replay_datasets,
                original = original,
                real_action_size = real_action_size,
                real_observation_size = real_observation_size,
                eval_episodess=eval_datasets,
                n_epochs=args.n_epochs if not args.test else 1,
                experiment_name=experiment_name + algos_name,
                scorers={
                    "real_env": evaluate_on_environment(envs, end_points, task_nums, draw_path),
                },
                test=args.test,
                train_dynamics=dynamics_path[task_id] is None,
            )
            if args.algos == 'co':
                if args.mb_replay:
                    replay_datasets[task_id], save_datasets[task_id] = co.generate_replay_data(task_id, task_datasets[task_id], original, in_task=False, max_save_num=args.max_save_num, real_action_size=real_action_size, real_observation_size=real_observation_size)
                    print(f"len(replay_datasets[task_id]): {len(replay_datasets[task_id])}")
                else:
                    replay_datasets[task_id], save_datasets[task_id] = co.generate_replay_data_random(task_id, task_datasets[task_id], max_save_num=args.max_save_num, real_action_size=real_action_size, in_task=False)
                    print(f"replay_datasets[task_id].shape[0]: {replay_datasets[task_id].shape[0]}")
            else:
                raise NotImplementedError
            co.save_model(args.model_path + algos_name + '_' + str(task_id) + '.pt')
            if args.test and task_id >= 1:
                break
        torch.save(save_datasets, f=args.model_path + algos_name + '_datasets.pt')
    else:
        assert args.model_path
        eval_datasets = dict()
        if co._impl is None:
            co.create_impl([real_observation_size + task_nums], real_action_size)
        for task_id, dataset in task_datasets.items():
            draw_path = args.model_path + algos_name + '_trajectories_' + str(task_id) + '_'
            eval_datasets[task_id] = dataset
            co.load_model(args.model_path + algos_name + '_' + str(task_id) + '.pt')
            replay_datasets = torch.load(args.model_path + algos_name + '_datasets.pt')
            co.test(
                replay_datasets,
                eval_episodess=eval_datasets,
                scorers={
                    # 'environment': evaluate_on_environment(env),
                    "real_env": evaluate_on_environment(envs, end_points, task_nums, draw_path),
                },
            )
    print('finish')

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Experimental evaluation of lifelong PG learning')
    parser.add_argument("--dataset", default='ant_maze', type=str)
    parser.add_argument('--inner_buffer_size', default=-1, type=int)
    parser.add_argument('--task_config', default='task_config/cheetah_dir.json', type=str)
    parser.add_argument('--siamese_hidden_size', default=100, type=int)
    parser.add_argument('--near_threshold', default=1, type=float)
    parser.add_argument('--siamese_threshold', default=1, type=float)
    parser.add_argument('--eval_batch_size', default=256, type=int)
    parser.add_argument('--batch_size', default=256, type=int)
    parser.add_argument('--topk', default=4, type=int)
    parser.add_argument('--max_save_num', default=1000, type=int)
    parser.add_argument('--task_split_type', default='undirected', type=str)
    parser.add_argument('--dataset_name', default='antmaze-large-play-v0', type=str)
    parser.add_argument('--algos', default='co', type=str)
    parser.add_argument('--eval', action='store_true')
    parser.add_argument('--test', action='store_true')
    parser.add_argument("--n_epochs", default=1000, type=int)
    parser.add_argument("--n_action_samples", default=4, type=int)
    parser.add_argument('--top_euclid', default=64, type=int)
    orl_parser = parser.add_mutually_exclusive_group(required=True)
    orl_parser.add_argument('--orl', dest='orl', action='store_true')
    orl_parser.add_argument('--no_orl', dest='orl', action='store_false')
    mb_generate_parser = parser.add_mutually_exclusive_group(required=True)
    mb_generate_parser.add_argument('--mb_generate', dest='mb_generate', action='store_true')
    mb_generate_parser.add_argument('--no_mb_generate', dest='mb_generate', action='store_false')
    mb_replay_parser = parser.add_mutually_exclusive_group(required=True)
    mb_replay_parser.add_argument('--mb_replay', dest='mb_replay', action='store_true')
    mb_replay_parser.add_argument('--no_mb_replay', dest='mb_replay', action='store_false')
    args = parser.parse_args()
    args.model_path = 'd3rlpy_mb_' + ('test' if args.test else ('train' if not args.eval else 'eval')) + '/model_'
    if args.orl:
        args.cql_loss = True
        args.td3_loss = True
        args.q_bc_loss = False
        args.policy_bc_loss = False
    else:
        args.cql_loss = False
        args.td3_loss = False
        args.q_bc_loss = True
        args.policy_bc_loss = True
    global DATASET_PATH
    DATASET_PATH = './.d4rl/datasets/'
    device = torch.device('cuda:0')
    random.seed(12345)
    np.random.seed(12345)
    main(args, device)
