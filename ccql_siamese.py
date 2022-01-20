import os
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


replay_name = ['observations', 'actions', 'rewards', 'next_observations', 'terminals', 'means', 'std_logs', 'qs', 'phis', 'psis']
# 暂时只练出来一个。
def main(args, device):
    np.set_printoptions(precision=1, suppress=True)
    if args.dataset in ['antmaze_umaze', 'antmaze_medium', 'antmaze_large', 'maze2d_open', 'maze2d_umaze', 'maze2d_medium', 'maze2d_large']:
        if args.dataset == 'antmaze_umaze':
            from dataset.split_antmaze import split_navigate_antmaze_umaze_v2 as split_navigate
        elif args.dataset == 'antmaze_medium':
            from dataset.split_antmaze import split_navigate_antmaze_medium_v2 as split_navigate
        elif args.dataset == 'antmaze_large':
            from dataset.split_antmaze import split_navigate_antmaze_large_v2 as split_navigate
        if args.dataset == 'maze2d_open':
            from dataset.split_maze import split_navigate_maze2d_open_v0 as split_navigate
        elif args.dataset == 'maze2d_umaze':
            from dataset.split_maze import split_navigate_maze2d_umaze_v1 as split_navigate
        elif args.dataset == 'maze2d_medium':
            from dataset.split_maze import split_navigate_maze2d_medium_v1 as split_navigate
        elif args.dataset == 'maze2d_large':
            from dataset.split_maze import split_navigate_maze2d_large_v1 as split_navigate
        else:
            raise NotImplementedError
        task_datasets, envs, end_points, original, real_action_size, real_observation_size, indexes_euclids, task_nums = split_navigate(args.task_split_type, args.top_euclid, device, args.dense)
    elif args.dataset in ['hopper_expert_v0', 'hopper_medium_v0', 'hopper_medium_expert_v0', 'hopper_medium_replay_v0', 'hopper_random_v0', 'halfcheetah_expert_v0', 'halfcheetah_medium_v0', 'halfcheetah_medium_expert_v0', 'halfcheetah_medium_replay_v0', 'halfcheetah_random_v0', 'walker2d_expert_v0', 'walker2d_medium_v0', 'walker2d_medium_expert_v0', 'walker2d_medium_replay_v0', 'walker2d_random_v0']:
        if args.dataset in ['hopper_expert_v0', 'hopper_medium_v0', 'hopper_medium_expert_v0', 'hopper_medium_replay_v0', 'hopper_random_v0']:
            from dataset.split_gym import split_hopper as split_gym
        elif args.dataset in ['halfcheetah_expert_v0', 'halfcheetah_medium_v0', 'halfcheetah_medium_expert_v0', 'halfcheetah_medium_replay_v0', 'halfcheetah_random_v0']:
            from dataset.split_gym import split_cheetah as split_gym
        elif args.dataset in ['walker2d_expert_v0', 'walker2d_medium_v0', 'walker2d_medium_expert_v0', 'walker2d_medium_replay_v0', 'walker2d_random_v0']:
            from dataset.split_gym import split_walker as split_gym
        else:
            raise NotImplementedError
        task_datasets, envs, end_points, original, real_action_size, real_observation_size, indexes_euclids, task_nums = split_gym(args.top_euclid, args.dataset.replace('_', '-'))
    else:
        raise NotImplementedError

    # prepare algorithm
    if args.algos == 'co':
        from myd3rlpy.algos.co import CO
        co = CO(use_gpu=True, batch_size=args.batch_size, n_action_samples=args.n_action_samples, id_size=task_nums, cql_loss=args.cql_loss, q_bc_loss=args.q_bc_loss, td3_loss=args.td3_loss, policy_bc_loss=args.policy_bc_loss, generate_type=args.generate_type, reduce_replay=args.reduce_replay, double_data=args.double_data, change_reward=args.change_reward)
    else:
        raise NotImplementedError
    experiment_name = "CO"
    algos_name = "_orl" if args.orl else "_noorl"
    algos_name += "_" + args.generate_type
    algos_name += "_" + args.replay_type
    algos_name += '_' + args.dataset

    if not args.eval:
        replay_datasets = dict()
        save_datasets = dict()
        eval_datasets = dict()
        for task_id, dataset in task_datasets.items():
            eval_datasets[task_id] = dataset
            draw_path = args.model_path + algos_name + '_trajectories_' + str(task_id)

            co._origin = original
            # train
            co.fit(
                task_id,
                dataset,
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
            )
            if args.algos == 'co':
                if args.replay_type == 'siamese':
                    replay_datasets[task_id], save_datasets[task_id] = co.generate_replay_data_phi(task_id, task_datasets[task_id], original, in_task=False, max_save_num=args.max_save_num, real_action_size=real_action_size, real_observation_size=real_observation_size)
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
    parser.add_argument("--dataset", default='ant_maze', type=str, choices=["antmaze_umaze", "antmaze_medium", "antmaze_large", "maze2d_open", "maze2d_umaze", "maze2d_medium", "maze2d_large", 'hopper_expert_v0', 'hopper_medium_v0', 'hopper_medium_expert_v0', 'hopper_medium_replay_v0', 'hopper_random_v0', 'halfcheetah_expert_v0', 'halfcheetah_medium_v0', 'halfcheetah_medium_expert_v0', 'halfcheetah_medium_replay_v0', 'halfcheetah_random_v0', 'walker2d_expert_v0', 'walker2d_medium_v0', 'walker2d_medium_expert_v0', 'walker2d_medium_replay_v0', 'walker2d_random_v0'])
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
    parser.add_argument('--algos', default='co', type=str)
    parser.add_argument('--eval', action='store_true')
    parser.add_argument('--test', action='store_true')
    parser.add_argument("--n_epochs", default=1000, type=int)
    parser.add_argument("--n_action_samples", default=4, type=int)
    parser.add_argument('--top_euclid', default=64, type=int)
    parser.add_argument('--orl', default='orl', choices=['orl', 'no_orl'])
    parser.add_argument('--replay_type', default='siamese', type=str, choices=['siamese', 'random'])
    parser.add_argument('--generate_type', default='siamese', type=str, choices=['siamese', 'random'])
    parser.add_argument('--reduce_replay', default='retrain', type=str, choices=['retrain', 'no_retrain'])
    parser.add_argument('--double_data', default='double_data', type=str)
    parser.add_argument('--change_reward', default='change', type=str)
    parser.add_argument('--dense', default='dense', type=str)
    args = parser.parse_args()
    args.model_path = 'd3rlpy_' + args.replay_type + '_' + args.generate_type + '_' + args.orl + '_' + args.reduce_replay + '_' + args.dataset + '_' + ('test' if args.test else ('train' if not args.eval else 'eval'))
    if not os.path.exists(args.model_path):
        os.makedirs(args.model_path)
    args.model_path +=  '/model_'
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
