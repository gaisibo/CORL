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
from d3rlpy.metrics.scorer import dynamics_observation_prediction_error_scorer, dynamics_reward_prediction_error_scorer, dynamics_prediction_variance_scorer
from d3rlpy.dataset import MDPDataset
# from myd3rlpy.datasets import get_d4rl
from utils.k_means import kmeans
from dataset.split_navigate import split_navigate_antmaze_large_play_v0
from myd3rlpy.metrics.scorer import bc_error_scorer, td_error_scorer, evaluate_on_environment
from myd3rlpy.siamese_similar import similar_psi, similar_phi
from myd3rlpy.dynamics.probabilistic_ensemble_dynamics import ProbabilisticEnsembleDynamics


replay_name = ['observations', 'actions', 'rewards', 'next_observations', 'next_actions', 'next_rewards', 'terminals', 'means', 'std_logs', 'qs', 'phis', 'psis']
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

    # prepare algorithm
    if args.algos == 'co':
        from myd3rlpy.algos.comb import COMB
        co = COMB(use_gpu=True, batch_size=args.batch_size, n_action_samples=args.n_action_samples, cql_loss=args.cql_loss, q_bc_loss=args.q_bc_loss, td3_loss=args.td3_loss, policy_bc_loss=args.policy_bc_loss, mb_generate=args.mb_generate)
    else:
        raise NotImplementedError
    experiment_name = "COMB_dynamics"
    algos_name = "_orl" if args.orl else "_noorl"
    algos_name += "_mb_generate" if args.mb_generate else "_no_mb_generate"
    algos_name += "_mb_replay" if args.mb_replay else "_no_mb_replay"
    algos_name += args.dataset_name

    if not args.eval:
        replay_datasets = dict()
        save_datasets = dict()
        eval_datasets = dict()
        for task_id, dataset in task_datasets.items():
            eval_datasets[task_id] = dataset
            draw_path = args.model_path + algos_name + '_trajectories_' + str(task_id) + '_'

            dynamics = ProbabilisticEnsembleDynamics(task_id=task_id, original=original, learning_rate=1e-4, use_gpu=True, id_size=task_nums)
            dynamics.fit(
                origin_task_datasets[task_id].episodes,
                n_epochs=1000,
                scorers={
                   'observation_error': dynamics_observation_prediction_error_scorer,
                   'reward_error': dynamics_reward_prediction_error_scorer,
                   'variance': dynamics_prediction_variance_scorer,
                },
                pretrain=True,
            )
            dynamics.save_model(args.model_path + algos_name + '_' + str(task_id) + '.pt')

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
