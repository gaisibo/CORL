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
from dataset.split_navigate import split_navigate_antmaze_large_play_v0
from myd3rlpy.metrics.scorer import bc_error_scorer, td_error_scorer, evaluate_on_environment
from myd3rlpy.siamese_similar import similar_psi, similar_phi
from myd3rlpy.dynamics.probabilistic_ensemble_dynamics import ProbabilisticEnsembleDynamics


replay_name = ['observations', 'actions', 'rewards', 'next_observations', 'next_actions', 'next_rewards', 'terminals', 'means', 'std_logs', 'qs', 'phis', 'psis']
def main(args, device):
    np.set_printoptions(precision=1, suppress=True)
    if args.dataset == 'ant_maze':
        origin_dataset, task_datasets, taskid_task_datasets, origin_task_datasets, envs, end_points, original, real_action_size, real_observation_size, indexes_euclids, task_nums = split_navigate_antmaze_large_play_v0(args.task_split_type, args.top_euclid, device)
    else:
        assert False

    # prepare algorithm
    if args.algos == 'co':
        from myd3rlpy.algos.comb import COMB
        co = COMB(use_gpu=True, batch_size=args.batch_size, n_action_samples=args.n_action_samples, cql_loss=args.cql_loss, q_bc_loss=args.q_bc_loss, td3_loss=args.td3_loss, policy_bc_loss=args.policy_bc_loss)
    else:
        raise NotImplementedError
    experiment_name = "COMB_"
    algos_name = "update" if args.use_phi_update else "noupdate"
    algos_name += "_replay" if args.use_phi_replay else "_noreplay"
    algos_name += "_orl" if args.orl else "_noorl"
    algos_name += '_pretrain' if args.pretrain_phi_epoch else ""

    if not args.eval:
        replay_datasets = dict()
        eval_datasets = dict()
        for dataset_num, dataset in task_datasets.items():
            eval_datasets[dataset_num] = dataset
            draw_path = args.model_path + algos_name + '_trajectories_' + str(dataset_num) + '_'

            dynamics = ProbabilisticEnsembleDynamicsWithLogStd(learning_rate=1e-4, use_gpu=True)
# same as algorithms
            co._dynamics = dynamics
            co._origin = original
            # train
            co.fit(
                dataset_num,
                dataset,
                origin_task_datasets[dataset_num],
                replay_datasets,
                real_action_size = real_action_size,
                real_observation_size = real_observation_size,
                eval_episodess=eval_datasets,
                n_epochs=args.n_epochs if not args.test else 1,
                experiment_name=experiment_name + algos_name,
                scorers={
                    "real_env": evaluate_on_environment(envs, end_points, task_nums, draw_path),
                },
            )
            if args.algos == 'co':
                if args.use_mb_generate:
                    replay_datasets[dataset_num] = co.generate_replay_data(dataset_num, origin_task_datasets[dataset_num], original, in_task=False, max_save_num=args.max_save_num)
                else:
                    replay_datasets[dataset_num] = co.generate_new_data_random(dataset_num, origin_task_datasets[dataset_num], args.max_save_num)
            else:
                raise NotImplementedError
            co.save_model(args.model_path + algos_name + '_' + str(dataset_num) + '.pt')
            if args.test and dataset_num >= 1:
                break
        torch.save(replay_datasets, f=args.model_path + algos_name + '_datasets.pt')
    else:
        assert args.model_path
        eval_datasets = dict()
        if co._impl is None:
            co.create_impl([real_observation_size + task_nums], real_action_size)
        for dataset_num, dataset in task_datasets.items():
            eval_datasets[dataset_num] = dataset
            co.load_model(args.model_path + algos_name + '_' + str(dataset_num) + '.pt')
            replay_datasets = torch.load(args.model_path + algos_name + '_datasets.pt')
            co.test(
                replay_datasets,
                eval_episodess=eval_datasets,
                scorers={
                    # 'environment': evaluate_on_environment(env),
                    "real_env": evaluate_on_environment(envs, end_points, task_nums, draw_path),
                },
            )

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
    parser.add_argument('--pretrain_phi_epoch', default=0, type=int)
    parser.add_argument("--n_epochs", default=200, type=int)
    parser.add_argument("--sample_num", default=4, type=int)
    parser.add_argument('--top_euclid', default=8, type=int)
    orl_parser = parser.add_mutually_exclusive_group(required=True)
    orl_parser.add_argument('--orl', dest='orl', action='store_true')
    orl_parser.add_argument('--no_orl', dest='orl', action='store_false')
    use_phi_replay_parser = parser.add_mutually_exclusive_group(required=True)
    use_phi_replay_parser.add_argument('--use_phi_replay', dest='use_phi_replay', action='store_true')
    use_phi_replay_parser.add_argument('--no_use_phi_replay', dest='use_phi_replay', action='store_false')
    use_phi_update_parser = parser.add_mutually_exclusive_group(required=True)
    use_phi_update_parser.add_argument('--use_phi_update', dest='use_phi_update', action='store_true')
    use_phi_update_parser.add_argument('--no_use_phi_update', dest='use_phi_update', action='store_false')
    args = parser.parse_args()
    args.model_path = 'd3rlpy_mb_' + ('test' if args.test else ('train' if not args.eval else 'eval')) + '/model_'
    if args.orl:
        args.cql_loss = True
        args.td3_loss = True
        args.q_bc_loss = False
        args.policy_bc_loss = False
        args.phi_bc_loss = False
        args.psi_bc_loss = False
    else:
        args.cql_loss = False
        args.td3_loss = False
        args.q_bc_loss = True
        args.policy_bc_loss = True
        args.phi_bc_loss = True
        args.psi_bc_loss = True
    global DATASET_PATH
    DATASET_PATH = './.d4rl/datasets/'
    device = torch.device('cuda:0')
    random.seed(12345)
    np.random.seed(12345)
    main(args, device)
