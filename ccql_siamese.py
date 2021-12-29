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
    origin_dataset, task_datasets, taskid_task_datasets, envs, end_points, original, real_action_size, real_observation_size, indexes_euclids, task_nums = split_navigate_antmaze_large_play_v0(args.task_split_type, args.top_euclid, device)
    np.set_printoptions(precision=1, suppress=True)

    # prepare algorithm
    if args.algos == 'co':
        from myd3rlpy.algos.co import CO
        train_phi = True
        if not args.use_phi_update and not args.use_phi_replay:
            train_phi = False
        co = CO(use_gpu=True, batch_size=args.batch_size, use_phi_update=args.use_phi_update, train_phi=train_phi)
        if args.use_phi_replay:
            from myd3rlpy.finish_task.finish_task_co import finish_task_co as finish_task
        else:
            from myd3rlpy.finish_task.finish_task_mi import finish_task_mi as finish_task
    else:
        raise NotImplementedError
    experiment_name = "CO_"
    algos_name = "update" if args.use_phi_update else "noupdate"
    algos_name += '_'
    algos_name += "replay" if args.use_phi_replay else "noreplay"
    algos_name += '_pretrain' if args.pretrain_phi_epoch else ""

    if not args.eval:
        replay_datasets = dict()
        eval_datasets = dict()
        for dataset_num, dataset in task_datasets.items():
            eval_datasets[dataset_num] = dataset
            # train
            co.fit(
                dataset,
                replay_datasets,
                dataset,
                real_action_size = real_action_size,
                real_observation_size = real_observation_size,
                eval_episodess=eval_datasets,
                n_epochs=20,
                pretrain_phi_epoch=args.pretrain_phi_epoch,
                scorers={
                    # 'environment': evaluate_on_environment(env),
                    'td_error': td_error_scorer(real_action_size=real_action_size)
                },
                replay_scorers={
                    'bc_error': bc_error_scorer(real_action_size=real_action_size)
                },
                experiment_name=experiment_name + algos_name
            )
            assert co._impl is not None
            assert co._impl._q_func is not None
            assert co._impl._policy is not None
            if args.algos == 'co':
                replay_datasets[dataset_num] = finish_task(dataset_num, task_nums, dataset, original, co, indexes_euclids[dataset_num], real_action_size, args.topk, device)
            else:
                raise NotImplementedError
            co.save_model(args.model_path + algos_name + '_' + str(dataset_num) + '.pt')
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
            draw_path = args.model_path + algos_name + '_trajectories_' + str(dataset_num) + '_'
            co.test(
                replay_datasets,
                eval_episodess=eval_datasets,
                scorers={
                    # 'environment': evaluate_on_environment(env),
                    'td_error': td_error_scorer(real_action_size=real_action_size),
                    "real_env": evaluate_on_environment(envs, end_points, task_nums, draw_path),
                },
                replay_scorers={
                    'bc_error': bc_error_scorer(real_action_size=real_action_size)
                },

            )

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Experimental evaluation of lifelong PG learning')
    parser.add_argument('--inner_buffer_size', default=-1, type=int)
    parser.add_argument('--task_config', default='task_config/cheetah_dir.json', type=str)
    parser.add_argument('--siamese_hidden_size', default=100, type=int)
    parser.add_argument('--near_threshold', default=1, type=float)
    parser.add_argument('--siamese_threshold', default=1, type=float)
    parser.add_argument('--eval_batch_size', default=256, type=int)
    parser.add_argument('--batch_size', default=256, type=int)
    parser.add_argument('--topk', default=4, type=int)
    parser.add_argument('--task_split_type', default='undirected', type=str)
    parser.add_argument('--dataset_name', default='antmaze-large-play-v0', type=str)
    parser.add_argument('--algos', default='co', type=str)
    parser.add_argument('--eval', action='store_true')
    parser.add_argument('--test', action='store_true')
    parser.add_argument('--n_epochs', default=20, type=int)
    parser.add_argument('--pretrain_phi_epoch', default=0, type=int)
    parser.add_argument('--top_euclid', default=8, type=int)
    use_phi_replay_parser = parser.add_mutually_exclusive_group(required=True)
    use_phi_replay_parser.add_argument('--use_phi_replay', dest='use_phi_replay', action='store_true')
    use_phi_replay_parser.add_argument('--no_use_phi_replay', dest='use_phi_replay', action='store_false')
    use_phi_update_parser = parser.add_mutually_exclusive_group(required=True)
    use_phi_update_parser.add_argument('--use_phi_update', dest='use_phi_update', action='store_true')
    use_phi_update_parser.add_argument('--no_use_phi_update', dest='use_phi_update', action='store_false')
    args = parser.parse_args()
    args.model_path = 'd3rlpy_' + ('test' if args.test else ('train' if not args.eval else 'eval')) + '/model_'
    global DATASET_PATH
    DATASET_PATH = './.d4rl/datasets/'
    device = torch.device('cuda:0')
    random.seed(12345)
    np.random.seed(12345)
    main(args, device)
