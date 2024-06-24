import copy
import os
import sys
import argparse
import json
import random
from collections import namedtuple
import pickle
import time
from functools import partial
import numpy as np
import gym
from mygym.envs.halfcheetah_block import HalfCheetahBlockEnv

import torch
from torch.optim.lr_scheduler import CosineAnnealingLR

from d4rl.locomotion import maze_env, ant
from d4rl.locomotion.wrappers import NormalizedBoxEnv

import d3rlpy
from d3rlpy.ope import FQE
from d3rlpy.dataset import MDPDataset
from d3rlpy.torch_utility import get_state_dict, set_state_dict
from d3rlpy.online.iterators import train_single_env
from d3rlpy.models.optimizers import AdamFactory
from d3rlpy.online.buffers import ReplayBuffer
# from myd3rlpy.datasets import get_d4rl

from utils.k_means import kmeans
# from myd3rlpy.metrics.scorer import evaluate_on_environment_help, single_evaluate_on_environment, q_dataset_scorer, q_play_scorer, q_online_diff_scorer, q_offline_diff_scorer, q_id_diff_scorer, q_ood_diff_scorer, policy_replay_scorer, policy_dataset_scorer, policy_online_diff_scorer, policy_offline_diff_scorer, policy_id_diff_scorer, policy_ood_diff_scorer
from dataset.load_d4rl import get_d4rl_local, get_antmaze_local, get_dataset, get_macaw_local
from rlkit.torch import pytorch_util as ptu
from config.single_config import get_st_dict

from mygym.envs.envs import HalfCheetahDirEnv, HalfCheetahVelEnv, AntDirEnv, AntGoalEnv, HumanoidDirEnv, WalkerRandParamsWrappedEnv, ML45Env


RESET = R = 'r'  # Reset position.
GOAL = G = 'g'

mazes = {
    'umaze':
        [[[[1, 1, 1, 1, 1],
           [1, R, 0, 0, 1],
           [1, 1, 1, G, 1],
           [1, 0, 0, 0, 1],
           [1, 1, 1, 1, 1]],

          [[1, 1, 1, 1, 1],
           [1, R, 0, 0, 1],
           [1, 1, 1, 0, 1],
           [1, G, 0, 0, 1],
           [1, 1, 1, 1, 1]]],
         None,
        ],
    'medium':
	[[[[1, 1, 1, 1, 1, 1, 1, 1],
           [1, R, 0, 1, 1, 0, 0, 1],
           [1, 0, 0, 1, 0, 0, 0, 1],
           [1, 1, 0, 0, 0, 1, 1, 1],
           [1, 0, 0, 1, G, 0, 0, 1],
           [1, 0, 1, 0, 0, 1, 0, 1],
           [1, 0, 0, 0, 1, 0, 0, 1],
           [1, 1, 1, 1, 1, 1, 1, 1]],
	  [[1, 1, 1, 1, 1, 1, 1, 1],
           [1, R, 0, 1, 1, 0, 0, 1],
           [1, 0, 0, 1, 0, 0, 0, 1],
           [1, 1, 0, 0, 0, 1, 1, 1],
           [1, 0, 0, 1, 0, 0, 0, 1],
           [1, 0, 1, 0, 0, 1, 0, 1],
           [1, 0, 0, 0, 1, 0, G, 1],
           [1, 1, 1, 1, 1, 1, 1, 1]]],

	 [[[1, 1, 1, 1, 1, 1, 1, 1],
           [1, R, 0, 1, 1, 0, 0, 1],
           [1, 0, 0, 1, 0, 0, 0, 1],
           [1, 1, G, 0, 0, 1, 1, 1],
           [1, 0, 0, 1, 0, 0, 0, 1],
           [1, 0, 1, 0, 0, 1, 0, 1],
           [1, 0, 0, 0, 1, 0, 0, 1],
           [1, 1, 1, 1, 1, 1, 1, 1]],
	  [[1, 1, 1, 1, 1, 1, 1, 1],
           [1, R, 0, 1, 1, 0, 0, 1],
           [1, 0, 0, 1, 0, 0, 0, 1],
           [1, 1, 0, 0, 0, 1, 1, 1],
           [1, 0, 0, 1, G, 0, 0, 1],
           [1, 0, 1, 0, 0, 1, 0, 1],
           [1, 0, 0, 0, 1, 0, 0, 1],
           [1, 1, 1, 1, 1, 1, 1, 1]],
	  [[1, 1, 1, 1, 1, 1, 1, 1],
           [1, R, 0, 1, 1, 0, 0, 1],
           [1, 0, 0, 1, 0, 0, 0, 1],
           [1, 1, 0, 0, 0, 1, 1, 1],
           [1, 0, 0, 1, 0, 0, 0, 1],
           [1, 0, 1, 0, 0, 1, G, 1],
           [1, 0, 0, 0, 1, 0, 0, 1],
           [1, 1, 1, 1, 1, 1, 1, 1]],
	  [[1, 1, 1, 1, 1, 1, 1, 1],
           [1, R, 0, 1, 1, 0, 0, 1],
           [1, 0, 0, 1, 0, 0, 0, 1],
           [1, 1, 0, 0, 0, 1, 1, 1],
           [1, 0, 0, 1, 0, 0, 0, 1],
           [1, 0, 1, 0, 0, 1, 0, 1],
           [1, 0, 0, 0, 1, 0, G, 1],
           [1, 1, 1, 1, 1, 1, 1, 1]]],
        ],
    'large':
	[[[[1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
	   [1, R, 0, 0, 0, 1, 0, 0, 0, 0, 0, 1],
	   [1, 0, 1, 1, 0, 1, 0, 1, 0, 1, 0, 1],
	   [1, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 1],
	   [1, 0, 1, 1, 1, 1, G, 1, 1, 1, 0, 1],
	   [1, 0, 0, 1, 0, 1, 0, 0, 0, 0, 0, 1],
	   [1, 1, 0, 1, 0, 1, 0, 1, 0, 1, 1, 1],
	   [1, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 1],
	   [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1]],
	  [[1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
	   [1, R, 0, 0, 0, 1, 0, 0, 0, 0, 0, 1],
	   [1, 0, 1, 1, 0, 1, 0, 1, 0, 1, 0, 1],
	   [1, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 1],
	   [1, 0, 1, 1, 1, 1, 0, 1, 1, 1, 0, 1],
	   [1, 0, 0, 1, 0, 1, 0, 0, 0, 0, 0, 1],
	   [1, 1, 0, 1, 0, 1, 0, 1, 0, 1, 1, 1],
	   [1, 0, 0, 1, 0, 0, 0, 1, 0, G, 0, 1],
	   [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1]]],

	 [[[1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
	   [1, R, 0, 0, 0, 1, 0, 0, 0, 0, 0, 1],
	   [1, 0, 1, 1, 0, 1, 0, 1, 0, 1, 0, 1],
	   [1, 0, 0, G, 0, 0, 0, 1, 0, 0, 0, 1],
	   [1, 0, 1, 1, 1, 1, 0, 1, 1, 1, 0, 1],
	   [1, 0, 0, 1, 0, 1, 0, 0, 0, 0, 0, 1],
	   [1, 1, 0, 1, 0, 1, 0, 1, 0, 1, 1, 1],
	   [1, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 1],
	   [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1]],
	  [[1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
	   [1, R, 0, 0, 0, 1, G, 0, 0, 0, 0, 1],
	   [1, 0, 1, 1, 0, 1, 0, 1, 0, 1, 0, 1],
	   [1, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 1],
	   [1, 0, 1, 1, 1, 1, 0, 1, 1, 1, 0, 1],
	   [1, 0, 0, 1, 0, 1, 0, 0, 0, 0, 0, 1],
	   [1, 1, 0, 1, 0, 1, 0, 1, 0, 1, 1, 1],
	   [1, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 1],
	   [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1]],
	  [[1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
	   [1, R, 0, 0, 0, 1, 0, 0, 0, 0, 0, 1],
	   [1, 0, 1, 1, 0, 1, 0, 1, 0, 1, 0, 1],
	   [1, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 1],
	   [1, 0, 1, 1, 1, 1, 0, 1, 1, 1, 0, 1],
	   [1, 0, 0, 1, 0, 1, 0, 0, 0, 0, 0, 1],
	   [1, 1, 0, 1, 0, 1, 0, 1, 0, 1, 1, 1],
	   [1, 0, 0, 1, 0, 0, G, 1, 0, 0, 0, 1],
	   [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1]],
	  [[1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
	   [1, R, 0, 0, 0, 1, 0, 0, 0, 0, 0, 1],
	   [1, 0, 1, 1, 0, 1, 0, 1, 0, 1, 0, 1],
	   [1, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 1],
	   [1, 0, 1, 1, 1, 1, 0, 1, 1, 1, 0, 1],
	   [1, 0, 0, 1, 0, 1, 0, 0, 0, 0, 0, 1],
	   [1, 1, 0, 1, 0, 1, 0, 1, 0, 1, 1, 1],
	   [1, 0, 0, 1, 0, 0, 0, 1, 0, G, 0, 1],
	   [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1]]],
        ]
    }

mazes_start = {'umaze': [[(1, 1), (1, 1)], None], 'medium': [[(1, 1), (3, 4)], [(1, 1), (2, 2), (3, 4), (4, 6)]], 'large': [[(1, 1), (3, 6)], [(1, 1), (3, 2), (1, 7), (7, 5)]]}

def read_dict(state_dict, prename):
    for key, value in state_dict.items():
        if not isinstance(value, dict):
            if isinstance(value, torch.Tensor):
                print(f"{prename}.{str(key)}: {value.shape}")
            else:
                print(f"{prename}.{str(key)}: {value}")
        else:
            read_dict(value, prename + '.' + str(key))

def update(args, step_dict, experiment_name, algos_name, dataset_name, dataset_num, env, i, fs, fs_dict):

    # h5_path = 'dataset/d4rl/' + args.dataset + '/' + dataset_num + '.hdf5'
    h5_path = f'dataset/macaw/{dataset_name}/buffers_{dataset_name}_train_{dataset_num}_sub_task_0.hdf5'
    dataset = get_macaw_local(h5_path)
    env.reset_task(i)

    # training
    replay_dataset = None
    learned_datasets = []
    if not args.test:
        pretrain_path_eval = "pretrained_network/" + f"ST_{args.algo}_" + args.dataset + '_d4rl.pt'

    if args.clear_network:
        fs = FS(**fs_dict)

    learned_datasets.append(dataset)
    add_one_learned_datasets = [None] + learned_datasets

    # if env is not None:
    #     # scorers_list = [{'environment': d3rlpy.metrics.evaluate_on_environment(env), 'fune_tuned_environment': single_evaluate_on_environment(env)}]
    #     scorers_env = {'environment_{dataset_name}_{i}': d3rlpy.metrics.evaluate_on_environment(env)}
    #     scorers_list.append(scorers_env)
    # else:
    #     raise NotImplementedError

    eval_env = env

    start_time = time.perf_counter()
    print(f'Start Training {dataset_num}')
    if dataset_num <= args.read_policy:
        iterator, replay_iterator, n_epochs = fs.make_iterator(dataset, replay_dataset, step_dict['merge_n_steps'], step_dict['n_steps_per_epoch'], None, True)
        if args.read_policy == 0:
            pretrain_path = "pretrained_network/" + "ST_" + args.algo_kind + '_0.9_' + args.dataset + '_' + args.dataset_nums[0] + '.pt'
            if not os.path.exists(pretrain_path):
                pretrain_path = "pretrained_network/" + "ST_" + args.algo_kind + '_' + args.dataset + '_' + args.dataset_nums[0] + '.pt'
                assert os.path.exists(pretrain_path)
        else:
            pretrain_path = args.model_path + algos_name + '_' + str(dataset_num) + '.pt'

        fs.build_with_dataset(dataset, dataset_num)
        fs._impl.save_clone_data()
        fs.load_model(pretrain_path)
        fs._impl.save_clone_data()
        # if (args.critic_replay_type not in ['ewc', 'si', 'rwalk'] or args.actor_replay_type not in ['ewc', 'si', 'rwalk']) and args.read_policy != 0:
        #     try:
        #         replay_dataset = torch.load(f=args.model_path + algos_name + '_' + str(dataset_num) + '_datasets.pt')
        #     except BaseException as e:
        #         print(f'Don\' have replay_dataset')
        #         raise e
    # elif args.merge and dataset_num == args.read_merge_policy:
    #     iterator, replay_iterator, n_epochs = fs.make_iterator(dataset, replay_dataset, step_dict['merge_n_steps'], step_dict['n_steps_per_epoch'], None, True)
    #     pretrain_path = "pretrained_network/" + "ST_" + args.algo_kind + '_' + args.dataset + '_' + args.dataset_nums[0] + '.pt'
    #     fs.build_with_dataset(dataset, dataset_num)
    #     fs.load_model(pretrain_path)
    #     for param_group in fs._impl._actor_optim.param_groups:
    #         param_group["lr"] = fs_dict['actor_learning_rate']
    #     if args.algo in ['iql', 'iqln']:
    #         scheduler = CosineAnnealingLR(fs._impl._actor_optim, step_dict['n_steps'])

    #         def callback(algo, epoch, total_step):
    #             scheduler.step()
    #     else:
    #         callback = None
    #     fs.fit(
    #         dataset_num,
    #         dataset=dataset,
    #         iterator=iterator,
    #         replay_dataset=replay_dataset,
    #         replay_iterator=replay_iterator,
    #         eval_episodes_list=add_one_learned_datasets,
    #         # n_epochs=args.n_epochs if not args.test else 1,
    #         n_epochs=n_epochs,
    #         coldstart_steps=step_dict['coldstart_steps'],
    #         save_interval=args.save_interval,
    #         experiment_name=experiment_name + algos_name + '_' + str(dataset_num),
    #         scorers_list = scorers_list,
    #         callback=callback,
    #         test=args.test,
    #     )
    elif dataset_num > args.read_policy:
        # train
        print(f'fitting dataset {dataset_num}')
        # if args.merge and args.read_merge_policy >= 0 and dataset_num > 0:
        #     iterator, replay_iterator, n_epochs = fs.make_iterator(dataset, replay_dataset, step_dict['n_steps'] + step_dict['merge_n_steps'], step_dict['n_steps_per_epoch'], None, True)
        # else:
        iterator, n_epochs = fs.make_iterator(dataset, step_dict['n_steps'], step_dict['n_steps_per_epoch'], None, True)
        fs.build_with_dataset(dataset, dataset_num)
        for param_group in fs._impl._actor_optim.param_groups:
            param_group["lr"] = fs_dict['actor_learning_rate']
        for param_group in fs._impl._critic_optim.param_groups:
            param_group["lr"] = fs_dict['critic_learning_rate']
        if args.use_vae:
            for param_group in fs._impl._vae_optim.param_groups:
                param_group["lr"] = fs_dict['vae_learning_rate']
        if args.algo in ['iql', 'iqln', 'iqln2', 'iqln3', 'iqln4', 'sql', 'sqln']:
            scheduler = CosineAnnealingLR(fs._impl._actor_optim, 1000000)

            def callback(algo, epoch, total_step):
                scheduler.step()
            # fs_dict['expectile'] = 1
        else:
            callback = None

        fs.fit(
            dataset_num,
            dataset=dataset,
            iterator=iterator,
            eval_episodes_list=add_one_learned_datasets,
            # n_epochs=args.n_epochs if not args.test else 1,
            n_epochs=n_epochs,
            coldstart_steps=step_dict['coldstart_steps'],
            save_interval=args.save_interval,
            experiment_name=experiment_name + algos_name + '_' + str(dataset_num),
            # scorers_list = scorers_list,
            score = False,
            callback=callback,
            test=args.test,
        )
    # fs.after_learn(iterator, experiment_name + algos_name + '_' + str(dataset_num), scorers_list, add_one_learned_datasets)
    print(f'Training task {dataset_num} time: {time.perf_counter() - start_time}')
    fs.save_model(args.model_path + algos_name + '_' + str(dataset_num) + '.pt')
    # if args.test and i >= 2:
    #     break

replay_name = ['observations', 'actions', 'rewards', 'next_observations', 'terminals', 'means', 'std_logs', 'qs', 'phis', 'psis']
def main(args, device):
    np.set_printoptions(precision=1, suppress=True)
    ask_indexes = False

    # prepare algorithm
    if args.algo in ['td3_plus_bc', 'td3']:
        from myd3rlpy.algos.fs_td3_plus_bc import FS
    elif args.algo_kind == 'cql':
        from myd3rlpy.algos.fs_cql import FS
    elif args.algo in ['iql', 'iqln', 'iqln2', 'iqln3', 'iqln4', 'sql', 'sqln']:
        from myd3rlpy.algos.fs_iql import FS
    elif args.algo in ['sacn', 'edac']:
        from myd3rlpy.algos.fs_sacn import FS
    else:
        raise NotImplementedError
    fs_dict, online_fs_dict, step_dict = get_st_dict(args, 'd4rl', args.algo)
    if args.n_steps is not None:
        step_dict['n_steps'] = args.n_steps
    if args.algo in ['iql', 'sql', 'iqln', 'iqln2', 'iqln3', 'iqln4', 'sqln']:
        fs_dict['weight_temp'] = args.weight_temp
        fs_dict['expectile'] = args.expectile
        fs_dict['expectile_min'] = args.expectile_min
        fs_dict['expectile_max'] = args.expectile_max
        if args.algo in ['sql', 'sqln']:
            fs_dict['alpha'] = args.alpha
        if args.algo in ['iqln', 'iqln2', 'iqln3', 'iqln4', 'sqln']:
            fs_dict['n_ensemble'] = args.n_ensemble
            fs_dict['std_time'] = args.std_time
            fs_dict['std_type'] = args.std_type
            fs_dict['entropy_time'] = args.entropy_time
    elif args.algo == 'cql':
        fs_dict['std_time'] = args.std_time
        fs_dict['std_type'] = args.std_type
        fs_dict['entropy_time'] = args.entropy_time
    elif args.algo in ['sacn', 'edac']:
        fs_dict['n_ensemble'] = args.n_ensemble
        if args.algo == 'edac':
            fs_dict['eta'] = args.eta
    fs_dict['embed'] = args.embed

    fs_dict['actor_learning_rate'] = args.actor_learning_rate
    fs_dict['critic_learning_rate'] = args.critic_learning_rate

    fs = FS(**fs_dict)

    experiment_name = "FS" + '_'
    algos_name = args.algo
    algos_name += '_' + str(args.weight_temp)
    algos_name += '_' + str(args.expectile)
    algos_name += '_' + str(args.expectile_min)
    algos_name += '_' + str(args.expectile_max)
    algos_name += '_' + args.actor_replay_type
    algos_name += '_' + str(args.actor_replay_lambda)
    algos_name += '_' + str(args.actor_learning_rate)
    algos_name += '_' + args.critic_replay_type
    algos_name += '_' + str(args.critic_replay_lambda)
    algos_name += '_' + str(args.critic_learning_rate)
    algos_name += '_' + str(args.max_save_num)
    if args.add_name != '':
        algos_name += '_' + args.add_name

    pretrain_name = args.model_path

    if not args.eval:
        pklfile = {}
        max_itr_num = 3000
        task_datasets = []
        eval_envs = []
        dataset_num_counter = dict()
        scorers_list = []
        learned_dataset_names = {}
        if args.actor_replay_type == 'orl':
            replay_datasets = []
        for env_num, (dataset_name, dataset_nums) in enumerate(args.dataset_kinds.items()):
            print(f"Start dataset {dataset_name}")
            tasks = []
            for dataset_num in dataset_nums:
                env_path = f'dataset/macaw/{dataset_name}/env_{dataset_name}_train_task{dataset_num}.pkl'
                with open(env_path, 'rb') as f:
                    task_info = pickle.load(f)
                    assert len(task_info) == 1, f'Unexpected task info: {task_info}'
                    tasks.append(task_info[0])

            if dataset_name == 'ant_dir':
                env = AntDirEnv(tasks, 50, include_goal = False)
            elif dataset_name == 'cheetah_dir':
                env = HalfCheetahDirEnv(tasks, include_goal = False)
            elif dataset_name == 'cheetah_vel':
                env = HalfCheetahVelEnv(tasks, include_goal = False, one_hot_goal=False)
            elif dataset_name == 'walker_dir':
                env = WalkerRandParamsWrappedEnv(tasks, 50, include_goal = False)
            else:
                raise RuntimeError(f'Invalid env name {dataset_name}')
            eval_envs.append(env)
            if dataset_name == 'ant_dir':
                test_env = AntDirEnv(tasks, 50, include_goal = False)
            elif dataset_name == 'cheetah_dir':
                test_env = HalfCheetahDirEnv(tasks, include_goal = False)
            elif dataset_name == 'cheetah_vel':
                test_env = HalfCheetahVelEnv(tasks, include_goal = False, one_hot_goal=False)
            elif dataset_name == 'walker_dir':
                test_env = WalkerRandParamsWrappedEnv(tasks, 50, include_goal = False)
            else:
                raise RuntimeError(f'Invalid env name {dataset_name}')

            learned_dataset_names.update({dataset_name: (dataset_nums, test_env)})

            if len(args.dataset_kinds.items()) == env_num + 1:
                print(f"Testing after {dataset_name}")
                for test_dataset_name, (test_dataset_nums, test_env) in learned_dataset_names.items():
                    for j, dataset_num in enumerate(test_dataset_nums[-1:] if not args.test else test_dataset_nums[-1:]):
                        print(f"Testing {test_dataset_name}, {dataset_num}")

                        h5_path = f'dataset/macaw/{test_dataset_name}/buffers_{test_dataset_name}_train_{dataset_num}_sub_task_0.hdf5'
                        dataset, _, _ = get_macaw_local(h5_path)
                        fs.build_with_dataset(dataset, dataset_num)
                        fs.load_model(args.pretrain_path)
                        replay_dataset = None
                        iterator, n_epochs = fs.make_iterator(dataset, step_dict['n_steps'], step_dict['n_steps_per_epoch'], None, True)
                        test_env.reset_task(j)
                        fs.fit(
                            dataset_num,
                            dataset=dataset,
                            iterator=iterator,
                            # eval_episodes_list=add_one_learned_datasets,
                            # n_epochs=args.n_epochs if not args.test else 1,
                            n_epochs=n_epochs,
                            coldstart_steps=step_dict['coldstart_steps'],
                            save_interval=args.save_interval,
                            experiment_name=experiment_name + algos_name + '_' + str(dataset_num),
                            # scorers_list = scorers_list,
                            score = True,
                            # callback=callback,
                            test=args.test,
                        )

            # eval
            # scorers = dict(zip(['real_env' + str(n) for n in datasets.keys()], [evaluate_on_environment(eval_envs[n], test_id=str(n), mix='mix' in args.dataset and n == '0', add_on=args.add_on, clone_actor=args.clone_actor) for n in learned_tasks]))

            # 比较的测试没必要对新的数据集做。
        # if online_st_dict['n_steps'] > 0:
        #     for param_group in fs._impl._actor_optim.param_groups:
        #         param_group["lr"] = fs_dict['actor_learning_rate']
        #     for param_group in fs._impl._critic_optim.param_groups:
        #         param_group["lr"] = fs_dict['critic_learning_rate']
        #     if args.use_vae:
        #         for param_group in fs._impl._vae_optim.param_groups:
        #             param_group["lr"] = fs_dict['vae_learning_rate']
        #     buffer_ = ReplayBuffer(maxlen=online_st_dict['buffer_size'], env=env)
        #     fs.online_fit(env, eval_env, buffer_, n_steps=online_st_dict['n_steps'], n_steps_per_epoch=online_st_dict['n_steps_per_epoch'], experiment_name = experiment_name + algos_name, test=args.test)
    print('finish')

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Experimental evaluation of lifelong PG learning')
    parser.add_argument('--add_name', default='', type=str)
    parser.add_argument('--epoch', default='500', type=int)
    parser.add_argument("--dataset", default='antmaze-large-play-v2', type=str)
    parser.add_argument('--dataset_nums', default="0", type=str)
    parser.add_argument('--inner_path', default='', type=str)
    parser.add_argument('--env_path', default=None, type=str)
    parser.add_argument('--pretrain_path', default=None, type=str)
    parser.add_argument('--inner_buffer_size', default=-1, type=int)
    parser.add_argument('--task_config', default='task_config/cheetah_dir.json', type=str)
    parser.add_argument('--siamese_hidden_size', default=100, type=int)
    parser.add_argument('--near_threshold', default=1, type=float)
    parser.add_argument('--siamese_threshold', default=1, type=float)
    parser.add_argument('--eval_batch_size', default=256, type=int)
    parser.add_argument('--topk', default=4, type=int)
    parser.add_argument('--max_save_num', default=1, type=int)
    parser.add_argument('--task_split_type', default='undirected', type=str)
    parser.add_argument('--algo', default='iql', type=str, choices=['combo', 'td3_plus_bc', 'cql', 'mgcql', 'mrcql', 'iql', 'iqln', 'iqln2', 'iqln3', 'iqln4', 'sql', 'sqln', 'sacn', 'edac'])
    parser.add_argument('--weight_temp', default=3.0, type=float)
    parser.add_argument('--expectile', default=0.7, type=float)
    parser.add_argument('--expectile_min', default=0.7, type=float)
    parser.add_argument('--expectile_max', default=0.7, type=float)
    parser.add_argument('--alpha', default=2, type=float)
    parser.add_argument('--eval', action='store_true')
    parser.add_argument('--test', action='store_true')


    parser.add_argument("--n_steps", default=None, type=int)
    parser.add_argument("--online_n_steps", default=100000, type=int)
    parser.add_argument("--online_maxlen", default=1000000, type=int)

    parser.add_argument("--save_interval", default=10, type=int)
    parser.add_argument("--n_action_samples", default=10, type=int)
    parser.add_argument('--top_euclid', default=64, type=int)

    parser.add_argument('--critic_replay_type', default='bc', type=str, choices=['orl', 'bc', 'generate', 'generate_orl', 'lwf', 'ewc', 'gem', 'agem', 'rwalk', 'si', 'none'])
    parser.add_argument('--critic_replay_lambda', default=100, type=float)
    parser.add_argument('--critic_learning_rate', default=3e-4, type=float)
    parser.add_argument('--actor_replay_type', default='orl', type=str, choices=['orl', 'bc', 'generate', 'generate_orl', 'lwf', 'lwf_orl', 'ewc', 'gem', 'agem', 'rwalk', 'si', 'none'])
    parser.add_argument('--actor_replay_lambda', default=1, type=float)
    parser.add_argument('--actor_learning_rate', default=1e-5, type=float)

    parser.add_argument('--n_ensemble', default=2, type=int)
    parser.add_argument('--eta', default=1.0, type=int)
    parser.add_argument('--std_time', default=1, type=float)
    parser.add_argument('--std_type', default='none', type=str, choices=['clamp', 'none', 'linear', 'entropy'])
    parser.add_argument('--entropy_time', default=0.2, type=float)
    parser.add_argument('--update_ratio', default=0.3, type=float)

    parser.add_argument('--fine_tuned_step', default=1, type=int)
    parser.add_argument('--clone_actor', action='store_true')
    parser.add_argument('--vae_replay_type', default='generate', type=str, choices=['orl', 'bc', 'generate', 'ewc', 'gem', 'agem', 'rwalk', 'si', 'none'])
    parser.add_argument('--vae_replay_lambda', default=1, type=float)
    parser.add_argument('--mix_type', default='q', type=str, choices=['q', 'v', 'random', 'vq_diff', 'all'])

    parser.add_argument('--experience_type', default='random_episode', type=str, choices=['all', 'none', 'single', 'online', 'generate', 'model_prob', 'model_next', 'model', 'model_this', 'coverage', 'random_transition', 'random_episode', 'max_reward', 'max_match', 'max_supervise', 'max_model', 'max_reward_end', 'max_reward_mean', 'max_match_end', 'max_match_mean', 'max_supervise_end', 'max_supervise_mean', 'max_model_end', 'max_model_mean', 'min_reward', 'min_match', 'min_supervise', 'min_model', 'min_reward_end', 'min_reward_mean', 'min_match_end', 'min_match_mean', 'min_supervise_end', 'min_supervise_mean', 'min_model_end', 'min_model_mean'])
    parser.add_argument('--max_export_step', default=1000, type=int)
    parser.add_argument('--dense', default='dense', type=str)
    parser.add_argument('--sum', default='no_sum', type=str)
    parser.add_argument('--d_threshold', type=float, default=0.1)
    parser.add_argument('--use_cpu', action='store_true')
    parser.add_argument('--gpu', type=str, default='0')
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--read_policy', type=int, default=-1)
    parser.add_argument('--read_merge_policy', type=int, default=-1)
    # 作为对照实验，证明er算法不是重新学习了重放缓存而是具备持续学习能力
    parser.add_argument('--clear_network', action='store_true')
    parser.add_argument('--embed', action='store_true')
    # parser.add_argument('--merge', action='store_true')
    args = parser.parse_args()

    args.algo_kind = args.algo
    if args.algo_kind in ['cql', 'mrcql', 'mgcql']:
        args.algo_kind = 'cql'

    # ant_dir, cheetah_vel, walker_dir
    args.dataset_kinds = {'ant_dir': np.random.permutation(5), 'cheetah_vel': np.random.permutation(5), 'walker_dir': np.random.permutation(5)}

    args.model_path = 'd3rlpy' + '_' + args.dataset
    if not os.path.exists(args.model_path):
        os.makedirs(args.model_path)
    args.model_path += '/model_'
    # if args.experience_type == 'model':
    #     args.experience_type = 'model_next'

    if args.critic_replay_type in ['generate', 'generate_orl'] or args.actor_replay_type in ['generate', 'generate_orl']:
        args.use_vae = True
    else:
        args.use_vae = False

    global DATASET_PATH
    DATASET_PATH = './.d4rl/datasets/'
    if args.use_cpu:
        device = torch.device('cpu')
    else:
        device = torch.device('cuda')
    ptu.set_gpu_mode(True)
    args.clone_critic = True
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu
    seeds = [12345, 1234, 123, 12, 1]
    random.seed(seeds[args.seed])
    np.random.seed(seeds[args.seed])
    torch.manual_seed(seeds[args.seed])
    torch.cuda.manual_seed(seeds[args.seed])
    main(args, device)
