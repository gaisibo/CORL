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
from envs import HalfCheetahDirEnv
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
from myd3rlpy.metrics.scorer import evaluate_on_environment_help, single_evaluate_on_environment, q_dataset_scorer, q_play_scorer, q_online_diff_scorer, q_offline_diff_scorer, q_id_diff_scorer, q_ood_diff_scorer, policy_replay_scorer, policy_dataset_scorer, policy_online_diff_scorer, policy_offline_diff_scorer, policy_id_diff_scorer, policy_ood_diff_scorer
from dataset.load_d4rl import get_d4rl_local, get_dataset
from rlkit.torch import pytorch_util as ptu
from config.single_config import get_st_dict


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
           [1, 0, 0, 0, 1, G, 0, 1],
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
           [1, 0, 0, 0, 1, G, 0, 1],
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

mazes_start = {'umaze': [[(1, 1), (2, 3)], None], 'medium': [[(1, 1), (2, 5)], [(1, 1), (2, 2), (2, 5), (4, 6)]], 'large': [[(1, 1), (3, 6)], [(1, 1), (3, 2), (1, 7), (7, 5)]]}

def read_dict(state_dict, prename):
    for key, value in state_dict.items():
        if not isinstance(value, dict):
            if isinstance(value, torch.Tensor):
                print(f"{prename}.{str(key)}: {value.shape}")
            else:
                print(f"{prename}.{str(key)}: {value}")
        else:
            read_dict(value, prename + '.' + str(key))
replay_name = ['observations', 'actions', 'rewards', 'next_observations', 'terminals', 'means', 'std_logs', 'qs', 'phis', 'psis']
def main(args, device):
    np.set_printoptions(precision=1, suppress=True)
    ask_indexes = False
    if args.dataset_kind in ['d4rl', 'antmaze']:
        _, env = d3rlpy.datasets.get_dataset(args.dataset)
        _, eval_env = d3rlpy.datasets.get_dataset(args.dataset)
    elif args.dataset_kind == 'block':
        env = gym.make(args.dataset)
        eval_env = gym.make(args.dataset)
    else:
        raise NotImplementedError

    # prepare algorithm
    if args.algo in ['td3_plus_bc', 'td3']:
        from myd3rlpy.algos.st_td3_plus_bc import ST
    elif args.algo_kind == 'cql':
        from myd3rlpy.algos.st_cql import ST
    elif args.algo in ['iql', 'iqln']:
        from myd3rlpy.algos.st_iql import ST
    elif args.algo == 'sacn':
        from myd3rlpy.algos.st_sacn import ST
    else:
        raise NotImplementedError
    st_dict, online_st_dict, step_dict = get_st_dict(args, args.dataset_kind, args.algo)
    if args.algo in ['iql', 'iqln']:
        st_dict['expectile'] = args.expectile
        st_dict['expectile_min'] = args.expectile_min
        st_dict['expectile_max'] = args.expectile_max
    st = ST(**st_dict)
    st_dict_eval, *_ = get_st_dict(args, args.dataset_kind, 'iql')

    experiment_name = "ST" + '_'
    algos_name = args.algo
    algos_name += '_' + args.dataset
    algos_name += '_' + args.dataset_nums_str
    algos_name += '_' + str(args.max_save_num)
    algos_name += '_' + str(args.critic_replay_type)
    algos_name += '_' + str(args.critic_replay_lambda)
    algos_name += '_' + str(args.actor_replay_type)
    algos_name += '_' + str(args.actor_replay_lambda)
    algos_name += '_' + str(args.seed)
    if args.add_name != '':
        algos_name += '_' + args.add_name

    pretrain_name = args.model_path

    if not args.eval:
        pklfile = {}
        max_itr_num = 3000
        task_datasets = []
        eval_envs = []
        for i, dataset_num in enumerate(args.dataset_nums):
            if dataset_num != 'd4rl':
                h5_path = 'dataset/d4rl/' + args.dataset + '/' + dataset_num + '.hdf5'
                dataset, eval_env = get_d4rl_local(get_dataset(h5_path), dataset_num + '-' + args.dataset, mazes[args.maze][args.part_times_num][int(dataset_num.split('_')[-1])])
                task_datasets.append((dataset_num, dataset))
                eval_envs.append(eval_env)
            else:
                task_datasets.append((dataset_num, d3rlpy.datasets.get_dataset(args.dataset)[0]))
                eval_envs.append(env)
        replay_dataset = None
        learned_id = []
        learned_datasets = []
        if not args.test:
            pretrain_path_eval = "pretrained_network/" + "ST_iql_" + args.dataset + '_d4rl.pt'

        for dataset_id, (dataset_num, dataset) in enumerate(task_datasets):
            learned_id.append(dataset_id)
            learned_datasets.append(dataset)
            add_one_learned_datasets = [None] + learned_datasets

            if env is not None:
                # scorers_list = [{'environment': d3rlpy.metrics.evaluate_on_environment(env), 'fune_tuned_environment': single_evaluate_on_environment(env)}]
                scorers_env = {'environment': d3rlpy.metrics.evaluate_on_environment(env)}
                scorers_part = dict(zip(['environment_part' + str(n) for n in range(dataset_id + 1)], [evaluate_on_environment_help(eval_envs[n], mazes_start[args.maze][args.part_times_num][n]) for n in learned_id]))
                scorers_env.update(scorers_part)
                scorers_list = [scorers_env]
            else:
                raise NotImplementedError

            start_time = time.perf_counter()
            print(f'Start Training {dataset_id}')
            if dataset_id <= args.read_policy:
                iterator, replay_iterator, n_epochs = st.make_iterator(dataset, replay_dataset, step_dict['merge_n_steps'], step_dict['n_steps_per_epoch'], None, True)
                if args.read_policy == 0:
                    pretrain_path = "pretrained_network/" + "ST_" + args.algo_kind + '_' + str(args.expectile) + '_' + args.dataset + '_' + args.dataset_nums[0] + '.pt'
                    if not os.path.exists(pretrain_path):
                        pretrain_path = "pretrained_network/" + "ST_" + args.algo_kind + '_' + args.dataset + '_' + args.dataset_nums[0] + '.pt'
                        assert os.path.exists(pretrain_path)
                else:
                    pretrain_path = args.model_path + algos_name + '_' + str(dataset_id) + '.pt'
                st.build_with_dataset(dataset, dataset_id)
                st._impl.save_clone_data()
                st.load_model(pretrain_path)
                st._impl.save_clone_data()
                # if (args.critic_replay_type not in ['ewc', 'si', 'rwalk'] or args.actor_replay_type not in ['ewc', 'si', 'rwalk']) and args.read_policy != 0:
                #     try:
                #         replay_dataset = torch.load(f=args.model_path + algos_name + '_' + str(dataset_id) + '_datasets.pt')
                #     except BaseException as e:
                #         print(f'Don\' have replay_dataset')
                #         raise e
            elif args.merge and dataset_id == args.read_merge_policy:
                iterator, replay_iterator, n_epochs = st.make_iterator(dataset, replay_dataset, step_dict['merge_n_steps'], step_dict['n_steps_per_epoch'], None, True)
                pretrain_path = "pretrained_network/" + "ST_" + args.algo_kind + '_' + args.dataset + '_' + args.dataset_nums[0] + '.pt'
                st.build_with_dataset(dataset, dataset_id)
                st.load_model(pretrain_path)
                for param_group in st._impl._actor_optim.param_groups:
                    param_group["lr"] = st_dict['actor_learning_rate']
                if args.algo in ['iql', 'iqln']:
                    scheduler = CosineAnnealingLR(st._impl._actor_optim, step_dict['n_steps'])

                    def callback(algo, epoch, total_step):
                        scheduler.step()
                else:
                    callback = None
                st.fit(
                    dataset_id,
                    dataset=dataset,
                    iterator=iterator,
                    replay_dataset=replay_dataset,
                    replay_iterator=replay_iterator,
                    eval_episodes_list=add_one_learned_datasets,
                    # n_epochs=args.n_epochs if not args.test else 1,
                    n_epochs=n_epochs,
                    coldstart_steps=step_dict['coldstart_steps'],
                    save_interval=args.save_interval,
                    experiment_name=experiment_name + algos_name,
                    scorers_list = scorers_list,
                    callback=callback,
                    test=args.test,
                )
            elif dataset_id > args.read_policy:
                # train
                print(f'fitting dataset {dataset_id}')
                if args.merge and args.read_merge_policy >= 0 and dataset_id > 0:
                    iterator, replay_iterator, n_epochs = st.make_iterator(dataset, replay_dataset, step_dict['n_steps'] + step_dict['merge_n_steps'], step_dict['n_steps_per_epoch'], None, True)
                else:
                    iterator, replay_iterator, n_epochs = st.make_iterator(dataset, replay_dataset, step_dict['n_steps'], step_dict['n_steps_per_epoch'], None, True)
                st.build_with_dataset(dataset, dataset_id)
                for param_group in st._impl._actor_optim.param_groups:
                    param_group["lr"] = st_dict['actor_learning_rate']
                for param_group in st._impl._critic_optim.param_groups:
                    param_group["lr"] = st_dict['critic_learning_rate']
                if args.use_vae:
                    for param_group in st._impl._vae_optim.param_groups:
                        param_group["lr"] = st_dict['vae_learning_rate']
                if args.algo in ['iql', 'iqln']:
                    scheduler = CosineAnnealingLR(st._impl._actor_optim, 1000000)

                    def callback(algo, epoch, total_step):
                        scheduler.step()
                    # st_dict['expectile'] = 1
                else:
                    callback = None

                st.fit(
                    dataset_id,
                    dataset=dataset,
                    iterator=iterator,
                    replay_dataset=replay_dataset,
                    replay_iterator=replay_iterator,
                    eval_episodes_list=add_one_learned_datasets,
                    # n_epochs=args.n_epochs if not args.test else 1,
                    n_epochs=n_epochs,
                    coldstart_steps=step_dict['coldstart_steps'],
                    save_interval=args.save_interval,
                    experiment_name=experiment_name + algos_name,
                    scorers_list = scorers_list,
                    callback=callback,
                    test=args.test,
                )
            st.after_learn(iterator, experiment_name + algos_name)
            print(f'Training task {dataset_id} time: {time.perf_counter() - start_time}')
            # st.save_model(args.model_path + algos_name + '_' + str(dataset_id) + '.pt')
            if args.critic_replay_type in ['bc', 'orl', 'gem', 'agem'] or args.actor_replay_type in ['bc', 'orl', 'gem', 'agem'] or (args.use_vae and args.vae_replay_type in ['bc', 'orl', 'gem', 'agem']):
                # if args.mix_type == 'random':
                #     slide_dataset_length = args.max_save_num // (dataset_id + 1)
                # elif args.mix_type in ['q', 'vq_diff']:
                #     slide_dataset_length = args.max_save_num
                # else:
                #     raise NotImplementedError
                # new_replay_dataset = st.generate_replay(dataset_id, dataset, env, args.critic_replay_type, args.actor_replay_type, args.experience_type, slide_dataset_length, args.max_export_step, args.test)
                # if replay_dataset is not None:
                replay_dataset = st.select_replay(dataset, replay_dataset, dataset_id, args.max_save_num, args.mix_type)
                # else:
                #     replay_dataset = new_replay_dataset
            else:
                replay_dataset = None
            if args.test and dataset_id >= 2:
                break
            # 比较的测试没必要对新的数据集做。
        if online_st_dict['n_steps'] > 0:
            for param_group in st._impl._actor_optim.param_groups:
                param_group["lr"] = st_dict['actor_learning_rate']
            for param_group in st._impl._critic_optim.param_groups:
                param_group["lr"] = st_dict['critic_learning_rate']
            if args.use_vae:
                for param_group in st._impl._vae_optim.param_groups:
                    param_group["lr"] = st_dict['vae_learning_rate']
            buffer_ = ReplayBuffer(maxlen=online_st_dict['buffer_size'], env=env)
            st.online_fit(env, eval_env, buffer_, n_steps=online_st_dict['n_steps'], n_steps_per_epoch=online_st_dict['n_steps_per_epoch'], experiment_name = experiment_name + algos_name, test=args.test)
    print('finish')

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Experimental evaluation of lifelong PG learning')
    parser.add_argument('--add_name', default='', type=str)
    parser.add_argument("--dataset", default='antmaze-large-play-v2', type=str)
    parser.add_argument('--dataset_nums', default="0", type=str)
    parser.add_argument('--inner_path', default='', type=str)
    parser.add_argument('--env_path', default=None, type=str)
    parser.add_argument('--inner_buffer_size', default=-1, type=int)
    parser.add_argument('--task_config', default='task_config/cheetah_dir.json', type=str)
    parser.add_argument('--siamese_hidden_size', default=100, type=int)
    parser.add_argument('--near_threshold', default=1, type=float)
    parser.add_argument('--siamese_threshold', default=1, type=float)
    parser.add_argument('--eval_batch_size', default=256, type=int)
    parser.add_argument('--topk', default=4, type=int)
    parser.add_argument('--max_save_num', default=1000, type=int)
    parser.add_argument('--task_split_type', default='undirected', type=str)
    parser.add_argument('--algo', default='iql', type=str, choices=['combo', 'td3_plus_bc', 'cql', 'mgcql', 'mrcql', 'iql', 'iqln', 'sacn'])
    parser.add_argument('--expectile', default=0.7, type=float)
    parser.add_argument('--expectile_min', default=0.7, type=float)
    parser.add_argument('--expectile_max', default=0.7, type=float)
    parser.add_argument('--eval', action='store_true')
    parser.add_argument('--test', action='store_true')

    parser.add_argument("--online_n_steps", default=100000, type=int)
    parser.add_argument("--online_maxlen", default=1000000, type=int)

    parser.add_argument("--save_interval", default=10, type=int)
    parser.add_argument("--n_action_samples", default=10, type=int)
    parser.add_argument('--top_euclid', default=64, type=int)

    parser.add_argument('--critic_replay_type', default='bc', type=str, choices=['orl', 'bc', 'generate', 'generate_orl', 'lwf', 'ewc', 'gem', 'agem', 'rwalk', 'si', 'none'])
    parser.add_argument('--critic_replay_lambda', default=100, type=float)
    parser.add_argument('--actor_replay_type', default='orl', type=str, choices=['orl', 'bc', 'generate', 'generate_orl', 'lwf', 'lwf_orl', 'ewc', 'gem', 'agem', 'rwalk', 'si', 'none'])
    parser.add_argument('--actor_replay_lambda', default=1, type=float)
    parser.add_argument('--fine_tuned_step', default=1, type=int)
    parser.add_argument('--clone', default='none', type=str)
    parser.add_argument('--vae_replay_type', default='generate', type=str, choices=['orl', 'bc', 'generate', 'ewc', 'gem', 'agem', 'rwalk', 'si', 'none'])
    parser.add_argument('--vae_replay_lambda', default=1, type=float)
    parser.add_argument('--mix_type', default='q', type=str, choices=['q', 'random', 'vq_diff', 'all'])

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
    parser.add_argument('--merge', action='store_true')
    args = parser.parse_args()

    args.dataset_nums_str = args.dataset_nums
    args.dataset_nums = [x for x in args.dataset_nums.split('-')]
    args.algo_kind = args.algo
    if args.algo_kind in ['cql', 'mrcql', 'mgcql']:
        args.algo_kind = 'cql'
    if args.dataset in ['halfcheetah-random-v0', 'hopper-random-v0', 'walker2d-random-v0']:
        args.dataset_kind = 'd4rl'
        args.dataset_nums = ['itr_' + dataset_num for dataset_num in args.dataset_nums]
    elif args.dataset in ['HalfCheetahBlock-v2', 'Walker2dBlock-v4', 'HopperBlock-v4']:
        args.dataset_kind = 'block'
        args.dataset_nums = ['itr_' + dataset_num for dataset_num in args.dataset_nums]
    elif 'antmaze' in args.dataset:
        args.dataset_kind = 'antmaze'
        args.maze = args.dataset.split('-')[1]
        assert args.maze in ['umaze', 'medium', 'large']
        args.part_times_num = 0 if len(args.dataset_nums) == 2 else 1
    else:
        raise NotImplementedError

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
    if args.clone == 'none':
        args.clone_actor = False
        args.clone_critic = False
    elif args.clone == 'clone':
        args.clone_actor = True
        args.clone_critic = True
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu
    seeds = [12345, 1234, 123, 12, 1]
    random.seed(seeds[args.seed])
    np.random.seed(seeds[args.seed])
    torch.manual_seed(seeds[args.seed])
    torch.cuda.manual_seed(seeds[args.seed])
    main(args, device)
