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
from d3rlpy.torch_utility import get_state_dict, set_state_dict
# from myd3rlpy.datasets import get_d4rl
from utils.k_means import kmeans
from myd3rlpy.metrics.scorer import bc_error_scorer, td_error_scorer, evaluate_on_environment, q_error_scorer
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
        elif args.dataset == 'maze2d_open':
            from dataset.split_maze import split_navigate_maze2d_open_v0 as split_navigate
        elif args.dataset == 'maze2d_umaze':
            from dataset.split_maze import split_navigate_maze2d_umaze_v1 as split_navigate
        elif args.dataset == 'maze2d_medium':
            from dataset.split_maze import split_navigate_maze2d_medium_v1 as split_navigate
        elif args.dataset == 'maze2d_large':
            from dataset.split_maze import split_navigate_maze2d_large_v1 as split_navigate
        else:
            raise NotImplementedError
        # task_datasets, origin_datasets, taskid_datasets, action_datasets, envs, end_points, original, real_action_size, real_observation_size, indexes_euclids, task_nums = split_gym(args.top_euclid, args.dataset.replace('_', '-'), device=device)
        origin_datasets, taskid_datasets, envs, end_points, original, real_action_size, real_observation_size, task_nums = split_gym(args.top_euclid, args.dataset.replace('_', '-'), device=device)
    elif args.dataset in ['hopper_expert_v0', 'hopper_medium_v0', 'hopper_medium_expert_v0', 'hopper_medium_replay_v0', 'hopper_random_v0', 'halfcheetah_expert_v0', 'halfcheetah_medium_v0', 'halfcheetah_medium_expert_v0', 'halfcheetah_medium_replay_v0', 'halfcheetah_random_v0', 'walker2d_expert_v0', 'walker2d_medium_v0', 'walker2d_medium_expert_v0', 'walker2d_medium_replay_v0', 'walker2d_random_v0', 'mix_expert_v0', 'mix_medium_expert_v0', 'mix_medium_v0', 'mix_random_v0']:
        if args.dataset in ['hopper_expert_v0', 'hopper_medium_v0', 'hopper_medium_expert_v0', 'hopper_medium_replay_v0', 'hopper_random_v0']:
            from dataset.split_gym import split_hopper as split_gym
        elif args.dataset in ['halfcheetah_expert_v0', 'halfcheetah_medium_v0', 'halfcheetah_medium_expert_v0', 'halfcheetah_medium_replay_v0', 'halfcheetah_random_v0']:
            from dataset.split_gym import split_cheetah as split_gym
        elif args.dataset in ['walker2d_expert_v0', 'walker2d_medium_v0', 'walker2d_medium_expert_v0', 'walker2d_medium_replay_v0', 'walker2d_random_v0']:
            from dataset.split_gym import split_walker as split_gym
        elif args.dataset in ['mix_expert_v0', 'mix_medium_expert_v0', 'mix_medium_v0', 'mix_random_v0']:
            from dataset.split_gym import split_mix as split_gym
        else:
            raise NotImplementedError
        # task_datasets, origin_datasets, taskid_datasets, action_datasets, envs, end_points, original, real_action_size, real_observation_size, indexes_euclids, task_nums = split_gym(args.top_euclid, args.dataset.replace('_', '-'), device=device)
        origin_datasets, taskid_datasets, envs, end_points, original, real_action_size, real_observation_size, task_nums = split_gym(args.top_euclid, args.dataset.replace('_', '-'), device=device)
        env = None
    elif args.dataset in ['ant_dir', 'cheetah_dir', 'walker_dir', 'cheetah_vel']:
        from dataset.split_macaw import split_macaw
        inner_paths = ['dataset/macaw/' + args.inner_path.replace('num', str(i)) for i in range(args.task_nums)]
        env_paths = ['dataset/macaw/' + args.env_path.replace('num', str(i)) for i in range(args.task_nums)]
        origin_datasets, taskid_datasets, env, end_points, original, real_action_size, real_observation_size = split_macaw(args.top_euclid, args.dataset.replace('_', '-'), inner_paths, env_paths, device=device)
        envs = None

    else:
        raise NotImplementedError

    # prepare algorithm
    if args.algos == 'co':
        from myd3rlpy.algos.co import CO
        if args.experience_type == 'siamese':
            use_phi = True
        else:
            use_phi = False
        co = CO(use_gpu=not args.use_cpu, batch_size=args.batch_size, id_size=args.task_nums, replay_type=args.replay_type, experience_type=args.experience_type, sample_type=args.sample_type, reduce_replay=args.reduce_replay, use_phi=use_phi, use_model=args.use_model, replay_critic=args.replay_critic, replay_model=args.replay_model, replay_alpha=args.replay_alpha, generate_step=args.generate_step, model_noise=args.model_noise, retrain_time=args.retrain_time, orl_alpha=args.orl_alpha, single_head=args.single_head)
    else:
        raise NotImplementedError
    experiment_name = "CO" + '_'
    algos_name = "_" + args.replay_type
    algos_name += "_" + args.experience_type
    algos_name += '_' + args.sample_type
    algos_name += '_' + args.dataset
    algos_name += '_' + str(args.max_save_num)
    algos_name += '_' + str(args.replay_alpha)
    if args.add_name != '':
        algos_name += '_' + args.add_name
    algos_name += '_' + 'singlehead' if args.single_head else 'multihead'

    pretrain_name = args.model_path

    if not args.eval:
        replay_datasets = dict()
        save_datasets = dict()
        eval_datasets = dict()
        learned_tasks = []
        for task_id, dataset in origin_datasets.items():
            learned_tasks.append(task_id)
            task_id = str(task_id)
            start_time = time.perf_counter()
            print(f'Start Training {task_id}')
            eval_datasets[task_id] = dataset
            draw_path = args.model_path + algos_name + '_trajectories_' + str(task_id)
            dynamic_path = args.model_path + args.dataset + '_' + str(task_id) + '_dynamic.pt'
            dynamic_state_dict = torch.load(dynamic_path, map_location=device)
            print(dynamic_path)
            assert dynamic_state_dict is not None, dynamic_path
            if task_id == '0':
                pretrain_path = args.model_path + args.dataset + '_' + str(task_id) + '.pt'
                try:
                    pretrain_state_dict = torch.load(pretrain_path, map_location=device)
                except:
                    pretrain_state_dict = None
            else:
                pretrain_state_dict = None

            # train
            if env is not None:
                scorers = dict(zip(['real_env' + str(n) for n in origin_datasets.keys()], [evaluate_on_environment(env, test_id=str(n), mix='mix' in args.dataset and n == '0') for n in learned_tasks]))
            elif envs is not None:
                scorers = dict(zip(['real_env' + str(n) for n in origin_datasets.keys()], [evaluate_on_environment(envs[str(n)], test_id=str(n), mix='mix' in args.dataset and n == '0') for n in learned_tasks]))
            else:
                raise NotImplementedError
            co.fit(
                task_id,
                dataset,
                replay_datasets,
                real_action_size = real_action_size,
                real_observation_size = real_observation_size,
                eval_episodes=origin_datasets,
                # n_epochs=args.n_epochs if not args.test else 1,
                n_steps=args.n_steps,
                n_steps_per_epoch=args.n_steps_per_epoch,
                n_dynamic_epochs=100,
                # n_dynamic_steps=args.n_dynamic_steps,
                # n_dynamic_steps_per_epoch=args.n_dynamic_steps_per_epoch,
                n_begin_steps=args.n_begin_steps,
                n_begin_steps_per_epoch=args.n_begin_steps_per_epoch,
                dynamic_state_dict=dynamic_state_dict,
                pretrain_state_dict=pretrain_state_dict,
                experiment_name=experiment_name + algos_name,
                scorers = scorers,
                test=args.test,
            )
            print(f'Training task {task_id} time: {time.perf_counter() - start_time}')
            # co.save_model(args.model_path + algos_name + '_' + str(task_id) + '.pt')

            if args.algos == 'co':
                start_time = time.perf_counter()
                if args.experience_type in ['model', 'siamese']:
                    replay_datasets[task_id], save_datasets[task_id] = co.generate_replay_data_trajectory(origin_datasets[task_id], original[task_id], max_save_num=args.max_save_num, real_action_size=real_action_size, real_observation_size=real_observation_size)
                    print(f"len(replay_datasets[task_id]): {len(replay_datasets[task_id])}")
                elif args.experience_type in ['random_transition', 'max_reward', 'max_match', 'max_model', 'min_reward', 'min_match', 'min_model']:
                    replay_datasets[task_id], save_datasets[task_id] = co.generate_replay_data_transition(origin_datasets[task_id], max_save_num=args.max_save_num, real_action_size=real_action_size, real_observation_size=real_observation_size)
                    print(f"len(replay_datasets[task_id]): {len(replay_datasets[task_id])}")
                elif args.experience_type in ['random_episode', 'max_reward_end', 'max_reward_mean', 'max_match_end', 'max_match_mean', 'max_model_end', 'max_model_mean', 'min_reward_end', 'min_reward_mean', 'min_match_end', 'min_match_mean', 'min_model_end', 'min_model_mean']:
                    replay_datasets[task_id], save_datasets[task_id] = co.generate_replay_data_episode(origin_datasets[task_id], max_save_num=args.max_save_num, real_action_size=real_action_size, real_observation_size=real_observation_size)
                    print(f"len(replay_datasets[task_id]): {len(replay_datasets[task_id])}")
                elif args.experience_type == 'generate':
                    replay_datasets[task_id], save_datasets[task_id] = co.generate_new_data(origin_datasets[task_id], original[task_id], max_save_num=args.max_save_num, real_observation_size=real_observation_size, real_action_size=real_action_size)
                    print(f"len(replay_datasets[task_id]): {len(replay_datasets[task_id])}")
                elif args.experience_type == 'model_generate':
                    replay_datasets[task_id], save_datasets[task_id] = co.generate_replay_data_trajectory_with_model(origin_datasets[task_id], original[task_id], max_save_num=args.max_save_num, real_action_size=real_action_size, real_observation_size=real_observation_size)
                    print(f"len(replay_datasets[task_id]): {len(replay_datasets[task_id])}")
                elif args.experience_type in ['random_transition_generate', 'max_reward_generate', 'max_match_generate', 'max_model_generate', 'min_reward_generate', 'min_match_generate', 'min_model_generate']:
                    replay_datasets[task_id], save_datasets[task_id] = co.generate_replay_data_transition_with_model(origin_datasets[task_id], original[task_id], max_save_num=args.max_save_num, real_action_size=real_action_size, real_observation_size=real_observation_size)
                elif args.experience_type in ['random_episode_generate', 'max_reward_end_generate', 'max_reward_mean_generate', 'max_match_end_generate', 'max_match_mean_generate', 'max_model_end_generate', 'max_model_mean_generate', 'min_reward_end_generate', 'min_reward_mean_generate', 'min_match_end_generate', 'min_match_mean_generate', 'min_model_end_generate', 'min_model_mean_generate']:
                    replay_datasets[task_id], save_datasets[task_id] = co.generate_replay_data_episode_with_model(origin_datasets[task_id], original[task_id], max_save_num=args.max_save_num, real_action_size=real_action_size, real_observation_size=real_observation_size)
                    print(f"len(replay_datasets[task_id]): {len(replay_datasets[task_id])}")
                else:
                    replay_datasets = None
                print(f'Select Replay Buffer Time: {time.perf_counter() - start_time}')
            else:
                raise NotImplementedError
            # co.save_model(args.model_path + algos_name + '_' + str(task_id) + '.pt')
            if args.test and int(task_id) >= 1:
                break
        torch.save(save_datasets, f=args.model_path + algos_name + '_datasets.pt')
    else:
        assert args.model_path
        eval_datasets = dict()
        if co._impl is None:
            co.create_impl([real_observation_size + task_nums], real_action_size)
        for task_id, dataset in action_datasets.items():
            draw_path = args.model_path + algos_name + '_trajectories_' + str(task_id) + '_'
            eval_datasets[task_id] = dataset
            co.load_model(args.model_path + algos_name + '_' + str(task_id) + '.pt')
            co.test(
                eval_episodess=eval_datasets,
                scorers={
                    # 'environment': evaluate_on_environment(env),
                    "real_env": evaluate_on_environment(envs, end_points, task_nums, draw_path, dense=args.dense=='dense'),
                },
            )
    print('finish')

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Experimental evaluation of lifelong PG learning')
    parser.add_argument('--add_name', default='', type=str)
    parser.add_argument("--dataset", default='ant_maze', type=str, choices=['hopper_expert_v0', 'hopper_medium_v0', 'hopper_medium_expert_v0', 'hopper_medium_replay_v0', 'hopper_random_v0', 'halfcheetah_expert_v0', 'halfcheetah_medium_v0', 'halfcheetah_medium_expert_v0', 'halfcheetah_medium_replay_v0', 'halfcheetah_random_v0', 'walker2d_expert_v0', 'walker2d_medium_v0', 'walker2d_medium_expert_v0', 'walker2d_medium_replay_v0', 'walker2d_random_v0', 'mix_expert_v0', 'mix_medium_v0', 'mix_medium_expert_v0', 'mix_random_v0', 'walker_dir', 'ant_dir', 'cheetah_dir', 'cheetah_vel'])
    parser.add_argument('--inner_path', default='', type=str)
    parser.add_argument('--env_path', default=None, type=str)
    parser.add_argument('--inner_buffer_size', default=-1, type=int)
    parser.add_argument('--task_config', default='task_config/cheetah_dir.json', type=str)
    parser.add_argument('--siamese_hidden_size', default=100, type=int)
    parser.add_argument('--near_threshold', default=1, type=float)
    parser.add_argument('--siamese_threshold', default=1, type=float)
    parser.add_argument('--eval_batch_size', default=256, type=int)
    parser.add_argument('--batch_size', default=1024, type=int)
    parser.add_argument('--topk', default=4, type=int)
    parser.add_argument('--max_save_num', default=1000, type=int)
    parser.add_argument('--task_split_type', default='undirected', type=str)
    parser.add_argument('--algos', default='co', type=str)
    parser.add_argument('--eval', action='store_true')
    parser.add_argument('--test', action='store_true')
    parser.add_argument("--n_steps", default=500000, type=int)
    parser.add_argument("--n_steps_per_epoch", default=5000, type=int)
    parser.add_argument("--n_dynamic_steps", default=500000, type=int)
    parser.add_argument("--n_dynamic_steps_per_epoch", default=5000, type=int)
    parser.add_argument("--n_begin_steps", default=50000, type=int)
    parser.add_argument("--n_begin_steps_per_epoch", default=5000, type=int)
    parser.add_argument("--n_action_samples", default=4, type=int)
    parser.add_argument('--top_euclid', default=64, type=int)
    parser.add_argument('--replay_type', default='orl', type=str, choices=['orl', 'bc', 'ewc', 'gem', 'agem', 'r_walk', 'si'])
    parser.add_argument('--experience_type', default='siamese', type=str, choices=['siamese', 'model', 'generate', 'model_generate', 'random_transition', 'random_episode', 'max_reward', 'max_match', 'max_model', 'max_reward_end', 'max_reward_mean', 'max_match_end', 'max_match_mean', 'max_model_end', 'max_model_mean', 'min_reward', 'min_match', 'min_model', 'min_reward_end', 'min_reward_mean', 'min_match_end', 'min_match_mean', 'min_model_end', 'min_model_mean'])
    parser.add_argument('--sample_type', default='none', type=str, choices=['retrain', 'noise', 'none'])
    parser.add_argument('--use_model', action='store_true')
    parser.add_argument('--reduce_replay', default='retrain', type=str, choices=['retrain', 'no_retrain'])
    parser.add_argument('--dense', default='dense', type=str)
    parser.add_argument('--sum', default='no_sum', type=str)
    parser.add_argument('--replay_critic', action='store_true')
    parser.add_argument('--replay_model', action='store_true')
    parser.add_argument('--generate_step', default=1000, type=int)
    parser.add_argument('--model_noise', default=0, type=float)
    parser.add_argument('--retrain_time', type=int, default=1)
    parser.add_argument('--orl_alpha', type=float, default=1)
    parser.add_argument('--replay_alpha', type=float, default=1)
    parser.add_argument('--single_head', action='store_true')
    parser.add_argument('--task_nums', default=50, type=int)
    parser.add_argument('--use_cpu', action='store_true')
    parser.add_argument('--gpu', type=str, default='0')
    args = parser.parse_args()
    # if 'maze' in args.dataset:
    #     args.model_path = 'd3rlpy_' + args.experience_type + '_' + args.replay_type + '_' + args.reduce_replay + '_' + args.dense + '_' + args.dataset + '_' + ('test' if args.test else ('train' if not args.eval else 'eval'))
    # else:
    #     args.model_path = 'd3rlpy_' + args.experience_type + '_' + args.replay_type + '_' + args.reduce_replay + '_' + args.dataset + '_' + ('test' if args.test else ('train' if not args.eval else 'eval'))
    if 'maze' in args.dataset:
        args.model_path = 'd3rlpy' + '_' + args.dense + '_' + args.dataset + '/model_'
    else:
        args.model_path = 'd3rlpy' + '_' + args.dataset + '/model_'
    if not os.path.exists(args.model_path):
        os.makedirs(args.model_path)
    if 'model' in args.experience_type:
        args.use_model = True
    if args.replay_type == 'orl':
        args.replay_critic = True
    global DATASET_PATH
    DATASET_PATH = './.d4rl/datasets/'
    if args.use_cpu:
        device = torch.device('cpu')
    else:
        device = torch.device('cuda:0')
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu
    random.seed(12345)
    np.random.seed(12345)
    main(args, device)
