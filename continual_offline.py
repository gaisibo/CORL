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
from myd3rlpy.metrics.scorer import bc_error_scorer, td_error_scorer, evaluate_on_environment, match_on_environment, dis_on_environment, q_mean_scorer, q_replay_scorer
from myd3rlpy.siamese_similar import similar_psi, similar_phi
from myd3rlpy.dynamics.probabilistic_ensemble_dynamics import ProbabilisticEnsembleDynamics


replay_name = ['observations', 'actions', 'rewards', 'next_observations', 'terminals', 'means', 'std_logs', 'qs', 'phis', 'psis']
def main(args, device):
    np.set_printoptions(precision=1, suppress=True)
    ask_indexes = False
    if args.experience_type in ['model_prob', 'model_next', 'model_this', 'coverage'] and not args.eval:
        ask_indexes = True
    if args.dataset in ['hopper_expert_v0', 'hopper_medium_v0', 'hopper_medium_expert_v0', 'hopper_medium_replay_v0', 'hopper_random_v0', 'halfcheetah_expert_v0', 'halfcheetah_medium_v0', 'halfcheetah_medium_expert_v0', 'halfcheetah_medium_replay_v0', 'halfcheetah_random_v0', 'walker2d_expert_v0', 'walker2d_medium_v0', 'walker2d_medium_expert_v0', 'walker2d_medium_replay_v0', 'walker2d_random_v0', 'mix_expert_v0', 'mix_medium_expert_v0', 'mix_medium_v0', 'mix_random_v0']:
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
        # task_datasets, origin_datasets, taskid_datasets, action_datasets, envs, real_action_size, real_observation_size, indexes_euclids, task_nums = split_gym(args.top_euclid, args.dataset.replace('_', '-'), device=device)
        origin_datasets, taskid_datasets, indexes_euclids, distances_euclids, envs, real_action_size, real_observation_size, task_nums = split_gym(args.top_euclid, args.dataset.replace('_', '-'), device=device)
        if args.single_head:
            datasets = taskid_datasets
        else:
            datasets = origin_datasets
        env = None
    elif args.dataset in ['ant_dir_expert', 'cheetah_dir_expert', 'walker_dir_expert', 'cheetah_vel_expert', 'ant_dir_medium', 'cheetah_dir_medium', 'walker_dir_medium', 'cheetah_vel_medium', 'ant_dir_random', 'cheetah_dir_random', 'walker_dir_random', 'cheetah_vel_random', 'ant_dir_medium_random', 'cheetah_dir_medium_random', 'walker_dir_medium_random', 'cheetah_vel_medium_random', 'ant_dir_medium_replay', 'cheetah_dir_medium_replay', 'walker_dir_medium_replay', 'cheetah_vel_medium_replay']:
        from dataset.split_macaw import split_macaw
        inner_paths = ['dataset/macaw/' + args.inner_path.replace('num', str(i)) for i in range(args.task_nums)]
        env_paths = ['dataset/macaw/' + args.env_path.replace('num', str(i)) for i in range(args.task_nums)]
        origin_datasets, taskid_datasets, indexes_euclids, distances_euclids, env, real_action_size, real_observation_size = split_macaw(args.top_euclid, args.dataset, inner_paths, env_paths, ask_indexes=ask_indexes, device=device)
        if args.single_head:
            datasets = taskid_datasets
        else:
            datasets = origin_datasets
        envs = None
    elif args.dataset in ['ant_umaze_random', 'ant_umaze_medium', 'ant_umaze_expert']:
        strs = args.dataset.split('_')
        if strs[1] == 'umaze':
            from dataset.split_antmaze import split_navigate_antmaze_umaze_v2
            origin_datasets, taskid_datasets, indexes_euclids, distances_euclids, envs, real_action_size, real_observation_size, task_nums = split_navigate_antmaze_umaze_v2(args.top_euclid, device, strs[2])
            if args.single_head:
                datasets = taskid_datasets
            else:
                datasets = origin_datasets
        else:
            raise NotImplementedError
        env = None

    else:
        raise NotImplementedError

    # prepare algorithm
    if args.algo in ['td3_plus_bc', 'td3']:
        from myd3rlpy.algos.co_td3_plus_bc import CO
    elif args.algo == 'combo':
        from myd3rlpy.algos.co_combo import CO
    elif args.algo == 'cql':
        from myd3rlpy.algos.co_cql import CO
    else:
        raise NotImplementedError
    if args.experience_type == 'siamese':
        use_phi = True
    else:
        use_phi = False
    # co = CO(impl_name=args.algo, use_gpu=not args.use_cpu, batch_size=args.batch_size, id_size=args.task_nums, replay_type=args.replay_type, experience_type=args.experience_type, sample_type=args.sample_type, reduce_replay=args.reduce_replay, use_phi=use_phi, use_model=args.use_model, replay_critic=args.replay_critic, replay_model=args.replay_model, replay_alpha=args.replay_alpha, generate_step=args.generate_step, model_noise=args.model_noise, retrain_time=args.retrain_time, orl_alpha=args.orl_alpha, single_head=args.single_head, clone_actor=args.clone_actor, clone_finish=args.clone_finish)
    co = CO(impl_name=args.algo, use_gpu=not args.use_cpu, batch_size=args.batch_size, id_size=args.task_nums, replay_type=args.replay_type, experience_type=args.experience_type, reduce_replay=args.reduce_replay, use_phi=use_phi, use_model=args.use_model, replay_critic=args.replay_critic, replay_model=args.replay_model, replay_alpha=args.replay_alpha, model_noise=args.model_noise, retrain_time=args.retrain_time, orl_alpha=args.orl_alpha, single_head=args.single_head, clone_actor=args.clone_actor, clone_finish=args.clone_finish)

    experiment_name = "CO" + '_'
    algos_name = args.replay_type
    algos_name += "_" + args.algo
    algos_name += "_" + args.experience_type
    algos_name += '_' + args.distance_type
    # algos_name += '_' + args.sample_type
    algos_name += '_' + args.dataset
    algos_name += '_' + str(args.max_save_num)
    algos_name += '_' + str(args.replay_alpha)
    algos_name += '_' + str(args.seed)
    if args.add_name != '':
        algos_name += '_' + args.add_name
    algos_name += '_singlehead' if args.single_head else '_multihead'
    algos_name += '_clone' if args.clone_actor else '_noclone'
    algos_name += '_finish' if args.clone_finish else '_nofinish'

    pretrain_name = args.model_path

    if not args.eval:
        replay_datasets = dict()
        save_datasets = dict()
        eval_datasets = dict()
        learned_tasks = []
        if args.experience_type == 'all':
            for epoch in range(int(args.n_steps // args.n_steps_per_epoch)):
                for task_id, dataset in datasets.items():
                    if int(task_id) < args.read_policy:
                        replay_datasets[task_id] = torch.load(args.model_path + algos_name + '_' + str(task_id) + '_datasets.pt')
                        co._impl.change_task(int(task_id))
                        continue
                    learned_tasks.append(task_id)
                    task_id = str(task_id)
                    start_time = time.perf_counter()
                    print(f'Start Training {task_id}')
                    eval_datasets[task_id] = dataset
                    draw_path = args.model_path + algos_name + '_trajectories_' + str(task_id)
                    dynamic_path = args.model_path + args.dataset + '_' + str(task_id) + '_dynamic.pt'
                    print(dynamic_path)
                    try:
                        dynamic_state_dict = torch.load(dynamic_path, map_location=device)
                    except:
                        dynamic_state_dict = None
                        raise NotImplementedError
                    pretrain_state_dict = None
                    # train
                    if env is not None:
                        scorers = dict(zip(['real_env' + str(n) for n in datasets.keys()], [evaluate_on_environment(env, test_id=str(n), mix='mix' in args.dataset and n == '0', add_on=args.add_on, clone_actor=args.clone_actor, task_id_dim=0 if not args.single_head else len(datasets.keys())) for n in learned_tasks]))
                    elif envs is not None:
                        scorers = dict(zip(['real_env' + str(n) for n in datasets.keys()], [evaluate_on_environment(envs[str(n)], test_id=str(n), mix='mix' in args.dataset and n == '0', add_on=args.add_on, clone_actor=args.clone_actor) for n in learned_tasks]))
                    else:
                        raise NotImplementedError
                    co.fit(
                        task_id,
                        dataset,
                        replay_datasets,
                        real_action_size = real_action_size,
                        real_observation_size = real_observation_size,
                        eval_episodes=datasets,
                        # n_epochs=args.n_epochs if not args.test else 1,
                        n_steps=args.n_steps_per_epoch,
                        n_steps_per_epoch=args.n_steps_per_epoch,
                        n_dynamic_steps=args.n_dynamic_steps,
                        n_dynamic_steps_per_epoch=args.n_dynamic_steps_per_epoch,
                        dynamic_state_dict=dynamic_state_dict,
                        pretrain_state_dict=pretrain_state_dict,
                        pretrain_task_id=args.read_policy,
                        experiment_name=experiment_name + algos_name,
                        scorers = scorers,
                        test=args.test,
                        epoch_num = epoch,
                    )
                    print(f'Training task {task_id} time: {time.perf_counter() - start_time}')
                    co.save_model(args.model_path + algos_name + '_' + str(task_id) + '.pt')
        elif args.experience_type == 'single':
            for task_id, dataset in datasets.items():
                co._impl = None
                if int(task_id) < args.read_policy:
                    replay_datasets[task_id] = torch.load(args.model_path + algos_name + '_' + str(task_id) + '_datasets.pt')
                    co._impl.change_task(int(task_id))
                    continue
                learned_tasks.append(task_id)
                task_id = str(task_id)
                start_time = time.perf_counter()
                print(f'Start Training {task_id}')
                eval_datasets[task_id] = dataset
                draw_path = args.model_path + algos_name + '_trajectories_' + str(task_id)
                dynamic_path = args.model_path + args.dataset + '_' + str(task_id) + '_dynamic.pt'
                print(dynamic_path)
                try:
                    dynamic_state_dict = torch.load(dynamic_path, map_location=device)
                except:
                    dynamic_state_dict = None
                    raise NotImplementedError
                pretrain_state_dict = None
                # train
                if env is not None:
                    scorers = dict(zip(['real_env' + str(n) for n in datasets.keys()], [evaluate_on_environment(env, test_id=str(n), mix='mix' in args.dataset and n == '0', add_on=args.add_on, clone_actor=args.clone_actor, task_id_dim=0 if not args.single_head else len(datasets.keys())) for n in learned_tasks]))
                elif envs is not None:
                    scorers = dict(zip(['real_env' + str(n) for n in datasets.keys()], [evaluate_on_environment(envs[str(n)], test_id=str(n), mix='mix' in args.dataset and n == '0', add_on=args.add_on, clone_actor=args.clone_actor) for n in learned_tasks]))
                else:
                    raise NotImplementedError
                co.fit(
                    task_id,
                    dataset,
                    replay_datasets,
                    real_action_size = real_action_size,
                    real_observation_size = real_observation_size,
                    eval_episodes=datasets,
                    # n_epochs=args.n_epochs if not args.test else 1,
                    n_steps=args.n_steps,
                    n_steps_per_epoch=args.n_steps_per_epoch,
                    n_dynamic_steps=args.n_dynamic_steps,
                    n_dynamic_steps_per_epoch=args.n_dynamic_steps_per_epoch,
                    dynamic_state_dict=dynamic_state_dict,
                    pretrain_state_dict=pretrain_state_dict,
                    pretrain_task_id=args.read_policy,
                    experiment_name=experiment_name + algos_name,
                    scorers = scorers,
                    test=args.test,
                )
                print(f'Training task {task_id} time: {time.perf_counter() - start_time}')
                co.save_model(args.model_path + algos_name + '_' + str(task_id) + '.pt')
        else:
            print(f'datasets.items()')
            for task_id, dataset in datasets.items():
                # train
                max_transition_len = max(list([len(episode.transitions) for episode in dataset.episodes]))
                learned_tasks.append(task_id)
                if int(task_id) < args.read_policy:
                    replay_datasets[task_id] = torch.load(args.model_path + algos_name + '_' + str(task_id) + '_datasets.pt')
                    co._impl.change_task(int(task_id))
                    continue
                task_id = str(task_id)
                start_time = time.perf_counter()
                print(f'Start Training {task_id}')
                eval_datasets[task_id] = dataset
                draw_path = args.model_path + algos_name + '_trajectories_' + str(task_id)
                dynamic_path = args.model_path + args.dataset + '_' + str(task_id) + '_dynamic.pt'
                try:
                    dynamic_state_dict = torch.load(dynamic_path, map_location=device)
                except Exception as e:
                    print(e)
                    dynamic_state_dict = None
                    raise NotImplementedError
                if env is not None:
                    scorers = dict(zip(['real_env' + str(n) for n in datasets.keys()], [evaluate_on_environment(env, test_id=str(n), mix='mix' in args.dataset and n == '0', add_on=args.add_on, clone_actor=args.clone_actor, task_id_dim=0 if not args.single_head else len(datasets.keys())) for n in learned_tasks]))
                elif envs is not None:
                    scorers = dict(zip(['real_env' + str(n) for n in datasets.keys()], [evaluate_on_environment(envs[str(n)], test_id=str(n), mix='mix' in args.dataset and n == '0', add_on=args.add_on, clone_actor=args.clone_actor) for n in learned_tasks]))
                else:
                    raise NotImplementedError
                if int(task_id) == args.read_policy:
                    pretrain_path = 'offline_pretrained_models/' + experiment_name + args.dataset + '_' + str(task_id) + '.pt'
                    if args.replay_type not in ['fix', 'ewc', 'si', 'rwalk']:
                        for past_task_id in range(int(task_id)):
                            try:
                                replay_datasets[str(past_task_id)] = torch.load(f=args.model_path + algos_name + '_' + str(past_task_id) + '_datasets.pt')
                            except BaseException as e:
                                print(f'Don\' have replay_datasets[{past_task_id}]')
                                raise e
                    co.build_with_dataset(dataset, real_action_size, real_observation_size, task_id)
                    co.load_model(pretrain_path)
                else:
                    print("Start Fitting")
                    co.fit(
                        task_id,
                        dataset,
                        replay_datasets,
                        real_action_size = real_action_size,
                        real_observation_size = real_observation_size,
                        eval_episodes=datasets,
                        # n_epochs=args.n_epochs if not args.test else 1,
                        n_steps=args.n_steps,
                        n_steps_per_epoch=args.n_steps_per_epoch,
                        n_dynamic_steps=args.n_dynamic_steps,
                        n_dynamic_steps_per_epoch=args.n_dynamic_steps_per_epoch,
                        dynamic_state_dict=dynamic_state_dict,
                        # pretrain_state_dict=pretrain_state_dict,
                        # pretrain_task_id=args.read_policy,
                        experiment_name=experiment_name + algos_name,
                        scorers = scorers,
                        test=args.test,
                    )
                    print(f'Training task {task_id} time: {time.perf_counter() - start_time}')
                    co.save_model(args.model_path + algos_name + '_' + str(task_id) + '.pt')
                max_transition_len = 1000
                if env is not None:
                    co.generate_replay(task_id, datasets, env, args.replay_type, args.experience_type, replay_datasets, save_datasets, args.max_save_num, max_transition_len, real_action_size, real_observation_size, indexes_euclids[task_id], distances_euclids[task_id], args.d_threshold, args.test, args.model_path, algos_name, learned_tasks)
                else:
                    co.generate_replay(task_id, datasets, envs[task_id], args.replay_type, args.experience_type, replay_datasets, save_datasets, args.max_save_num, max_transition_len, real_action_size, real_observation_size, indexes_euclids[task_id], distances_euclids[task_id], args.d_threshold, args.test, args.model_path, algos_name, learned_tasks)
                # # eval
                # if args.replay_type not in ['fix', 'ewc', 'si', 'rwalk']:
                #     if env is not None:
                #         scorers = dict(zip(['dis_env' + str(n) for n in datasets.keys()], [dis_on_environment(env, replay_dataset = replay_datasets[n], test_id=str(n), mix='mix' in args.dataset and n == '0', clone_actor=args.clone_actor, task_id_dim=0 if not args.single_head else len(datasets.keys())) for n in learned_tasks]))
                #     elif envs is not None:
                #         scorers = dict(zip(['dis_env' + str(n) for n in datasets.keys()], [dis_on_environment(envs[str(n)], replay_dataset = replay_datasets[n], test_id=str(n), mix='mix' in args.dataset and n == '0', clone_actor=args.clone_actor) for n in learned_tasks]))
                #     else:
                #         raise NotImplementedError
                # # setup logger
                # logger = co._prepare_logger(True, experiment_name, True, "d3rply_logs", True, None,)
                # eval_episodes = datasets
                # if scorers and eval_episodes:
                #     co._evaluate(eval_episodes, scorers, logger)

                # logger.commit(int(task_id), 0)
                # if args.test and int(task_id) >= 2:
                #     break
    else:
        replay_datasets = dict()
        learned_tasks = []
        if args.replay_type not in ['fix', 'ewc', 'si', 'rwalk']:
            for past_task_id in datasets.keys():
                try:
                    replay_datasets[str(past_task_id)] = torch.load(f=args.model_path + algos_name + '_' + str(past_task_id) + '_datasets.pt')
                except BaseException as e:
                    print(f'Don\' have replay_datasets[{past_task_id}]')
        for task_id, dataset in datasets.items():
            learned_tasks.append(task_id)
            draw_path = args.model_path + algos_name + '_trajectories_' + str(task_id)
            dynamic_path = args.model_path + args.dataset + '_' + str(task_id) + '_dynamic.pt'
            try:
                dynamic_state_dict = torch.load(dynamic_path, map_location=device)
            except:
                raise NotImplementedError
            pretrain_path = args.model_path + algos_name + '_' + str(task_id) + '_no_clone.pt'
            try:
                pretrain_state_dict = torch.load(pretrain_path, map_location=device)
            except BaseException as e:
                print(f'Don\'t have pretrain_state_dict[{task_id}]')
                raise e
            if args.replay_type not in ['fix', 'ewc', 'si', 'rwalk']:
                for past_task_id in range(int(task_id)):
                    try:
                        replay_datasets[str(past_task_id)] = torch.load(f=args.model_path + algos_name + '_' + str(past_task_id) + '_datasets.pt')
                    except BaseException as e:
                        print(f'Don\' have replay_datasets[{past_task_id}]')
                        raise e
            co.build_with_dataset(dataset, real_action_size, real_observation_size, task_id)
            co.load_state_dict(pretrain_state_dict, task_id)
            logger = co._prepare_logger(True, experiment_name, True, "d3rply_logs", True, None,)

            # eval
            if env is not None:
                scorers = dict(zip(['real_env' + str(n) for n in datasets.keys()], [evaluate_on_environment(env, test_id=str(n), mix='mix' in args.dataset and n == '0', add_on=args.add_on, clone_actor=args.clone_actor, task_id_dim=0 if not args.single_head else len(datasets.keys())) for n in learned_tasks]))
            elif envs is not None:
                scorers = dict(zip(['real_env' + str(n) for n in datasets.keys()], [evaluate_on_environment(envs[str(n)], test_id=str(n), mix='mix' in args.dataset and n == '0', add_on=args.add_on, clone_actor=args.clone_actor) for n in learned_tasks]))
            else:
                raise NotImplementedError
            # setup logger
            eval_episodes = datasets
            if scorers and eval_episodes:
                co._evaluate(eval_episodes, scorers, logger)

            # eval
            if env is not None:
                scorers = dict(zip(['dis_env' + str(n) for n in datasets.keys()], [dis_on_environment(env, replay_dataset = replay_datasets[n], test_id=str(n), mix='mix' in args.dataset and n == '0', clone_actor=args.clone_actor, task_id_dim=0 if not args.single_head else len(datasets.keys())) for n in learned_tasks]))
            elif envs is not None:
                scorers = dict(zip(['dis_env' + str(n) for n in datasets.keys()], [dis_on_environment(envs[str(n)], replay_dataset = replay_datasets[n], test_id=str(n), mix='mix' in args.dataset and n == '0', clone_actor=args.clone_actor) for n in learned_tasks]))
            else:
                raise NotImplementedError
            # setup logger
            eval_episodes = datasets
            if scorers and eval_episodes:
                co._evaluate(eval_episodes, scorers, logger)

            logger.commit(int(task_id), 0)
            if args.test and int(task_id) >= 2:
                break
    print('finish')

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Experimental evaluation of lifelong PG learning')
    parser.add_argument('--add_name', default='', type=str)
    parser.add_argument("--dataset", default='ant_dir', type=str)
    parser.add_argument("--quality", default='medium', type=str)
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
    parser.add_argument('--algo', default='td3_plus_bc', type=str, choices=['combo', 'td3_plus_bc', 'cql'])
    parser.add_argument('--eval', action='store_true')
    parser.add_argument('--test', action='store_true')
    parser.add_argument("--n_steps", default=150000, type=int)
    parser.add_argument("--n_steps_per_epoch", default=5000, type=int)
    parser.add_argument("--n_dynamic_steps", default=500000, type=int)
    parser.add_argument("--n_dynamic_steps_per_epoch", default=5000, type=int)
    parser.add_argument("--n_begin_steps", default=50000, type=int)
    parser.add_argument("--n_begin_steps_per_epoch", default=5000, type=int)
    parser.add_argument("--n_action_samples", default=4, type=int)
    parser.add_argument('--top_euclid', default=64, type=int)
    parser.add_argument('--replay_type', default='orl', type=str, choices=['none', 'fix', 'orl', 'bc', 'ewc', 'gem', 'agem', 'rwalk', 'si'])
    parser.add_argument('--experience_type', default='siamese', type=str, choices=['all', 'none', 'single', 'online', 'model_prob', 'model_next', 'model', 'model_this', 'coverage', 'random_transition', 'random_episode', 'max_reward', 'max_match', 'max_supervise', 'max_model', 'max_reward_end', 'max_reward_mean', 'max_match_end', 'max_match_mean', 'max_supervise_end', 'max_supervise_mean', 'max_model_end', 'max_model_mean', 'min_reward', 'min_match', 'min_supervise', 'min_model', 'min_reward_end', 'min_reward_mean', 'min_match_end', 'min_match_mean', 'min_supervise_end', 'min_supervise_mean', 'min_model_end', 'min_model_mean'])
    parser.add_argument("--distance_type", default='l2', type=str, choices=['l2', 'feature'])
    # parser.add_argument('--generate_type', default='none', type=str)
    parser.add_argument('--clone_actor', action='store_true')
    parser.add_argument('--clone_finish', action='store_true')
    # parser.add_argument('--sample_type', default='none', type=str, choices=['retrain_model', 'retrain_actor', 'noise', 'none'])
    parser.add_argument('--use_model', action='store_true')
    parser.add_argument('--reduce_replay', default='retrain', type=str, choices=['retrain', 'no_retrain'])
    parser.add_argument('--dense', default='dense', type=str)
    parser.add_argument('--sum', default='no_sum', type=str)
    parser.add_argument('--replay_critic', action='store_true')
    parser.add_argument('--replay_model', action='store_true')
    # parser.add_argument('--generate_step', default=10, type=int)
    parser.add_argument('--model_noise', default=0, type=float)
    parser.add_argument('--retrain_time', type=int, default=1)
    parser.add_argument('--orl_alpha', type=float, default=1)
    parser.add_argument('--replay_alpha', type=float, default=1)
    parser.add_argument('--d_threshold', type=float, default=0.1)
    parser.add_argument('--single_head', action='store_true')
    parser.add_argument('--task_nums', default=50, type=int)
    parser.add_argument('--use_cpu', action='store_true')
    parser.add_argument('--gpu', type=str, default='0')
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--read_policy', type=int, default=0)
    args = parser.parse_args()
    args.env_path = f"{args.dataset}/env_{args.dataset}_train_tasknum.pkl"
    args.inner_path = f"sac_{args.dataset}_num/{args.quality}.hdf5"
    args.dataset = f"{args.dataset}_{args.quality}"
    args.model_path = 'd3rlpy' + '_' + args.dataset
    if not os.path.exists(args.model_path):
        os.makedirs(args.model_path)
    args.model_path += '/model_'
    # if args.experience_type == 'model':
    #     args.experience_type = 'model_next'
    if 'model' in args.experience_type:#  or args.experience_type == 'generate' or args.generate_type in ['generate', 'model', 'model_generate']:
        args.use_model = True
    args.use_model = True
    if args.replay_type == 'orl':
        args.replay_critic = True
    if 'maze' in args.dataset:
        args.add_on = False
    else:
        args.add_on = True
    if args.single_head:
        args.clone_actor = False
        args.clone_finish = False

    if args.replay_type == 'fix':
        assert not args.clone_actor

    global DATASET_PATH
    DATASET_PATH = './.d4rl/datasets/'
    if args.use_cpu:
        device = torch.device('cpu')
    else:
        device = torch.device('cuda:0')
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu
    seeds = [12345, 1234, 123, 12, 1]
    random.seed(seeds[args.seed])
    np.random.seed(seeds[args.seed])
    torch.manual_seed(seeds[args.seed])
    torch.cuda.manual_seed(seeds[args.seed])

    # args.test = True

    main(args, device)
