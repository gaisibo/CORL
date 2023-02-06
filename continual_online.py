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
from dataset.sequential_mujoco import sequential_mujoco


replay_name = ['observations', 'actions', 'rewards', 'next_observations', 'terminals', 'means', 'std_logs', 'qs', 'phis', 'psis']
def main(args, device):
    np.set_printoptions(precision=1, suppress=True)
    inner_paths = [f'dataset/online/sac_{args.dataset}/itr_{x}.hdf5' for x in args.dataset_num]
    datasets, env = sequential_mujoco(args.dataset, inner_paths)
    real_observation_size = datasets['0'].observations.shape[1]
    real_action_size = datasets['0'].actions.shape[1]

    # prepare algorithm
    if args.algo in ['td3_plus_bc', 'td3n']:
        from myd3rlpy.algos.co_td3_plus_bc import CO
        if args.algo == 'td3_plus_bc':
            n_critics = 2
        else:
            n_critics = 500
    elif args.algo == 'combo':
        n_critics = 2
        from myd3rlpy.algos.co_combo import CO
    elif args.algo == 'cql':
        n_critics = 2
        from myd3rlpy.algos.co_cql import CO
    elif args.algo == 'sacn':
        n_critics = 100
        from myd3rlpy.algos.co_sacn import CO
    else:
        raise NotImplementedError
    if args.experience_type == 'siamese':
        use_phi = True
    else:
        use_phi = False
    co = CO(use_gpu=not args.use_cpu, impl_name=args.algo, batch_size=args.batch_size, n_critics=n_critics, id_size=args.task_nums, replay_type=args.replay_type, experience_type=args.experience_type, sample_type=args.sample_type, reduce_replay=args.reduce_replay, use_phi=use_phi, use_model=args.use_model, replay_critic=args.replay_critic, replay_model=args.replay_model, replay_alpha=args.replay_alpha, generate_step=args.generate_step, model_noise=args.model_noise, retrain_time=args.retrain_time, orl_alpha=args.orl_alpha, single_head=args.single_head, clone_actor=args.clone_actor, clone_finish=args.clone_finish)

    experiment_name = "CO" + '_'
    algos_name = args.replay_type
    algos_name += "_" + args.algo
    algos_name += "_" + args.experience_type
    algos_name += '_' + args.sample_type
    algos_name += '_' + args.dataset
    algos_name += '_' + args.dataset_num_str
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
        for task_id, dataset in datasets.items():
            max_transition_len = max(list([len(episode.transitions) for episode in dataset.episodes]))
            if int(task_id) < args.read_policies:
                replay_datasets[task_id] = torch.load(args.model_path + algos_name + '_' + str(task_id) + '_datasets.pt')
                co._impl.change_task(int(task_id))
                continue
            start_time = time.perf_counter()
            print(f'Start Training {task_id}')
            eval_datasets[task_id] = dataset
            # draw_path = args.model_path + algos_name + '_trajectories_' + str(task_id)
            # dynamic_path = args.model_path + args.dataset + '_' + str(task_id) + '_dynamic.pt'
            # print(dynamic_path)
            # try:
            #     dynamic_state_dict = torch.load(dynamic_path, map_location=device)
            # except:
            #     dynamic_state_dict = None
            #     raise NotImplementedError
            dynamic_state_dict = None
            if int(task_id) == args.read_policies:
                pretrain_path = args.model_path + args.algo + '_' + args.dataset + '_' + str(task_id) + '.pt'
                try:
                    pretrain_state_dict = torch.load(pretrain_path, map_location=device)
                except BaseException as e:
                    print(f'Don\'t have pretrain_state_dict[{task_id}]')
                    raise e
                # if args.replay_type not in ['ewc', 'si', 'rwalk']:
                #     for past_task_id in range(int(task_id)):
                #         try:
                #             replay_datasets[str(past_task_id)] = torch.load(f=args.model_path + algos_name + '_' + str(past_task_id) + '_datasets.pt')
                #         except BaseException as e:
                #             print(f'Don\' have replay_datasets[{past_task_id}]')
                #             raise e
            else:
                pretrain_state_dict = None

            # train
            if not args.test:
                scorers = dict()
                scorers['real_env'] = evaluate_on_environment(env)
            else:
                scorers = None
            co.fit(
                task_id,
                dataset,
                replay_datasets,
                real_action_size=real_action_size,
                real_observation_size=real_observation_size,
                eval_episodes=datasets,
                # n_epochs=args.n_epochs if not args.test else 1,
                n_steps=args.n_steps,
                n_steps_per_epoch=args.n_steps_per_epoch,
                n_dynamic_steps=args.n_dynamic_steps,
                n_dynamic_steps_per_epoch=args.n_dynamic_steps_per_epoch,
                dynamic_state_dict=dynamic_state_dict,
                pretrain_state_dict=pretrain_state_dict,
                pretrain_task_id=args.read_policies,
                experiment_name=experiment_name + algos_name,
                scorers = scorers,
                test=args.test,
            )
            print(f'Training task {task_id} time: {time.perf_counter() - start_time}')
            co.save_model(args.model_path + algos_name + '_' + str(task_id) + '.pt')

            # random_select replay buffer
            if args.replay_type in ['bc', 'orl']:
                if isinstance(dataset, MDPDataset):
                    episodes = dataset.episodes
                else:
                    episodes = dataset
                transitions = [transition for episode in episodes for transition in episode.transitions]
                random.shuffle(transitions)
                transitions = transitions[:args.max_save_num]
                replay_observations = torch.stack([torch.from_numpy(transition.observation) for transition in transitions], dim=0).detach().to('cpu')
                replay_actions = torch.stack([torch.from_numpy(transition.action) for transition in transitions], dim=0).detach().to('cpu')
                replay_rewards = torch.stack([torch.from_numpy(np.array([transition.reward])) for transition in transitions], dim=0).detach().to(torch.float32).to('cpu')
                replay_next_observations = torch.stack([torch.from_numpy(transition.next_observation) for transition in transitions], dim=0).detach().to('cpu')
                replay_terminals = torch.stack([torch.from_numpy(np.array([transition.terminal])) for transition in transitions], dim=0).detach().to(torch.float32).to('cpu')
                replay_policy_actions = co._impl._policy(replay_observations.to(co._impl.device)).detach().to('cpu')
                replay_qs = co._impl._q_func(replay_observations.to(co._impl.device), replay_actions.to(co._impl.device)).detach().to('cpu')
                replay_dataset = torch.utils.data.TensorDataset(replay_observations, replay_actions, replay_rewards, replay_next_observations, replay_terminals, replay_policy_actions, replay_qs)
            elif args.replay_type == 'ewc':
                replay_dataset = None
            elif args.replay_type == 'none':
                replay_dataset = None
            replay_datasets[task_id] = replay_dataset
    else:
        replay_datasets = dict()
        learned_tasks = []
        if args.replay_type not in ['ewc', 'si', 'rwalk']:
            for past_task_id in datasets.keys():
                try:
                    replay_datasets[str(past_task_id)] = torch.load(f=args.model_path + algos_name + '_' + str(past_task_id) + '_datasets.pt')
                except BaseException as e:
                    print(f'Don\' have replay_datasets[{past_task_id}]')
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
        if args.replay_type not in ['ewc', 'si', 'rwalk']:
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
        if not args.test:
            if env is not None:
                scorers = dict(zip(['real_env' + str(n) for n in datasets.keys()], [evaluate_on_environment(env, test_id=str(n), mix='mix' in args.dataset and n == '0', add_on=args.add_on, clone_actor=args.clone_actor, task_id_dim=0 if not args.single_head else len(datasets.keys())) for n in learned_tasks]))
            elif envs is not None:
                scorers = dict(zip(['real_env' + str(n) for n in datasets.keys()], [evaluate_on_environment(envs[str(n)], test_id=str(n), mix='mix' in args.dataset and n == '0', add_on=args.add_on, clone_actor=args.clone_actor) for n in learned_tasks]))
            else:
                raise NotImplementedError
        else:
            raise NotImplementedError
        # setup logger
        eval_episodes = datasets
        if scorers and eval_episodes:
            co._evaluate(eval_episodes, scorers, logger)

        # eval
        if not args.test:
            if env is not None:
                scorers = dict(zip(['dis_env' + str(n) for n in datasets.keys()], [dis_on_environment(env, replay_dataset = replay_datasets[n], test_id=str(n), mix='mix' in args.dataset and n == '0', clone_actor=args.clone_actor, task_id_dim=0 if not args.single_head else len(datasets.keys())) for n in learned_tasks]))
            elif envs is not None:
                scorers = dict(zip(['dis_env' + str(n) for n in datasets.keys()], [dis_on_environment(envs[str(n)], replay_dataset = replay_datasets[n], test_id=str(n), mix='mix' in args.dataset and n == '0', clone_actor=args.clone_actor) for n in learned_tasks]))
            else:
                raise NotImplementedError
        else:
            raise NotImplementedError
        # setup logger
        eval_episodes = datasets
        if scorers and eval_episodes:
            co._evaluate(eval_episodes, scorers, logger)

        logger.commit(int(task_id), 0)
    print('finish')

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Experimental evaluation of lifelong PG learning')
    parser.add_argument('--add_name', default='', type=str)
    parser.add_argument("--dataset", required=True, type=str)
    parser.add_argument("--dataset_num", default='0,100,300,600,900', type=str)
    parser.add_argument('--inner_path', default='', type=str)
    parser.add_argument('--env_path', default=None, type=str)
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
    parser.add_argument('--algo', default='td3_plus_bc', type=str, choices=['combo', 'td3_plus_bc', 'td3n', 'cql', 'sacn'])
    parser.add_argument('--eval', action='store_true')
    parser.add_argument('--test', action='store_true')
    parser.add_argument("--n_steps", default=9000000, type=int)
    parser.add_argument("--n_steps_per_epoch", default=3000, type=int)
    parser.add_argument("--n_dynamic_steps", default=500000, type=int)
    parser.add_argument("--n_dynamic_steps_per_epoch", default=5000, type=int)
    parser.add_argument("--n_action_samples", default=4, type=int)
    parser.add_argument('--top_euclid', default=64, type=int)
    parser.add_argument('--replay_type', default='bc', type=str, choices=['none', 'orl', 'bc', 'ewc', 'gem', 'agem', 'rwalk', 'si'])
    parser.add_argument('--experience_type', default='none', type=str, choices=['all', 'none', 'single', 'online', 'generate', 'model_prob', 'model_next', 'model', 'model_this', 'coverage', 'random_transition', 'random_episode', 'max_reward', 'max_match', 'max_supervise', 'max_model', 'max_reward_end', 'max_reward_mean', 'max_match_end', 'max_match_mean', 'max_supervise_end', 'max_supervise_mean', 'max_model_end', 'max_model_mean', 'min_reward', 'min_match', 'min_supervise', 'min_model', 'min_reward_end', 'min_reward_mean', 'min_match_end', 'min_match_mean', 'min_supervise_end', 'min_supervise_mean', 'min_model_end', 'min_model_mean'])
    parser.add_argument('--generate_type', default='none', type=str)
    parser.add_argument('--clone_actor', action='store_true')
    parser.add_argument('--clone_finish', action='store_true')
    parser.add_argument('--sample_type', default='none', type=str, choices=['retrain_model', 'retrain_actor', 'noise', 'none'])
    parser.add_argument('--use_model', action='store_true')
    parser.add_argument('--reduce_replay', default='no_retrain', type=str, choices=['retrain', 'no_retrain'])
    parser.add_argument('--dense', default='dense', type=str)
    parser.add_argument('--sum', default='no_sum', type=str)
    parser.add_argument('--replay_critic', action='store_true')
    parser.add_argument('--replay_model', action='store_true')
    parser.add_argument('--generate_step', default=10, type=int)
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
    parser.add_argument('--read_policies', type=int, default=-1)
    args = parser.parse_args()
    args.model_path = 'd3rlpy' + '_' + args.dataset
    if not os.path.exists(args.model_path):
        os.makedirs(args.model_path)
    args.model_path += '/model_'
    # if 'model' in args.experience_type or args.experience_type == 'generate' or args.generate_type in ['generate', 'model', 'model_generate']:
    #     args.use_model = True
    # args.use_model = True
    args.single_head = True
    if args.single_head:
        args.clone_actor = False
        args.clone_finish = False
    args.dataset_num_str = args.dataset_num
    args.dataset_num = args.dataset_num.split('-')
    args.dataset_num = [x for x in args.dataset_num]
    if args.replay_type == 'orl':
        args.replay_critic = True

    global DATASET_PATH
    DATASET_PATH = './.d4rl/datasets/'
    if args.use_cpu:
        device = torch.device('cpu')
    else:
        device = torch.device('cuda:0')
    if not args.use_cpu:
        os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu
    seeds = [12345, 1234, 123, 12, 1]
    random.seed(seeds[args.seed])
    np.random.seed(seeds[args.seed])
    torch.manual_seed(seeds[args.seed])
    if not args.use_cpu:
        torch.cuda.manual_seed(seeds[args.seed])
    main(args, device)
