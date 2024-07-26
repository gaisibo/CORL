import os
import argparse
import random
from collections import namedtuple
import time
from functools import partial
import numpy as np
import gym
from mygym.envs.halfcheetah_block import HalfCheetahBlockEnv

import torch
from torch.optim.lr_scheduler import CosineAnnealingLR

import d3rlpy
from d3rlpy.online.buffers import ReplayBuffer

from myd3rlpy.metrics.scorer import evaluate_on_environment_help
from dataset.load_d4rl import get_d4rl_local, get_antmaze_local, get_dataset
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
replay_name = ['observations', 'actions', 'rewards', 'next_observations', 'terminals', 'means', 'std_logs', 'qs']
def main(args, device):
    np.set_printoptions(precision=1, suppress=True)
    ask_indexes = False
    if args.dataset_kind in ['d4rl', 'antmaze']:
        env = gym.make(args.dataset)
        eval_env = gym.make(args.dataset)
    elif args.dataset_kind == 'block':
        env = gym.make(args.dataset)
        eval_env = gym.make(args.dataset)
    else:
        raise NotImplementedError

    # prepare algorithm
    if args.algo in ['td3_plus_bc', 'td3']:
        from myd3rlpy.algos.st_td3 import ST
    elif args.algo_kind == 'cql':
        from myd3rlpy.algos.st_cql import ST
    elif args.algo in ['iql', 'iqln', 'iqln2', 'iqln3', 'iqln4', 'sql', 'sqln']:
        from myd3rlpy.algos.st_iql import ST
    elif args.algo in ['sacn', 'edac']:
        from myd3rlpy.algos.st_sac import ST
    else:
        raise NotImplementedError
    st_dict, online_st_dict, step_dict = get_st_dict(args, args.dataset_kind, args.algo)
    print(f"{st_dict['actor_learning_rate']=}")
    if args.n_steps is not None:
        step_dict['n_steps'] = args.n_steps
    if args.algo in ['iql', 'sql', 'iqln', 'iqln2', 'iqln3', 'iqln4', 'sqln']:
        st_dict['weight_temp'] = args.weight_temp
        st_dict['expectile'] = args.expectile
        st_dict['expectile_min'] = args.expectile_min
        st_dict['expectile_max'] = args.expectile_max
        if args.algo in ['sql', 'sqln']:
            st_dict['alpha'] = args.alpha
        if args.algo in ['iqln', 'iqln2', 'iqln3', 'iqln4', 'sqln']:
            st_dict['n_critics'] = args.n_critics
            st_dict['std_time'] = args.std_time
            st_dict['std_type'] = args.std_type
            st_dict['entropy_time'] = args.entropy_time
    elif args.algo == 'cql':
        st_dict['std_time'] = args.std_time
        st_dict['std_type'] = args.std_type
        st_dict['entropy_time'] = args.entropy_time
    elif args.algo in ['sacn', 'edac']:
        st_dict['n_critics'] = args.n_critics
        if args.algo == 'edac':
            st_dict['eta'] = args.eta
    st = ST(**st_dict)

    experiment_name = "ST" + '_'
    algos_name = args.algo
    algos_name += '_' + str(args.weight_temp)
    algos_name += '_' + str(args.expectile)
    algos_name += '_' + str(args.expectile_min)
    algos_name += '_' + str(args.expectile_max)
    algos_name += '_' + args.actor_replay_type
    algos_name += '_' + str(args.actor_replay_lambda)
    algos_name += '_' + args.critic_replay_type
    algos_name += '_' + str(args.critic_replay_lambda)
    algos_name += '_' + args.dataset
    algos_name += '_' + args.dataset_nums_str
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
        for i, dataset_num in enumerate(args.dataset_nums):

            if dataset_num not in dataset_num_counter.keys():
                dataset_num_counter[dataset_num] = 0
            else:
                dataset_num_counter[dataset_num] += 1
            if args.dataset_kind == 'antmaze':
                # h5_path = 'dataset/d4rl/' + args.dataset + '/' + dataset_num + '.hdf5'
                h5_path = 'dataset/d4rl/origin/' + args.dataset + '.hdf5'
                print(f"h5_path: {h5_path}")
                dataset = get_antmaze_local(get_dataset(h5_path), epoch_num=dataset_num, epoch_sum=len(args.dataset_nums))
                # eval_envs.append(eval_env)
            elif args.dataset_kind == 'd4rl':
                epoch_sum = 3
                h5_path = 'dataset/d4rl/' + args.dataset + '/' + dataset_num + '.hdf5'
                print(f"h5_path: {h5_path}")
                dataset = get_d4rl_local(get_dataset(h5_path), epoch_num=(dataset_num_counter[dataset_num]) % epoch_sum, epoch_sum=epoch_sum)
            task_datasets.append((dataset_num, dataset))
        if args.dataset_kind == 'antmaze':
            for i in range(args.dataset_sum):
                # 每一段都要至少学过才行。
                assert i in dataset_num_counter.keys()
        replay_dataset = None
        learned_id = []
        learned_datasets = []
        if not args.test:
            pretrain_path_eval = "pretrained_network/" + f"ST_{args.algo}_" + args.dataset + '_d4rl.pt'

        for dataset_num, (dataset_id, dataset) in enumerate(task_datasets):
            if args.clear_network:
                st = ST(**st_dict)
            learned_id.append((dataset_num, dataset_id))

            learned_datasets.append(dataset)
            add_one_learned_datasets = [None] + learned_datasets

            if env is not None:
                # scorers_list = [{'environment': d3rlpy.metrics.evaluate_on_environment(env), 'fune_tuned_environment': single_evaluate_on_environment(env)}]
                scorers_env = {'environment': d3rlpy.metrics.evaluate_on_environment(env)}
                if len(eval_envs) > 0:
                    scorers_part = dict(zip(['environment_part' + str(n) for n in learned_id], [evaluate_on_environment_help(eval_envs[num], mazes_start[args.maze][args.part_times_num][int(id_)]) for num, id_ in learned_id]))
                    scorers_env.update(scorers_part)
                scorers_list = [scorers_env]
            else:
                raise NotImplementedError

            start_time = time.perf_counter()
            print(f'Start Training {dataset_num}')
            if dataset_num <= args.read_policy:
                iterator, replay_iterator, n_epochs = st.make_iterator(dataset, replay_dataset, step_dict['n_steps_per_epoch'], None, True)
                if args.read_policy == 0:
                    pretrain_path = "pretrained_network/" + "ST_" + args.algo_kind + '_0.9_' + args.dataset + '_' + args.dataset_nums[0] + '.pt'
                    if not os.path.exists(pretrain_path):
                        pretrain_path = "pretrained_network/" + "ST_" + args.algo_kind + '_' + args.dataset + '_' + args.dataset_nums[0] + '.pt'
                        assert os.path.exists(pretrain_path)
                else:
                    pretrain_path = args.model_path + algos_name + '_' + str(dataset_num) + '.pt'

                st.build_with_dataset(dataset, dataset_num)
                st._impl.save_clone_data()
                st.load_model(pretrain_path)
                st._impl.save_clone_data()
                # if (args.critic_replay_type not in ['ewc', 'si', 'rwalk'] or args.actor_replay_type not in ['ewc', 'si', 'rwalk']) and args.read_policy != 0:
                #     try:
                #         replay_dataset = torch.load(f=args.model_path + algos_name + '_' + str(dataset_num) + '_datasets.pt')
                #     except BaseException as e:
                #         print(f'Don\' have replay_dataset')
                #         raise e
            elif dataset_num > args.read_policy:
                # train
                print(f'fitting dataset {dataset_num}')
                iterator, replay_iterator, n_epochs = st.make_iterator(dataset, replay_dataset, step_dict['n_steps'], step_dict['n_steps_per_epoch'], None, True)
                st.build_with_dataset(dataset, dataset_num)
                for param_group in st._impl._actor_optim.param_groups:
                    param_group["lr"] = st_dict['actor_learning_rate']
                for param_group in st._impl._critic_optim.param_groups:
                    param_group["lr"] = st_dict['critic_learning_rate']
                if args.algo in ['iql', 'iqln', 'iqln2', 'iqln3', 'iqln4', 'sql', 'sqln']:
                    scheduler = CosineAnnealingLR(st._impl._actor_optim, 1000000)

                    def callback(algo, epoch, total_step):
                        scheduler.step()
                    # st_dict['expectile'] = 1
                else:
                    callback = None
                if args.offline:
                    st.fit(
                        dataset_num,
                        dataset=dataset,
                        iterator=iterator,
                        replay_dataset=replay_dataset,
                        replay_iterator=replay_iterator,
                        eval_episodes_list=add_one_learned_datasets,
                        # n_epochs=args.n_epochs if not args.test else 1,
                        n_epochs=n_epochs,
                        save_interval=args.save_interval,
                        experiment_name=experiment_name + algos_name + '_' + str(dataset_num),
                        scorers_list = scorers_list,
                        callback=callback,
                        test=args.test,
                    )
                else:
                    st.online_fit(
                        env,
                        eval_env,
                        dataset_num,
                    )
            st.after_learn(iterator, experiment_name + algos_name + '_' + str(dataset_num), scorers_list, add_one_learned_datasets)
            print(f'Training task {dataset_num} time: {time.perf_counter() - start_time}')
            # st.save_model(args.model_path + algos_name + '_' + str(dataset_num) + '.pt')
            if args.critic_replay_type in ['bc', 'orl', 'gem', 'agem'] or args.actor_replay_type in ['bc', 'orl', 'gem', 'agem']:
                replay_dataset = st.select_replay(dataset, replay_dataset, dataset_num, args.max_save_num, args.mix_type)
            else:
                replay_dataset = None
            if args.test and dataset_num >= 2:
                break

            # 比较的测试没必要对新的数据集做。
        if online_st_dict['n_steps'] > 0:
            for param_group in st._impl._actor_optim.param_groups:
                param_group["lr"] = st_dict['actor_learning_rate']
            for param_group in st._impl._critic_optim.param_groups:
                param_group["lr"] = st_dict['critic_learning_rate']
            buffer_ = ReplayBuffer(maxlen=online_st_dict['buffer_size'], env=env)
            st.online_fit(env, eval_env, buffer_, n_steps=online_st_dict['n_steps'], n_steps_per_epoch=online_st_dict['n_steps_per_epoch'], experiment_name = experiment_name + algos_name, test=args.test)
    print('finish')

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Experimental evaluation of lifelong PG learning')
    parser.add_argument('--add_name', default='', type=str)
    parser.add_argument('--epoch', default='500', type=int)
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

    parser.add_argument('--critic_replay_type', default='bc', type=str, choices=['orl', 'bc', 'lwf', 'ewc', 'gem', 'agem', 'rwalk', 'si', 'none'])
    parser.add_argument('--critic_replay_lambda', default=100, type=float)
    parser.add_argument('--actor_replay_type', default='orl', type=str, choices=['orl', 'bc', 'lwf', 'lwf_orl', 'ewc', 'gem', 'agem', 'rwalk', 'si', 'none'])
    parser.add_argument('--actor_replay_lambda', default=1, type=float)

    parser.add_argument('--n_critics', default=2, type=int)
    parser.add_argument('--eta', default=1.0, type=int)
    parser.add_argument('--std_time', default=1, type=float)
    parser.add_argument('--std_type', default='none', type=str, choices=['clamp', 'none', 'linear', 'entropy'])
    parser.add_argument('--entropy_time', default=0.2, type=float)
    parser.add_argument('--update_ratio', default=0.3, type=float)

    parser.add_argument('--fine_tuned_step', default=1, type=int)
    parser.add_argument('--clone_actor', action='store_true')
    parser.add_argument('--mix_type', default='q', type=str, choices=['q', 'v', 'random', 'vq_diff', 'all'])

    parser.add_argument('--experience_type', default='random_episode', type=str, choices=['all', 'none', 'single', 'online', 'model_prob', 'model_next', 'model', 'model_this', 'coverage', 'random_transition', 'random_episode', 'max_reward', 'max_match', 'max_supervise', 'max_model', 'max_reward_end', 'max_reward_mean', 'max_match_end', 'max_match_mean', 'max_supervise_end', 'max_supervise_mean', 'max_model_end', 'max_model_mean', 'min_reward', 'min_match', 'min_supervise', 'min_model', 'min_reward_end', 'min_reward_mean', 'min_match_end', 'min_match_mean', 'min_supervise_end', 'min_supervise_mean', 'min_model_end', 'min_model_mean'])
    parser.add_argument('--max_export_step', default=1000, type=int)
    parser.add_argument('--dense', default='dense', type=str)
    parser.add_argument('--sum', default='no_sum', type=str)
    parser.add_argument('--d_threshold', type=float, default=0.1)
    parser.add_argument('--use_cpu', action='store_true')
    parser.add_argument('--gpu', type=str, default='0')
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--read_policy', type=int, default=-1)
    # 作为对照实验，证明er算法不是重新学习了重放缓存而是具备持续学习能力
    parser.add_argument('--clear_network', action='store_true')
    args = parser.parse_args()

    args.algo_kind = args.algo
    if args.algo_kind in ['cql', 'mrcql', 'mgcql']:
        args.algo_kind = 'cql'
    args.dataset_nums_str = args.dataset_nums
    args.dataset_nums = [x for x in args.dataset_nums.split('_')]
    if args.dataset in ['halfcheetah-random-v0', 'hopper-random-v0', 'walker2d-random-v0', 'ant-random-v0']:
        args.dataset_kind = 'd4rl'
        args.dataset_nums = ['itr_' + dataset_num for dataset_num in args.dataset_nums]
    elif args.dataset in ['HalfCheetahBlock-v2', 'Walker2dBlock-v4', 'HopperBlock-v4']:
        args.dataset_kind = 'block'
        args.dataset_nums = ['itr_' + dataset_num for dataset_num in args.dataset_nums]
    elif 'antmaze' in args.dataset:
        args.dataset_kind = 'antmaze'
        args.dataset_sum = int(args.dataset_nums[0])
        args.dataset_nums = [int(dataset_num) for dataset_num in args.dataset_nums[1:]]
        # args.maze = args.dataset.split('-')[1]
        # assert args.maze in ['umaze', 'medium', 'large']
        # args.part_times_num = 0 if len(args.dataset_nums) == 2 else 1
    else:
        raise NotImplementedError

    args.model_path = 'd3rlpy' + '_' + args.dataset
    if not os.path.exists(args.model_path):
        os.makedirs(args.model_path)
    args.model_path += '/model_'
    # if args.experience_type == 'model':
    #     args.experience_type = 'model_next'

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
