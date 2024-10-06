import os
import argparse
import random
import numpy as np
import gym

import torch
from torch.optim.lr_scheduler import CosineAnnealingLR
from d3rlpy.online.buffers import ReplayBuffer
from myd3rlpy.datasets import get_d4rl
from d3rlpy.metrics import evaluate_on_environment

from myd3rlpy.algos.o2o_td3 import O2OTD3
from myd3rlpy.algos.o2o_sac import O2OSAC
from myd3rlpy.algos.o2o_iql import O2OIQL
from myd3rlpy.algos.o2o_cql import O2OCQL
from mygym.envs.online_offline_wrapper import online_offline_wrapper
from config.o2o_config import get_o2o_dict, online_algos, offline_algos


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
def main(args, use_gpu):
    print("Start")
    np.set_printoptions(precision=1, suppress=True)
    dataset0, env = get_d4rl(args.dataset + '-' + args.qualities[0].replace("_", "-") + '-v0')
    _, eval_env = get_d4rl(args.dataset + '-' + args.qualities[0].replace("_", "-") + '-v0')

    # prepare algorithm
    # st_dict, online_st_dict, step_dict = get_st_dict(args, args.dataset_kind, args.algo)

    experiment_name = "ST" + '_'
    algos_name = args.dataset
    algos_name += '_' + str(args.first_n_steps)
    algos_name += '_' + str(args.n_buffer)
    algos_name += '_' + args.algorithms_str
    algos_name += '_' + args.qualities_str
    algos_name += ('_' + "test") if args.test else ""
    if args.add_name != '':
        algos_name += '_' + args.add_name
    experiment_name += algos_name

    # For saving and loading
    load_name = args.dataset
    load_name += '_' + str(args.first_n_steps)
    if args.algorithms[0] in online_algos:
        load_name += '_' + str(args.n_buffer)
    load_name += '_' + args.algorithms[0]
    if args.algorithms[0] in offline_algos:
        load_name += '_' + args.qualities[0]
    if args.add_name != '':
        load_name += '_' + args.add_name

    if not args.eval:
        print(f'Start Training')
        if args.test:
            o2o0_path = "save_algos/" + load_name + '.pt.test'
        else:
            o2o0_path = "save_algos/" + load_name + '.pt'
            #if os.path.exists(o2o0_path):
            #    return 0
        print(f"{o2o0_path}")
        print(f'Start Training Algo 0')
        o2o0_dict = get_o2o_dict(args.algorithms[0], args.qualities[0])
        # Task 0
        o2o0_dict['use_gpu'] = use_gpu
        o2o0_dict['impl_name'] = args.algorithms[0]
        if args.algorithms[0] in ['td3', 'td3_plus_bc']:
            o2o0 = O2OTD3(**o2o0_dict)
        elif args.algorithms[0] == 'sac':
            o2o0 = O2OSAC(**o2o0_dict)
        elif args.algorithms[0] == 'iql':
            o2o0 = O2OIQL(**o2o0_dict)
        elif args.algorithms[0] in ['cql', 'cal']:
            o2o0 = O2OCQL(**o2o0_dict)
        else:
            raise NotImplementedError
        if args.algorithms[0] in online_algos:
            buffer = ReplayBuffer(args.n_buffer, env)
            o2o0.build_with_env(env)
            o2o0.fit_online(
                env,
                eval_env,
                buffer,
                n_steps = args.first_n_steps,
                n_steps_per_epoch = args.n_steps_per_epoch,
                save_steps=args.save_steps,
                save_path=o2o0_path,
                test = args.test,
            )
            torch.save({'buffer': buffer.to_mdp_dataset(), 'algo': o2o0}, o2o0_path)
        elif args.algorithms[0] in offline_algos:
            scorers_env = {'evaluation': evaluate_on_environment(online_offline_wrapper(env))}
            scorers_list = [scorers_env]
            o2o0.build_with_env(online_offline_wrapper(env))
            iterator, _, n_epochs = o2o0.make_iterator(dataset0, None, args.first_n_steps, args.n_steps_per_epoch, None, True)
            fitter_dict = dict()
            if args.algorithms[0] == 'iql':
                scheduler = CosineAnnealingLR(o2o0._impl._actor_optim, 1000000)
                def callback(algo, epoch, total_step):
                    scheduler.step()
                fitter_dict['callback'] = callback
            if args.algorithms[0] in ['ppo', 'bppo']:
                value_iterator, _, n_value_epochs = o2o0.make_iterator(dataset0, None, args.first_n_value_steps, args.n_value_steps_per_epoch, None, True)
                bc_iterator, _, n_bc_epochs = o2o0.make_iterator(dataset0, None, args.first_n_bc_steps, args.n_bc_steps_per_epoch, None, True)
                fitter_dict['value_iterator'] = value_iterator
                fitter_dict['bc_iterator'] = bc_iterator
                fitter_dict['n_value_epochs'] = n_value_epochs
                fitter_dict['n_bc_epochs'] = n_bc_epochs
            save_epochs = []
            for save_step in args.save_steps:
                save_epochs.append(save_step // args.n_steps_per_epoch)
            o2o0.fitter(
                dataset=dataset0,
                iterator=iterator,
                n_epochs=n_epochs,
                n_steps_per_epoch=args.n_steps_per_epoch,
                experiment_name=experiment_name + "_0",
                scorers_list = scorers_list,
                eval_episodes_list = [None],
                save_epochs=save_epochs,
                save_path=o2o0_path,
                test = args.test,
                **fitter_dict,
            )
            torch.save({'buffer': None, 'algo': o2o0}, o2o0_path)
        else:
            raise NotImplementedError
    print('finish')

if __name__ == '__main__':
    print(1)
    parser = argparse.ArgumentParser(description='Experimental evaluation of lifelong PG learning')
    parser.add_argument('--add_name', default='', type=str)
    parser.add_argument('--epoch', default='500', type=int)
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
    parser.add_argument('--weight_temp', default=3.0, type=float)
    parser.add_argument('--expectile', default=0.7, type=float)
    parser.add_argument('--expectile_min', default=0.7, type=float)
    parser.add_argument('--expectile_max', default=0.7, type=float)
    parser.add_argument('--alpha', default=2, type=float)
    parser.add_argument('--eval', action='store_true')
    parser.add_argument('--test', action='store_true')


    parser.add_argument("--n_buffer", default=1000000, type=int)
    # For ppo
    parser.add_argument("--n_value_steps_per_epoch", default=1000, type=int)
    parser.add_argument("--n_bc_steps_per_epoch", default=1000, type=int)
    parser.add_argument("--online_maxlen", default=1000000, type=int)

    parser.add_argument("--save_interval", default=1, type=int)
    parser.add_argument("--n_action_samples", default=10, type=int)
    parser.add_argument('--top_euclid', default=64, type=int)

    parser.add_argument('--critic_replay_type', default='bc', type=str, choices=['orl', 'bc', 'generate', 'generate_orl', 'lwf', 'ewc', 'gem', 'agem', 'rwalk', 'si', 'none'])
    parser.add_argument('--critic_replay_lambda', default=100, type=float)
    parser.add_argument('--actor_replay_type', default='orl', type=str, choices=['orl', 'bc', 'generate', 'generate_orl', 'lwf', 'lwf_orl', 'ewc', 'gem', 'agem', 'rwalk', 'si', 'none'])
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

    parser.add_argument('--algorithms', type=str, required=True)
    parser.add_argument('--qualities', type=str, default="medium")
    parser.add_argument('--buffer_mix_type', type=str, choices=['all', 'policy', 'value'], default='all')
    parser.add_argument("--dataset", default='halfcheetah', type=str)

    parser.add_argument('--experience_type', default='random_episode', type=str, choices=['all', 'none', 'single', 'online', 'generate', 'model_prob', 'model_next', 'model', 'model_this', 'coverage', 'random_transition', 'random_episode', 'max_reward', 'max_match', 'max_supervise', 'max_model', 'max_reward_end', 'max_reward_mean', 'max_match_end', 'max_match_mean', 'max_supervise_end', 'max_supervise_mean', 'max_model_end', 'max_model_mean', 'min_reward', 'min_match', 'min_supervise', 'min_model', 'min_reward_end', 'min_reward_mean', 'min_match_end', 'min_match_mean', 'min_supervise_end', 'min_supervise_mean', 'min_model_end', 'min_model_mean'])
    parser.add_argument('--max_export_step', default=1000, type=int)
    parser.add_argument('--dense', default='dense', type=str)
    parser.add_argument('--sum', default='no_sum', type=str)
    parser.add_argument('--d_threshold', type=float, default=0.1)
    parser.add_argument('--gpu', type=int, default=0)
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--read_policy', type=int, default=-1)


    args = parser.parse_args()

    if args.algorithms not in ['ppo', 'bppo']:
        args.first_n_steps = 1000000
        args.n_steps_per_epoch = 1000
    else:
        args.first_n_steps = 100
        args.n_steps_per_epoch = 10
        args.first_n_value_steps = 2000000
        args.first_n_bc_steps = 500000
        args.first_n_value_steps_per_epoch = 1000
        args.first_n_value_steps_per_epoch = 1000

    args.algorithms_str = args.algorithms
    args.algorithms = args.algorithms.split('-')
    assert len(args.algorithms) == 1
    for algo in args.algorithms:
        assert algo in offline_algos + online_algos
    if args.qualities is not None:
        args.qualities_str = args.qualities
        args.qualities = args.qualities.split('-')
        assert len(args.qualities) == 1
        assert args.qualities[0] in ['medium', 'expert', 'medium_replay', 'medium_expert', 'random']

    args.save_steps = [300000, 100000]

    args.model_path = 'd3rlpy' + '_' + args.dataset
    if not os.path.exists(args.model_path):
        os.makedirs(args.model_path)
    args.model_path += '/model_'
    # if args.experience_type == 'model':
    #     args.experience_type = 'model_next'

    global DATASET_PATH
    DATASET_PATH = './.d4rl/datasets/'
    if args.gpu < 0:
        use_gpu = False
    else:
        os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu)
        use_gpu = 0
    args.clone_critic = True
    seeds = [12345, 1234, 123, 12, 1]
    random.seed(seeds[args.seed])
    np.random.seed(seeds[args.seed])
    torch.manual_seed(seeds[args.seed])
    torch.cuda.manual_seed(seeds[args.seed])
    print(f"use_gpu: {use_gpu}")
    main(args, use_gpu)
