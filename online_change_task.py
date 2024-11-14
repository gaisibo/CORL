import os
import argparse
import random
import numpy as np

import torch
from d4rl.locomotion import ant
from d3rl.locomotion.wrappers import NormalizedBoxEnv

from d3rlpy.online.buffers import ReplayBuffer
from myd3rlpy.datasets import get_d4rl
from d3rlpy.metrics import evaluate_on_environment

from myd3rlpy.algos.o2o_td3 import O2OTD3
from myd3rlpy.algos.o2o_sac import O2OSAC
from myd3rlpy.algos.o2o_iql import O2OIQL
from myd3rlpy.algos.o2o_cql import O2OCQL
from myd3rlpy.algos.o2o_test import O2OTEST
from mygym.envs.online_offline_wrapper import online_offline_wrapper
from config.o2o_config import get_o2o_dict, online_algos, offline_algos
from dataset.expand_maze import maze_maps

from myd3rlpy.dataset import MDPDataset
from d3rlpy.dataset import MDPDataset as OldMDPDataset


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
    if args.dataset_kind == "d4rl":
        #print(args.dataset + '-' + args.qualities[0].replace("_", "-") + '-v0')
        dataset0, env = get_d4rl(args.dataset + '-' + args.qualities[0].replace("_", "-") + '-v0')
        _, eval_env = get_d4rl(args.dataset + '-' + args.qualities[1].replace("_", "-") + '-v0')
    elif args.dataset_kind == "antmaze":
        assert "expand" not in args.qualities[0]
        #print(args.dataset + '-' + args.qualities[0].replace("_", "-") + '-v0')
        if "expand" not in args.qualities[0]:
            dataset0, env = get_d4rl(args.dataset + '-' + args.qualities[0].replace("_", "-") + '-v0')
            _, eval_env = get_d4rl(args.dataset + '-' + args.qualities[1].replace("_", "-") + '-v0')
        else:
            assert args.algorithms[1] in online_algos
            # qualities 应该为类似expand_medium_play这样的。
            maze_map = args.qualities[1].split("_")
            # maze_map 就像是expand-medium这样。
            maze_map = maze_map[0] + "-" + maze_map[1]
            maze_map = maze_maps[maze_map]
            env = NormalizedBoxEnv(ant.AntMazeEnv(maze_map=maze_map, maze_size_scaling=4.0, non_zero_reset=False))
            eval_env = NormalizedBoxEnv(ant.AntMazeEnv(maze_map=maze_map, maze_size_scaling=4.0, non_zero_reset=False))
    else:
        raise NotImplementedError

    # prepare algorithm
    # st_dict, online_st_dict, step_dict = get_st_dict(args, args.dataset_kind, args.algo)

    experiment_name = "ST" + '_'
    algos_name = args.dataset
    algos_name += '_' + str(args.first_n_steps)
    algos_name += '_' + str(args.second_n_steps)
    algos_name += '_' + str(args.n_buffer)
    algos_name += '_' + args.algorithms_str
    algos_name += '_' + str(args.n_critics)
    algos_name += '_' + args.qualities_str
    algos_name += '_' + args.continual_type
    algos_name += ('_' + "copy_optim") if args.copy_optim else ""
    algos_name += ('_' + "test") if args.test else ""
    if args.add_name != '':
        algos_name += '_' + args.add_name
    experiment_name += algos_name

    # For saving and loading
    load_name = args.dataset
    load_name += '_' + str(args.first_n_steps)
    if args.algorithms[0] not in offline_algos:
        load_name += '_' + str(args.n_buffer)
    load_name += '_' + args.algorithms[0]
    load_name += '_' + str(args.n_critics)
    if args.algorithms[0] in offline_algos:
        load_name += '_' + args.qualities[0]
    if args.add_name != '':
        load_name += '_' + args.add_name

    if not args.eval:
        print(f'Start Training')
        #if args.test:
        #    o2o0_path = "save_algos/" + load_name + '.pt.test'
        #else:
        o2o0_path = "save_algos/" + load_name + '.pt'
        assert os.path.exists(o2o0_path)
        print(f'Start Loading Algo 0')
        loaded_data = torch.load(o2o0_path, map_location="cuda:" + str(use_gpu))
        o2o0 = loaded_data['algo']

        o2o1_path = "save_algos/" + algos_name + '.pt'
        # Task 1
        print(f'Start Training Algo 1')
        o2o1_dict = get_o2o_dict(args.algorithms[1], args.qualities[1])
        # Each algo a half.
        o2o1_dict['use_gpu'] = use_gpu
        o2o1_dict['impl_name'] = args.algorithms[1]
        o2o1_dict["critic_replay_type"] = args.critic_replay_type
        o2o1_dict["actor_replay_type"] = args.actor_replay_type
        if args.algorithms[1] in ['td3', 'td3_plus_bc']:
            o2o1 = O2OTD3(**o2o1_dict)
        elif args.algorithms[1] == 'sac':
            o2o1 = O2OSAC(**o2o1_dict)
        elif args.algorithms[1] in ['iql', 'iqln', 'iql_online', 'iqle_online', 'iqln_online', 'iqlne_online']:
            o2o1 = O2OIQL(**o2o1_dict)
        elif args.algorithms[1] in ['test_online']:
            o2o1 = O2OTEST(**o2o1_dict)
        elif args.algorithms[1] in ['cql', 'cal']:
            o2o1 = O2OCQL(**o2o1_dict)
        else:
            raise NotImplementedError
        o2o1.build_with_env(env)
        o2o1.copy_from_past(args.algorithms[0], o2o0._impl, args.copy_optim)
        if args.algorithms[1] in online_algos:
            if args.algorithms[0] in online_algos:
                loaded_mdp = loaded_data['buffer']
                if isinstance(loaded_mdp, MDPDataset):
                    loaded_mdp = OldMDPDataset(loaded_mdp.observations, loaded_mdp.actions, loaded_mdp.rewards, loaded_mdp.terminals, loaded_mdp.episode_terminals)
                loaded_buffer = ReplayBuffer(args.n_buffer, env)
                for episode in loaded_mdp.episodes:
                    loaded_buffer.append_episode(episode)
            elif args.algorithms[0] in offline_algos:
                if isinstance(dataset0, MDPDataset):
                    loaded_mdp = OldMDPDataset(dataset0.observations, dataset0.actions, dataset0.rewards, dataset0.terminals, dataset0.episode_terminals)
                if args.continual_type in ['copy'] or (args.buffer_replay_type == 'same' and args.continual_type in ['mix', 'ewc']):
                    loaded_buffer = ReplayBuffer(args.n_buffer, env)
                elif args.buffer_replay_type == 'all' and args.continual_type in ['mix', 'ewc']:
                    loaded_buffer = ReplayBuffer(dataset0.observations.shape[0], env)
                if args.continual_type != 'none':
                    for episode in loaded_mdp.episodes:
                        loaded_buffer.append_episode(episode)
            else:
                raise NotImplementedError
            ## For making scalers
            #o2o1.make_transitions(loaded_mdp)
            if args.continual_type in ['copy']:
                buffer = loaded_buffer
                old_buffer = None
            elif args.continual_type in ['mix', 'ewc']:
                buffer = ReplayBuffer(args.n_buffer, env)
                old_buffer = loaded_buffer
            elif args.continual_type == 'none':
                buffer = ReplayBuffer(args.n_buffer, env)
                old_buffer = None
            else:
                raise NotImplementedError
            #if args.algorithms[1] == 'ppo':
            #    n_steps = 1000
            #    n_steps_per_epoch = 1
            #else:
            n_steps = args.second_n_steps
            n_steps_per_epoch = args.n_steps_per_epoch
            scorers_env = {'evaluation': evaluate_on_environment(online_offline_wrapper(env))}
            scorers_list = [scorers_env]
            if old_buffer is not None:
                o2o1.after_learn(old_buffer, args.test)
            o2o1.fit_online(
                env,
                eval_env,
                buffer,
                old_buffer = old_buffer,
                n_steps = args.second_n_steps,
                n_steps_per_epoch = args.n_steps_per_epoch,
                save_steps=args.save_steps,
                save_path=o2o1_path,
                random_step=0,
                test = args.test,
                start_epoch = args.first_n_steps // args.n_steps_per_epoch + 1,
                experiment_name=experiment_name + "_1",

                scorers_list = scorers_list,
                eval_episodes_list = [None],
            )
        #elif args.algorithms[1] in offline_algos:
        #    if args.algorithms[0] in online_algos:
        #        loaded_mdp = loaded_data['buffer']
        #        if isinstance(loaded_mdp, MDPDataset):
        #            loaded_mdp = OldMDPDataset(loaded_mdp.observations, loaded_mdp.actions, loaded_mdp.rewards, loaded_mdp.terminals, loaded_mdp.episode_terminals)
        #    else:
        #        raise NotImplementedError
        #    if args.continual_type == 'none':
        #        old_dataset = None
        #    elif args.continual_type == 'copy':
        #        dataset1 = loaded_mdp
        #        if isinstance(dataset1, OldMDPDataset):
        #            dataset1 = MDPDataset(dataset1.observations, dataset1.actions, dataset1.rewards, dataset1.terminals, dataset1.episode_terminals)
        #        old_dataset = None
        #    elif args.continual_type in ['mix_all', 'mix_same']:
        #        old_dataset = loaded_mdp
        #    else:
        #        raise NotImplementedError
        #    scorers_env = {'evaluation': evaluate_on_environment(online_offline_wrapper(env))}
        #    scorers_list = [scorers_env]
        #    o2o1.build_with_env(online_offline_wrapper(env))
        #    iterator, _, n_epochs = o2o1.make_iterator(dataset1, None, args.first_n_steps, args.n_steps_per_epoch, None, True)
        #    if old_dataset is not None:
        #        old_iterator, _, n_epochs = o2o1.make_iterator(old_dataset, None, args.first_n_steps, args.n_steps_per_epoch, None, True)
        #    else:
        #        old_iterator = None
        #    fitter_dict = dict()
        #    if args.algorithms[0] == 'iql':
        #        scheduler = CosineAnnealingLR(o2o0._impl._actor_optim, 1000000)
        #        def callback(algo, epoch, total_step):
        #            scheduler.step()
        #        fitter_dict['callback'] = callback
        #    if args.algorithms[0] == 'ppo':
        #        value_iterator, _, n_value_epochs = o2o0.make_iterator(loaded_mdp, None, args.first_n_value_steps, args.n_value_steps_per_epoch, None, True)
        #        bc_iterator, _, n_bc_epochs = o2o0.make_iterator(loaded_mdp, None, args.first_n_bc_steps, args.n_bc_steps_per_epoch, None, True)
        #        fitter_dict['value_iterator'] = value_iterator
        #        fitter_dict['bc_iterator'] = bc_iterator
        #        fitter_dict['n_value_epochs'] = n_value_epochs
        #        fitter_dict['n_bc_epochs'] = n_bc_epochs
        #    save_epochs = []
        #    for save_step in args.save_steps:
        #        save_epochs.append(save_step // args.n_steps_per_epoch)
        #    o2o1.fitter(
        #        dataset1,
        #        iterator,
        #        continual_type = args.continual_type,
        #        old_iterator = old_iterator,
        #        buffer_replay_type = args.buffer_replay_type,
        #        n_epochs=n_epochs,
        #        experiment_name=experiment_name + "_1",
        #        scorers_list = scorers_list,
        #        eval_episodes_list = [None],
        #        save_epochs=save_epochs,
        #        save_path=o2o1_path,
        #        callback=callback,
        #        test = args.test,
        #        **fitter_dict,
        #    )
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
    parser.add_argument("--first_n_steps", default=1000000, type=int)
    parser.add_argument("--second_n_steps", default=1000000, type=int)
    parser.add_argument("--n_steps_per_epoch", default=1000, type=int)

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

    parser.add_argument('--copy_optim', action='store_true')
    parser.add_argument('--algorithms', type=str, required=True)
    parser.add_argument('--qualities', type=str, default="medium-medium")
    parser.add_argument('--continual_type', type=str, choices=['none', 'copy', 'mix', 'ewc'], default="diff")
    parser.add_argument('--buffer_replay_type', type=str, choices=['all', 'same'], default='all')
    parser.add_argument("--dataset", default='halfcheetah', type=str)
    parser.add_argument('--explore', action='store_true')

    parser.add_argument('--experience_type', default='random_episode', type=str, choices=['all', 'none', 'single', 'online', 'generate', 'model_prob', 'model_next', 'model', 'model_this', 'coverage', 'random_transition', 'random_episode', 'max_reward', 'max_match', 'max_supervise', 'max_model', 'max_reward_end', 'max_reward_mean', 'max_match_end', 'max_match_mean', 'max_supervise_end', 'max_supervise_mean', 'max_model_end', 'max_model_mean', 'min_reward', 'min_match', 'min_supervise', 'min_model', 'min_reward_end', 'min_reward_mean', 'min_match_end', 'min_match_mean', 'min_supervise_end', 'min_supervise_mean', 'min_model_end', 'min_model_mean'])
    parser.add_argument('--max_export_step', default=1000, type=int)
    parser.add_argument('--dense', default='dense', type=str)
    parser.add_argument('--sum', default='no_sum', type=str)
    parser.add_argument('--d_threshold', type=float, default=0.1)
    parser.add_argument('--gpu', type=int, default=0)
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--read_policy', type=int, default=-1)

    args = parser.parse_args()

    if args.dataset in ['halfcheetah', 'hopper', 'walker2d', 'ant']:
        args.dataset_kind = 'd4rl'
    elif 'antmaze' in args.dataset:
        args.dataset_kind = 'antmaze'
    else:
        raise NotImplementedError
    args.algorithms_str = args.algorithms
    args.algorithms = args.algorithms.split('-')
    assert len(args.algorithms) == 2
    print(f"args.algorithms: {args.algorithms}")
    for algo in args.algorithms:
        assert algo in offline_algos + online_algos
    if args.qualities is not None:
        args.qualities_str = args.qualities
        args.qualities = args.qualities.split('-')
        assert len(args.qualities) == 2
        for quality in args.qualities:
            assert quality in ['medium', 'expert', 'medium_replay', 'medium_expert', 'random']

    #if args.algorithms[1] not in ['ppo', 'bppo']:
    #    args.second_n_steps = 1000000
    #    args.n_steps_per_epoch = 1000
    args.save_steps = [args.second_n_steps, 300000, 100000]
    #else:
    #args.second_n_steps = 100
    #args.n_steps_per_epoch = 10
    #args.second_n_value_steps = 2000000
    #args.second_n_bc_steps = 500000
    #args.second_n_value_steps_per_epoch = 1000
    #args.second_n_value_steps_per_epoch = 1000

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
    main(args, use_gpu)
