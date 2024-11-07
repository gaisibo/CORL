import os
import argparse
import random
from collections import namedtuple
import time
from functools import partial
import numpy as np
import gym
from dataset.continual_world import get_split_cl_env
from dataset.continual_atari import get_atari_envs
from mygym.envs.halfcheetah_block import HalfCheetahBlockEnv

import torch
from torch.optim.lr_scheduler import CosineAnnealingLR

import d3rlpy
from d3rlpy.online.buffers import ReplayBuffer

from myd3rlpy.metrics.scorer import evaluate_on_environment, critic_actor_diff, old_critic_actor_diff
from config.o2o_config import get_o2o_dict, online_algos, offline_algos
from dataset.load_d4rl import get_d4rl_local, get_antmaze_local, get_dataset
from rlkit.torch import pytorch_util as ptu
from config.single_config import get_st_dict


TASK_SEQS = {
    "CW10": [
        "hammer-v2",
        "push-wall-v2",
        "faucet-close-v2",
        "push-back-v2",
        "stick-pull-v2",
        "handle-press-side-v2",
        "push-v2",
        "shelf-place-v2",
        "window-close-v2",
        "peg-unplug-side-v2",
    ],
    "ASI6": ("ALE/SpaceInvaders-v5", list(range(6))),
    "ASI12": ("ALE/SpaceInvaders-v5", list(range(6)) + list(range(6))),
}

TASK_SEQS["CW20"] = TASK_SEQS["CW10"] + TASK_SEQS["CW10"]
continual_world_datasets = ["CW10", "CW20"]
atari_datasets = ["ASI6", "ASI12"]

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
    if args.dataset_kind == 'continual_world':
        tasks = TASK_SEQS[args.dataset]
        envs = get_split_cl_env(tasks, args.randomization)
        eval_envs = get_split_cl_env(tasks, args.randomization)
        if args.dataset == 'CW20':
            env_ids = list(range(10)) + list(range(10))
        else:
            raise NotImplementedError
    elif args.dataset_kind == "atari":
        task, dataset_num = TASK_SEQS[args.dataset]
        envs = get_atari_envs(task, task_nums=dataset_num, randomization=args.randomization)
        eval_envs = get_atari_envs(task, task_nums=dataset_num, randomization=args.randomization)
        if args.dataset == 'SI20':
            env_ids = list(range(10)) + list(range(10))
        else:
            raise NotImplementedError
    else:
        raise NotImplementedError

    # prepare algorithm
    if args.algo in ['td3']:
        from myd3rlpy.algos.o2o_td3 import O2OTD3 as O2O
    elif args.algo in ['sac']:
        from myd3rlpy.algos.o2o_sac import O2OSAC as O2O
    else:
        raise NotImplementedError
    algo_dict = get_o2o_dict(args.algo, None)
    algo_dict["use_gpu"] = use_gpu
    algo_dict['impl_name'] = args.algo
    algo_dict["actor_replay_type"] = args.actor_replay_type
    algo_dict["critic_replay_type"] = args.critic_replay_type
    algo = O2O(**algo_dict)

    experiment_name = "Online" + '_'
    algos_name = args.algo
    #algos_name += "_" + args.continual_type
    algos_name += '_' + args.actor_replay_type
    #algos_name += '_' + str(args.actor_replay_lambda)
    algos_name += '_' + args.critic_replay_type
    #algos_name += '_' + str(args.critic_replay_lambda)
    algos_name += '_' + args.dataset
    algos_name += '_' + str(args.max_save_num)
    if args.add_name != '':
        algos_name += '_' + args.add_name

    if not args.eval:

        learned_env_id = []
        old_algos = []
        buffer = ReplayBuffer(args.n_buffer, envs[0])
        old_buffer = None
        for task_id, (env, eval_env, env_id) in enumerate(zip(envs, eval_envs, env_ids)):
            if args.clear_network:
                algo = O2O(**algo_dict)
            if task_id > 0:
                if args.actor_replay_type == "bc" or args.critic_replay_type == "bc":
                    old_buffer = buffer
                    buffer = ReplayBuffer(args.n_buffer, env)
                elif args.actor_replay_type != "prefect_memory" and args.critic_replay_type != "prefect_memory":
                    old_buffer = None
                    buffer = ReplayBuffer(args.n_buffer, env)
            if env_id not in learned_env_id:
                learned_env_id.append(env_id)
            save_path = experiment_name + algos_name + '_' + args.dataset + "_" + str(task_id)
            if task_id != 0:
                old_algo = O2O(**algo_dict)
                old_algo.build_with_env(env)
                # origin continual world do not copy optim as default
                old_algo.copy_from_past(args.algo, algo._impl, False)
                old_algos.append(old_algo)

            if env is not None:
                # scorers_list = [{'environment': d3rlpy.metrics.evaluate_on_environment(env), 'fune_tuned_environment': single_evaluate_on_environment(env)}]
                scorers_env = dict()
                #for _, old_id in learned_env_id:
                scorers_env["environment_" + str(env_id)] = evaluate_on_environment(envs[env_id])
                scorers_list = [scorers_env]
                eval_episodes_list = [None]
                for old_id in learned_env_id[:-1]:
                    scorers_env = dict()
                    scorers_env["environment_" + str(old_id) + '-critic_diff_' + str(old_id) + "-actor_diff_" + str(old_id)] = critic_actor_diff(envs[old_id])
                    scorers_list.append(scorers_env)
                    eval_episodes_list.append(old_algos[old_id])
                for old_id in learned_env_id[:-1]:
                    scorers_env = dict()
                    scorers_env['old_critic_diff_' + str(old_id) + "-old_actor_diff_" + str(old_id)] = old_critic_actor_diff(envs[old_id])
                    scorers_list.append(scorers_env)
                    eval_episodes_list.append(old_algos[old_id])
            else:
                raise NotImplementedError

            print(f'Start Training {task_id}')
            algo.before_learn(buffer, args.actor_replay_type, args.critic_replay_type, args.test)
            if task_id <= args.read_policy:
                if args.read_policy == 0:
                    pretrain_path = "pretrained_network/" + "ST_" + args.algo_kind + '_' + args.dataset + '.pt'
                    assert os.path.exists(pretrain_path)
                else:
                    pretrain_path = args.model_path + algos_name + '_' + args.dataset + " " + str(task_id) + '.pt'

                algo.build_with_env(env)
                algo.change_task(task_id)
                algo._impl.save_clone_data()
                algo.load_model(pretrain_path)
                algo._impl.save_clone_data()
                # if (args.critic_replay_type not in ['ewc', 'si', 'rwalk'] or args.actor_replay_type not in ['ewc', 'si', 'rwalk']) and args.read_policy != 0:
                #     try:
                #         replay_dataset = torch.load(f=args.model_path + algos_name + '_' + str(dataset_num) + '_datasets.pt')
                #     except BaseException as e:
                #         print(f'Don\' have replay_dataset')
                #         raise e
            elif task_id > args.read_policy:
                # train
                print(f'learning {task_id}')
                algo.build_with_env(env)
                algo.change_task(task_id)
                #for param_group in st._impl._actor_optim.param_groups:
                #    param_group["lr"] = st_dict['actor_learning_rate']
                #for param_group in st._impl._critic_optim.param_groups:
                #    param_group["lr"] = st_dict['critic_learning_rate']
                save_path = experiment_name + algos_name + '_' + args.dataset + "_" + str(task_id)
                algo.fit_online(
                    env,
                    eval_env,
                    buffer,
                    old_buffer = old_buffer,
                    #buffer_mix_type = args.buffer_mix_type,
                    n_steps = args.n_steps,
                    n_steps_per_epoch = args.n_steps_per_epoch,
                    save_steps=args.save_steps,
                    save_path=save_path,
                    random_step=100000 if args.non_explore else 0,
                    test = args.test,
                    start_epoch = 0,
                    experiment_name=experiment_name + algos_name + "_" + args.dataset + "_" + str(task_id),
                    scorers_list = scorers_list,
                    eval_episodes_list = eval_episodes_list,
                )
            algo.after_learn(buffer, args.buffer_mix_type, args.test)
    print('finish')

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Experimental evaluation of lifelong PG learning')
    parser.add_argument('--add_name', default='', type=str)
    parser.add_argument('--epoch', default='500', type=int)
    parser.add_argument("--dataset", default='CW20', type=str)
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
    parser.add_argument('--algo', default='sac', type=str, choices=['sac', 'td3'])
    parser.add_argument('--weight_temp', default=3.0, type=float)
    parser.add_argument('--expectile', default=0.7, type=float)
    parser.add_argument('--expectile_min', default=0.7, type=float)
    parser.add_argument('--expectile_max', default=0.7, type=float)
    parser.add_argument('--alpha', default=2, type=float)
    parser.add_argument('--eval', action='store_true')
    parser.add_argument('--test', action='store_true')


    parser.add_argument("--n_steps", default=1000000, type=int)
    parser.add_argument("--n_steps_per_epoch", default=1000, type=int)
    parser.add_argument("--online_n_steps", default=100000, type=int)
    parser.add_argument("--online_maxlen", default=1000000, type=int)

    parser.add_argument("--save_interval", default=10, type=int)
    parser.add_argument("--n_action_samples", default=10, type=int)
    parser.add_argument('--top_euclid', default=64, type=int)

    parser.add_argument('--critic_replay_type', default='bc', type=str, choices=['orl', 'bc', 'er', 'prefect_memory', 'lwf', 'ewc', 'gem', 'agem', 'rwalk', 'si', 'none'])
    parser.add_argument('--critic_replay_lambda', default=100, type=float)
    parser.add_argument('--actor_replay_type', default='orl', type=str, choices=['orl', 'bc', 'er', 'prefect_memory', 'lwf', 'lwf_orl', 'ewc', 'gem', 'agem', 'rwalk', 'si', 'none'])
    parser.add_argument('--actor_replay_lambda', default=1, type=float)

    parser.add_argument("--continual_type", default="ewc", type=str)

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
    # 仅continual world使用
    parser.add_argument('--n_buffer', type=int, default=20000)
    parser.add_argument('--randomization', type=str, default="random_init_all")
    parser.add_argument('--buffer_mix_type', type=str, default="all")
    parser.add_argument('--non_explore', action='store_true')
    args = parser.parse_args()

    args.save_steps = [args.n_steps, 300000, 100000]
    assert args.algo in online_algos
    args.algo_kind = args.algo
    if args.algo_kind in ['cql', 'mrcql', 'mgcql']:
        args.algo_kind = 'cql'
    if args.dataset in continual_world_datasets:
        args.dataset_kind = "continual_world"
    elif args.dataset in atari_datasets:
        args.dataset_kind = "atari"

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
        use_gpu = False
    else:
        device = torch.device('cuda')
        use_gpu = 0
    ptu.set_gpu_mode(True)
    args.clone_critic = True
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu
    seeds = [12345, 1234, 123, 12, 1]
    random.seed(seeds[args.seed])
    np.random.seed(seeds[args.seed])
    torch.manual_seed(seeds[args.seed])
    torch.cuda.manual_seed(seeds[args.seed])
    main(args, device)
