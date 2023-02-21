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
from torch.optim.lr_scheduler import CosineAnnealingLR

import d3rlpy
from d3rlpy.ope import FQE
from d3rlpy.dataset import MDPDataset
from d3rlpy.torch_utility import get_state_dict, set_state_dict
from d3rlpy.online.buffers import ReplayBuffer
from d3rlpy.online.iterators import train_single_env
from d3rlpy.models.optimizers import AdamFactory
# from myd3rlpy.datasets import get_d4rl
from utils.k_means import kmeans
from myd3rlpy.metrics.scorer import merge_evaluate_on_environment
from dataset.load_d4rl import get_d4rl_local, get_dataset
from rlkit.torch import pytorch_util as ptu


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
    offline_name = ['hopper-random-v0', 'walker2d-random-v0', 'halfcheetah-random-v0']
    if args.dataset in offline_name:
        task_datasets = []
        _, env = d3rlpy.datasets.get_dataset(args.dataset)
        _, eval_env = d3rlpy.datasets.get_dataset(args.dataset)

    else:
        raise NotImplementedError

    # prepare algorithm
    if args.algo in ['td3_plus_bc', 'td3']:
        from myd3rlpy.algos.st_td3_plus_bc import ST
    elif args.algo_kind == 'cql':
        from myd3rlpy.algos.st_cql import ST
    elif args.algo == 'iql':
        from myd3rlpy.algos.st_iql import ST
    else:
        raise NotImplementedError
    encoder = d3rlpy.models.encoders.VectorEncoderFactory([256, 256, 256])
    conservative_weight = 5.0
    st_dict = dict()
    st_dict['impl_name'] = args.algo
    st_dict['batch_size'] = args.batch_size
    st_dict['critic_replay_type'] = args.critic_replay_type
    st_dict['critic_replay_lambda'] = args.critic_replay_lambda
    st_dict['actor_replay_type'] = args.actor_replay_type
    st_dict['actor_replay_lambda'] = args.actor_replay_lambda
    st_dict['vae_replay_lambda'] = args.vae_replay_lambda
    st_dict['vae_replay_type'] = args.vae_replay_type
    st_dict['clone_actor'] = args.clone_actor
    st_dict['clone_critic'] = args.clone_critic
    st_dict['fine_tuned_step'] = args.fine_tuned_step
    st_dict['experience_type'] = args.experience_type
    st_dict['actor_learning_rate'] = 1e-4
    st_dict['critic_learning_rate'] = 3e-4
    st_dict['vae_learning_rate'] = 1e-3
    st_dict['actor_encoder_factory'] = encoder
    st_dict['critic_encoder_factory'] = encoder
    st_dict['n_action_samples'] = args.n_action_samples
    st_dict['use_gpu'] = True
    if args.critic_replay_type in ['generate', 'generate_orl'] or args.actor_replay_type == 'generate':
        st_dict['use_vae'] = args.use_vae
    if args.algo_kind == 'cql':
        st_dict['temp_learning_rate'] = 1e-4
        st_dict['alpha_learning_rate'] = 0.0
        st_dict['conservative_weight'] = conservative_weight
        st_dict['conservative_threshold'] = 0.1
    elif args.algo == 'iql':
        st_dict['actor_encoder_factory'] = "default"
        st_dict['critic_encoder_factory'] = "default"
        st_dict['value_encoder_factory'] = "default"
        st_dict['actor_learning_rate'] = 3e-4
        st_dict['weight_temp'] = 3.0
        st_dict['max_weight'] = 100.0
        st_dict['expectile'] = 0.7
        reward_scaler = d3rlpy.preprocessing.ReturnBasedRewardScaler(
            multiplier=1000.0)
        st_dict['reward_scaler'] = reward_scaler
    st1 = ST(**st_dict)
    st2 = ST(**st_dict)
    sts = [st1, st2]

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
        for dataset_id, (dataset_num, st) in enumerate(zip(args.dataset_nums, sts)):
            h5_path = 'dataset/d4rl/' + args.dataset + '/itr_' + str(dataset_num) + '.hdf5'
            dataset = get_d4rl_local(get_dataset(h5_path))
            pretrain_path = "pretrained_network/" + "ST_" + args.algo_kind + '_' + args.dataset + '_' + str(dataset_num) + '.pt'
            st.build_with_dataset(dataset, dataset_id)
            st.load_model(pretrain_path)
        scorer = merge_evaluate_on_environment(env)
        scorer(sts[0], sts[1])
        print(f"merge score: {scorer(sts[0], sts[1])}")
    print('finish')

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Experimental evaluation of lifelong PG learning')
    parser.add_argument('--add_name', default='', type=str)
    parser.add_argument("--dataset", default='halfcheetah-random-v0', type=str)
    parser.add_argument('--dataset_nums', default="0", type=str)
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
    parser.add_argument('--algo', default='iql', type=str, choices=['combo', 'td3_plus_bc', 'cql', 'mgcql', 'mrcql', 'iql'])
    parser.add_argument('--eval', action='store_true')
    parser.add_argument('--test', action='store_true')
    parser.add_argument("--n_steps", default=500000, type=int)
    parser.add_argument("--n_steps_per_epoch", default=5000, type=int)

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

    parser.add_argument('--experience_type', default='siamese', type=str, choices=['all', 'none', 'single', 'online', 'generate', 'model_prob', 'model_next', 'model', 'model_this', 'coverage', 'random_transition', 'random_episode', 'max_reward', 'max_match', 'max_supervise', 'max_model', 'max_reward_end', 'max_reward_mean', 'max_match_end', 'max_match_mean', 'max_supervise_end', 'max_supervise_mean', 'max_model_end', 'max_model_mean', 'min_reward', 'min_match', 'min_supervise', 'min_model', 'min_reward_end', 'min_reward_mean', 'min_match_end', 'min_match_mean', 'min_supervise_end', 'min_supervise_mean', 'min_model_end', 'min_model_mean'])
    parser.add_argument('--max_export_step', default=1000, type=int)
    parser.add_argument('--dense', default='dense', type=str)
    parser.add_argument('--sum', default='no_sum', type=str)
    parser.add_argument('--d_threshold', type=float, default=0.1)
    parser.add_argument('--use_cpu', action='store_true')
    parser.add_argument('--gpu', type=str, default='0')
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--read_policy', type=int, default=-1)
    args = parser.parse_args()

    args.algo_kind = args.algo
    if args.algo_kind in ['cql', 'mrcql', 'mgcql']:
        args.algo_kind = 'cql'

    args.dataset_nums_str = args.dataset_nums
    args.dataset_nums = [int(x) for x in args.dataset_nums.split('-')]
    args.model_path = 'd3rlpy' + '_' + args.dataset
    if not os.path.exists(args.model_path):
        os.makedirs(args.model_path)
    args.model_path += '/model_'
    # if args.experience_type == 'model':
    #     args.experience_type = 'model_next'

    if args.critic_replay_type in ['generate', 'generate_orl'] or args.actor_replay_type == 'generate':
        args.use_vae = True

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
        args.clone_actor = False
        args.clone_critic = True
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu
    seeds = [12345, 1234, 123, 12, 1]
    random.seed(seeds[args.seed])
    np.random.seed(seeds[args.seed])
    torch.manual_seed(seeds[args.seed])
    torch.cuda.manual_seed(seeds[args.seed])
    main(args, device)
