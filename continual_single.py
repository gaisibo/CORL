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
from d3rlpy.dataset import MDPDataset
from d3rlpy.torch_utility import get_state_dict, set_state_dict
from d3rlpy.online.buffers import ReplayBuffer
from d3rlpy.online.iterators import train_single_env
# from myd3rlpy.datasets import get_d4rl
from utils.k_means import kmeans
from myd3rlpy.metrics.scorer import dataset_value_scorer, single_evaluate_on_environment
from dataset.load_d4rl import get_d4rl_local, get_dataset


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
    if args.dataset in ['hopper-random-v0', 'walker2d-random-v0', 'halfcheetah-random-v0']:
        task_datasets = {}
        for dataset_num in args.dataset_nums:
            h5_path = 'dataset/d4rl/' + args.dataset + '/itr_' + str(dataset_num) + '.hdf5'
            task_datasets[dataset_num] = get_d4rl_local(get_dataset(h5_path))
        _, env = d3rlpy.datasets.get_dataset(args.dataset)
        _, eval_env = d3rlpy.datasets.get_dataset(args.dataset)

    else:
        raise NotImplementedError

    # prepare algorithm
    if args.algo in ['td3_plus_bc', 'td3']:
        from myd3rlpy.algos.st_td3_plus_bc import ST
    elif args.algo in ['cql', 'mgcql', 'mrcql']:
        from myd3rlpy.algos.st_cql import ST
    else:
        raise NotImplementedError
    encoder = d3rlpy.models.encoders.VectorEncoderFactory([256, 256, 256])
    conservative_weight = 5.0
    st = ST(impl_name=args.algo, batch_size=args.batch_size, critic_replay_type=args.critic_replay_type, critic_replay_lambda=args.critic_replay_lambda, actor_replay_type=args.actor_replay_type, actor_replay_lambda=args.actor_replay_lambda, clone_actor=args.clone_actor, clone_critic=args.clone_critic, fine_tuned_step=args.fine_tuned_step, experience_type=args.experience_type, actor_learning_rate=1e-4, critic_learning_rate=3e-4, temp_learning_rate=1e-4, actor_encoder_factory=encoder, critic_encoder_factory=encoder, n_action_samples=args.n_action_samples, alpha_learning_rate=0.0, conservative_weight=conservative_weight, conservative_threshold=0.1, use_gpu=True)

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
        replay_dataset = None
        learned_datasets = []
        if not args.test:
            if env is not None:
                scorers_list = [{'environment': d3rlpy.metrics.evaluate_on_environment(env)}]
            else:
                raise NotImplementedError
        else:
            scorers_list = None
        print(f"task_datasets: {task_datasets}")
        for dataset_id, (dataset_num, dataset) in enumerate(task_datasets.items()):
            learned_datasets.append(dataset)
            if dataset_id != 0:
                scorers_list += [{'dataset_value_scorer': dataset_value_scorer, 'dataset_scale': d3rlpy.metrics.average_value_estimation_scorer, "action_scorer": d3rlpy.metrics.continuous_action_diff_scorer}]
            else:
                scorers_list += [{'dataset_value_scorer': dataset_value_scorer, 'dataset_scale': d3rlpy.metrics.average_value_estimation_scorer, "action_scorer": d3rlpy.metrics.continuous_action_diff_scorer}]

            add_one_learned_datasets = [None] + learned_datasets
            if dataset_id < args.read_policy:
                continue
            start_time = time.perf_counter()
            print(f'Start Training {dataset_id}')
            iterator, replay_iterator, replay_dataloader, n_epochs = st.make_iterator(dataset, replay_dataset, args.n_steps, args.n_steps_per_epoch, None, True)
            if dataset_id == args.read_policy:
                if args.read_policy == 0:
                    pretrain_path = args.model_path + 'cql' + '_' + args.dataset + '_' + str(args.dataset_nums[0]) + '.pt'
                else:
                    pretrain_path = args.model_path + args.algo + '_' + args.dataset + '_' + args.dataset_nums_str + '_' + str(dataset_id) + '.pt'
                st.build_with_dataset(dataset, dataset_id)
                st.load_model(pretrain_path)
                st._impl.rebuild()

                if args.read_policy == 0:
                    pretrain_path = args.model_path + 'cql' + '_' + args.dataset + '_' + str(args.dataset_nums[1]) + '.pt'
                else:
                    pretrain_path = args.model_path + args.algo + '_' + args.dataset + '_' + args.dataset_nums_str + '_' + str(dataset_id + 1) + '.pt'
                pretrain_state_dict = torch.load(pretrain_path, map_location=device)
                st._impl.init_copy_network()
                st.copy_load_model(pretrain_state_dict)
                if (args.critic_replay_type not in ['ewc', 'si', 'rwalk'] or args.actor_replay_type not in ['ewc', 'si', 'rwalk']) and args.read_policy != 0:
                    try:
                        replay_dataset = torch.load(f=args.model_path + algos_name + '_' + str(dataset_id) + '_datasets.pt')
                    except BaseException as e:
                        print(f'Don\' have replay_dataset')
                        raise e
            else:

                # train
                print(f'fitting dataset {dataset_id}')
                st.fit(
                    dataset_id,
                    dataset=dataset,
                    iterator=iterator,
                    replay_dataset=replay_dataset,
                    replay_iterator=replay_iterator,
                    replay_dataloader=replay_dataloader,
                    eval_episodes_list=add_one_learned_datasets,
                    # n_epochs=args.n_epochs if not args.test else 1,
                    n_epochs=n_epochs,
                    save_interval=args.save_interval,
                    experiment_name=experiment_name + algos_name,
                    scorers_list = scorers_list,
                    test=args.test,
                )
            st.after_learn(iterator, add_one_learned_datasets, scorers_list, experiment_name + algos_name)
            print(f'Training task {dataset_id} time: {time.perf_counter() - start_time}')
            st.save_model(args.model_path + algos_name + '_' + str(dataset_id) + '.pt')
            slide_dataset_length = args.max_save_num // (dataset_id + 1)
            new_replay_dataset = st.generate_replay(dataset_id, dataset, env, args.critic_replay_type, args.actor_replay_type, args.experience_type, slide_dataset_length, args.max_export_step, args.test)
            if replay_dataset is not None:
                replay_dataset_length = len(replay_dataset)
                slide_dataset_length = args.max_save_num // (dataset_id)
                indices = torch.cat([torch.range(slide_dataset_length * i, slide_dataset_length *(i + 1), device=device)[torch.randperm(slide_dataset_length)[: slide_dataset_length // 2]] for i in range(dataset_id)])
                replay_dataset = torch.utils.data.Subset(replay_dataset, indices)
                replay_dataset = torch.utils.data.ConcatDataset([replay_dataset, new_replay_dataset])
            else:
                replay_dataset = new_replay_dataset
            if args.test and dataset_id >= 2:
                break
        if args.online_n_steps > 0:
            buffer = ReplayBuffer(maxlen=args.online_maxlen, env=env)
            st.online_fit(env, eval_env, buffer, n_steps=args.online_n_steps, n_steps_per_epoch=args.n_steps_per_epoch, experiment_name = experiment_name + algos_name, test=args.test)
    print('finish')

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Experimental evaluation of lifelong PG learning')
    parser.add_argument('--add_name', default='', type=str)
    parser.add_argument("--dataset", default='ant_dir', type=str)
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
    parser.add_argument('--algo', default='combo', type=str, choices=['combo', 'td3_plus_bc', 'cql', 'mgcql', 'mrcql'])
    parser.add_argument('--eval', action='store_true')
    parser.add_argument('--test', action='store_true')
    parser.add_argument("--n_steps", default=500000, type=int)
    parser.add_argument("--n_steps_per_epoch", default=1000, type=int)

    parser.add_argument("--online_n_steps", default=100000, type=int)
    parser.add_argument("--online_maxlen", default=1000000, type=int)

    parser.add_argument("--save_interval", default=10, type=int)
    parser.add_argument("--n_action_samples", default=10, type=int)
    parser.add_argument('--top_euclid', default=64, type=int)

    parser.add_argument('--critic_replay_type', default='bc', type=str, choices=['orl', 'bc', 'ewc', 'gem', 'agem', 'rwalk', 'si', 'none'])
    parser.add_argument('--critic_replay_lambda', default=100, type=float)
    parser.add_argument('--actor_replay_type', default='orl', type=str, choices=['orl', 'bc', 'ewc', 'gem', 'agem', 'rwalk', 'si', 'none'])
    parser.add_argument('--actor_replay_lambda', default=1, type=float)
    parser.add_argument('--fine_tuned_step', default=1, type=int)
    parser.add_argument('--clone', default='none', type=str)

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
    args.dataset_nums_str = args.dataset_nums
    args.dataset_nums = [int(x) for x in args.dataset_nums.split('-')]
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
