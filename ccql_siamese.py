import argparse
import json
import random
from collections import namedtuple
import pickle
from dataset.d4rl import cut_antmaze
from envs import HalfCheetahDirEnv
from utils.utils import ReplayBuffer
import numpy as np
import torch

import d3rlpy
from d3rlpy.ope import FQE
from d3rlpy.datasets import get_d4rl
from d3rlpy.metrics import evaluate_on_environment, td_error_scorer
from d3rlpy.metrics.scorer import initial_state_value_estimation_scorer
from d3rlpy.metrics.scorer import soft_opc_scorer
from d3rlpy.dataset import MDPDataset
from myd3rlpy.algos.co import CO
# from myd3rlpy.datasets import get_d4rl
from utils.siamese_similar import similar_euclid, similar_phi
from utils.k_means import kmeans


def build_networks_and_buffers(args, env, task_config):
    obs_dim = env.observation_space.shape[0]
    action_dim = env.action_space.shape[0]
    buffer_paths = [
        task_config.train_buffer_paths.format(idx) for idx in task_config.train_tasks
    ]

    buffers = [
        ReplayBuffer(
            args.inner_buffer_size,
            obs_dim,
            action_dim,
            discount_factor=0.99,
            immutable=True,
            load_from=buffer_paths[i],
        )
        for i, task in enumerate(task_config.train_tasks)
    ]

    return buffers
def get_env(task_config):
    tasks = []
    for task_idx in range(task_config.total_tasks):
        with open(task_config.task_paths.format(task_idx), "rb") as f:
            task_info = pickle.load(f)
            assert len(task_info) == 1, f"Unexpected task info: {task_info}"
            tasks.append(task_info[0])

    return HalfCheetahDirEnv(tasks, include_goal=False)

def main(args, device):
#     with open(f"./{args.task_config}", "r") as f:
#         task_config = json.load(
#             f, object_hook=lambda d: namedtuple("X", d.keys())(*d.values())
#         )
#     env = get_env(task_config)
#     buffers = build_networks_and_buffers(args, env, task_config)
#     task_num = len(buffers)
#     datasets = []
#     for task_id, buffer_ in enumerate(buffers):
#         task_id_np = np.zeros((buffer_._obs.shape[0], task_num), dtype=np.float32)
#         task_id_np[:, task_id] = 1
#         buffer_._obs = np.hstack((buffer_._obs, task_id_np))
#         datasets.append(MDPDataset(buffer_._obs, buffer_._actions, buffer_._rewards, buffer_._terminals))
#         break
    np.set_printoptions(precision=1, suppress=True)
    dataset, env = get_d4rl('antmaze-large-play-v0')
    obses = []
    for episode in dataset.episodes:
        if len(episode) != 1:
            obses.append(torch.from_numpy(episode[-1].observation[:2]).cuda())
            # if (obses[-1][0] < 32 or obses[-1][0] > 34) and (obses[-1][1] < 23 or obses[-1][1] > 25):
            #     print(obses[-1])
    obses = torch.stack(obses)
    # end_points, _ = kmeans(obses, args.task_nums)
    end_points = [np.array([32.41604, 24.43354]), np.array([21.3771, 17.4113]), np.array([20.8545, 25.0958]), np.array([4.5582, 17.7067]), np.array([18.1493, 8.9290]), np.array([0.1346, 13.3144]), np.array([37.0817, 12.0133])]
    print('start task split')
    assert False

    task_datasets = {k: [] for k in range(args.task_nums)}
    if args.task_split_type == 'directed':
        for episode in dataset.episodes:
            if len(episode) == 1:
                continue
            position = episode[-1].observation[:2]
            for end_point in end_points:
                end_point_distance = np.linalg.norm(position, end_point)
            min_index = np.argmin(end_point_distance)
            task_datasets[min_index].append(episode)
    else:
        all_episodes = dataset.episodes
        episodes = []
        for episode in all_episodes:
            if len(episode) != 1:
                episodes.append(episode)
        random.shuffle(episodes)
        task_length = len(episodes) // args.task_nums
        task_mod = len(episodes) - args.task_nums * task_length
        i = 0
        for task_num in range(args.task_nums):
            if task_num < task_mod:
                task_datasets[task_num] = episodes[i: i + task_length + 1]
                i += task_length + 1
            else:
                task_datasets[task_num] = episodes[i: i + task_length]
                i += task_length
    print('finish task split')
    assert False
    for task_index, task_episodes in task_datasets.items():
        for episode in task_episodes:
            for time_index, observation in enumerate(episode.observations):
                if np.linalg.norm(observation[:2] - end_points[task_index]) < 0.5:
                    episode.rewards[time_index] = 1
                else:
                    episode.rewards[time_index] = 0
        observations = [episode.observations for episode in task_episodes]
        observations = np.concatenate(observations, axis=0)
        actions = [episode.actions for episode in task_episodes]
        actions = np.concatenate(actions, axis=0)
        rewards = [episode.rewards for episode in task_episodes]
        rewards = np.concatenate(rewards, axis=0)
        terminals = [episode.terminal for episode in task_episodes]
        terminals = np.array(terminals)
        task_datasets[task_index] = MDPDataset(observations, actions, rewards, terminals)
    print(f'finish task rereward')
    assert False

    original = torch.zeros([2]).to(device)
    destination = [torch.from_numpy(end_point).to(device) for end_point in end_points]

# prepare algorithm
    print('start co')
    co = CO(use_gpu=True)

    replay_datasets = None
    for dataset_num, dataset in task_datasets.items():
        print(f'start dataset {dataset_num}')
        indexes_euclid = similar_euclid(torch.from_numpy(dataset.observations).cpu())
        assert False
        # 用action保存一下indexes_euclid
        dataset.actions = np.hstack(dataset.actions, indexes_euclid)
        # train
        co.fit(
            dataset,
            replay_datasets,
            dataset_num,
            dataset,
            eval_episodes=dataset,
            replay_eval_episodess = replay_datasets,
            n_epochs=1,
            scorers={
                # 'environment': evaluate_on_environment(env),
                'td_error': td_error_scorer
            },
            replay_scorers={
                'bc_error': bc_error_scorer
            }
        )

        # 关键算法
        start_indexes = [original]
        start_observations = [dataset._observations[original, :]]
        start_actionss, start_action_log_probss = co._impl._policy.sample_n_with_log_prob(start_observations, dataset_num, args.sample_times)
        replay_indexes, replay_actionss, replay_action_log_probss, replay_observations, replay_means, replay_stddevs, replay_qss = np.empty(), np.empty(), np.empty(), np.empty(), np.empty(), np.empty(), np.empty()
        while len(start_indexes) != 0:
            near_observations = torch.from_numpy(dataset._observations)[indexes_euclid[start_indexes]]
            near_actions = torch.from_numpy(dataset._actions)[indexes_euclid[start_indexes]]
            this_observations = start_observations.unsqueeze(dim=1).expand(-1, indexes_euclid.shape[1], -1)
            near_indexes = []
            for sample_time in range(args.sample_times):
                start_actions = start_actionss[:, sample_time, :]
                this_actions = start_actions.unsqueeze(dim=1).expand(-1, indexes_euclid.shape[1], -1)
                for o in near_observations.shape[0]:
                    near_indexes.append(similar_phi(this_observations[o], this_actions[o], near_observations[o], near_actions[o], co._impl._phi, indexes_euclid[start_indexes[o]]))
            start_actionss = torch.stack(start_actionss).permute(1, 0, 2)
            near_indexes = torch.cat(near_indexes, dim=1).numpy()
            near_indexes = np.unique(near_indexes)
            start_indexes = np.setdiff1d(near_indexes, replay_indexes)
            start_observations = torch.from_numpy(dataset._observation[start_indexes])
            start_actionss, start_action_log_probss = co._impl._policy.sample_n_with_log_prob(start_observations, dataset_num, args.sample_times)

            start_means = co._impl.dist(start_observations).mean.numpy()
            start_stddevs = co._impl.dist(start_observations).stddev.numpy()
            start_qss = []
            for sample_time in range(args.sample_times):
                start_qs = co._impl._q_func.sample_q_function(start_observations, start_actions, dataset_num).numpy()
                start_qss.append(start_qs)
            start_qss = torch.stack(start_qss).permute(1, 0)
            replay_indexes.extend(start_indexes)
            replay_observations.extend(start_observations.numpy())
            replay_actionss.extend(start_actionss)
            replay_means.extend(start_means)
            replay_stddevs.extend(start_stddevs)
            replay_qss.extend(start_qss)
            start_terminals = dataset.terminal[start_indexes]
            start_indexes = torch.from_numpy(start_indexes[start_terminals == False])
        replay_dataset = torch.utils.data.TensorDataset([torch.from_numpy(replay_observations), torch.from_numpy(replay_actionss), torch.from_numpy(replay_means), torch.from_numpy(replay_stddevs), torch.from_numpy(replay_qss)])
        if replay_datasets is None:
            replay_datasets = [replay_dataset]
        else:
            replay_datasets.append(replay_dataset)

    for dataset in datasets:
        # off-policy evaluation algorithm
        fqe = FQE(algo=co)

        # metrics to evaluate with

        # train estimators to evaluate the trained policy
        fqe.fit(dataset.episodes,
                eval_episodes=dataset.episodes,
                n_epochs=1,
                scorers={
                   'init_value': initial_state_value_estimation_scorer,
                   'soft_opc': soft_opc_scorer(return_threshold=600)
                })

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Experimental evaluation of lifelong PG learning')
    parser.add_argument('--inner_buffer_size', default=-1, type=int)
    parser.add_argument('--task_config', default='task_config/cheetah_dir.json', type=str)
    parser.add_argument('--siamese_hidden_size', default=100, type=int)
    parser.add_argument('--near_threshold', default=1, type=float)
    parser.add_argument('--siamese_threshold', default=1, type=float)
    parser.add_argument('--eval_batch_size', default=256, type=int)
    parser.add_argument('--sample_times', default=20, type=int)
    parser.add_argument('--task_split_type', default='undirected', type=str)
    parser.add_argument('--task_nums', default=7, type=int)
    args = parser.parse_args()
    global DATASET_PATH
    DATASET_PATH = './.d4rl/datasets/'
    device = torch.device('cuda:0')
    main(args, device)
