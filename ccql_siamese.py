import argparse
import json
from collections import namedtuple
import pickle
from envs import HalfCheetahDirEnv
from utils import ReplayBuffer
import numpy as np
import torch

import d3rlpy
from d3rlpy.ope import FQE
from d3rlpy.datasets import get_d4rl
from d3rlpy.metrics import evaluate_on_environment, td_error_scorer
from d3rlpy.metrics.scorer import initial_state_value_estimation_scorer
from d3rlpy.metrics.scorer import soft_opc_scorer
from d3rlpy.dataset import MDPDataset
from myd3rlpy.algos.siamese import Siamese
from myd3rlpy.algos.cqlbc import CQLBC


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

def main(args):
    with open(f"./{args.task_config}", "r") as f:
        task_config = json.load(
            f, object_hook=lambda d: namedtuple("X", d.keys())(*d.values())
        )
    env = get_env(task_config)
    buffers = build_networks_and_buffers(args, env, task_config)
    task_num = len(buffers)
    datasets = []
    for task_id, buffer_ in enumerate(buffers):
        task_id_np = np.zeros((buffer_._obs.shape[0], task_num), dtype=np.float32)
        task_id_np[:, task_id] = 1
        buffer_._obs = np.hstack((buffer_._obs, task_id_np))
        datasets.append(MDPDataset(buffer_._obs, buffer_._actions, buffer_._rewards, buffer_._terminals))
        break
    original = torch.from_numpy(buffers[0]._obs[0, :])
    destination = torch.from_numpy(buffers[0]._obs[-1, :])

# prepare algorithm
    cql = CQLBC(use_gpu=True)

    replay_datasets = []
    for dataset_num, dataset in enumerate(datasets):
        # train
        mix_dataset = MDPDataset(np.hstack([dataset._observations] + [x._observations for x in replay_datasets]), np.hstack([dataset._actions] + [x._actions for x in replay_datasets]), np.hstack([dataset._rewards] + [x._rewards for x in replay_datasets]), np.hstack([dataset._terminals] + [x._terminals for x in replay_datasets]))
        cql.fit(
            mix_dataset,
            eval_episodes=mix_dataset,
            replay_datasets = replay_datasets,
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
        siamese = Siamese(cql, use_gpu=True)
        siamese.fit(
            mix_dataset,
            eval_episodes=mix_dataset,
            n_epochs=10,
        )
        start_points = []
        start_points.append(original)
        replay_observations, replay_actions, replay_means, replay_stddevs, replay_qs = [], [], [], [], []
        while len(start_points) is not None:
            start_observations = []
            start_actions = []
            for start_point in start_points:
                replay_observations.append(start_point)
                start_observations.append(start_point)
                act_tmp = cql.predict([start_point])
                replay_actions.append(act_tmp)
                start_actions.append(act_tmp)
                replay_means.append(cql.dist(start_point).mean)
                replay_stddevs.append(cql.dist(start_point).stddev)
                replay_qs.append(cql.q_function(start_point, act_tmp))
            end_points = []
            for start_observation, start_actions in zip(start_observations, start_actions):
                near_observations = []
                near_actions = []
                for observation, action in zip(dataset.observations, dataset.actions):
                    near_distance = np.linalg.norm(observation, start_point)
                    print(f'near distance: {near_distance}')
                    print(f'near distance min: {np.min(near_distance)}')
                    print(f'near distance max: {np.max(near_distance)}')
                    print(f'near distance mean: {np.mean(near_distance)}')
                    if near_distance <= args.near_threshold:
                        near_observations.append(observation)
                        near_actions.append(action)
                near_observations = torch.from_numpy(np.array(near_observations))
                near_actions = torch.from_numpy(np.array(near_actions))
                action = cql.predict([start_point])
                this_observations = start_point.unsqueeze(dim=0).expand(args.eval_batch_size, 1)
                this_action = action.unsqueeze(dim=0).expand(args.eval_batch_size, 1)
                i = 0
                idx = None
                if i < near_observations.shape[0]:
                    siamese_distance = siamese.predict([near_observations[i: i + args.eval_batch_size], near_actions[i: i + args.eval_batch_size], this_observations, this_action])
                    print(f'siamese_distance: {siamese_distance}')
                    print(f'siamese distance min: {np.min(siamese_distance)}')
                    print(f'siamese distance max: {np.max(siamese_distance)}')
                    print(f'siamese distance mean: {np.mean(siamese_distance)}')
                    near_index = torch.where(siamese_distance < args.siamese_threshold, 1, 0)
                    near_index += i
                    if idx is not None:
                        idx = near_index
                    else:
                        idx = torch.cat([idx, near_index], dim=0)
                    i += i + args.eval_batch_size

                part_dataset = dataset[idx]
                part_episodes = part_dataset.episodes
                for part_episode in part_episodes:
                    part_transitions = part_episodes.transitions
                    for part_transition in part_transitions:
                        # 到达终点，没必要继续
                        if part_transition.next_observation == destination:
                            continue
                        # 与之前的点重复了。注意数据集中的点除了起点都是固定的，等于判断就可以了。
                        to_continue = False
                        for finished_points in replay_observations:
                            # 与过去的起点重复了。
                            if finished_points == part_transition.next_observation:
                                to_continue = True
                                break
                        if to_continue:
                            continue
                        for finished_points in end_points:
                            # 与这次探索出来的起点重复了。
                            if finished_points == part_transition.next_observation:
                                to_continue = True
                                break
                        if to_continue:
                            continue
                        end_points.append(part_transition.next_observation)
            start_points = end_points

    for dataset in datasets:
        # off-policy evaluation algorithm
        fqe = FQE(algo=cql)

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
    args = parser.parse_args()
    main(args)
