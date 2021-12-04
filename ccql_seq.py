import argparse
import json
from collections import namedtuple
import pickle
import d3rlpy
from d3rlpy.ope import FQE
from d3rlpy.algos import CQL
from d3rlpy.datasets import get_d4rl
from d3rlpy.metrics import evaluate_on_environment, td_error_scorer
from d3rlpy.metrics.scorer import initial_state_value_estimation_scorer
from d3rlpy.metrics.scorer import soft_opc_scorer
from d3rlpy.dataset import MDPDataset

from envs import HalfCheetahDirEnv
from utils import ReplayBuffer
import numpy as np


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
    trained_datasets = []
    with open(f"./{args.task_config}", "r") as f:
        task_config = json.load(
            f, object_hook=lambda d: namedtuple("X", d.keys())(*d.values())
        )
    tasks = []
    env = get_env(task_config)
    buffers = build_networks_and_buffers(args, env, task_config)
    task_num = len(buffers)
    datasets = []
    for task_id, buffer_ in enumerate(buffers):
        task_id_np = np.zeros((buffer_._obs.shape[0], task_num), dtype=np.float32)
        task_id_np[:, task_id] = 1
        buffer_._obs = np.hstack((buffer_._obs, task_id_np))
        datasets.append(MDPDataset(buffer_._obs, buffer_._actions, buffer_._rewards, buffer_._terminals))

# prepare algorithm
    cql = CQL(use_gpu=True)

    for dataset_num, dataset in enumerate(datasets):
        # train
        for epoch in range(10):
            cql.fit(
                dataset,
                eval_episodes=dataset,
                n_epochs=1,
                scorers={
                    # 'environment': evaluate_on_environment(env),
                    'td_error': td_error_scorer
                }
            )
            print(f'dataset: {dataset_num}, epoch: {epoch}')

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
    args = parser.parse_args()
    main(args)
