import gym
from typing import List
from dataset.continual_world import OneHotAdder, TimeLimit


ATARI_TIME_HORIZON = 27000
def get_atari_envs(task: str, task_nums: List[int]) -> List[gym.Env]:
    """Returns continual learning environment.

    Args:
      tasks: list of task names or MT50 numbers
      randomization: randomization kind, one of 'deterministic', 'random_init_all',
                     'random_init_fixed20', 'random_init_small_box'.

    Returns:
      gym.Env: continual learning environment
    """
    num_tasks = len(task_nums)
    envs = []
    env_ids = []
    for i, task_num in enumerate(task_nums):
        env = gym.make(task, mode=task_num)
        #env = RandomizationWrapper(env, get_subtasks(task_name), randomization)
        env = OneHotAdder(env, one_hot_idx=i, one_hot_len=num_tasks)
        env.name = task_name
        env = TimeLimit(env, ATARI_TIME_HORIZON)
        #env = SuccessCounter(env)
        env.reset()
        envs.append(env)
    return envs
