import numpy as np
from typing import List, Dict
from mygym.envs.walker2d_rand_params import Walker2DRandParamsEnv
from gym.spaces.box import Box
from gym.utils.ezpickle import EzPickle

from . import register_env


@register_env('walker-rand-params')
class WalkerRandParamsWrappedEnv(Walker2DRandParamsEnv):
    def __init__(self, tasks: List[Dict] = None, n_tasks: int = None, randomize_tasks=True):
        observation_shape = Box(low=-np.inf, high=np.inf, dtype=np.float32, shape=(17,))
        super(WalkerRandParamsWrappedEnv, self).__init__(observation_shape)
        if tasks is None and n_tasks is None:
            raise Exception("Either tasks or n_tasks must be specified")

        if tasks is not None:
            self.tasks = tasks
        else:
            self.tasks = self.sample_tasks(n_tasks)
        self.reset_task(0)

    def get_all_task_idx(self):
        return range(len(self.tasks))

    def reset_task(self, idx):
        self._task = self.tasks[idx]
        self._goal = idx
        self.set_task(self._task)
        self.reset()
