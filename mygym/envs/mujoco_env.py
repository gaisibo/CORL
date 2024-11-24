import os
from os import path

import mujoco_py
import numpy as np
from gym.envs.mujoco import mujoco_env


ENV_ASSET_DIR = os.path.join(os.path.dirname(__file__), 'assets')


class MujocoEnv(mujoco_env.MujocoEnv):
    """
    My own wrapper around MujocoEnv.

    The caller needs to declare
    """

    def init_serialization(self, locals):
        Serializable.quick_init(self, locals)

    def log_diagnostics(self, paths):
        pass


def get_asset_xml(xml_name):
    return os.path.join(ENV_ASSET_DIR, xml_name)
