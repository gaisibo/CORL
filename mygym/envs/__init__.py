import os
import importlib
import gym


ENVS = {}


def register_env(name):
    """Registers a env by name for instantiation in rlkit."""

    def register_env_fn(fn):
        if name in ENVS:
            raise ValueError("Cannot register duplicate env {}".format(name))
        if not callable(fn):
            raise TypeError("env {} must be callable".format(name))
        ENVS[name] = fn
        return fn

    return register_env_fn

gym.register(
    id="HalfCheetahBlock-v2",
    entry_point="mygym.envs.halfcheetah_block:HalfCheetahBlockEnv",
    max_episode_steps=1000,
    reward_threshold=4800.0,
)

gym.register(
    id="Walker2dBlock-v4",
    max_episode_steps=1000,
    entry_point="mygym.envs.walker2d_block:Walker2dBlockEnv",
)

gym.register(
    id="HopperBlock-v4",
    entry_point="mygym.envs.hopper_block:HopperBlockEnv",
    max_episode_steps=1000,
    reward_threshold=3800.0,
)

# automatically import any envs in the envs/ directory
#for file in os.listdir(os.path.dirname(__file__)):
#    if file.endswith('.py') and not file.startswith('_'):
#        module = file[:file.find('.py')]
#        importlib.import_module('rlkit.envs.' + module)
