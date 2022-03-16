from gym.envs.registration import register


register(
    id="HalfCheetahBack-v2",
    entry_point="mygym.envs.mujoco:HalfCheetahBackEnv",
    max_episode_steps=1000,
    reward_threshold=3800.0,
)
register(
    id="Walker2dBack-v2",
    entry_point="mygym.envs.mujoco:Walker2dBackEnv",
    max_episode_steps=1000,
    reward_threshold=3800.0,
)
register(
    id="HopperBack-v2",
    entry_point="mygym.envs.mujoco:HopperBackEnv",
    max_episode_steps=1000,
    reward_threshold=3800.0,
)
