from gym.envs.registration import register

register(
    id='collect-v0',
    entry_point='gym_collect.envs:CollectEnv',
)