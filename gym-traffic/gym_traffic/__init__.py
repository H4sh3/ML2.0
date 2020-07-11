from gym.envs.registration import register

register(
    id='traffic-v0',
    entry_point='gym_traffic.envs:TrafficEnv',
)