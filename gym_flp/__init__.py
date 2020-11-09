from gym.envs.registration import register

name = 'gym_flp'
register(id='qap-v1', entry_point='gym_flp.envs:qapEnv')
register(id='fbs-v0', entry_point='gym_flp.envs:qspEnv')
register(id='mip-v0', entry_point='gym_flp.envs:mipEnv')
