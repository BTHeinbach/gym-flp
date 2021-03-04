from gym.envs.registration import register

name = 'gym_flp'
register(id='qap-v0',
         entry_point='gym_flp.envs:qapEnv')

register(id='fbs-v0', entry_point='gym_flp.envs:fbsEnv')

register(id='ofp-v0', entry_point='gym_flp.envs:ofpEnv')

register(id='sts-v0', entry_point='gym_flp.envs:stsEnv')
