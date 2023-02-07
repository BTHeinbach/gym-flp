from gym.envs.registration import register

name = 'gym_flp'
register(id='qap-v0', entry_point='gym_flp.envs.QAP:QapEnv')
register(id='fbs-v0', entry_point='gym_flp.envs.FBS:FbsEnv')
register(id='ofp-v0', entry_point='gym_flp.envs.OFP:OfpEnv')
register(id='sts-v0', entry_point='gym_flp.envs.STS:StsEnv')
