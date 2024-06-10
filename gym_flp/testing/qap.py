import gym
import gym_flp
import matplotlib.pyplot as plt
import imageio
import numpy as np
from stable_baselines3.common.env_util import make_vec_env

env = gym.make('ofp-v0', instance='P6', mode='rgb_array', box=True)
env2 = make_vec_env(env_id='ofp-v0', env_kwargs={'instance': 'P6', 'mode':'rgb_array', 'box': True}, n_envs=1)

#env.reset()
env2.reset()

a = env2.action_space.sample()

print(a)
for _ in range(1):
    #s,r,d,i=env.step(a)
    s1, r1, d1, i1 = env2.step(a)

    print(s, s1)
