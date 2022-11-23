import gym
import gym_flp
import numpy as np
import matplotlib.pyplot as plt
from stable_baselines3.common.env_util import make_vec_env

#env = gym.make('ofp-v0', mode='rgb_array', instance='P6', aspace='box', multi=False)
env = make_vec_env('ofp-v0', env_kwargs={'mode': 'rgb_array', "instance": 'P6', "aspace": 'discrete', "multi": True}, n_envs=1)



fig, (ax1,ax2) = plt.subplots(2,1)

r = []
m = []

env.reset()
print(env.get_attr("internal_state"))
for _ in range(100):
    a = env.action_space.sample()
    s, reward, d, i = env.step(a)
    #print(a)
    print(env.get_attr("internal_state"),r, i)
    pix = np.array(env.render())
    #plt.imshow(pix)
    #plt.show()
    r.append(reward)
    m.append(i['mhc'])
    ax1.plot(r)
    ax2.plot(m)
fig.show()