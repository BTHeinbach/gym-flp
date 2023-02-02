import gym
import gym_flp
import matplotlib.pyplot as plt
import imageio
import numpy as np

env = gym.make('qap-v0', instance='Neos-n7', mode='rgb_array')
env.reset()
s0 = env.internal_state
for _ in range(1):
    s,r,d,i=env.step(env.action_space.n-1)
    img = env.render()
    plt.imshow(img)
    plt.show()
    print(s0, env.internal_state,r,d,i)
