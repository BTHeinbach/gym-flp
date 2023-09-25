import gym
import gym_flp
import matplotlib.pyplot as plt
import imageio
import numpy as np

env = gym.make('ofp-v0', instance='P6', mode='rgb_array', randomize='randomize')
env.reset()
s0 = env.state
for _ in range(1):
    s, r, d, i=env.step(env.action_space.n-1)
    print(s0, env.state,r,d,i)
img = env.render()
plt.axis('off')
plt.imshow(img)

plt.show()
