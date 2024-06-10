from gym_flp.envs import OfpEnv, FbsEnv, StsEnv
import numpy as np
import matplotlib.pyplot as plt

env = FbsEnv(instance='WG12', mode='rgb_array')
env.reset()
img = env.render()
plt.imshow(img)
plt.show()

for _ in range(100):
    s, r, d, i = env.step(env.action_space.sample())
    print(i)
    img = env.render()
    plt.imshow(img)
    plt.show()