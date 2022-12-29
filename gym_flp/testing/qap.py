import gym
import gym_flp
import matplotlib.pyplot as plt
import imageio

env = gym.make('ofp-v0', instance='BME15', mode='rgb_array')
env.reset()
img = env.render()
plt.imshow(img)
