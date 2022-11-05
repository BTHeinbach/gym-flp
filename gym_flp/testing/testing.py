import gym
import gym_flp
import matplotlib as plt

env = gym.make('ofp-v0', mode='human', instance = 'P6')
s0 = env.reset()

print("Reset done")

for _ in range(1):
    a = env.action_space.sample()
    s1, r, i, d = env.step(a)
    print("Step done")
    img = env.render()
    img.show()