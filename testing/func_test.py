import gym
import gym_flp

env = gym.make('ofp-v0', instance='P6', mode='rgb_array')

print(env.action_space)

s0 = env.reset()

a0 = env.action_space.sample()
print(a0)

s1, r, d, i = env.step(a0)

