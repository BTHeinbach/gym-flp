import gym
import gym_flp


env = gym.make('ofp-v0', mode='human', instance='P6', aspace='box')
env.reset()
for _ in range(10):
    s, r, d, i = env.step(env.action_space.sample())
    print(s,r,d,i)