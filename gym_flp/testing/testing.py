import gym
import gym_flp


env = gym.make('ofp-v0', mode='human', instance='P6', aspace='box')
env.reset()
for _ in range(1000):
    a = env.action_space.sample()
    s, r, d, i = env.step(a)
    print(env.counter, d)
    #img = env.render()
    #img.show()