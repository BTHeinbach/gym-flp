import gym
import gym_flp


env = gym.make('ofp-v0', mode='human', instance='P6', aspace='box')
env.reset()
for _ in range(10):
    a = env.action_space.sample()
    s, r, d, i = env.step(a)
    print(a, env.internal_state,r,d,i)
    #img = env.render()
    #img.show()