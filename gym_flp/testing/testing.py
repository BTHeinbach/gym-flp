import gym
import gym_flp


env = gym.make('ofp-v0', mode='human', instance='P6', aspace='multi-discrete')
env.reset()
print(env.action_space.shape)
env.step(env.action_space.sample())