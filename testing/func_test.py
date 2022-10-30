import gym
import gym_flp
from stable_baselines3.common.env_util import make_vec_env

#env = gym.make('ofp-v0', instance='P6', mode='rgb_array')
env = make_vec_env('ofp-v0', env_kwargs={'mode': 'rgb_array', "instance": 'P6'}, n_envs=1)


#print(env.action_space)

env.reset()
s0 = env.get_attr('internal_state')

a0 = env.action_space.sample()


s1, r, d, i = env.step(a0)
print(s0)
print(a0)
print(env.get_attr('internal_state'))

