import json
import gymnasium as gym
import inspect
import gym_flp


# Simulate parser
algo = 'ppo'

if algo == 'ppo':
    from stable_baselines3 import PPO as RL
elif algo == 'a2c':
    from stable_baselines3 import A2C as RL
elif algo == 'dqn':
    from stable_baselines3 import DQN as RL
elif algo == 'sac':
    from stable_baselines3 import SAC as RL
elif algo == 'ddpg':
    from stable_baselines3 import DDPG as RL
elif algo == 'td3':
    from stable_baselines3 import TD3 as RL
else:
    raise Exception("Algorithm not recognized or supported")

env = gym.make('ofp-v0')
f = open('./config/algo-conf.json')
cfg = json.load(f)
config = cfg[algo]

model = RL('MlpPolicy', env)

for k,v in config.items():
    model.__setattr__(k, v)

print(cfg[algo])

s0 = env.reset()
s, r, d, i = env.step(env.action_space.sample())
print(s, r)
