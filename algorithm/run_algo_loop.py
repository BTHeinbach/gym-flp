from a2c import A2C_FLP


A2C_FLP(train_steps = [1e6], aspace = 'discrete', multi = True)

A2C_FLP(train_steps = [1e6], aspace = 'discrete', multi = False)

A2C_FLP(train_steps = [1e6], aspace = 'box', multi = True)

A2C_FLP(train_steps = [1e6], aspace = 'box', multi = False)

from ppo import PPO_FLP

PPO_FLP(train_steps = [1e6], aspace = 'discrete', multi = True)

PPO_FLP(train_steps = [1e6], aspace = 'discrete', multi = False)

PPO_FLP(train_steps = [1e6], aspace = 'box', multi = True)

PPO_FLP(train_steps = [1e6], aspace = 'box', multi = False)

from ddpg import DDPG_FLP

DDPG_FLP(train_steps = [1e6], aspace = 'box', multi = True)

DDPG_FLP(train_steps = [1e6], aspace = 'box', multi = False)


