import gym
import gym_flp
import imageio
import datetime
import os

import numpy as np
import matplotlib.pyplot as plt
import torch as th

from stable_baselines3 import TD3
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.callbacks import EvalCallback, BaseCallback
from stable_baselines3.common.vec_env import VecTransposeImage
from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3.common.logger import Video
from PIL import Image
from typing import Any, Dict

instance = 'P6'
timestamp = datetime.datetime.now().strftime("%y%m%d_%H%M")
environment = 'ofp'
algo = 'ppo'
mode = 'rgb_array'
aspace = 'box'
multi = False
train_steps = [5e4]
vec_env = make_vec_env('ofp-v0',
                       env_kwargs={'mode': mode, "instance": instance, "aspace": aspace, "multi": multi},
                       n_envs=1)
wrap_env = VecTransposeImage(vec_env)
print(wrap_env.action_space.sample())
vec_eval_env = make_vec_env('ofp-v0',
                            env_kwargs={'mode': mode, "instance": instance, "aspace": aspace, "multi": multi},
                            n_envs=1)
wrap_eval_env = VecTransposeImage(vec_eval_env)
model = '221127_2132_P6_td3_ofp_box_single_50000'
path=fr'C:\Users\Benny\PycharmProjects\gym-flp\algorithm\models\best_model\{model}\best_model.zip'

best_model = TD3.load(path)
fig, (ax1, ax2) = plt.subplots(2, 1)

obs = wrap_env.reset()
start_cost = wrap_env.get_attr("last_cost")[0]

rewards = []
mhc = []
images = []
gain = 0
gains = []
c = []
actions = []
done = False

eval_steps = 1000
while not done:
    # for _ in range(eval_steps):
    action, _states = best_model.predict(obs, deterministic=True)
    actions.append(action)
    obs, reward, done, info = wrap_env.step(action)
    img = Image.fromarray(wrap_env.render(mode='rgb_array'))
    rewards.append(reward[0])
    mhc.append(info[0]['mhc'])
    c.append(info[0]['collisions'])
    images.append(img)

final_cost = mhc[-1]

print(start_cost, final_cost)
cost_saved = final_cost - start_cost
cost_saved_rel = 1 - (start_cost / final_cost)
ax1.plot(np.arange(1, len(rewards) + 1), rewards)
ax2.plot(np.arange(1, len(rewards) + 1), mhc)
imageio.mimsave(f'gifs/{model}_eval_env.gif',
                [np.array(img.resize((200, 200), Image.NEAREST))
                 for i, img in enumerate(images) if i % 2 == 0], fps=15)
wrap_env.close()

del best_model

fig.show()
