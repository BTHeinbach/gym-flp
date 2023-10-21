from stable_baselines3.common.env_util import make_vec_env
from PIL import Image
from typing import Any, Dict
import argparse
import matplotlib.pyplot as plt
from stable_baselines3.common.vec_env import VecTransposeImage, DummyVecEnv
import numpy as np
import json
import os
import imageio
import tkinter as tk
from tkinter import filedialog
from copy import deepcopy
import gym_flp


def run(save_path=None):

    trg_path = os.path.dirname(os.path.abspath(__file__))

    if save_path is None:
        root = tk.Tk()
        root.withdraw()

        path = filedialog.askopenfilename().split('/')[-1]
        save_path = path.split('.')[0]

    algo = save_path.split('_')[3]
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
        raise Exception(f"{algo}: Algorithm not recognized or supported")

    final_model = RL.load(f"{trg_path}/models/{save_path}")
    best_model = RL.load(f"{trg_path}/models/best_model/{save_path}/best_model.zip")

    env_kwargs = final_model.env_kwargs
    test_env_final = make_vec_env(env_id=env_kwargs['envId'], env_kwargs=env_kwargs, n_envs=1)
    # test_env_best = make_vec_env(env_id=env_kwargs['envId'], env_kwargs=env_kwargs, n_envs=1)


    obs_final = test_env_final.reset()
    #obs_best = test_env_best.reset()

    #test_env_best.set_attr('internal_state', test_env_final.get_attr('internal_state'))
    #test_env_best.set_attr('state', test_env_final.get_attr('state'))
    obs_best = obs_final.copy()

    test_env_best = deepcopy(test_env_final)


    img_first = Image.fromarray(test_env_best.render(mode='rgb_array'))

    if isinstance(test_env_final, VecTransposeImage) or isinstance(test_env_final, DummyVecEnv):
        start_cost_final = test_env_final.get_attr("last_cost")[0]
    else:
        start_cost_final = test_env_final.last_cost
    start_cost_best = start_cost_final

    collisions=[]
    rewards = []
    mhc_final = []
    mhc_best = []
    images = []
    imgs = []
    actions = []
    dones = [False, False]
    counter = 0
    experiment_results = {
        'start_cost_final_model': start_cost_final,
        'start_cost_best_model': start_cost_best
    }

    fig, axs = plt.subplots(2, 2)
    while False in dones:
        counter += 1

        if not dones[0]:
            action_final, _states_final = final_model.predict(obs_final, deterministic=True)
            obs_final, reward_final, done_final, info_final = test_env_final.step(action_final)
            img_final = Image.fromarray(test_env_final.render(mode='rgb_array'))
            dones[0] = done_final

        if not dones[1]:
            action_best, _states_best = best_model.predict(obs_best, deterministic=True)
            obs_best, reward_best, done_best, info_best = test_env_best.step(action_best)
            img_best = Image.fromarray(test_env_best.render(mode='rgb_array'))
            imgs.append(img_best)
            dones[1] = done_best
        
        # print(test_env_final.get_attr('internal_state'), test_env_best.get_attr('internal_state'))


        rewards.append([reward_final[0], reward_best[0]])
        mhc_final.append(info_final[0]['mhc'])
        mhc_best.append(info_best[0]['mhc'])
        collisions.append(info_final[0]['collisions'])


        axs[0, 0].imshow(img_final)
        axs[1, 0].imshow(img_best)
        # axs[0, 0].axis('off')
        # axs[1, 0].axis('off')
        # plt.show()

        axs[0, 1].plot(np.arange(1, len(mhc_final)+1), mhc_final)
        axs[1, 1].plot(np.arange(1, len(mhc_best)+1), mhc_best)
        # fig.show()

        fig.canvas.draw()
        # Now we can save it to a numpy array.
        data = np.frombuffer(fig.canvas.tostring_rgb(), dtype=np.uint8)
        data = data.reshape(fig.canvas.get_width_height()[::-1] + (3,))

        images.append(data)
        if counter > 500:
            print("kill process")
            break

    experiment_results['cost_final'] = mhc_final
    experiment_results['cost_best'] = mhc_best

    imageio.mimsave(f'{trg_path}/gifs/{save_path}_test_env.gif', images, duration=100)

   # new_path = os.path.join(os.getcwd(), f'{trg_path}/experiments', save_path + '.png')
   # plt.imsave(new_path, images[-2])
   # plt.imsave(os.path.join(os.getcwd(), f'{trg_path}/experiments', 'layout_ppo.png'), imgs[-2])
   # plt.imsave(os.path.join(os.getcwd(), f'{trg_path}/experiments', 'start_layout_ppo.png'), img_first)

    print("Evaluation finished")

    return experiment_results

if __name__ == '__main__':

    run()
