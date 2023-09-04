import gym
import gym_flp
import imageio
import datetime
import tkinter as tk
import json
import os

import numpy as np
import matplotlib.pyplot as plt
import torch as th

from tkinter import filedialog
from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.callbacks import EvalCallback, BaseCallback, StopTrainingOnNoModelImprovement
from stable_baselines3.common.vec_env import VecTransposeImage, DummyVecEnv
from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3.common.logger import Video
from PIL import Image
from typing import Any, Dict
import argparse

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--train', action="store_true", help='leave empty for debugging')
    parser.add_argument('--env', type=str, default='ofp-v0', help='environment name')
    parser.add_argument('--mode', type=str, default='rgb_array', help='state representation mode')
    parser.add_argument('--instance', type=str, default='P6', help='problem instance')
    parser.add_argument('--distance', type=str, default='r', help='distance metric')
    parser.add_argument('--step_size', type=int, default=1, help='step size for ofp envs')
    parser.add_argument('--box', action="store_true",  help='input box to use box env, if omitted uses discrete')
    parser.add_argument('--multi', action="store_true", help='whether to move one or more machines per step')
    parser.add_argument('--train_steps', type=int, default=1e5, help='number of training steps')
    parser.add_argument('--num_workers', type=int, default=1, help='number of parallel envs')
    parser.add_argument('--algo', type=str, default=parser.prog)
    args = parser.parse_args()
    return args


class TensorboardCallback(BaseCallback):
    """
    Custom callback for plotting additional values in tensorboard.
    """

    def __init__(self, verbose=0):
        super(TensorboardCallback, self).__init__(verbose)

    def _on_step(self) -> bool:
        # Log scalar value (here a random variable)
        self.logger.dump(self.num_timesteps)
        self.logger.record("monitor/counter", self.training_env.get_attr('counter')[0])
        self.logger.record("monitor/mhc", self.training_env.venv.buf_infos[0]['mhc'])
        self.logger.record("monitor/collisions", self.training_env.venv.buf_infos[0]['collisions'])
        return True


class VideoRecorderCallback(BaseCallback):
    def __init__(self, eval_env: gym.Env, render_freq: int, n_eval_episodes: int = 1, deterministic: bool = True):
        """
        Records a video of an agent's trajectory traversing ``eval_env`` and logs it to TensorBoard

        :param eval_env: A gym environment from which the trajectory is recorded
        :param render_freq: Render the agent's trajectory every eval_freq call of the callback.
        :param n_eval_episodes: Number of episodes to render
        :param deterministic: Whether to use deterministic or stochastic policy
        """
        super().__init__()
        self._eval_env = eval_env
        self._render_freq = render_freq
        self._n_eval_episodes = n_eval_episodes
        self._deterministic = deterministic

    def _on_step(self) -> bool:
        if self.n_calls % self._render_freq == 0:
            screens = []

            def grab_screens(_locals: Dict[str, Any], _globals: Dict[str, Any]) -> None:
                """
                Renders the environment in its current state, recording the screen in the captured `screens` list

                :param _locals: A dictionary containing all local variables of the callback's scope
                :param _globals: A dictionary containing all global variables of the callback's scope
                """
                screen = self._eval_env.render(mode="rgb_array")
                # PyTorch uses CxHxW vs HxWxC gym (and tensorflow) image convention
                screens.append(screen.transpose(2, 0, 1))

            evaluate_policy(
                self.model,
                self._eval_env,
                callback=grab_screens,
                n_eval_episodes=self._n_eval_episodes,
                deterministic=self._deterministic,
            )
            screens_array = np.array(screens)
            self.logger.record(
                f"trajectory/video",
                Video(th.ByteTensor([screens_array]), fps=20),
                exclude=("stdout", "log", "json", "csv"),
            )
        return True

stop_train_callback = StopTrainingOnNoModelImprovement(max_no_improvement_evals=20, min_evals=5, verbose=1)

if __name__ == '__main__':
    timestamp = datetime.datetime.now().strftime("%y%m%d_%H%M")
    args = parse_args()


    env_kwargs = {
        'mode': args.mode,
        'instance': args.instance,
        'box': args.box,
        'multi': args.multi
    }
    env = make_vec_env(env_id=args.env, env_kwargs=env_kwargs, n_envs=1)
    eval_env = make_vec_env(env_id=args.env, env_kwargs=env_kwargs, n_envs=1)
    test_env_final = make_vec_env(env_id=args.env, env_kwargs=env_kwargs, n_envs=1)
    test_env_best = make_vec_env(env_id=args.env, env_kwargs=env_kwargs, n_envs=1)

    if args.mode == 'rgb_array':
        env = VecTransposeImage(env)
        eval_env = VecTransposeImage(eval_env)
        test_env_final = VecTransposeImage(test_env_final)
        test_env_best = VecTransposeImage(test_env_best)

    a = timestamp
    b = args.instance
    c = args.algo.split('.')[0]
    d = args.mode
    e = args.env
    f = 'box' if args.box else 'discrete'
    g = 'multi' if args.multi else 'single'
    h = int(args.train_steps)

    if args.train:
        save_path = f"{a}_{b}_{c}_{d}_{e}_{f}_{g}_{h}"

        model = PPO("CnnPolicy",
                    env,
                    learning_rate=9e-5,
                    n_steps=2048,
                    batch_size=1024,
                    n_epochs=10,
                    gamma=0.99,
                    gae_lambda=0.95,
                    clip_range=0.2,
                    clip_range_vf=None,
                    ent_coef=0.0,
                    vf_coef=0.5,
                    max_grad_norm=0.5,
                    use_sde=False,
                    sde_sample_freq=- 1,
                    target_kl=None,
                    tensorboard_log=f'logs/{save_path}',
                    policy_kwargs=None,
                    verbose=1,
                    seed=42,
                    device='auto',
                    _init_setup_model=True)
        video_recorder = VideoRecorderCallback(eval_env, render_freq=5)
        eval_callback = EvalCallback(eval_env,
                                     best_model_save_path=f'./models/best_model/{save_path}',
                                     log_path='./logs/',
                                     eval_freq=10000,
                                     deterministic=True,
                                     render=False,
                                     #callback_after_eval=stop_train_callback
                                     )

        model.learn(total_timesteps=args.train_steps, callback=eval_callback, progress_bar=True)
        model.save(f"./models/{save_path}")

        del model
        env.close()
        eval_env.close()

    else:
        root = tk.Tk()
        root.withdraw()

        path = filedialog.askopenfilename().split('/')[-1]
        save_path = path.split('.')[0]
    final_model = PPO.load(f"./models/{save_path}")
    best_model = PPO.load(f"./models/best_model/{save_path}/best_model.zip")

    obs_final = test_env_final.reset()
    obs_best = test_env_best.reset()
    img_first = Image.fromarray(test_env_best.render(mode='rgb_array'))

    if isinstance(test_env_final, VecTransposeImage) or isinstance(test_env_final, DummyVecEnv):
        start_cost_final = test_env_final.get_attr("last_cost")[0]
    else:
        start_cost_final = test_env_final.last_cost

    if isinstance(test_env_final, VecTransposeImage) or isinstance(test_env_final, DummyVecEnv):
        start_cost_best = test_env_final.get_attr("last_cost")[0]
    else:
        start_cost_best = test_env_best.last_cost

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


        rewards.append([reward_final[0], reward_best[0]])
        mhc_final.append(info_final[0]['mhc'])
        mhc_best.append(info_best[0]['mhc'])

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

    imageio.mimsave(f'gifs/{save_path}_test_env.gif', images, fps=10)

    new_path = os.path.join(os.getcwd(), 'experiments', save_path + '.png')
    plt.imsave(new_path, images[-2])
    plt.imsave(os.path.join(os.getcwd(), 'experiments', 'layout_ppo.png'), imgs[-2])
    plt.imsave(os.path.join(os.getcwd(), 'experiments', 'start_layout_ppo.png'), img_first)
    with open(f'{save_path}.json', 'w') as outfile:
        json.dump(experiment_results, outfile)

