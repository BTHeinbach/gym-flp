import gym
import gym_flp
import imageio
import datetime

import numpy as np
import matplotlib.pyplot as plt
import torch as th

from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.callbacks import EvalCallback, BaseCallback, StopTrainingOnNoModelImprovement
from stable_baselines3.common.vec_env import VecTransposeImage
from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3.common.logger import Video
from PIL import Image
from typing import Any, Dict


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


stop_train_callback = StopTrainingOnNoModelImprovement(max_no_improvement_evals=100, min_evals=10, verbose=1)

instance = 'P12'
timestamp = datetime.datetime.now().strftime("%y%m%d_%H%M")
environment = 'ofp'
algo = 'ppo'
mode = 'rgb_array'
train_steps = [10e6]
aspace = 'discrete'
multi = False
vec_env = make_vec_env('ofp-v0',
                       env_kwargs={'mode': mode, "instance": instance, "aspace": aspace, "multi": multi},
                       n_envs=1)

vec_eval_env = make_vec_env('ofp-v0',
                            env_kwargs={'mode': mode, "instance": instance, "aspace": aspace, "multi": multi},
                            n_envs=1)

vec_test_env = make_vec_env('ofp-v0',
                            env_kwargs={'mode': mode, "instance": instance, "aspace": aspace, "multi": multi},
                            n_envs=1)

wrap_env = VecTransposeImage(vec_env)
wrap_eval_env = VecTransposeImage(vec_eval_env)
test_env_final = VecTransposeImage(vec_test_env)
test_env_best = VecTransposeImage(vec_test_env)


for ts in train_steps:
    ts = int(ts)
    save_path = f"{timestamp}_{instance}_{algo}_{mode}_{environment}_{aspace}_multi_{multi}_{ts}"

    model = PPO("CnnPolicy", 
                wrap_env,
                learning_rate=8.98e-5,
                n_steps=2048, 
                batch_size=1024,
                n_epochs=20,
                gamma=0.99, 
                gae_lambda=0.95, 
                clip_range=0.1,
                clip_range_vf=None, 
                ent_coef=0.0, 
                vf_coef=0.5, 
                max_grad_norm=0.5, 
                use_sde=False, 
                sde_sample_freq=- 1, 
                target_kl=0.2,
                tensorboard_log=f'logs/{save_path}', 
                create_eval_env=False, 
                policy_kwargs=None, 
                verbose=1,
                seed=None, 
                device='cuda',
                _init_setup_model=True)
    video_recorder = VideoRecorderCallback(wrap_eval_env, render_freq=5)
    eval_callback = EvalCallback(wrap_eval_env,
                                 best_model_save_path=f'./models/best_model/{save_path}',
                                 log_path='./logs/',
                                 eval_freq=10000,
                                 deterministic=True,
                                 render=False,
                                 callback_after_eval=stop_train_callback,
                                 n_eval_episodes=20)

    model.learn(total_timesteps=ts, callback=eval_callback, progress_bar=True)
    #model.set_env(wrap_env, force_reset=True)
    model.save(f"./models/{save_path}")

    del model
    wrap_env.close()
    wrap_eval_env.close()

    final_model = PPO.load(f"./models/{save_path}")
    best_model = PPO.load(f"./models/best_model/{save_path}/best_model.zip")

    obs_final = test_env_final.reset()
    obs_best = np.array(obs_final)

    start_cost_final = test_env_final.get_attr("last_cost")[0]
    start_cost_best = test_env_best.get_attr("last_cost")[0]

    rewards = []
    mhc_final = []
    mhc_best =  []
    images = []
    gain = 0
    gains = []
    c = []
    actions = []
    dones = [False, False]
    counter = 0

    fig, axs = plt.subplots(2, 2)
    while False in dones:
        counter += 1

        if not dones[0]:
            action_final, _states_final = final_model.predict(obs_final, deterministic=True)
            obs_final, reward_final, done_final, info_final = test_env_final.step(action_final)
            img_final = Image.fromarray(test_env_final.render(mode='rgb_array'))
            dones[0] = done_final

        if not dones[1]:
            action_best, _states_final = final_model.predict(obs_best, deterministic=True)
            obs_best, reward_best, done_best, info_best = test_env_best.step(action_best)
            img_best = Image.fromarray(test_env_best.render(mode='rgb_array'))
            dones[1] = done_best

        rewards.append([reward_final[0], reward_best[0]])
        mhc_final.append(info_final[0]['mhc'])
        mhc_best.append(info_best[0]['mhc'])

        axs[0, 0].imshow(img_final)
        axs[1, 0].imshow(img_best)
        #plt.show()

        axs[0, 1].plot(np.arange(1, len(mhc_final)+1), mhc_final)
        axs[1, 1].plot(np.arange(1, len(mhc_best)+1), mhc_best)
        #fig.show()

        fig.canvas.draw()
        # Now we can save it to a numpy array.
        data = np.frombuffer(fig.canvas.tostring_rgb(), dtype=np.uint8)
        data = data.reshape(fig.canvas.get_width_height()[::-1] + (3,))

        images.append(data)
        #dones[0], dones[1] = done_final, done_best
        if counter > 100:
            print("kill process")
            break

    imageio.mimsave(f'gifs/{save_path}_test_env.gif', images, fps=10)

