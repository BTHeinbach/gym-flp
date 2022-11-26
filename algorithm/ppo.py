import gym
import gym_flp
import imageio
import datetime

import numpy as np
import matplotlib.pyplot as plt
import torch as th

from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.callbacks import EvalCallback, BaseCallback
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

instance = 'P6'
timestamp = datetime.datetime.now().strftime("%y%m%d_%H%M")
environment = 'ofp'
algo = 'ppo'
mode = 'rgb_array'
train_steps = [2e6]
vec_env = make_vec_env('ofp-v0',
                       env_kwargs={'mode': mode, "instance": instance, "aspace": 'discrete', "multi": True},
                       n_envs=1)

vec_eval_env = make_vec_env('ofp-v0',
                            env_kwargs={'mode': mode, "instance": instance, "aspace": 'discrete', "multi": True},
                            n_envs=1)

wrap_env = VecTransposeImage(vec_env)
wrap_eval_env = VecTransposeImage(vec_eval_env)

for ts in train_steps:
    ts = int(ts)
    save_path = f"{timestamp}_{instance}_{algo}_{mode}_{environment}_movingavg_nocollisions_{ts}"

    model = PPO("CnnPolicy", 
                wrap_env,
                learning_rate=0.0003, 
                n_steps=2048, 
                batch_size=64,
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
                                 render=False)

    model.learn(total_timesteps=ts)
    model.save(f"./models/{save_path}")
    
    # model = PPO.load(f"./models/221015_2201_P6_ppo_rgb_array_ofp_movingavg_nocollisions_1000000")
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
        #for _ in range(eval_steps):
        action, _states = model.predict(obs, deterministic=True)
        actions.append(action)
        obs, reward, done, info = wrap_env.step(action)
        img = Image.fromarray(wrap_env.render(mode='rgb_array'))
        rewards.append(reward[0])
        mhc.append(info[0]['mhc'])
        c.append(info[0]['collisions'])
        images.append(img)
    
    final_cost = mhc[-1]

    print(start_cost, final_cost)
    cost_saved = final_cost-start_cost
    cost_saved_rel = 1-(start_cost/final_cost)
    ax1.plot(np.arange(1, eval_steps+1), rewards)
    ax2.plot(np.arange(1, eval_steps+1), mhc)
    imageio.mimsave(f'gifs/{save_path}_test_env.gif',
                    [np.array(img.resize((200, 200), Image.NEAREST))
                     for i, img in enumerate(images) if i % 2 == 0], fps=29)
    wrap_env.close()

    del model

    fig.show()
