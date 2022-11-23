import gym
import gym_flp
import matplotlib.pyplot as plt
from stable_baselines3 import DDPG
from stable_baselines3.common.noise import NormalActionNoise, OrnsteinUhlenbeckActionNoise
from stable_baselines3.common.env_util import make_vec_env
import matplotlib.animation as animation
import datetime
from stable_baselines3.common.callbacks import CheckpointCallback, EvalCallback
from matplotlib.animation import FuncAnimation, PillowWriter
import numpy as np
from stable_baselines3.common.vec_env import VecEnv, VecTransposeImage, DummyVecEnv
import imageio
from PIL import Image

instance = 'P6'
timestamp = datetime.datetime.now().strftime("%y%m%d_%H%M")
environment = 'ofp'
algo = 'ddpg'
mode = 'rgb_array'
multi = True
aspace = 'box'
train_steps = np.append(np.outer(10.0 ** (np.arange(4, 6)), np.arange(1, 10, 1)).flatten(), 10 ** 6)
train_steps = [1e5]
vec_env = make_vec_env('ofp-v0',
                       env_kwargs={'mode': mode, "instance": instance, "aspace": aspace, "multi": multi},
                       n_envs=1)
wrap_env = VecTransposeImage(vec_env)

vec_eval_env = make_vec_env('ofp-v0',
                            env_kwargs={'mode': mode, "instance": instance, "aspace": aspace, "multi": multi},
                            n_envs=1)
wrap_eval_env = VecTransposeImage(vec_eval_env)

#env = gym.make('ofp-v0', instance='P6', mode='rgb_array', aspace='continuous', multi=False)
#eval_env = gym.make('ofp-v0', instance='P6', mode='rgb_array', aspace='continuous', multi=False)

experiment_results = {}

# The noise objects for DDPG
n_actions = wrap_env.get_attr('action_space')[0].shape[-1]
action_noise = NormalActionNoise(mean=np.zeros(n_actions), sigma=0.1 * np.ones(n_actions))

for ts in train_steps:
    ts = int(ts)
    # print(ts)
    save_path = f"{timestamp}_{instance}_{algo}_{mode}_{aspace}_multi-{multi}_{ts}"

    eval_callback = EvalCallback(wrap_eval_env,
                                 best_model_save_path=f'./models/best_model/{save_path}',
                                 log_path='./logs/',
                                 eval_freq=10000,
                                 deterministic=True,
                                 render=False,
                                 n_eval_episodes=10)

    model = DDPG("CnnPolicy",
                wrap_env,
                learning_rate=0.001,
                buffer_size=10000,
                learning_starts=100,
                batch_size=100,
                tau=0.005,
                gamma=0.99,
                train_freq=(1, 'episode'),
                gradient_steps=-1,
                action_noise=action_noise,
                replay_buffer_class=None,
                replay_buffer_kwargs=None,
                optimize_memory_usage=False,
                policy_kwargs=None,
                seed=None,
                device='auto',
                _init_setup_model=True,
                tensorboard_log=f'logs/{save_path}',
                verbose=5,)
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
    while done != True:
        action, _states = model.predict(obs, deterministic=True)
        actions.append(action)
        obs, reward, done, info = wrap_env.step(action)
        gain += reward
        img = wrap_env.render(mode='rgb_array')
        rewards.append(reward[0])
        mhc.append(info[0]['mhc'])
        c.append(info[0]['collisions'])
        gains.append(gain)
        images.append(img)

    final_cost = mhc[-1]

    cost_saved = final_cost - start_cost
    cost_saved_rel = 1 - (start_cost / final_cost)
    # print(cost_saved, cost_saved_rel, '%')
    experiment_results[ts] = [cost_saved, cost_saved_rel]
    ax1.plot(rewards)
    ax2.plot(mhc)
    imageio.mimsave(f'gifs/{save_path}_test_env.gif',
                    [np.array(img.resize((200, 200), Image.NEAREST)) for i, img in enumerate(images) if i % 2 == 0],
                    fps=29)
    print(mhc, rewards, c)
    vec_eval_env.close()
    del model
y = np.array([i for i in experiment_results.values()])
plt.plot(train_steps, abs(y[:, 0]), )
