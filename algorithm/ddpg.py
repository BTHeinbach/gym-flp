import gym
import gym_flp
import matplotlib.pyplot as plt
from stable_baselines3 import DDPG
from stable_baselines3.common.noise import NormalActionNoise, OrnsteinUhlenbeckActionNoise
from stable_baselines3.common.env_util import make_vec_env
import matplotlib.animation as animation
import datetime
from stable_baselines3.common.monitor import Monitor

from stable_baselines3.common.callbacks import CheckpointCallback, EvalCallback
from matplotlib.animation import FuncAnimation, PillowWriter
import numpy as np
from stable_baselines3.common.vec_env import VecEnv, VecTransposeImage #, DummyVecEnv
import imageio
from PIL import Image
import os
from stable_baselines3.common import results_plotter
from stable_baselines3.common.results_plotter import load_results, ts2xy
os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"
instance = 'P6'
timestamp = datetime.datetime.now().strftime("%y%m%d_%H%M")
environment = 'ofp'
algo = 'ddpg'
mode = 'rgb_array'
aspace = 'box'
multi = False
monitor_dir = "./Monitor/"
train_steps = np.append(np.outer(10.0 ** (np.arange(4, 6)), np.arange(1, 10, 1)).flatten(), 10 ** 6)
train_steps = [3000]
vec_env = make_vec_env('ofp-v0',
                       env_kwargs={'mode': mode, "instance": instance, "aspace": aspace, "multi": multi},
                       n_envs=1, monitor_dir = monitor_dir)
# wrap_env = VecTransposeImage(vec_env)
wrap_env = VecTransposeImage(vec_env)

vec_eval_env = make_vec_env('ofp-v0',
                            env_kwargs={'mode': mode, "instance": instance, "aspace": aspace, "multi": multi},
                            n_envs=1, monitor_dir = monitor_dir)
# wrap_eval_env = VecTransposeImage(vec_eval_env)
wrap_eval_env = VecTransposeImage(vec_eval_env)

#env = gym.make('ofp-v0', instance='P6', mode='rgb_array', aspace='continuous', multi=False)
#eval_env = gym.make('ofp-v0', instance='P6', mode='rgb_array', aspace='continuous', multi=False)

experiment_results = {}
# The noise objects for DDPG
# print(wrap_env.get_attr('action_space')[0].shape)
# n_actions = wrap_env.get_attr('action_space')[0].shape[-1]

# action_noise = NormalActionNoise(mean=np.zeros(n_actions), sigma=0.1 * np.ones(n_actions))

for ts in train_steps:
    ts = int(ts)
    # print(ts)
    save_path = f"{timestamp}_{instance}_{algo}_{mode}_{environment}_{aspace}_{multi}_movingavg_nocollisions_{ts}"

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
                batch_size=500,
                tau=0.005,
                gamma=0.99,
                train_freq=(1, 'episode'),
                gradient_steps=-1,
                # action_noise=action_noise,
                replay_buffer_class=None,
                replay_buffer_kwargs=None,
                optimize_memory_usage=False,
                policy_kwargs=None,
                seed=None,
                device='auto',
                _init_setup_model=True,
                tensorboard_log=f'logs/{save_path}',
                verbose=5,)
    # print("model learn start")
    # model.learn(total_timesteps=ts)
    model.save(f"./models/{save_path}")

    # model = PPO.load(f"./models/221015_2201_P6_ppo_rgb_array_ofp_movingavg_nocollisions_1000000")
    # model = DDPG.load(f"./models/221123_0036_P6_ddpg_rgb_array_ofp_movingavg_nocollisions_10000")
    model = DDPG.load(f"./models/221123_2255_P6_ddpg_rgb_array_ofp_box_False_movingavg_nocollisions_2500")
    
    fig, (ax1, ax2) = plt.subplots(2, 1)

    obs = wrap_env.reset()
    start_cost = wrap_env.get_attr("last_cost")[0] #wrap_env.last_cost 

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
        rewards.append(reward)
        mhc.append(info[0]['mhc'])
        c.append(info[0]['collisions'])
        gains.append(gain)
        images.append(img)

    final_cost = mhc[-1]

    cost_saved = final_cost - start_cost
    cost_saved_rel = 1 - (start_cost / final_cost)
    # print(cost_saved, cost_saved_rel, '%')
    experiment_results[ts] = [cost_saved, cost_saved_rel]
    # results_plotter.plot_results([monitor_dir], 10000, results_plotter.X_EPISODES, "DDPG Mpt")
    ax1.plot(rewards)
    ax2.plot(mhc)
    print("images",images)
    for i, img in enumerate(images):
        imageio.imsave(f'gifpics/test{i}.jpg',img)
        
    imageio.mimsave(f'gifs/{save_path}_test_env.gif',
                    [np.array(img.resize((200, 200), Image.Resampling.NEAREST)) for i, img in enumerate(images) if i % 2 == 0],
                    fps=29)
    # print(mhc, rewards, c)
    wrap_env.close()
    del model
y = np.array([i for i in experiment_results.values()])
# plt.plot(train_steps, abs(y[:, 0]), )

def moving_average(values, window):
    """
    Smooth values by doing a moving average
    :param values: (numpy array)
    :param window: (int)
    :return: (numpy array)
    """
    weights = np.repeat(1.0, window) / window
    #values.astype(float)
    return np.convolve(values, weights, 'valid')


def plot_results(log_folder, title='Learning Curve'):
    """
    plot the results
    :param log_folder: (str) the save location of the results to plot
    :param title: (str) the title of the task to plot
    """
    x, y= ts2xy(load_results(log_folder), 'episodes')
    #print(load_results(log_folder))
    print(x,y)
    y = moving_average(y, window=1)
    # Truncate x
    #print('length',len(x),len(y))
    x = x[len(x) - len(y):]
    print(x,y)
    # x_error, y_reward= ts2xy(load_results(log_folder), 'errors')
    
    # x_error = x_error[len(x_error) - len(y_reward):]
    # print(len(x_error),len(y_reward))
    plt.figure(title)
    plt.plot(x, y)
    plt.xlabel('Number of episodes')
    plt.ylabel('Rewards')
    plt.title(title + " Smoothed")
    plt.show()
    # plt.plot(x_error,y_reward)
    
    # plt.xlabel('Cumulative errors')
    # plt.ylabel('Rewards')
    # plt.title(" reward function")
    # plt.show()

# plot_results(monitor_dir)



