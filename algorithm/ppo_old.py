import gym
import gym_flp
import matplotlib.pyplot as plt
from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_vec_env
import matplotlib.animation as animation
import datetime
from stable_baselines3.common.callbacks import CheckpointCallback, EvalCallback
from matplotlib.animation import FuncAnimation, PillowWriter
import numpy as np
from stable_baselines3.common.vec_env import VecEnv, VecTransposeImage, DummyVecEnv
import imageio
from PIL import Image
from stable_baselines3.common import results_plotter
from stable_baselines3.common.results_plotter import load_results, ts2xy

instance = 'P6'
timestamp = datetime.datetime.now().strftime("%y%m%d_%H%M")
environment = 'ofp'
algo = 'ppo'
mode = 'rgb_array'
aspace = 'box'
multi = False
monitor_dir = "./Monitor/"
train_steps = np.append(np.outer(10.0**(np.arange(4, 6)), np.arange(1,10,1)).flatten(), 10**6)
train_steps = [1e6]
vec_env = make_vec_env('ofp-v0', env_kwargs={'mode': mode, "instance":instance, "aspace":'discrete', "multi":True}, n_envs=1, monitor_dir = monitor_dir)
wrap_env = VecTransposeImage(vec_env)

vec_eval_env = make_vec_env('ofp-v0', env_kwargs={'mode': mode, "instance":instance, "aspace":'discrete', "multi":True}, n_envs=1, monitor_dir = monitor_dir)
wrap_eval_env = VecTransposeImage(vec_eval_env)

experiment_results={}

for ts in train_steps:
    ts = int(ts)
    #print(ts)
    save_path = f"{timestamp}_{instance}_{algo}_{mode}_{environment}_{aspace}_{multi}_movingavg_nocollisions_{ts}"
    
    eval_callback = EvalCallback(wrap_eval_env , 
                             best_model_save_path=f'./models/best_model/{save_path}',
                             log_path='./logs/',
                             eval_freq=10000,
                             deterministic=True, 
                             render=False,
                             n_eval_episodes = 10)
    
    model = PPO("CnnPolicy", 
                wrap_env,
                learning_rate=0.0003, 
                n_steps=2048, 
                batch_size=2048, 
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
                verbose=5,
                seed=None, 
                device='cuda',
                _init_setup_model=True)
    model.learn(total_timesteps=ts)
    model.save(f"./models/{save_path}")
    
    #model = PPO.load(f"./models/221015_2201_P6_ppo_rgb_array_ofp_movingavg_nocollisions_1000000")
    fig, (ax1,ax2) = plt.subplots(2,1)
    
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
    
        action, _states = model.predict(obs, deterministic = True)
        actions.append(action)
        obs, reward, done, info = wrap_env.step(action)
        gain += reward
        img =  wrap_env.render(mode='rgb_array')
        rewards.append(reward[0])
        mhc.append(info[0]['mhc'])
        c.append(info[0]['collisions'])
        gains.append(gain)
        images.append(img)
    
    final_cost = mhc[-1]
    
    cost_saved = final_cost-start_cost
    cost_saved_rel = 1-(start_cost/final_cost)
    #print(cost_saved, cost_saved_rel, '%')
    experiment_results[ts]=[cost_saved, cost_saved_rel]
    ax1.plot(rewards)
    ax2.plot(mhc)
    for i, img in enumerate(images):
        imageio.imsave(f'gifpics/test{i}.jpg',img)
    imageio.mimsave(f'gifs/{save_path}_test_env.gif', [np.array(img.resize((200,200),Image.NEAREST)) for i, img in enumerate(images) if i%2 == 0], fps=29)
    print(mhc, rewards, c)
    vec_eval_env.close()
    del model
y = np.array([i for i in experiment_results.values()])
plt.plot(train_steps,abs(y[:,0]),)

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

plot_results(monitor_dir)
