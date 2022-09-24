# -*- coding: utf-8 -*-
"""
Created on Wed May 11 11:34:23 2022

@author: Shimraz
"""

# import gym
# import gym_flp
# env = gym.make("ofp-v1",mode='rgb_array',instance='P6')
# obs = env.reset()
# for _ in range(5):
#     # env.render()
#     # print(env.action_space)
#     print("-----------------")
    
#     # env.step(ss) # take a random action
#     done = False
#     while done != True:
#         # action, _states = model.predict(obs, deterministic = True)
#         ss = env.action_space.sample()
#         # print("sss",ss)
#         obs, reward, done, info = env.step(ss)

#         img =  env.render(mode='rgb_array')
# env.close()

import gym
import gym.envs
import numpy as np
import matplotlib.pyplot as plt
from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3.ddpg.policies import MlpPolicy, CnnPolicy
from stable_baselines3 import PPO
from stable_baselines3.common import results_plotter
from stable_baselines3.common.callbacks import CheckpointCallback, EvalCallback
from stable_baselines3.common.monitor import Monitor
import os
from stable_baselines3.common.results_plotter import load_results, ts2xy
os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"

class Train_FLP_Env:
    def __init__(self,env_name,training_timesteps, ckpt_name_prefeix,
                    log_path,save_freq):
        """
        Description : 
            Training class for the agent.
            Here the agent is trained and saved with inbetween checkpoints.
            Log is also saved in a csv file.
            
        Input : 
            env_name : str
                Name of the Env to be trained on 
                
            save_freq : int
                After this many timesteps, model will be saved in between total training
                
            ckpt_name_prefeix : str
                Name prefix during checkpoint for saving the model
                
            log_path : str
                Folder where the log details will be saved
            
            database_file : str
                File name for saving data with csv extension
        
        """
        ckpt_name_prefeix = ckpt_name_prefeix
        timestamp = datetime.datetime.now().strftime("%y%m%d_%H%M")
        log_path = log_path
        env = gym.make(env_name,mode='rgb_array',instance='P6')   # defining model
        log_dir = log_path + "_ttsteps_"+ str(training_timesteps)
        os.makedirs(log_dir, exist_ok=True)
        #wrapping model to monitor class for training data
        env = Monitor(env, log_dir) 
        
        # Save a checkpoint every 5000 steps
        checkpoint_callback = CheckpointCallback(save_freq=save_freq, save_path=log_dir,
                                                 name_prefix=ckpt_name_prefeix)
        
        save_path = f"{timestamp}_{instance}_{algo}_{mode}_{environment}_movingavg_nocollisions_{ts}"
    
        eval_callback = EvalCallback(env , 
                                  best_model_save_path=f'./models/best_model/{save_path}',
                                  log_path='./logs/', 
                                  eval_freq=10000,
                                  deterministic=True, 
                                  render=False,
                                  n_eval_episodes = 5)
        
        model = PPO("CnnPolicy", 
                    env, 
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
                    verbose=1, 
                    seed=None, 
                    device='cuda',
                    _init_setup_model=True)
        model.learn(total_timesteps=ts, callback=eval_callback)
        model.save(f"./models/{save_path}")
        # Non smooth learning curve
        results_plotter.plot_results([log_dir], 10000, results_plotter.X_EPISODES, "DDPG Mpt")
        #Evaluation of the model
        evaluate_policy(model, env, n_eval_episodes=10)   # Evaluate the model
    
  
    def moving_average(self,values, window):
        """
        Description : 
            Smooth values by doing a moving average
            
        Input : 
            values: numpy array
                Rewards for corresponding episodes
            window: int
                Window size
        Output : numpy array
            Smoothed values of rewards
        """
        weights = np.repeat(1.0, window) / window
        #values.astype(float)
        return np.convolve(values, weights, 'valid')
    
    
    def plot_results(self,log_folder, title='Learning Curve'):
        """
        Description :
            Plot the results of the training
            
        Input : 
            log_folder : str 
                The save location of the results to plot
            title: str 
                The title of the task to plot
                
        Output : 
            Plot of  Smoothed Learning Curve
        """
        x, y= ts2xy(load_results(log_folder), 'episodes')
        y = self.moving_average(y, window=5)
        # Truncate x
        x = x[len(x) - len(y):]
        plt.figure(title)
        plt.plot(x, y)
        plt.xlabel('Number of episodes')
        plt.ylabel('Rewards')
        plt.title(title + " Smoothed")
        plt.show()
    
if __name__ == '__main__':
    """
    Description :
        Main file to train and save the agent.
    
    Parameters:
        training_timesteps : int
            Number of steps for the agent to train
            
            
        save_freq : int
            After this many timesteps, model will be saved in between total training
            
        ckpt_name_prefeix : str
            Name prefix during checkpoint for saving the model
            
        log_path : str
            Folder where the log details will be saved
            
        env_name : str
            Name of the Env to be trained on
    """
    training_timesteps = 50000   # Set accordingly

    save_freq=10000
    ckpt_name_prefeix = 'FLP_Training'
    log_path = r'./FLP_Training/FLP_Training_v0'
    #Name of the environment 
    env_name = 'ofp-v1'
    training = Train_FLP_Env(env_name,
                    training_timesteps, ckpt_name_prefeix, 
                    log_path, save_freq)
    
    # log_dir = training['log_dir']
    log_dir = log_path + "_ttsteps_"+ str(training_timesteps)
    #Plotting the Learning Curve 
    training.plot_results(log_dir)
"""    
def moving_average(values, window):
    """
"""   Description : 
        Smooth values by doing a moving average
        
    Input : 
        values: numpy array
            Rewards for corresponding episodes
        window: int
            Window size
    Output : numpy array
        Smoothed values of rewards"""
"""
    weights = np.repeat(1.0, window) / window
    #values.astype(float)
    return np.convolve(values, weights, 'valid')


def plot_results(log_folder, title='Learning Curve'):
    """
"""    Description :
        Plot the results of the training
        
    Input : 
        log_folder : str 
            The save location of the results to plot
        title: str 
            The title of the task to plot
            
    Output : 
        Plot of  Smoothed Learning Curve"""
"""
    x, y= ts2xy(load_results(log_folder), 'episodes')
    y = moving_average(y, window=5)
    # Truncate x
    x = x[len(x) - len(y):]
    plt.figure(title)
    plt.plot(x, y)
    plt.xlabel('Number of episodes')
    plt.ylabel('Rewards')
    plt.title(title + " Smoothed")
    plt.show()

log_path = r'./FLP_Training/FLP_Training_v0'
log_dir = log_path + "_ttsteps_"+ str(50000)
plot_results(log_dir)"""