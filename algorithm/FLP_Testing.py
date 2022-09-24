# -*- coding: utf-8 -*-
"""
Created on Sun Jun 19 12:25:32 2022

@author: Shimraz
"""

import gym
import gym.envs
import pandas as pd
from stable_baselines3 import DDPG
import numpy as np
import matplotlib.pyplot as plt
import itertools
import os
import scipy.interpolate 
import imageio
from PIL import Image
os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"

class Test_FLP_Env:
    def __init__(self,env_name,training_timesteps,ckpt_name_prefeix,
                    log_path):
        """
        Description : 
            Training class for the agent.
            Here the agent is trained and saved with inbetween checkpoints.
            Log is also saved in a csv file.
            
        Input : 
            env_name : str
                Name of the Env to be tested on 
         
            training_timesteps : int
                Number of steps agent has trained
                
            save_freq : int
                After this many timesteps, model will be saved in between total training
                
            ckpt_name_prefeix : str
                Name prefix during checkpoint for the saved model
                
            log_path : str
                Folder where the log details are saved

        
        """
        training_timesteps = training_timesteps
        ckpt_name_prefeix = ckpt_name_prefeix
        log_path = log_path
        env = gym.make(env_name,mode='rgb_array',instance='P6')

        log_dir = log_path + "_ttsteps_"+ str(training_timesteps)                 
        
        
        load_file = log_dir+"/FLP_Training_50000_steps.zip"              
        
        # model = DDPG.load(load_file, env = env)
        model = DDPG.load(load_file, env = env)
        print(load_file)
        self.Test(env, model, training_timesteps,load_file)
        
    

    def Test(self,env, model,  train_steps, load_file):
        """
        Description : 
            This function run the test file given the model and env
            
        Input :
            env : Env
                Env to be tested on
            model : DDPG Agent
                DDPG agent used to predict the pressure
        """
        experiment_results={}
        # global model
        model = model
        print("here")
        # print("model", model)
        for ts in range(train_steps):
            ts = int(ts)
            print(ts)
            fig, (ax1,ax2) = plt.subplots(2,1)
                
            obs = env.reset()
            start_cost = getattr(env,"last_cost") #wrap_env.get_attr("last_cost")[0]
            print(start_cost)
            rewards = []
            mhc = []
            images = []
            gain = 0
            gains = []
            actions = []
            done = False
            while done != True:
                action, _states = model.predict(obs, deterministic = True)
                actions.append(action)
                obs, reward, done, info = env.step(action)
                
                gain += reward
                img =  env.render(mode='rgb_array')
                rewards.append(reward)
                # print(info)
                mhc.append(info['mhc'])
                gains.append(gain)
                images.append(img)
            print(mhc)
            print(rewards)
            final_cost = mhc[-1]
            print("finalcost",final_cost)
            cost_saved = final_cost-start_cost
            cost_saved_rel = 1-(start_cost/final_cost)
            print(cost_saved, cost_saved_rel, '%')
            experiment_results[ts]=[cost_saved, cost_saved_rel]
            ax1.plot(rewards)
            ax2.plot(mhc)
            print(actions)
            imageio.mimsave(f'gifs/test_env.gif', [np.array(img.resize((200,200),Image.NEAREST)) for i, img in enumerate(images) if i%2 == 0], fps=29)
            plt.show()
            env.close()
            del model
        y = np.array([i for i in experiment_results.values()])
        print(experiment_results, y)
        print(train_steps, abs(y[:,0]))
        fig1, axs = plt.subplots(1,1)
        axs.plot(train_steps,abs(y[:,0]),)
        plt.title("Plot flp")
        print("end")

        

if __name__ == '__main__':
    """
    Description :
        Main file to test and plot the agent's prediction.
    
    Parameters:
            
        training_timesteps : int
            Number of steps agent has trained
        
        ckpt_name_prefeix : str
            Name prefix during checkpoint for the saved model
            
        log_path : str
            Folder where the log details are saved
            
        env_name : str
            Name of the Env to be tested on
    """   
    training_timesteps = 10000   # Set accordingly
    ckpt_name_prefeix = 'FLP_Training'
    log_path = r'./FLP_Training/FLP_Training_v0'
    #Name of the environment 
    env_name = 'ofp-v1'

    testing = Test_FLP_Env(env_name,training_timesteps,ckpt_name_prefeix,log_path)
