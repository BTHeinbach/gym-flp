import json
import gym_flp
import inspect
from algorithm import eval
import datetime
import argparse

from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.vec_env import VecTransposeImage, DummyVecEnv
from stable_baselines3.common.callbacks import EvalCallback, BaseCallback, StopTrainingOnNoModelImprovement

import matplotlib.pyplot as plt

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--env', type=str, default='ofp-v0', help='environment name')
    parser.add_argument('--mode', type=str, default='rgb_array', help='state representation mode')
    parser.add_argument('--instance', type=str, default='P6', help='problem instance')
    parser.add_argument('--distance', type=str, default='r', help='distance metric')
    parser.add_argument('--step_size', type=int, default=1, help='step size for ofp envs')
    parser.add_argument('--box', action="store_true",  help='input box to use box env, if omitted uses discrete')
    parser.add_argument('--multi', action="store_true", help='whether to move one or more machines per step')
    parser.add_argument('--randomize', action="store_true", help='whether to move one or more machines per step')
    parser.add_argument('--train_steps', type=int, default=1e5, help='number of training steps')
    parser.add_argument('--num_workers', type=int, default=1, help='number of parallel envs')
    parser.add_argument('--algo', type=str, default='ppo', help='Pick any one out of PPO, A2C, DQN, DDPG, SAC, TD3')
    args = parser.parse_args()
    return args


if __name__ == '__main__':
    timestamp = datetime.datetime.now().strftime("%y%m%d_%H%M")
    args = parse_args()
    # Simulate parser
    algo = args.algo

    env_kwargs = {
        'mode': args.mode,
        'instance': args.instance,
        'box': args.box,
        'multi': args.multi,
        'envId': args.env,
        'randomize': args.randomize
    }

    env = make_vec_env(env_id=args.env, env_kwargs=env_kwargs, n_envs=1)
    eval_env = make_vec_env(env_id=args.env, env_kwargs=env_kwargs, n_envs=1)

    if args.mode == 'rgb_array':
        env = VecTransposeImage(env)
        eval_env = VecTransposeImage(eval_env)

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
        raise Exception("Algorithm not recognized or supported")

    a = timestamp
    b = args.instance
    c = args.algo.split('.')[0]
    d = args.mode
    e = args.env
    f = 'box' if args.box else 'discrete'
    g = 'multi' if args.multi else 'single'
    h = int(args.train_steps)

    save_path = f"{a}_{b}_{c}_{d}_{e}_{f}_{g}_{h}"

    eval_callback = EvalCallback(eval_env,
                                 best_model_save_path=f'./models/best_model/{save_path}',
                                 log_path='./logs/',
                                 eval_freq=10000,
                                 deterministic=True,
                                 render=False,
                                 # callback_after_eval=stop_train_callback
                                 )

    f = open('./config/algo-conf.json')
    cfg = json.load(f)
    config = cfg[algo]

    model = RL('MlpPolicy', env)

    #for k, v in config.items():
    #     if hasattr(model, k):
    #         model.__setattr__(k, v)
    #     else:
    #         print("model does not have attribute ", k)

    model.learn(total_timesteps=args.train_steps, callback=eval_callback, progress_bar=True)
    model.set_env(env)
    model.__setattr__('env_kwargs', env_kwargs)
    model.save(f"./models/{save_path}")
    del RL
    env.close()
    eval_env.close()

    eval.run(save_path=save_path)


