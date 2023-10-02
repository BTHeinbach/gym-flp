import json
import os
import gym_flp
import inspect
import datetime
import argparse
import optuna

from algorithm import eval
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.vec_env import VecTransposeImage, DummyVecEnv
from stable_baselines3.common.callbacks import EvalCallback, BaseCallback, StopTrainingOnNoModelImprovement
from optuna.visualization import plot_contour
from optuna.visualization import plot_edf
from optuna.visualization import plot_intermediate_values
from optuna.visualization import plot_optimization_history
from optuna.visualization import plot_parallel_coordinate
from optuna.visualization import plot_param_importances
from optuna.visualization import plot_slice

import matplotlib.pyplot as plt

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--env', type=str, default='ofp-v0', help='environment name')
    parser.add_argument('--mode', type=str, default='human', help='state representation mode')
    parser.add_argument('--instance', type=str, default='P6', help='problem instance')
    parser.add_argument('--distance', type=str, default='r', help='distance metric')
    parser.add_argument('--step_size', type=int, default=1, help='step size for ofp envs')
    parser.add_argument('--box', action="store_true",  help='input box to use box env, if omitted uses discrete')
    parser.add_argument('--multi', action="store_true", help='whether to move one or more machines per step')
    parser.add_argument('--randomize', action="store_true", help='whether to move one or more machines per step')
    parser.add_argument('--num_workers', type=int, default=1, help='number of parallel envs')
    parser.add_argument('--algo', type=str, default='dqn', help='Pick any one out of PPO, A2C, DQN, DDPG, SAC, TD3')
    parser.add_argument('--train_steps', type=list, default=[1e5], help='number of training steps')
    parser.add_argument('--learning_rate', type=list, default=[1e-6, 1e-1], help='number of training steps')
    parser.add_argument('--policy', type=list, default=['MlpPolicy'], help='Policy to use for training')
    parser.add_argument('--batch_size', type=list, default=[10, 1000], help='Policy to use for training')
    args = parser.parse_args()
    return args

class Objective:
    def __init__(self, args):
        # Hold this implementation specific arguments as the fields of the class.
        self.env = args.env
        self.env_kwargs = {
            'mode': args.mode,
            'instance': args.instance,
            'box': args.box,
            'multi': args.multi,
            'lr': args.learning_rate,
            'policy': args.policy,
            'bs': args.batch_size
        }
    
    def __call__(self, trial):
        # Calculate an objective value by using the extra arguments.
        #n_steps = trial.suggest_categorical("n_steps", [16 ,32, 64, 128, 256, 1024, 2048, 4096])
        batch_size = trial.suggest_int("batch_size", self.env_kwargs['bs'][0], self.env_kwargs['bs'][-1], log=True)
        #n_epochs = trial.suggest_int("n_epochs", 1, 101, step=5)

        # Floating point parameter (log)
        learning_rate = trial.suggest_float("learning_rate", self.env_kwargs['lr'][0], self.env_kwargs['lr'][-1], log=True)

        print(batch_size, learning_rate)
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
        h = int(args.train_steps[-1])
        i = trial.number

        save_path = f"{a}_{b}_{c}_{d}_{e}_{f}_{g}_{h}_{i}"
        trg_path = os.path.dirname(os.path.abspath(__file__))
        eval_callback = EvalCallback(eval_env,
                                    best_model_save_path=f'{trg_path}/models/best_model/{save_path}',
                                    log_path='{trg_path}/logs/',
                                    eval_freq=10000,
                                    deterministic=True,
                                    render=False,
                                    # callback_after_eval=stop_train_callback
                                    )

        cfg = open(f'{trg_path}/config/algo-conf.json')
        cfg = json.load(cfg)
        config = cfg[algo]

        model = RL(args.policy[0], 
                   env, 
                   seed=42, 
                   learning_rate=learning_rate,
                   batch_size = batch_size)

        #for k, v in config.items():
        #     if hasattr(model, k):
        #         model.__setattr__(k, v)
        #     else:
        #         print("model does not have attribute ", k)

        model.learn(total_timesteps=args.train_steps[-1], callback=eval_callback, progress_bar=True)
        model.set_env(env)
        model.__setattr__('env_kwargs', env_kwargs)
        model.save(f"{trg_path}/models/{save_path}")
        del RL
        env.close()
        eval_env.close()

        experiment_result =  { 
            trial.number: eval.run(save_path=save_path)
        }

        with open(f'{save_path}.json', 'w') as outfile:
            json.dump(experiment_result, outfile)

        final_start = experiment_result[trial.number]['start_cost_final_model']
        best_start = experiment_result[trial.number]['start_cost_best_model']

        
        final_end = experiment_result[trial.number]['cost_final'][-1]
        best_end = experiment_result[trial.number]['cost_best'][-1]

        print('final:', (final_start-final_end)/final_start, 'best:', (best_start-best_end)/best_start)
        return (final_start-final_end)/final_start


if __name__ == '__main__':
    timestamp = datetime.datetime.now().strftime("%y%m%d_%H%M")
    args = parse_args()

    study = optuna.create_study(direction="maximize",
                                sampler=optuna.samplers.TPESampler(seed=42))
    optuna.logging.set_verbosity(optuna.logging.INFO)
    study.optimize(Objective(args), n_trials=20)
    x=plot_optimization_history(study)
    y=plot_slice(study)
    x.show()
    y.show()
    #plot_intermediate_values(study)

