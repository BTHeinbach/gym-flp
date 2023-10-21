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
    parser.add_argument('--instance', type=str, default='P12', help='problem instance')
    parser.add_argument('--distance', type=str, default='r', help='distance metric')
    parser.add_argument('--step_size', type=int, default=2, help='step size for ofp envs')
    parser.add_argument('--box', action="store_true",  help='input box to use box env, if omitted uses discrete')
    parser.add_argument('--multi', action="store_true", help='whether to move one or more machines per step')
    parser.add_argument('--randomize', action="store_true", help='whether to move one or more machines per step')
    parser.add_argument('--num_workers', type=int, default=1, help='number of parallel envs')
    parser.add_argument('--algo', action='append', default=['ppo'], help='Pick any one out of PPO, A2C, DQN, DDPG, SAC, TD3')
    parser.add_argument('--train_steps', action='append', default=[1e5], help='number of training steps')
    parser.add_argument('--learning_rate', type=list, default=[9e-5], help='number of training steps')
    parser.add_argument('--policy', type=list, default=['MlpPolicy'], help='Policy to use for training')
    parser.add_argument('--batch_size', type=list, default=[32], help='Size of the mini-batch')
    parser.add_argument('--learning_starts', type=list, default=[0.1], help='Determines after which percentage of buffer size learning begins')
    parser.add_argument('--buffer', type=list, default=[200000], help='Size of the Replay Buffer')
    parser.add_argument('--tau', type=list, default=[0.005], help='Update Fraction')
    parser.add_argument('--gamma', type=list, default=[0.99], help='Discount Factor')
    parser.add_argument('--epochs', type=list, default=[10], help='Number of epochs')
    parser.add_argument('--n_steps', type=list, default=[32], help='Number of trainings steps to forecast in gradient methods')




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
        n_steps = trial.suggest_int("n_steps", args.n_steps[0], args.n_steps[-1])
        n_epochs = trial.suggest_int("n_epochs", args.epochs[0], args.epochs[-1])
        batch_size = trial.suggest_int("batch_size", int(args.batch_size[0]), int(args.batch_size[-1]), log=True)
        policy = trial.suggest_categorical("policy", args.policy)
        learning_rate = trial.suggest_float("learning_rate",  args.learning_rate[0],  args.learning_rate[-1], log=True)
        learning_starts = trial.suggest_float("learning_states", args.learning_starts[0], args.learning_starts[-1])
        buffer_size = trial.suggest_int('buffer_size', args.buffer[0], args.buffer[-1])
        tau = trial.suggest_float("tau", args.tau[0], args.tau[-1])
        gamma = trial.suggest_float("gamma", float(args.gamma[0]), float(args.gamma[-1]))
        
        algo_options = [args.algo if len(args.algo)>2 else args.algo[-1]]
        algo = trial.suggest_categorical("algo", [args.algo if len(args.algo)>2 else args.algo[-1]])
        #print(batch_size, learning_rate)
        #print(args.train_steps, int(args.train_steps[0]), int(args.train_steps[-1]), type(args.train_steps[0]))
        train_steps = trial.suggest_int('train_steps', int(args.train_steps[0]), int(args.train_steps[-1]), log=True)
        
        
        
        # Simulate parser
        # algo = args.algo if len(args.algo)<1 else args.algo[0]

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

        a = timestamp
        b = args.instance
        c = algo
        d = args.mode
        e = args.env
        f = 'box' if args.box else 'discrete'
        g = 'multi' if args.multi else 'single'
        h = train_steps
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

        # cfg = open(f'{trg_path}/config/algo-conf.json')
        # cfg = json.load(cfg)
        # config = cfg[algo]

        if algo == 'ppo':
            from stable_baselines3 import PPO
            model = PPO(policy,
                    env,
                    learning_rate=learning_rate,
                    n_steps=n_steps,
                    batch_size=batch_size,
                    n_epochs=n_epochs,
                    gamma=gamma,
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
                    seed=None,
                    device='auto',
                    _init_setup_model=True)
        elif algo == 'a2c':
            from stable_baselines3 import A2C
            model = A2C(policy,
                    env,
                    learning_rate=learning_rate,
                    n_steps=n_steps,
                    gamma=gamma,
                    gae_lambda=1.0,
                    ent_coef=0.0,
                    vf_coef=0.5,
                    max_grad_norm=0.5,
                    rms_prop_eps=1e-05,
                    use_rms_prop=True,
                    use_sde=False,
                    sde_sample_freq=-1,
                    normalize_advantage=True,
                    policy_kwargs=None,
                    verbose=1,
                    seed=None,
                    device='auto',
                    _init_setup_model=True,
                    tensorboard_log=f'logs/{save_path}')
        elif algo == 'dqn':
            from stable_baselines3 import DQN
            model = DQN(policy,
                    env,
                    learning_rate=learning_rate,
                    buffer_size=buffer_size,
                    learning_starts=buffer_size*learning_starts,
                    batch_size=batch_size,
                    tau=tau,
                    gamma=gamma,
                    train_freq=4,
                    gradient_steps=1,
                    replay_buffer_class=None,
                    replay_buffer_kwargs=None,
                    optimize_memory_usage=False,
                    target_update_interval=10000,
                    exploration_fraction=0.9,
                    exploration_initial_eps=1.0,
                    exploration_final_eps=0.05,
                    max_grad_norm=10,
                    policy_kwargs=None,
                    verbose=1,
                    seed=None,
                    device='auto',
                    _init_setup_model=True,
                    tensorboard_log=f'logs/{save_path}')
        elif algo == 'sac':
            from stable_baselines3 import SAC
            model = SAC(policy,
                    env,
                    learning_rate=learning_rate,
                    buffer_size=buffer_size,
                    learning_starts=buffer_size*learning_starts,
                    batch_size=batch_size,
                    tau=tau,
                    gamma=gamma,
                    train_freq=1,
                    gradient_steps=1,
                    action_noise=None,
                    replay_buffer_class=None,
                    replay_buffer_kwargs=None,
                    optimize_memory_usage=False,
                    ent_coef='auto',
                    target_update_interval=1,
                    target_entropy='auto',
                    use_sde=False,
                    sde_sample_freq=-1,
                    use_sde_at_warmup=False,
                    policy_kwargs=None,
                    verbose=1,
                    seed=None,
                    device='auto',
                    _init_setup_model=True,
                    tensorboard_log=f'logs/{save_path}')
        elif algo == 'ddpg':
            from stable_baselines3 import DDPG
            from stable_baselines3.common.noise import NormalActionNoise
            import numpy as np

            n_actions = env.get_attr('action_space')[0].shape[-1]
            action_noise = NormalActionNoise(mean=np.zeros(n_actions), sigma=0.1 * np.ones(n_actions))

            model = DDPG(args.policy[0],
                     env,
                     learning_rate=learning_rate,
                     buffer_size=buffer_size,
                     learning_starts=buffer_size*learning_starts,
                     batch_size=batch_size,
                     tau=tau,
                     gamma=gamma,
                     train_freq=(10, 'episode'),
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
                     verbose=5, )
        elif algo == 'td3':
            from stable_baselines3 import TD3

            model = TD3(args.policy[0],
                    env,
                    learning_rate=learning_rate,
                    buffer_size=buffer_size,
                    learning_starts=buffer_size*learning_starts,
                    batch_size=batch_size,
                    tau=tau,
                    gamma=gamma,
                    train_freq=(10, 'episode'),
                    gradient_steps=-1,
                    action_noise=None,
                    replay_buffer_class=None,
                    replay_buffer_kwargs=None,
                    optimize_memory_usage=False,
                    policy_delay=2,
                    target_policy_noise=0.2,
                    target_noise_clip=0.5,
                    policy_kwargs=None,
                    verbose=2,
                    seed=None,
                    device='auto',
                    _init_setup_model=True,
                    tensorboard_log=f'logs/{save_path}')
        else:
            raise Exception("Algorithm not recognized or supported")
        
        model.learn(total_timesteps=train_steps, 
                    callback=eval_callback, 
                    # progress_bar=True
                    )
        model.set_env(env)
        model.__setattr__('env_kwargs', env_kwargs)
        model.save(f"{trg_path}/models/{save_path}")
        del model
        env.close()
        eval_env.close()

        experiment_result =  { 
            trial.number: eval.run(save_path=save_path)
        }

        with open(f'{trg_path}/experiments/{save_path}.json', 'w') as outfile:
            json.dump(experiment_result, outfile)

        final_start = experiment_result[trial.number]['start_cost_final_model']
        best_start = experiment_result[trial.number]['start_cost_best_model']

        
        final_end = experiment_result[trial.number]['cost_final'][-1]
        best_end = experiment_result[trial.number]['cost_best'][-1]

        print('final:', (final_start-final_end)/final_start, 'best:', (best_start-best_end)/best_start)
        return (best_start-best_end)/best_start


if __name__ == '__main__':
    timestamp = datetime.datetime.now().strftime("%y%m%d_%H%M")
    args = parse_args()

    study = optuna.create_study(direction="maximize",
                                sampler=optuna.samplers.TPESampler(seed=None))
    optuna.logging.set_verbosity(optuna.logging.INFO)
    study.optimize(Objective(args), n_trials=50)
    x=plot_optimization_history(study)
    y=plot_slice(study)
    x.show()
    y.show()
    #plot_intermediate_values(study)

