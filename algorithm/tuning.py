import optuna
import gym
import gym_flp
import argparse

from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.vec_env import VecTransposeImage, DummyVecEnv
from optuna.visualization import plot_contour
from optuna.visualization import plot_edf
from optuna.visualization import plot_intermediate_values
from optuna.visualization import plot_optimization_history
from optuna.visualization import plot_parallel_coordinate
from optuna.visualization import plot_param_importances
from optuna.visualization import plot_slice

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--env', type=str, default='ofp-v0', help='environment name')
    parser.add_argument('--mode', type=str, default='rgb_array', help='state representation mode')
    parser.add_argument('--instance', type=str, default='P6', help='problem instance')
    parser.add_argument('--distance', type=str, default='r', help='distance metric')
    parser.add_argument('--step_size', type=int, default=1, help='step size for ofp envs')
    parser.add_argument('--box', action="store_true",  help='input box to use box env, if omitted uses discrete')
    parser.add_argument('--multi', action="store_true", help='whether to move one or more machines per step')
    parser.add_argument('--train_steps', type=int, default=1e5, help='number of training steps')
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
            'multi': args.multi
        }



    def __call__(self, trial):
        # Calculate an objective value by using the extra arguments.
        n_steps = trial.suggest_categorical("n_steps", [16 ,32, 64, 128, 256, 1024, 2048, 4096])
        batch_size = trial.suggest_categorical("batch_size", [16 ,32, 64, 128, 256, 1024, 2048, 4096])
        n_epochs = trial.suggest_int("n_epochs", 1, 101, step=5)

        # Floating point parameter (log)
        learning_rate = trial.suggest_float("learning_rate", 1e-5, 1e-2, log=True)

        env = make_vec_env(env_id=args.env, env_kwargs=self.env_kwargs, n_envs=1)
        eval_env = make_vec_env(env_id=args.env, env_kwargs=self.env_kwargs, n_envs=1)


        if args.mode == 'rgb_array':
            env = VecTransposeImage(env)
            eval_env = VecTransposeImage(eval_env)

        model = PPO("CnnPolicy",
                    env,
                    learning_rate=learning_rate,
                    n_steps=2048,
                    batch_size=32,
                    n_epochs=10,
                    device='cuda',
                    seed=42,
                    _init_setup_model=True)

        model.learn(total_timesteps=args.train_steps, progress_bar=True)

        obs = eval_env.reset()

        if isinstance(eval_env, VecTransposeImage) or isinstance(eval_env, DummyVecEnv):
            start_cost_final = eval_env.get_attr("last_cost")[0]
        else:
            start_cost_final = eval_env.last_cost

        rewards = []
        mhc = []
        done = False
        counter = 0

        while not done:
            action, _states = model.predict(obs, deterministic=True)
            obs, reward, done, info = eval_env.step(action)

            mhc.append(info[0]['mhc'])
            rewards.append(reward[0])

        return (start_cost_final-mhc[-1])/start_cost_final


if __name__ == '__main__':
    #timestamp = datetime.datetime.now().strftime("%y%m%d_%H%M")
    args = parse_args()
    study = optuna.create_study(direction="maximize",
    sampler=optuna.samplers.TPESampler(seed=42))
    optuna.logging.set_verbosity(optuna.logging.INFO)
    study.optimize(Objective(args), n_trials=20)
    plot_optimization_history(study)
    plot_intermediate_values(study)

