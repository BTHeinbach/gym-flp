import optuna
import gym
import gym_flp
import argparse

from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.vec_env import VecTransposeImage, DummyVecEnv


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
        n_steps = trial.suggest_int("n_steps", 32, 2048, log=True)
        batch_size = trial.suggest_int("batch_size", 32, 2048, log=True)
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
                    n_steps=n_steps,
                    batch_size=batch_size,
                    n_epochs=n_epochs,
                    device='cuda',
                    _init_setup_model=True)

        model.learn(total_timesteps=args.train_steps)

        obs = eval_env.reset()

        if isinstance(eval_env, VecTransposeImage) or isinstance(eval_env, DummyVecEnv):
            start_cost_final = eval_env.get_attr("last_cost")[0]
        else:
            start_cost_final = eval_env.last_cost

        rewards = []
        mhc_final = []
        done = False
        counter = 0

        while not done:
            action_final, _states_final = final_model.predict(obs_final, deterministic=True)
            obs_final, reward_final, done_final, info_final = test_env_final.step(action_final)


if __name__ == '__main__':
    #timestamp = datetime.datetime.now().strftime("%y%m%d_%H%M")
    args = parse_args()
    study = optuna.create_study(pruner=optuna.pruners.MedianPruner())
    study.optimize(Objective(args), n_trials=20)
