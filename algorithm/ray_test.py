# Import the RL algorithm (Algorithm) we would like to use.
from gym_flp.envs import OfpEnv
from ray.rllib.algorithms.ppo import PPO
# Configure the algorithm.
from ray.tune.registry import register_env

env_creator = {
    'instance': 'P6'
}

def env_creator(env_config):
    return OfpEnv()  # return an env instance

register_env("flp", env_creator)


config = {
    # Environment (RLlib understands openAI gym registered strings).
    # Use 2 environment workers (aka "rollout workers") that parallelly
    # collect samples from their own environment clone(s).
    "num_workers": 1,
    # Change this to "framework: torch", if you are using PyTorch.
    # Also, use "framework: tf2" for tf2.x eager execution.
    "framework": "torch",
    # Tweak the default model provided automatically by RLlib,
    # given the environment's observation- and action spaces.
    "model": {
        "fcnet_hiddens": [64, 64],
        "fcnet_activation": "relu",
    },
    # Set up a separate evaluation worker set for the
    # `algo.evaluate()` call after training (see below).
    "evaluation_num_workers": 1,
    # Only for evaluation runs, render the env.
    "evaluation_config": {
        "render_env": True,
    },
}

# Create our RLlib Trainer.
algo = PPO(env='flp')