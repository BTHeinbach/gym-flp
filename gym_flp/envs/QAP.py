import numpy as np
import gym
import pickle
import os
import math
import matplotlib.pyplot as plt

from gym import spaces
from numpy.random import default_rng
from PIL import Image
from gym_flp import rewards


class QapEnv(gym.Env):
    metadata = {'render.modes': ['rgb_array', 'human']}

    def __init__(self,
                 mode=None,
                 instance=None,
                 distance=None,
                 aspect_ratio=None,
                 step_size=None,
                 greenfield=None,
                 box=False,
                 multi=False):
        __location__ = os.path.realpath(os.path.join(os.getcwd(), os.path.dirname(__file__)))
        self.DistanceMatrices, self.FlowMatrices = pickle.load(
            open(os.path.join(__location__, 'instances/discrete', 'qap_matrices.pkl'), 'rb'))
        self.transport_intensity = None
        self.instance = instance
        self.mode = mode

        while not (
                self.instance in self.DistanceMatrices.keys() or self.instance in self.FlowMatrices.keys() or self.instance in [
            'Neos-n6', 'Neos-n7', 'Brewery']):
            print('Available Problem Sets:', self.DistanceMatrices.keys())
            self.instance = input('Pick a problem:').strip()

        self.D = self.DistanceMatrices[self.instance]
        self.F = self.FlowMatrices[self.instance]

        # Determine problem size relevant for much stuff in here:
        self.n = len(self.D[0])
        self.x = math.ceil((math.sqrt(self.n)))
        self.y = math.ceil((math.sqrt(self.n)))
        self.size = int(self.x * self.y)
        self.max_steps = 5 * (self.n - 1)

        self.action_space = spaces.Discrete(int((self.n ** 2 - self.n) * 0.5) + 1)

        # If you are using images as input, the input values must be in [0, 255] as the observation is normalized (dividing by 255 to have values in [0, 1]) when using CNN policies.
        if self.mode == "rgb_array":
            self.observation_space = spaces.Box(low=0, high=255, shape=(36, 36, 3),
                                                dtype=np.uint8)  # Image representation
        elif self.mode == 'human':
            self.observation_space = spaces.Box(low=1, high=self.n, shape=(self.n,), dtype=np.float32)

        self.states = {}  # Create an empty dictonary where states and their respective reward will be stored for future reference
        self.actions = self.pairwiseExchange(self.n)

        # Initialize Environment with empty state and action
        self.action = None
        self.state = None
        self.internal_state = None

        # Initialize moving target to incredibly high value. To be updated if reward obtained is smaller.

        self.movingTargetReward = np.inf
        self.MHC = rewards.mhc.MHC()  # Create an instance of class MHC in module mhc.py from package rewards

    def reset(self):
        self.step_counter = 0  # Zählt die Anzahl an durchgeführten Aktionen

        self.internal_state = default_rng().choice(range(1, self.n + 1), size=self.n, replace=False)
        self.internal_state = np.array([5, 4, 7, 1, 2, 3, 6])
        MHC, self.TM = self.MHC.compute(self.D, self.F, np.array(self.internal_state))
        self.initial_MHC = MHC
        self.last_cost = self.initial_MHC
        self.counter = 0
        self.movingTargetReward = np.inf
        state = np.array(self.internal_state) if self.mode == 'human' else np.array(self.get_image())

        return state

    def step(self, action):
        # Create new State based on action
        self.step_counter += 1
        # self.counter = 0

        fromState = np.array(self.internal_state)

        swap = self.actions[action]
        fromState[swap[0] - 1], fromState[swap[1] - 1] = fromState[swap[1] - 1], fromState[swap[0] - 1]

        MHC, self.TM = self.MHC.compute(self.D, self.F, fromState)

        if self.movingTargetReward == np.inf:
            self.movingTargetReward = MHC

        # reward = self.last_cost - MHC
        if MHC < self.movingTargetReward:
            self.counter = 0
            reward = 1
            self.movingTargetReward = MHC
            self.best_state = np.array(fromState)
        else:
            reward = 0
            self.counter += 1

        self.last_cost = MHC
        self.Actual_Minimum = self.movingTargetReward

        if action == self.action_space.n - 1:
            done = True

        else:
            done = True if self.counter > self.max_steps else False

        self.internal_state = np.array(fromState)
        state = np.array(self.internal_state) if self.mode == 'human' else np.array(self.get_image())

        return state, reward, done, {'mhc': MHC}
        # return newState, reward, done

    def render(self, mode=None):

        img = self.get_image()

        # plt.imshow(img)
        plt.axis('off')
        # plt.show()
        return np.array(img)

    def close(self):
        pass

    def pairwiseExchange(self, x):
        actions = [(i, j) for i in range(1, x) for j in range(i + 1, x + 1) if not i == j]
        actions.append((1, 1))
        return actions

        # FOR CNN #

    def get_image(self):
        rgb = np.zeros((self.x, self.y, 3), dtype=np.uint8)

        sources = np.sum(self.TM, axis=1)
        sinks = np.sum(self.TM, axis=0)

        state = self.internal_state

        R = np.array((state - np.min(state)) / (np.max(state) - np.min(state)) * 255).astype(int)
        G = np.array((sources - np.min(sources)) / (np.max(sources) - np.min(sources)) * 255).astype(int)
        B = np.array((sinks - np.min(sinks)) / (np.max(sinks) - np.min(sinks)) * 255).astype(int)

        k = 0
        a = 0
        Zeilen_ZAEHLER = 0
        for s in range(len(state)):
            rgb[k][a] = [R[s], G[s], B[s]]
            a += 1
            if a > (self.x - 1):
                Zeilen_ZAEHLER += 1
                k = Zeilen_ZAEHLER
                a = 0

        img = Image.fromarray(rgb, 'RGB')

        return img.resize((36, 36), Image.NEAREST)
