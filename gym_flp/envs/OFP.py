import numpy as np
import gym
import pickle
import os
import math
import json

from gym import spaces
from numpy.random import default_rng

import gym_flp.envs.instances.envBuilder
from gym_flp import rewards, util
from gym_flp.util import preprocessing


class OfpEnv(gym.Env):
    metadata = {"render_modes": ["human", "rgb_array"], "render_fps": 4}

    '''
    - This environment class assumes a (bounded) planar area on which facilities are located on a continuum.
    - Facilities are describes by x and y centroids as well as length and width, see nomenclature below.

    Upper and lower bound for observation space:
    - min x position can be point of origin (0,0) [coordinates map to upper left corner]
    - min y position can be point of origin (0,0) [coordinates map to upper left corner]
    - min width can be smallest area divided by its length, or 1
    - min lenght can be smallest width (above) multiplied by aspect ratio
    - max x pos can be bottom right edge of grid
    - max y pos can be bottpm right edge of grid

        Nomenclature:

            plant_Y --> Width of Plant (y coordinate)
            plant_X --> Length of Plant (x coordinate)
            fac_width_y --> Width of facility/bay (y coordinate)
            fac_length_x --> Length of facility/bay (x coordinate)
            plant_area --> Area of Plant: X*Y
            fac_Area --> Area of facility x*y
            Point of origin analoguous to numpy indexing (top left corner of plant)
            beta --> aspect ratios (as alpha is reserved for learning rate)

       X --> Length
       (0|0) ____________________
       |                         |   Y
       |       x_                |   | Width
       |       |_|y              |   
       |_________________________|


      '''

    def __init__(self,
                 mode=None,
                 instance=None,
                 distance=None,
                 step_size=None,
                 greenfield=None,
                 box=False,
                 multi=False,
                 envId='ofp',
                 randomize = True):
        self.mode = mode if mode is not None else 'rgb_array'
        self.instance = instance.strip().lower() if instance is not None else 'P6'
        self.distance = distance
        self.step_size = 1 if step_size is None else step_size
        self.greenfield = False if greenfield is None else greenfield
        self.multi = multi
        __location__ = os.path.realpath(os.path.join(os.getcwd(), os.path.dirname(__file__)))

        if instance != 'custom':
            with open(os.path.join(__location__, 'instances', 'continuous.json')) as f:
                g = json.load(f)
                problem_information = g[instance]

            while not (self.instance in [key.lower() for key in g.keys()]):
                print('Could not find problem. Available problem sets:', problem_information.keys())
                instance = input('Pick a problem:').strip()
                problem_information = g[instance]
        else:
            n = int(input('Define problem size:').strip())
            problem_information = gym_flp.envs.instances.envBuilder.load_from_file(envId.split('-')[0], n)

        self.D = None
        self.F = np.array(problem_information['flows'], dtype=float)
        self.n = self.F.shape[0]

        self.fac_area = np.array(problem_information['areas'], dtype=float)
        self.fac_width_y = np.array(problem_information['widths'], dtype=float)
        self.fac_length_x = np.array(problem_information['lengths'], dtype=float)

        #self.fac_width_y = np.array([2, 4, 5, 3, 8, 6])
        #self.fac_length_x = np.array([7, 12, 5, 6, 5, 9])
        self.plant_X = int(problem_information['plantX'])
        self.plant_Y = int(problem_information['plantY'])

        if len(self.fac_width_y) == 0 or len(self.fac_length_x) == 0:
            y = [preprocessing.divisor(int(x)) for x in self.fac_area]

            y_ = [[x for x in z] for z in y]

            # Remove empty entries
            y_ = [x for x in y_ if x]
            self.fac_length_x = np.array([i[int(np.floor(len(i) / 2))] if len(i) > 1 else i[-1] for i in y_])
            # self.fac_length_x = np.random.randint(self.min_side_length * self.aspect_ratio, np.min(self.fac_area),
            #                                      size=(self.n,))
            self.fac_width_y = np.round(self.fac_area / self.fac_length_x)

        if self.plant_X is None or self.plant_Y is None:
            self.plant_area = np.sum(self.fac_area)
            # Design a squared plant layout
            self.plant_X = int(round(math.sqrt(self.plant_area),
                                     0))  # We want the plant dimensions to be integers to fit them into an image
            self.plant_Y = self.plant_X

        if self.greenfield:
            self.plant_X = 2 * self.plant_X
            self.plant_Y = 2 * self.plant_Y

        # These values need to be set manually, e.g. acc. to data from literature.
        # Following Eq. 1 in Ulutas & Kulturel-Konak (2012), the minimum side length can be determined by assuming the
        # smallest facility will occupy alone.
        # self.aspect_ratio = int(max(self.beta)) if not self.beta is None else self.aspect_ratio
        # self.min_side_length = 1
        # self.min_width = self.min_side_length * self.aspect_ratio

        # 3. Define the possible actions: 5 for each box

        # 4. Define observation_space for human and rgb_array mode
        # Formatting for the observation_space:
        # [facility y, facility x, facility width, facility length] -->
        # [self.fac_y, self.fac_x, self.fac_width_y, self.fac_length_x]

        self.lower_bounds = {'Y': np.zeros(self.n),
                             'X': np.zeros(self.n),
                             'y': self.fac_width_y,
                             'x': self.fac_length_x}

        self.upper_bounds = {'Y': self.plant_Y - self.fac_width_y,
                             'X': self.plant_X - self.fac_length_x,
                             'y': self.fac_width_y,
                             'x': self.fac_length_x}

        observation_low = np.zeros(4 * self.n)
        observation_high = np.zeros(4 * self.n)

        observation_low[0::4] = np.array([i for i in self.lower_bounds['Y']])
        observation_low[1::4] = np.array([i for i in self.lower_bounds['X']])
        observation_low[2::4] = np.array([i for i in self.lower_bounds['y']])
        observation_low[3::4] = np.array([i for i in self.lower_bounds['x']])

        observation_high[0::4] = np.array([i for i in self.upper_bounds['Y']])
        observation_high[1::4] = np.array([i for i in self.upper_bounds['X']])
        observation_high[2::4] = np.array([i for i in self.upper_bounds['y']])
        observation_high[3::4] = np.array([i for i in self.upper_bounds['x']])

        # Keep a version of this to sample initial states from in reset()
        self.state_space = spaces.Box(low=observation_low, high=observation_high, dtype=np.uint8)

        if self.mode == "rgb_array":
            self.observation_space = spaces.Box(low=0, high=255, shape=(self.plant_Y, self.plant_X, 3),
                                                dtype=np.uint8)  # Image representation, channel-last for PyTorch CNNs

        elif self.mode == "human":
            self.observation_space = spaces.Box(low=observation_low, high=observation_high,
                                                dtype=np.uint8)  # Vector representation of coordinates
        else:
            raise Exception("Could not find render mode")

        self.action_space = util.preprocessing.build_action_space(self, box, multi)

        # 5. Set some starting points
        self.reward = 0
        self.state = None  # Variable for state being returned to agent
        self.internal_state = None  # Placeholder for state variable for internal manipulation in rgb_array mode
        self.counter = 0
        self.pseudo_stability = 10  # If the reward has not improved in the last x steps, terminate the episode
        self.best_reward = None
        self.reset_counter = 0
        self.MHC = rewards.mhc.MHC()
        self.empty = np.zeros((self.plant_Y, self.plant_X, 3), dtype=np.uint8)
        self.last_cost = 0
        self.TM = None
        self.randomize = randomize

    def reset(self):
        state_prelim = self.state_space.sample()
        state_prelim[2::4] = self.fac_width_y
        state_prelim[3::4] = self.fac_length_x

        self.randomize = False
        # Create fixed positions for reset:
        if not self.randomize:
            # Check if plant can be made square
            if math.isqrt(self.n) ** 2 == self.n:
                Y = np.floor(
                    np.outer(np.arange(start=0, stop=1.001, step=1 / math.isqrt(self.n)), np.max(self.upper_bounds['Y'])))
                X = np.floor(
                    np.outer(np.arange(start=0, stop=1.001, step=1 / math.isqrt(self.n)), np.max(self.upper_bounds['X'])))

                state_prelim[0::4] = np.tile(Y.flatten()[:-1], len(Y.flatten()[:-1]))
                state_prelim[1::4] = np.tile(X.flatten()[:-1], len(X.flatten()[:-1]))

            elif len(util.preprocessing.divisor(self.n)) > 1:
                divisors = util.preprocessing.divisor(self.n)
                stepsize_index = int(np.floor(len(divisors) / 2))

                x_partition = divisors[stepsize_index]
                y_partition = divisors[stepsize_index - 1]
                X = np.floor(
                    np.outer(np.arange(start=0, stop=1.001, step=1 / x_partition), np.max(self.upper_bounds['X'])))
                Y = np.floor(
                    np.outer(np.arange(start=0, stop=1.001, step=1 / y_partition), np.max(self.upper_bounds['Y'])))

                # state_prelim[0::4] = np.tile(np.floor([(i+j)/2 for i, j in zip(Y[:, -1], Y[1:, ].flatten())]), x_partition)
                # state_prelim[1::4] = np.tile(np.floor([(i+j)/2 for i, j in zip(X[:, -1], X[1:, ].flatten())]), y_partition)

                state_prelim[0::4] = np.tile(Y.flatten()[:-1], x_partition)
                state_prelim[1::4] = np.tile(X.flatten()[:-1], y_partition)


        else:
            i = 0
            while np.sum(self.collision_test(state_prelim)) > 0:
                state_prelim = self.state_space.sample()
                state_prelim[2::4] = self.fac_width_y
                state_prelim[3::4] = self.fac_length_x
                i += 1
                if i > 10000:
                    print("no")
                    break

        self.internal_state = np.array(state_prelim)
        self.state = np.array(
            self.internal_state) if self.mode == 'human' else preprocessing.make_image_from_coordinates(
            coordinates=self.internal_state,
            canvas=self.empty,
            flows=self.F)
        self.counter = 0

        self.D = self.MHC.getDistances(state_prelim[1::4], state_prelim[0::4])
        mhc, self.TM = self.MHC.compute(self.D, self.F, np.array(range(1, self.n + 1)))
        self.last_cost = mhc
        return np.array(self.state)

    def collision_test(self, state):
        collisions = []
        n = int(len(state) / 4)
        y, x, h, b = state[0::4], state[1::4], state[2::4], state[3::4]
        mask = np.ones(n, dtype=bool)

        for i in range(n):
            A = np.zeros((self.plant_Y, self.plant_X), dtype=np.uint8)
            B = np.zeros((self.plant_Y, self.plant_X), dtype=np.uint8)

            A[y[i]:y[i] + h[i], x[i]:x[i] + b[i]] = 1

            mask[i] = False
            y_, x_, h_, b_ = y[mask], x[mask], h[mask], b[mask]

            for j in range(len(y_)):
                B[y_[j]:y_[j] + h_[j], x_[j]:x_[j] + b_[j]] = 1

            collisions.append(np.sum(A & B))
        return collisions

    def step(self, action):

        step_size = self.step_size
        temp_state = np.array(self.internal_state)  # Get copy of state to manipulate:
        old_state = np.array(self.internal_state)  # Keep copy of state to restore if boundary condition is met
        done = False
        mhcs = []
        multi = self.multi
        # print(action)
        # Disassemble action
        if isinstance(self.action_space, gym.spaces.Discrete):
            i = np.int32(np.floor(action / 4))  # Facility on which the action is

            if action != self.action_space.n - 1:
                if action % 4 == 0:
                    temp_state[4 * i] += step_size
                elif action % 4 == 1:
                    temp_state[4 * i + 1] += step_size
                if action % 4 == 2:
                    temp_state[4 * i] -= step_size
                if action % 4 == 3:
                    temp_state[4 * i + 1] -= step_size
            else:
                temp_state

        elif isinstance(self.action_space, gym.spaces.MultiDiscrete):
            for i in range(0, action.shape[0]):

                if action[i] == 0:
                    temp_state[4 * i] += step_size
                elif action[i] == 1:
                    temp_state[4 * i + 1] += step_size
                elif action[i] == 2:
                    temp_state[4 * i] -= step_size
                elif action[i] == 3:
                    temp_state[4 * i + 1] -= step_size
                elif action[i] == 4:
                    temp_state

        elif isinstance(self.action_space, gym.spaces.Box):
            if multi:
                for i in range(0, self.n):
                    a_y = np.floor(preprocessing.rescale_actions(a=-1, b=1, x_min=self.lower_bounds['Y'],
                                                                 x_max=self.upper_bounds['Y'], x=action[2 * i])).astype(
                        int)
                    a_x = np.floor(preprocessing.rescale_actions(a=-1, b=1, x_min=self.lower_bounds['X'],
                                                                 x_max=self.upper_bounds['X'],
                                                                 x=action[2 * i + 1])).astype(int)

                    temp_state[4 * i] = a_y
                    temp_state[4 * i + 1] = a_x

            else:
                i = np.floor(preprocessing.rescale_actions(a=-1, b=1, x_min=0, x_max=self.n - 1, x=action[0])).astype(
                    int)

                a_y = np.floor(preprocessing.rescale_actions(a=-1, b=1, x_min=self.lower_bounds['Y'][i],
                                                             x_max=self.upper_bounds['Y'][i], x=action[1])).astype(int)
                a_x = np.floor(preprocessing.rescale_actions(a=-1, b=1, x_min=self.lower_bounds['X'][i],
                                                             x_max=self.upper_bounds['X'][i], x=action[2])).astype(int)

                temp_state[4 * i] = a_y
                temp_state[4 * i + 1] = a_x

        else:
            raise ValueError("Received invalid action={} which is not part of the action space".format(action))

        penalty = 0

        self.D = self.MHC.getDistances(temp_state[1::4], temp_state[0::4])
        mhc, self.TM = self.MHC.compute(D=self.D, F=self.F, s=np.array(range(1, self.n + 1)))

        # #2 Test if initial state causing a collision. If yes than initialize a new state until there is no collision
        collisions = self.collision_test(
            temp_state)  # Pass every 4th item starting at 0 (x pos) and 1 (y pos) for checking
        # collision_penalty = -1 if collisions > 0 else 0

        # print(self.internal_state)
        # Make new state for observation
        self.internal_state = np.array(temp_state)  # Keep a copy of the vector representation for future steps
        self.state = preprocessing.make_image_from_coordinates(
            coordinates=np.array(self.internal_state),
            canvas=np.zeros((self.plant_Y, self.plant_X, 3), dtype=np.uint8),
            flows=self.F) if self.mode == 'rgb_array' else np.array(self.internal_state)

        # Make rewards for observation

        if not self.state_space.contains(temp_state):
            done = True
            p1 = -2
            temp_state = np.array(old_state)
        else:
            p1 = 0

        if np.sum(collisions) > 0:
            p2 = -2
            # print(np.sum(collisions))
        else:
            p2 = 0

        if mhc < self.last_cost:
            self.last_cost = mhc
            self.counter = 0
            reward = 1
        else:
            self.counter += 1
            reward = 0

        # self.counter +=1 if mhc >= self.last_cost else 0
        # self.last_cost = mhc if mhc < self.last_cost else self.last_cost
        # mhc_penalties = [1 if x < self.last_cost else 0 for x in mhcs]
        # collision_penalties = [1 if x == 0 else -1 for x in collisions]

        # Check for terminality for observation
        if self.counter >= self.pseudo_stability:
            done = True
        # elif np.sum(collisions)!=0:
        #    done = True

        return np.array(self.state), reward + p1+ p2, done, {'mhc': mhc, 'collisions': sum(collisions), 'r': reward}

    def render(self, mode=None):
        return preprocessing.make_image_from_coordinates(coordinates=self.internal_state,
                                                         canvas=255 * np.zeros((self.plant_Y, self.plant_X, 3),
                                                                              dtype=np.uint8),
                                                         flows=self.F)

    def close(self):
        pass


def getAreaData(df):
    import re

    # First check for area data
    if np.any(df.columns.str.contains('Area', na=False, case=False)):
        a = df.filter(regex=re.compile("Area", re.IGNORECASE)).to_numpy()
        # a = np.reshape(a, (a.shape[0],))

    else:
        a = None

    if np.any(df.columns.str.contains('Length', na=False, case=False)):
        l = df.filter(regex=re.compile("Length", re.IGNORECASE)).to_numpy()
        l = np.reshape(l, (l.shape[0],))

    else:
        l = None

    if np.any(df.columns.str.contains('Width', na=False, case=False)):
        w = df.filter(regex=re.compile("Width", re.IGNORECASE)).to_numpy()
        w = np.reshape(w, (w.shape[0],))

    else:
        w = None

    if np.any(df.columns.str.contains('Aspect', na=False, case=False)):
        ar = df.filter(regex=re.compile("Aspect", re.IGNORECASE)).to_numpy()
        # ar = np.reshape(a, (a.shape[0],))

    else:
        ar = None

    '''
    The following cases can apply in the implemented problem sets (as of 23.12.2020):
        1. Area data --> use as is
        2. Length and width data --> compute area as l * w
        3. Only length data --> check for minimum length or aspect ratio
        4. Several area columns (i.e. min/max) --> pick max
        5. Lower and Upper Bounds for _machine-wise_ aspect ratio --> pick random between bounds
    '''
    l_min = 1
    if a is None:
        if not l is None and not w is None:
            a = l * w
        elif not l is None:
            a = l * max(l_min, max(l))
        else:
            a = w * max(l_min, max(w))

    if not ar is None and ar.ndim > 1:
        ar = np.array([np.random.default_rng().uniform(min(ar[i]), max(ar[i])) for i in range(len(ar))])

    if not a is None and a.ndim > 1:
        # a = a[np.where(np.max(np.sum(a, axis = 0))),:]
        a = a[:, 0]  # We choose the maximum value here. Can be changed if something else is needed

    a = np.reshape(a, (a.shape[0],))

    return ar, l, w, a, l_min