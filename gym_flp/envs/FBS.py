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


class FbsEnv(gym.Env):
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
        self.problems, self.FlowMatrices, self.sizes, self.LayoutWidths, self.LayoutLengths = pickle.load(
            open(os.path.join(__location__,
                              'instances/continual', 'cont_instances.pkl'), 'rb'))
        self.mode = mode

        self.instance = instance
        while not (self.instance in self.FlowMatrices.keys() or self.instance in ['Brewery']):
            print('Available Problem Sets:', self.FlowMatrices.keys())
            self.instance = input('Pick a problem:').strip()

        self.F = self.FlowMatrices[self.instance]
        self.n = self.problems[self.instance]
        self.AreaData = self.sizes[self.instance]

        # Obtain size data: FBS needs a length and area
        self.beta, self.l, self.w, self.a, self.min_side_length = getAreaData(
            self.AreaData)  # Investigate available area data and compute missing values if needed

        '''
        Nomenclature:

            W --> Width of Plant (y coordinate)
            L --> Length of Plant (x coordinate)
            w --> Width of facility/bay (x coordinate)
            l --> Length of facility/bay (y coordinate)
            A --> Area of Plant
            a --> Area of facility
            Point of origin analoguous to numpy indexing (top left corner of plant)
            beta --> aspect ratios (as alpha is reserved for learning rate)
        '''

        # if self.l is None or self.w is None:
        # self.l = np.random.randint(max(self.min_side_length, np.min(self.a)/self.min_side_length), max(self.min_side_length, np.min(self.a)/self.min_side_length), size=(self.n,))
        #    self.l = np.sqrt(self.A/self.aspect_ratio)
        #    self.w = np.round(self.a/self.l)

        # Check if there are Layout Dimensions available, if not provide enough (sqrt(a)*1.5)
        if self.instance in self.LayoutWidths.keys() and self.instance in self.LayoutLengths.keys():
            self.L = int(
                self.LayoutLengths[self.instance])  # We need both values to be integers for converting into image
            self.W = int(self.LayoutWidths[self.instance])
        else:
            self.A = np.sum(self.a)

            # Design a squared plant layout
            self.L = int(
                round(math.sqrt(self.A), 0))  # We want the plant dimensions to be integers to fit them into an image
            self.W = self.L

            # Design a layout with l = 1,5 * w
            # self.L = divisor(int(self.A))
            # self.W = self.A/self.L

        # These values need to be set manually, e.g. acc. to data from literature. Following Eq. 1 in Ulutas & Kulturel-Konak (2012), the minimum side length can be determined by assuming the smallest facility will occupy alone.
        self.aspect_ratio = int(max(self.beta)) if not self.beta is None else 1
        self.min_length = np.min(self.a) / self.L
        self.min_width = np.min(self.a) / self.W

        # We define minimum side lengths to be 1 in order to be displayable in array
        self.min_length = 1
        self.min_width = 1

        self.action_space = spaces.Discrete(5)  # Taken from doi:10.1016/j.engappai.2020.103697
        self.actions = {0: 'Randomize', 1: 'Bit Swap', 2: 'Bay Exchange', 3: 'Inverse', 4: 'Idle'}
        # self.state_space = spaces.Box(low=1, high = self.n, shape=(self.n,), dtype=np.int)
        self.bay_space = spaces.Box(low=0, high=1, shape=(self.n,),
                                    dtype=np.int)  # binary vector indicating bay breaks (i = 1 means last facility in bay)

        self.state = None
        self.permutation = None  # Permutation of all n facilities, read from top to bottom
        self.bay = None
        self.done = False
        self.MHC = rewards.mhc.MHC()

        if self.mode == "rgb_array":
            self.observation_space = spaces.Box(low=0, high=255, shape=(self.W, self.L, 3),
                                                dtype=np.uint8)  # Image representation
        elif self.mode == "human":

            observation_low = np.tile(np.array([0, 0, self.min_length, self.min_width], dtype=int), self.n)
            observation_high = np.tile(np.array([self.W, self.L, self.W, self.L], dtype=int), self.n)
            self.observation_space = spaces.Box(low=observation_low, high=observation_high,
                                                dtype=int)  # Vector representation of coordinates
        else:
            print("Nothing correct selected")

    def reset(self):
        # 1. Get a random permutation and bays
        self.permutation, self.bay = self.sampler()

        # 2. Last position in bay break vector has to be 1 by default.
        self.bay[-1] = 1

        self.fac_x, self.fac_y, self.fac_b, self.fac_h = self.getCoordinates()
        self.D = self.MHC.getDistances(self.fac_x, self.fac_y)
        reward, self.TM = self.MHC.compute(self.D, self.F, self.permutation[:])

        self.state = self.constructState(self.fac_x, self.fac_y, self.fac_b, self.fac_h, self.n)

        return self.state

    def constructState(self, x, y, l, w, n):
        # Construct state
        state_prelim = np.zeros((4 * n,), dtype=float)
        state_prelim[0::4] = y
        state_prelim[1::4] = x
        state_prelim[2::4] = w
        state_prelim[3::4] = l

        if self.mode == "human":
            self.state = np.array(state_prelim)

        elif self.mode == "rgb_array":
            self.state = self.ConvertCoordinatesToState(state_prelim)

        return self.state[:]

    def ConvertCoordinatesToState(self, state_prelim):
        data = np.zeros((self.observation_space.shape)) if self.mode == 'rgb_array' else np.zeros((self.W, self.L, 3),
                                                                                                  dtype=np.uint8)

        sources = np.sum(self.TM, axis=1)
        sinks = np.sum(self.TM, axis=0)

        R = np.array((self.permutation - np.min(self.permutation)) / (
                np.max(self.permutation) - np.min(self.permutation)) * 255).astype(int)
        G = np.array((sources - np.min(sources)) / (np.max(sources) - np.min(sources)) * 255).astype(int)
        B = np.array((sinks - np.min(sinks)) / (np.max(sinks) - np.min(sinks)) * 255).astype(int)

        for x, p in enumerate(self.permutation):
            x_from = state_prelim[4 * x + 1] - 0.5 * state_prelim[4 * x + 3]
            y_from = state_prelim[4 * x + 0] - 0.5 * state_prelim[4 * x + 2]
            x_to = state_prelim[4 * x + 1] + 0.5 * state_prelim[4 * x + 3]
            y_to = state_prelim[4 * x + 0] + 0.5 * state_prelim[4 * x + 2]

            data[int(y_from):int(y_to), int(x_from):int(x_to)] = [R[p - 1], G[p - 1], B[p - 1]]

        return np.array(data, dtype=np.uint8)

    def sampler(self):
        return default_rng().choice(range(1, self.n + 1), size=self.n, replace=False), self.bay_space.sample()

    def getCoordinates(self):
        facilities = np.where(self.bay == 1)[0]  # Read all positions with a bay break
        bays = np.split(self.permutation, facilities[:-1] + 1)

        lengths = np.zeros((len(self.permutation, )))
        widths = np.zeros((len(self.permutation, )))
        fac_x = np.zeros((len(self.permutation, )))
        fac_y = np.zeros((len(self.permutation, )))

        x = 0
        start = 0
        for b in bays:  # Get the facilities that are located in the bay

            areas = self.a[b - 1]  # Get the area associated with the facilities
            end = start + len(areas)

            lengths[start:end] = np.sum(
                areas) / self.W  # Calculate all facility widhts in bay acc. to Eq. (1) in https://doi.org/10.1016/j.eswa.2011.11.046
            widths[start:end] = areas / lengths[start:end]

            fac_x[start:end] = lengths[start:end] * 0.5 + x
            x += np.sum(areas) / self.W

            y = np.ones(len(b))
            ll = 0
            for idx, l in enumerate(widths[start:end]):
                y[idx] = ll + 0.5 * l
                ll += l
            fac_y[start:end] = y

            start = end

        return fac_x, fac_y, lengths, widths

    def step(self, action):
        a = self.actions[action]
        # k = np.count_nonzero(self.bay)
        fromState = np.array(self.permutation)

        # Get lists with a bay positions and facilities in each bay
        facilities = np.where(self.bay == 1)[0]
        bay_breaks = np.split(self.bay, facilities[:-1] + 1)

        # Load indiv. facilities into bay acc. to breaks; omit break on last position to avoid empty array in list.
        bays = np.split(self.permutation, facilities[:-1] + 1)

        if a == 'Randomize':
            # Two vector elements randomly chosen are exchanged. Bay vector remains untouched.
            k = default_rng().choice(range(len(self.permutation - 1)), size=1, replace=False)
            l = default_rng().choice(range(len(self.permutation - 1)), size=1, replace=False)
            fromState[k], fromState[l] = fromState[l], fromState[k]
            self.permutation = np.array(fromState)

        elif a == 'Bit Swap':
            # One element randomly selected flips its value (1 to 0 or 0 to 1)
            j = default_rng().choice(range(len(self.bay - 1)), size=1, replace=False)

            temp_bay = np.array(self.bay)  # Make a copy of bay
            temp_bay[j] = 1 if temp_bay[j] == 0 else 0

            self.bay = np.array(temp_bay)

        elif a == 'Bay Exchange':
            # Two bays are randomly selected and exchange facilities contained in them

            o = int(default_rng().choice(range(len(bays)), size=1, replace=False))
            p = int(default_rng().choice(range(len(bays)), size=1, replace=False))

            while p == o:  # Make sure bays are not the same
                p = int(default_rng().choice(range(len(bays)), size=1, replace=False))

            # Swap bays and break points accordingly:
            bays[o], bays[p] = bays[p], bays[o]
            bay_breaks[o], bay_breaks[p] = bay_breaks[p], bay_breaks[o]

            new_bay = np.concatenate(bay_breaks)
            new_state = np.concatenate(bays)

            # Make sure state is saved as copy
            self.permutation = np.array(new_state)
            self.bay = np.array(new_bay)


        elif a == 'Inverse':
            # Facilities present in a certain bay randomly chosen are inverted.
            q = default_rng().choice(range(len(bays)))
            bays[q] = np.flip(bays[q])

            new_bay = np.concatenate(bay_breaks)
            new_state = np.concatenate(bays)

            # Make sure state is saved as copy
            self.permutation = np.array(new_state)
            self.bay = np.array(new_bay)

        elif a == 'Idle':
            pass  # Keep old state

        self.fac_x, self.fac_y, self.fac_b, self.fac_h = self.getCoordinates()
        self.D = self.MHC.getDistances(self.fac_x, self.fac_y)
        mhc, self.TM = self.MHC.compute(self.D, self.F, fromState)
        self.state = self.constructState(self.fac_x, self.fac_y, self.fac_b, self.fac_h, self.n)

        self.done = False  # Always false for continuous task

        return self.state[:], -mhc, self.done, {'mhc': mhc}

    def render(self, mode=None):
        if self.mode == "human":
            # Mode 'human' needs intermediate step to convert state vector into image array
            data = self.ConvertCoordinatesToState(self.state[:])
            img = Image.fromarray(data, 'RGB')

        if self.mode == "rgb_array":
            data = self.state[:]
            img = Image.fromarray(self.state, 'RGB')

        plt.imshow(img)
        plt.axis('off')
        plt.show()

        return img

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