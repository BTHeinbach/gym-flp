import numpy as np
import gym as gym
import pickle
import os
import math
import matplotlib.pyplot as plt
import anytree

from gym import spaces
from numpy.random import default_rng
from PIL import Image
from gym_flp import rewards, util
from anytree import Node
from gym_flp.util import preprocessing


class StsEnv(gym.Env):
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
        self.instance = instance
        self.mode = mode
        self.MHC = rewards.mhc.MHC()

        while not (self.instance in self.FlowMatrices.keys() or self.instance in ['Brewery']):
            print('Available Problem Sets:', self.FlowMatrices.keys())
            self.instance = input('Pick a problem:').strip()

        self.F = self.FlowMatrices[self.instance]
        self.n = self.problems[self.instance]
        self.AreaData = self.sizes[self.instance]

        # Obtain size data: FBS needs a length and area
        self.beta, self.l, self.w, self.a, self.min_side_length = getAreaData(
            self.AreaData)  # Investigate available area data and compute missing values if needed

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
        # Provide variables for layout encoding (epsilon in doi:10.1016/j.ejor.2018.01.001)
        self.permutation = None
        self.slicing = None
        self.orientation_space = spaces.Box(low=0, high=1, shape=(self.n - 1,),
                                            dtype=np.int)  # binary vector indicating bay breaks (i = 1 means last facility in bay)
        self.state = None

        if self.mode == "rgb_array":
            self.observation_space = spaces.Box(low=0, high=255, shape=(self.W, self.L, 3),
                                                dtype=np.uint8)  # Image representation
        elif self.mode == "human":

            # observation_low = np.tile(np.array([0,0,self.min_side_length, self.min_side_length],dtype=float), self.n)
            # observation_high = np.tile(np.array([self.L, self.W, max(self.l), max(self.w)], dtype=float), self.n)

            observation_low = np.zeros(4 * self.n)
            observation_high = np.zeros(4 * self.n)

            observation_low[0::4] = 0.0  # Top-left corner y
            observation_low[1::4] = 0.0  # Top-left corner x
            observation_low[2::4] = 1.0  # Width
            observation_low[3::4] = 1.0  # Length

            observation_high[0::4] = self.W
            observation_high[1::4] = self.L
            observation_high[2::4] = self.W
            observation_high[3::4] = self.L

            self.observation_space = spaces.Box(low=observation_low, high=observation_high,
                                                dtype=float)  # Vector representation of coordinates
        else:
            print("Nothing correct selected")

        self.action_space = spaces.Discrete(5)
        self.actions = {0: 'Permute', 1: 'Slice_Swap', 2: 'Shuffle', 3: 'Bit_Swap', 4: 'Idle'}

    def reset(self):
        # 1. Get a random permutation, slicing order and orientation
        self.permutation, self.slicing, self.orientation = self.sampler()

        # 2. Build the tree incl. size information
        s = self.TreeBuilder(self.permutation, self.slicing, self.orientation)
        centers = np.array([s[0::4] + 0.5 * s[2::4], s[1::4] + 0.5 * s[3::4]])
        self.D = self.MHC.getDistances(centers[0], centers[1])
        reward, self.TM = self.MHC.compute(self.D, self.F, np.array(range(1, self.n + 1)))

        if self.mode == "human":
            self.state = np.array(s)

        elif self.mode == "rgb_array":
            self.state = self.ConvertCoordinatesToState(s)

        return self.state

    def ConvertCoordinatesToState(self, s):
        data = np.zeros((self.observation_space.shape)) if self.mode == 'rgb_array' else np.zeros((self.W, self.L, 3),
                                                                                                  dtype=np.uint8)

        sources = np.sum(self.TM, axis=1)
        sinks = np.sum(self.TM, axis=0)

        p = self.permutation[:]
        R = np.array((p - np.min(p)) / (np.max(p) - np.min(p)) * 255).astype(int)
        G = np.array((sources - np.min(sources)) / (np.max(sources) - np.min(sources)) * 255).astype(int)
        B = np.array((sinks - np.min(sinks)) / (np.max(sinks) - np.min(sinks)) * 255).astype(int)

        for x in range(self.n):
            y_from = s[4 * x + 0]
            x_from = s[4 * x + 1]

            y_to = y_from + s[4 * x + 2]
            x_to = x_from + s[4 * x + 3]

            data[int(y_from):int(y_to), int(x_from):int(x_to)] = [R[x], G[x], B[x]]

        return np.array(data, dtype=np.uint8)

    def TreeBuilder(self, p, s, o):
        names = {0: 'V', 1: 'H'}
        contains = np.array(p)

        W = self.W
        L = self.L

        area = W * L

        self.STS = Node(name=None, contains=contains, parent=None, area=area, width=W, length=L,
                        upper_left=np.zeros((2,)), lower_right=np.array([W, L]), dtype=float)

        for i, r in enumerate(o):
            name = names[r]
            cut_after_pos = s[i]
            whats_in_pos = p[cut_after_pos - 1]

            parent = anytree.search.find(self.STS, lambda node: np.any(node.contains == whats_in_pos))
            parent.name = name
            starting_point = parent.upper_left

            cuts = np.split(parent.contains, [np.where(parent.contains == whats_in_pos)[0][0] + 1])

            for c in cuts:
                area = float(np.sum(self.a[c - 1]))
                length = area / parent.width if name == 'V' else parent.length
                width = area / parent.length if name == 'H' else parent.width

                starting_point = starting_point

                contains = c

                new_name = None if not len(c) == 1 else c[0]

                Node(name=new_name, \
                     contains=contains, \
                     parent=parent, \
                     area=area, \
                     width=width, \
                     length=length, \
                     upper_left=starting_point, \
                     lower_right=starting_point + np.array([width, length]), \
                     dtype=float)

                starting_point = starting_point + np.array(
                    [0, length]) if parent.name == 'V' else starting_point + np.array([width, 0])

            parent.contains = None
        self.STS.root.area = np.sum([i.area for i in self.STS.root.children])

        s = np.zeros((4 * self.n,))
        for l in self.STS.leaves:
            trg = int(l.name) - 1

            s[4 * trg] = l.upper_left[0]
            s[4 * trg + 1] = l.upper_left[1]
            s[4 * trg + 2] = l.width
            s[4 * trg + 3] = l.length

        return s

    def step(self, a):
        action = self.actions[a]

        '''
        Available actions in STS:
            - Random permutation change
            - Random slicing order change at two positions
            - Shuffle slicing order (new random array)
            - Bitswap in Orientation vector
            - Do Nothing
        '''

        if action == 'Permute':
            i = np.random.randint(0, len(self.permutation) - 1)
            j = np.random.randint(0, len(self.permutation) - 1)

            temp_perm = np.array(self.permutation)
            temp_perm[i], temp_perm[j] = temp_perm[j], temp_perm[i]

            self.permutation = np.array(temp_perm)

        elif action == 'Slice_Swap':
            i = np.random.randint(0, len(self.slicing) - 1)
            j = np.random.randint(0, len(self.slicing) - 1)

            temp_sli = np.array(self.slicing)
            temp_sli[i], temp_sli[j] = temp_sli[j], temp_sli[i]

            self.slicing = np.array(temp_sli)

        elif action == 'Shuffle':
            self.slicing = default_rng().choice(range(1, self.n), size=self.n - 1, replace=False)

        elif action == 'Bit_Swap':
            i = np.random.randint(0, len(self.orientation) - 1)

            if self.orientation[i] == 1:
                self.orientation[i] = 0
            elif self.orientation[i] == 0:
                self.orientation[i] = 1


        elif action == 'Idle':
            self.permutation = np.array(self.permutation)
            self.slicing = np.array(self.slicing)
            self.orientation = np.array(self.orientation)

        new_state = self.TreeBuilder(self.permutation, self.slicing, self.orientation)

        if self.mode == "human":
            self.state = np.array(new_state)

        elif self.mode == "rgb_array":
            self.state = self.ConvertCoordinatesToState(new_state)

        return self.state[:], 0, False, {}

    def render(self, mode=None):
        if self.mode == "human":
            data = self.ConvertCoordinatesToState(self.state[:])
            img = Image.fromarray(data, 'RGB')

        elif self.mode == "rgb_array":
            img = Image.fromarray(self.state, 'RGB')

        plt.imshow(img)
        plt.axis('off')
        plt.show()
        return img

    def sampler(self):
        return default_rng().choice(range(1, self.n + 1), size=self.n, replace=False), \
               default_rng().choice(range(1, self.n), size=self.n - 1, replace=False), \
               self.orientation_space.sample()

    def close(self):
        None


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