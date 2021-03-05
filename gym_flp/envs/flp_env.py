import numpy as np
import gym
from gym import spaces
from numpy.random import default_rng
import pickle
import os
import math
import matplotlib.pyplot as plt
from PIL import Image
from gym_flp import rewards
import pygame as pg
from IPython.display import display, clear_output
import anytree
from anytree import Node, RenderTree, PreOrderIter, LevelOrderIter, LevelOrderGroupIter
    

'''
v0.0.3
Significant changes:
    08.09.2020:
    - Dicrete option removed from spaces; only Box allowed 
    - Classes for quadtratic set covering and mixed integer programming (-ish) added
    - Episodic tasks: no more terminal states (exception: max. no. of trials reached)
    
    12.10.2020:
        - mip added
        - fbs added
'''

class qapEnv(gym.Env):
    metadata = {'render.modes': ['rgb_array', 'human']}  

    def __init__(self, mode=None, instance=None):
        __location__ = os.path.realpath(os.path.join(os.getcwd(), os.path.dirname(__file__)))
        self.DistanceMatrices, self.FlowMatrices = pickle.load(open(os.path.join(__location__,'discrete', 'qap_matrices.pkl'), 'rb'))
        self.transport_intensity = None
        self.instance = instance
        self.mode = mode
        
        while not (self.instance in self.DistanceMatrices.keys() or self.instance in self.FlowMatrices.keys() or self.instance in ['Neos-n6', 'Neos-n7', 'Brewery']):
            print('Available Problem Sets:', self.DistanceMatrices.keys())
            self.instance = input('Pick a problem:').strip()
     
        self.D = self.DistanceMatrices[self.instance]
        self.F = self.FlowMatrices[self.instance]
        
        # Determine problem size relevant for much stuff in here:
        self.n = len(self.D[0])
        
        # Action space has two option:
        # 1) Define as Box with shape (1, 2) and allow values to range from 1 through self.n 
        # 2) Define as Discrete with x = 1+((n^2-n)/2) actions (one half of matrix + 1 value from diagonal) --> Omit "+1" to obtain range from 0 to x!
        # self.action_space = spaces.Box(low=-1, high=6, shape=(1,2), dtype=np.int) # Doubles complexity of the problem as it allows the identical action (1,2) and (2,1)
        self.action_space = spaces.Discrete(int((self.n**2-self.n)*0.5)+1)
                
        # If you are using images as input, the input values must be in [0, 255] as the observation is normalized (dividing by 255 to have values in [0, 1]) when using CNN policies.       
        if self.mode == "rgb_array":
            self.observation_space = spaces.Box(low = 0, high = 255, shape=(1, self.n, 3), dtype = np.uint8) # Image representation
        elif self.mode == 'human':
            self.observation_space = spaces.Box(low=1, high = self.n, shape=(self.n,), dtype=np.float32)
        
        self.states = {}    # Create an empty dictonary where states and their respective reward will be stored for future reference
        self.actions = self.pairwiseExchange(self.n)
        
        # Initialize Environment with empty state and action
        self.action = None
        self.state = None
        
        #Initialize moving target to incredibly high value. To be updated if reward obtained is smaller. 
        
        self.movingTargetReward = np.inf 
        self.MHC = rewards.mhc.MHC()    # Create an instance of class MHC in module mhc.py from package rewards
    
    def reset(self):
        state = default_rng().choice(range(1,self.n+1), size=self.n, replace=False) 

        MHC, self.TM = self.MHC.compute(self.D, self.F, state)
        
        if self.mode == "human":
            state = np.array(state)
        
        elif self.mode == "rgb_array":
            rgb = np.zeros((1,self.n,3), dtype=np.uint8)
            
            sources = np.sum(self.TM, axis = 1)
            sinks = np.sum(self.TM, axis = 0)
            
            R = np.array((state-np.min(state))/(np.max(state)-np.min(state))*255).astype(int)
            G = np.array((sources-np.min(sources))/(np.max(sources)-np.min(sources))*255).astype(int)
            B = np.array((sinks-np.min(sinks))/(np.max(sinks)-np.min(sinks))*255).astype(int)
                        
            for i, s in enumerate(state):
                rgb[0, i] = [R[s-1], G[s-1], B[s-1]]

            state = np.array(rgb)
            
        return state[:]
    
    def step(self, action):
        # Create new State based on action 
        
        swap = self.actions[action]
        
        fromState = np.array(self.state)
        
        if self.mode == 'human':
            fromState[swap[0]-1], fromState[swap[1]-1] = fromState[swap[1]-1], fromState[swap[0]-1]
            current_permutation = np.array(fromState)
            
        elif self.mode == 'rgb_array':
            
            temp1 = np.array(fromState[0,swap[0]-1])
            temp2 =  np.array(fromState[0,swap[1]-1])
            
            fromState[0,swap[0]-1] = temp2
            fromState[0,swap[1]-1] = temp1

            rescale = fromState[:,:,0]/255*5 +1
            current_permutation = rescale[0]
        
        MHC, self.TM = self.MHC.compute(self.D, self.F, current_permutation)   
        
        if self.mode == 'human':
            self.states[tuple(fromState)] = MHC
        
        if self.movingTargetReward == np.inf:
            self.movingTargetReward = MHC 
     
        #reward = self.movingTargetReward - MHC
        reward = -1 if MHC > self.movingTargetReward else 10
        self.movingTargetReward = MHC if MHC < self.movingTargetReward else self.movingTargetReward

        newState = np.array(fromState)
        
        return newState, MHC, False, {}
          
    def render(self, mode=None):
        if self.mode == "human":
                
            SCALE = 1  # Scale size of pixels for displayability
            img_h, img_w = SCALE, (len(self.state))*SCALE
            data = np.zeros((img_h, img_w, 3), dtype=np.uint8)
            
            sources = np.sum(self.TM, axis = 1)
            sinks = np.sum(self.TM, axis = 0)
            
            R = np.array((self.state-np.min(self.state))/(np.max(self.state)-np.min(self.state))*255).astype(int)
            G = np.array((sources-np.min(sources))/(np.max(sources)-np.min(sources))*255).astype(int)
            B = np.array((sinks-np.min(sinks))/(np.max(sinks)-np.min(sinks))*255).astype(int)
            
            for i, s in enumerate(self.state):
                data[0*SCALE:1*SCALE, i*SCALE:(i+1)*SCALE] = [R[s-1], G[s-1], B[s-1]]
                       
            img = Image.fromarray(data, 'RGB')            
                    
        if self.mode == 'rgb_array':
            img = Image.fromarray(self.state, 'RGB')            

        
        plt.imshow(img)
        plt.axis('off')
        plt.show()
        return img
    
    def close(self):
        pass
        
    def pairwiseExchange(self, x):
        actions = [(i,j) for i in range(1,x) for j in range(i+1,x+1) if not i==j]
        actions.append((1,1))
        return actions
                
class fbsEnv(gym.Env):
    metadata = {'render.modes': ['rgb_array', 'human']}          
    
    def __init__(self, mode=None, instance = None):
        __location__ = os.path.realpath(os.path.join(os.getcwd(), os.path.dirname(__file__)))
        self.problems, self.FlowMatrices, self.sizes, self.LayoutWidths, self.LayoutLengths = pickle.load(open(os.path.join(__location__,'continual', 'cont_instances.pkl'), 'rb'))
        self.mode = mode
        
        self.instance = instance
        while not (self.instance in self.FlowMatrices.keys() or self.instance in ['Brewery']):
            print('Available Problem Sets:', self.FlowMatrices.keys())
            self.instance = input('Pick a problem:').strip()
     
        self.F = self.FlowMatrices[self.instance]
        self.n = self.problems[self.instance]
        self.AreaData = self.sizes[self.instance]

        # Obtain size data: FBS needs a length and area
        self.beta, self.l, self.w, self.a, self.min_side_length = getAreaData(self.AreaData) #Investigate available area data and compute missing values if needed
        
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
               
        #if self.l is None or self.w is None:
            # self.l = np.random.randint(max(self.min_side_length, np.min(self.a)/self.min_side_length), max(self.min_side_length, np.min(self.a)/self.min_side_length), size=(self.n,))
        #    self.l = np.sqrt(self.A/self.aspect_ratio)
        #    self.w = np.round(self.a/self.l)
        
        # Check if there are Layout Dimensions available, if not provide enough (sqrt(a)*1.5)
        if self.instance in self.LayoutWidths.keys() and self.instance in self.LayoutLengths.keys():
            self.L = int(self.LayoutLengths[self.instance]) # We need both values to be integers for converting into image
            self.W = int(self.LayoutWidths[self.instance]) 
        else:
            self.A = np.sum(self.a)
            
            # Design a squared plant layout
            self.L = int(round(math.sqrt(self.A),0)) # We want the plant dimensions to be integers to fit them into an image
            self.W = self.L 
            
            # Design a layout with l = 1,5 * w
            #self.L = divisor(int(self.A))
            #self.W = self.A/self.L
            
        # These values need to be set manually, e.g. acc. to data from literature. Following Eq. 1 in Ulutas & Kulturel-Konak (2012), the minimum side length can be determined by assuming the smallest facility will occupy alone. 
        self.aspect_ratio = int(max(self.beta)) if not self.beta is None else 1
        self.min_length = np.min(self.a) / self.L
        self.min_width = np.min(self.a) / self.W
        
        # We define minimum side lengths to be 1 in order to be displayable in array
        self.min_length = 1
        self.min_width = 1
        
        self.action_space = spaces.Discrete(5)  #Taken from doi:10.1016/j.engappai.2020.103697
        self.actions = {0: 'Randomize', 1: 'Bit Swap', 2: 'Bay Exchange', 3: 'Inverse', 4: 'Idle'}
        #self.state_space = spaces.Box(low=1, high = self.n, shape=(self.n,), dtype=np.int)
        self.bay_space = spaces.Box(low=0, high = 1, shape=(self.n,), dtype=np.int) # binary vector indicating bay breaks (i = 1 means last facility in bay)

        self.state = None 
        self.permutation = None # Permutation of all n facilities, read from top to bottom
        self.bay = None
        self.done = False
        self.MHC = rewards.mhc.MHC() 
        
        if self.mode == "rgb_array":
            self.observation_space = spaces.Box(low = 0, high = 255, shape= (self.W, self.L,3), dtype = np.uint8) # Image representation
        elif self.mode == "human":
            
            observation_low = np.tile(np.array([0,0,self.min_length,self.min_width],dtype=int), self.n)
            observation_high = np.tile(np.array([self.W, self.L, self.W, self.L], dtype=int), self.n)
            self.observation_space = spaces.Box(low=observation_low, high=observation_high, dtype = int) # Vector representation of coordinates
        else:
            print("Nothing correct selected")
    
    def reset(self):
        # 1. Get a random permutation and bays    
        self.permutation, self.bay = self.sampler()

        # 2. Last position in bay break vector has to be 1 by default.
        self.bay[-1] = 1
        
        self.fac_x, self.fac_y, self.fac_b, self.fac_h = self.getCoordinates()
        self.D = getDistances(self.fac_x, self.fac_y)
        reward, self.TM = self.MHC.compute(self.D, self.F, self.permutation[:])
        
        self.state = self.constructState(self.fac_x, self.fac_y, self.fac_b, self.fac_h, self.n)
        
        return self.state
    
    def constructState(self, x, y, l, w, n):
        # Construct state
        state_prelim = np.zeros((4*n,), dtype=float)
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
        data = np.zeros((self.observation_space.shape)) if self.mode == 'rgb_array' else np.zeros((self.W, self.L, 3),dtype=np.uint8)
        
        sources = np.sum(self.TM, axis = 1)
        sinks = np.sum(self.TM, axis = 0)
        
        R = np.array((self.permutation-np.min(self.permutation))/(np.max(self.permutation)-np.min(self.permutation))*255).astype(int)
        G = np.array((sources-np.min(sources))/(np.max(sources)-np.min(sources))*255).astype(int)
        B = np.array((sinks-np.min(sinks))/(np.max(sinks)-np.min(sinks))*255).astype(int)
        
        for x, p in enumerate(self.permutation):
            x_from = state_prelim[4*x+1] -0.5 * state_prelim[4*x+3]
            y_from = state_prelim[4*x+0] -0.5 * state_prelim[4*x+2]
            x_to = state_prelim[4*x+1] + 0.5 * state_prelim[4*x+3]
            y_to = state_prelim[4*x+0] + 0.5 * state_prelim[4*x+2]
        
            data[int(y_from):int(y_to), int(x_from):int(x_to)] = [R[p-1], G[p-1], B[p-1]]

        return np.array(data, dtype=np.uint8)
    
    def sampler(self):
            return default_rng().choice(range(1,self.n+1), size=self.n, replace=False), self.bay_space.sample()
    
    def getCoordinates(self):
        facilities = np.where(self.bay==1)[0]  #Read all positions with a bay break
        bays = np.split(self.permutation, facilities[:-1]+1)
        
        lengths = np.zeros((len(self.permutation,)))
        widths = np.zeros((len(self.permutation,)))
        fac_x = np.zeros((len(self.permutation,)))
        fac_y = np.zeros((len(self.permutation,)))
        
        x = 0
        start = 0
        for b in bays: #Get the facilities that are located in the bay
            
            areas = self.a[b-1] #Get the area associated with the facilities
            end = start + len(areas)
            
            lengths[start:end] = np.sum(areas)/self.W #Calculate all facility widhts in bay acc. to Eq. (1) in https://doi.org/10.1016/j.eswa.2011.11.046
            widths[start:end] = areas/lengths[start:end]

            fac_x[start:end] = lengths[start:end] * 0.5 + x 
            x += np.sum(areas)/self.W
            
            y = np.ones(len(b))
            ll = 0
            for idx, l in enumerate(widths[start:end]):
                y[idx] = ll + 0.5*l
                ll += l
            fac_y[start:end] = y       
            
            start = end
        
        return fac_x, fac_y, lengths, widths
    
    def step(self, action):
        a = self.actions[action]
        #k = np.count_nonzero(self.bay)
        fromState = np.array(self.permutation)
        
        # Get lists with a bay positions and facilities in each bay
        facilities = np.where(self.bay==1)[0]
        bay_breaks = np.split(self.bay, facilities[:-1]+1)
        
        # Load indiv. facilities into bay acc. to breaks; omit break on last position to avoid empty array in list.
        bays = np.split(self.permutation, facilities[:-1]+1)
        
        if a == 'Randomize':
            # Two vector elements randomly chosen are exchanged. Bay vector remains untouched.
            k = default_rng().choice(range(len(self.permutation-1)), size=1, replace=False)
            l = default_rng().choice(range(len(self.permutation-1)), size=1, replace=False)
            fromState[k], fromState[l] = fromState[l], fromState[k]
            self.permutation = np.array(fromState)
            
        elif a == 'Bit Swap':
            #One element randomly selected flips its value (1 to 0 or 0 to 1)
            j = default_rng().choice(range(len(self.bay-1)), size=1, replace=False)

            temp_bay = np.array(self.bay) # Make a copy of bay
            temp_bay[j] = 1 if temp_bay[j] == 0 else 0

            self.bay = np.array(temp_bay)
            
        elif a == 'Bay Exchange':
            #Two bays are randomly selected and exchange facilities contained in them
            
            o = int(default_rng().choice(range(len(bays)), size=1, replace=False))           
            p = int(default_rng().choice(range(len(bays)), size=1, replace=False)) 
            
            while p==o: # Make sure bays are not the same
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
            #Facilities present in a certain bay randomly chosen are inverted.
            q = default_rng().choice(range(len(bays)))
            bays[q] = np.flip(bays[q])
            
            new_bay = np.concatenate(bay_breaks)
            new_state = np.concatenate(bays)
            
            # Make sure state is saved as copy
            self.permutation = np.array(new_state)
            self.bay = np.array(new_bay)
        
        elif a == 'Idle':
            pass # Keep old state
        
        self.fac_x, self.fac_y, self.fac_b, self.fac_h = self.getCoordinates()
        self.D = getDistances(self.fac_x, self.fac_y)
        reward, self.TM = self.MHC.compute(self.D, self.F, fromState)
        self.state = self.constructState(self.fac_x, self.fac_y, self.fac_b, self.fac_h, self.n)
        
        self.done = False #Always false for continuous task
        
        return self.state[:], reward, self.done, {}
               
    def render(self, mode= None):     
        if self.mode== "human":
            
            # Mode 'human' needs intermediate step to convert state vector into image array
            data = self.ConvertCoordinatesToState(self.state[:])
            img = Image.fromarray(data, 'RGB')            

        if self.mode == "rgb_array":
            data = self.state[:]
            img = Image.fromarray(self.state, 'RGB')

        plt.imshow(img)
        plt.axis('off')
        plt.show()
        
        #23.02.21: Switched to data instead of img for testing video
        return img
    
    def close(self):
        pass
        #self.close()
        
class ofpEnv(gym.Env):
    metadata = {'render.modes': ['rgb_array', 'human']} 
          
    def __init__(self, mode = None, instance = None, distance = None, aspect_ratio = None, step_size = None, greenfield = None):
        __location__ = os.path.realpath(os.path.join(os.getcwd(), os.path.dirname(__file__)))
        self.problems, self.FlowMatrices, self.sizes, self.LayoutWidths, self.LayoutLengths = pickle.load(open(os.path.join(__location__,'continual', 'cont_instances.pkl'), 'rb'))
        self.mode = mode
        self.aspect_ratio = 2 if aspect_ratio is None else aspect_ratio
        self.step_size = 2 if step_size is None else step_size
        self.greenfield = greenfield
        self.instance = instance     
        while not (self.instance in self.FlowMatrices.keys() or self.instance in ['Brewery']):
            print('Available Problem Sets:', self.FlowMatrices.keys())
            self.instance = input('Pick a problem:').strip()
     
        self.F = self.FlowMatrices[self.instance]
        self.n = self.problems[self.instance]
        self.AreaData = self.sizes[self.instance]
        
        
        # Obtain size data: FBS needs a length and area
        self.beta, self.l, self.w, self.a, self.min_side_length = getAreaData(self.AreaData) #Investigate available area data and compute missing values if needed
        
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
               
        #if self.l is None or self.w is None:
            # self.l = np.random.randint(max(self.min_side_length, np.min(self.a)/self.min_side_length), max(self.min_side_length, np.min(self.a)/self.min_side_length), size=(self.n,))
        #    self.l = np.sqrt(self.A/self.aspect_ratio)
        #    self.w = np.round(self.a/self.l)
        
        # Check if there are Layout Dimensions available, if not provide enough (sqrt(a)*1.5)
        if self.instance in self.LayoutWidths.keys() and self.instance in self.LayoutLengths.keys():
            self.L = int(self.LayoutLengths[self.instance]) # We need both values to be integers for converting into image
            self.W = int(self.LayoutWidths[self.instance]) 
        else:
            self.A = np.sum(self.a)
            
            # Design a squared plant layout
            self.L = int(round(math.sqrt(self.A),0)) # We want the plant dimensions to be integers to fit them into an image
            self.W = self.L 
        
        if self.greenfield:
            self.L = 5*self.L
            self.W = 5*self.W
            
            # Design a layout with l = 1,5 * w
            #self.L = divisor(int(self.A))
            #self.W = self.A/self.L
            
        # These values need to be set manually, e.g. acc. to data from literature. Following Eq. 1 in Ulutas & Kulturel-Konak (2012), the minimum side length can be determined by assuming the smallest facility will occupy alone. 
        self.aspect_ratio = int(max(self.beta)) if not self.beta is None else self.aspect_ratio
        self.min_length = np.min(self.a) / self.L
        self.min_width = np.min(self.a) / self.W

        # 3. Define  the possible actions: 5 for each box [toDo: plus 2 to manipulate sizes] + 1 idle action for all
        self.actions = {}   
        for i in range(self.n):
            self.actions[0+(i)*5] = "up"
            self.actions[1+(i)*5] = "down"
            self.actions[2+(i)*5] = "right"
            self.actions[3+(i)*5] = "left"
            self.actions[4+(i)*5] = "rotate"
             
        self.actions[len(self.actions)] = "keep"
                   
        # 4. Define actions space as Discrete Space
        self.action_space = spaces.Discrete(1+5*self.n) #5 actions for each facility: left, up, down, right, rotate + idle action across all
        
        # 5. Set some starting points
        self.reward = 0
        self.state = None 
        self.internal_state = None #Placeholder for state variable for internal manipulation in rgb_array mode
                
        if self.w is None or self.l is None:
            self.l = np.random.randint(self.min_side_length*self.aspect_ratio, np.min(self.a), size=(self.n, ))
            self.w = np.round(self.a/self.l)
        
        # 6. Set upper and lower bound for observation space
        # min x position can be point of origin (0,0) [coordinates map to upper left corner]
        # min y position can be point of origin (0,0) [coordinates map to upper left corner]
        # min width can be smallest area divided by its length
        # min lenght can be smallest width (above) multiplied by aspect ratio
        # max x pos can be bottom right edge of grid
        # max y pos can be bottpm right edge of grid
        
        if self.mode == "rgb_array":
            self.observation_space = spaces.Box(low = 0, high = 255, shape= (self.W, self.L,3), dtype = np.uint8) # Image representation
        elif self.mode == "human":
           
            #observation_low = np.tile(np.array([0,0,self.min_side_length, self.min_side_length],dtype=float), self.n)
            #observation_high = np.tile(np.array([self.L, self.W, max(self.l), max(self.w)], dtype=float), self.n)
            
            observation_low = np.zeros(4* self.n)
            observation_high = np.zeros(4* self.n)
            
            observation_low[0::4] = 0 + self.l / 2
            observation_low[1::4] = 0 + self.w / 2
            observation_low[2::4] = self.min_length
            observation_low[3::4] = self.min_width
            
            observation_high[0::4] = self.L - self.l / 2
            observation_high[1::4] = self.W - self.w / 2
            observation_high[2::4] = max(self.l)
            observation_high[3::4] = max(self.w)
            
            self.observation_space = spaces.Box(low=observation_low, high=observation_high, dtype = float) # Vector representation of coordinates
        else:
            print("Nothing correct selected")
        
        self.MHC = rewards.mhc.MHC() 
        
        # Set Boundaries
        self.upper_bound = self.W - max(self.w) / 2
        self.lower_bound = 0 + max(self.w) / 2
        self.left_bound = 0 + max(self.l)/2
        self.right_bound = self.L - max(self.l)/2
    
    def offGrid(self):
        ...
            
    def reset(self):
        # Start with random x and y positions
         
        if self.mode == 'human':
            state_prelim = self.observation_space.sample()
        
            # Override length (l) and width (w) or facilities with data from instances 
            state_prelim[2::4] = self.w
            state_prelim[3::4] = self.l

            self.D = getDistances(state_prelim[0::4], state_prelim[1::4])
            self.internal_state = np.array(state_prelim)
            self.state = np.array(state_prelim)
            reward, self.TM = self.MHC.compute(self.D, self.F, np.array(range(1,self.n+1)))
        
            
        elif self.mode == 'rgb_array':
            state_prelim = np.zeros((self.W, self.L, 3),dtype=np.uint8)
            
            x = np.random.uniform(0, self.L, size=(self.n,))
            y = np.random.uniform(0, self.W, size=(self.n,))
            
            #s = self.constructState(y, x, self.w, self.l, self.n)
            s = np.zeros(4*self.n)
            s[0::4] = y
            s[1::4] = x
            s[2::4] = self.w
            s[3::4] = self.l
            
            self.internal_state = np.array(s)
            #self.state = self.constructState(y, x, self.w, self.l, self.n)
           
            
            self.D = getDistances(s[0::4], s[1::4])
            reward, self.TM = self.MHC.compute(self.D, self.F, np.array(range(1,self.n+1)))
            self.state = self.ConvertCoordinatesToState(self.internal_state)
            
        return self.state[:]

    def collision_test(self, y, x, w, l):      

        # collision test
        collision = False #initialize collision for each collision test

        for i in range (0, self.n-1):
            for j in range (i+1,self.n):                
                if x[i]+0.5*l[i] < x[j]-0.5*l[j] or x[i]-0.5*l[i] > x[j]+0.5*l[j] or y[i]-0.5*w[i] < y[j]+0.5*w[j] or y[i]+0.5*w[i] < y[j]+0.5*w[j]:
                    collision = True
                    break
                          
        return collision

    def step(self, action):        
        m = np.int(np.ceil((action+1)/5))   # Facility on which the action is
               
        # Get copy of state to manipulate:
        temp_state = self.internal_state[:]
        
        
        step_size = self.step_size
        
        # Do the action 
        if self.actions[action] == "up":          
            if temp_state[4*(m-1)+1] < self.upper_bound:
                temp_state[4*(m-1)+1] += step_size
            else:
                temp_state[4*(m-1)+1] += 0
                print('Forbidden action: machine', m, 'left grid on upper bound')

        elif self.actions[action] == "down":
            if temp_state[4*(m-1)+1] > self.lower_bound:
                temp_state[4*(m-1)+1] += step_size
            else:
                temp_state[4*(m-1)+1] += 0
                print('Forbidden action: machine', m, 'left grid on lower bound')
            
        elif self.actions[action] == "right":
            if temp_state[4*(m-1)] < self.right_bound:
                temp_state[4*(m-1)] += step_size
            else:
                temp_state[4*(m-1)] += 0
                print('Forbidden action: machine', m, 'left grid on right bound')
            
        elif self.actions[action] == "left":
            if temp_state[4*(m-1)] > self.left_bound:
                temp_state[4*(m-1)] -= step_size
            else:
                temp_state[4*(m-1)] += 0
                print('Forbidden action: machine', m, 'left grid on left bound')
            
        elif self.actions[action] == "keep":
            None #Leave everything as is
        
        elif self.actions[action] == "rotate":
            temp_state[4*(m-1)+2], temp_state[4*(m-1)+3] = temp_state[4*(m-1)+3], temp_state[4*(m-1)+2]
            
        else:
            raise ValueError("Received invalid action={} which is not part of the action space".format(action))
        
        self.fac_x, self.fac_y, self.fac_b, self.fac_h = temp_state[0::4], temp_state[1::4], temp_state[2::4], temp_state[3::4] # ToDo: Read this from self.state
        self.D = getDistances(self.fac_x, self.fac_y)
        
        fromState = np.array(range(1,self.n+1)) # Need to create permutation matrix 
        reward, self.TM = self.MHC.compute(self.D, self.F, fromState)   
        
        self.internal_state = np.array(temp_state) # Keep a copy of the vector representation for future steps
        self.state = self.internal_state[:]
        
        # Test if initial state causing a collision. If yes than initialize a new state until there is no collision
        collision = self.collision_test(temp_state[0::4],temp_state[1::4], temp_state[2::4], temp_state[3::4]) # Pass every 4th item starting at 0 (x pos) and 1 (y pos) for checking         
        print(collision)
        
        #reward = -1000 if collision else -reward 
        
        if self.mode == 'rgb_array':
            self.state = self.ConvertCoordinatesToState(self.internal_state) #Retain state for internal use
            
        return self.state, -reward, False, {}
    
    def ConvertCoordinatesToState(self, state_prelim):    
        data = np.zeros((self.observation_space.shape)) if self.mode == 'rgb_array' else np.zeros((self.W, self.L, 3),dtype=np.uint8)
        
        sources = np.sum(self.TM, axis = 1)
        sinks = np.sum(self.TM, axis = 0)
        
        p = np.arange(self.n)
        R = np.array((p-np.min(p))/(np.max(p)-np.min(p))*255).astype(int)
        G = np.array((sources-np.min(sources))/(np.max(sources)-np.min(sources))*255).astype(int)
        B = np.array((sinks-np.min(sinks))/(np.max(sinks)-np.min(sinks))*255).astype(int)
        
        for x, p in enumerate(p):
            x_from = state_prelim[4*x+0] -0.5 * state_prelim[4*x+2]
            y_from = state_prelim[4*x+1] -0.5 * state_prelim[4*x+3]
            x_to = state_prelim[4*x+0] + 0.5 * state_prelim[4*x+2]
            y_to = state_prelim[4*x+1] + 0.5 * state_prelim[4*x+3]
        
            data[int(y_from):int(y_to), int(x_from):int(x_to)] = [R[p-1], G[p-1], B[p-1]]

        return np.array(data, dtype=np.uint8)
    
    def constructState(self, x, y, b, h, n):
        # Construct state
        state_prelim = np.zeros((4*n,), dtype=float)
        state_prelim[0::4] = y
        state_prelim[1::4] = x
        state_prelim[2::4] = b
        state_prelim[3::4] = h
        
        if self.mode == "human":
            self.state = np.array(state_prelim)
        
        elif self.mode == "rgb_array":
            self.state = self.ConvertCoordinatesToState(state_prelim)
            
        return self.state[:]
    
    def render(self):       
         
        if self.mode == "human":
            data = self.ConvertCoordinatesToState(self.state[:])
            img = Image.fromarray(data, 'RGB')            

        if self.mode == "rgb_array":
            img = Image.fromarray(self.state, 'RGB')

        plt.imshow(img)
        plt.axis('off')
        plt.show()
        return img
        
    def close(self):
        self.close()
        

class stsEnv(gym.Env):

    metadata = {'render.modes': ['rgb_array', 'human']} 
          
    def __init__(self, mode = None, instance = None):
        __location__ = os.path.realpath(os.path.join(os.getcwd(), os.path.dirname(__file__)))
        self.problems, self.FlowMatrices, self.sizes, self.LayoutWidths, self.LayoutLengths = pickle.load(open(os.path.join(__location__,'continual', 'cont_instances.pkl'), 'rb'))
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
        self.beta, self.l, self.w, self.a, self.min_side_length = getAreaData(self.AreaData) #Investigate available area data and compute missing values if needed
        
        # Check if there are Layout Dimensions available, if not provide enough (sqrt(a)*1.5)
        if self.instance in self.LayoutWidths.keys() and self.instance in self.LayoutLengths.keys():
            self.L = int(self.LayoutLengths[self.instance]) # We need both values to be integers for converting into image
            self.W = int(self.LayoutWidths[self.instance]) 
        else:
            self.A = np.sum(self.a)
            
            # Design a squared plant layout
            self.L = int(round(math.sqrt(self.A),0)) # We want the plant dimensions to be integers to fit them into an image
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
        self.orientation_space = spaces.Box(low=0, high = 1, shape=(self.n-1,), dtype=np.int) # binary vector indicating bay breaks (i = 1 means last facility in bay)
        self.state = None
        
        if self.mode == "rgb_array":
            self.observation_space = spaces.Box(low = 0, high = 255, shape= (self.W, self.L,3), dtype = np.uint8) # Image representation
        elif self.mode == "human":
           
            #observation_low = np.tile(np.array([0,0,self.min_side_length, self.min_side_length],dtype=float), self.n)
            #observation_high = np.tile(np.array([self.L, self.W, max(self.l), max(self.w)], dtype=float), self.n)
            
            observation_low = np.zeros(4* self.n)
            observation_high = np.zeros(4* self.n)
            
            observation_low[0::4] = 0.0  #Top-left corner y
            observation_low[1::4] = 0.0  #Top-left corner x
            observation_low[2::4] = 1.0  #Width
            observation_low[3::4] = 1.0  #Length
            
            observation_high[0::4] = self.W
            observation_high[1::4] = self.L
            observation_high[2::4] = self.W
            observation_high[3::4] = self.L
            
            self.observation_space = spaces.Box(low=observation_low, high=observation_high, dtype = float) # Vector representation of coordinates
        else:
            print("Nothing correct selected")
        
        self.action_space = spaces.Discrete(5)
        self.actions = {0: 'Permute', 1: 'Slice_Swap', 2: 'Shuffle', 3: 'Bit_Swap', 4: 'Idle'}
        
    def reset(self):
        # 1. Get a random permutation, slicing order and orientation    
        self.permutation, self.slicing, self.orientation = self.sampler()
        
        # 2. Build the tree incl. size information
        s = self.TreeBuilder(self.permutation, self.slicing, self.orientation)
        centers = np.array([s[0::4] + 0.5*s[2::4], s[1::4] + 0.5* s[3::4]])
        self.D = getDistances(centers[0], centers[1])
        reward, self.TM = self.MHC.compute(self.D, self.F, np.array(range(1,self.n+1)))
        
        
        if self.mode == "human":
            self.state = np.array(s)
            
        elif self.mode == "rgb_array":
            self.state = self.ConvertCoordinatesToState(s)
        
            
        return self.state 

    def ConvertCoordinatesToState(self, s):    
        data = np.zeros((self.observation_space.shape)) if self.mode == 'rgb_array' else np.zeros((self.W, self.L, 3),dtype=np.uint8)
        
        sources = np.sum(self.TM, axis = 1)
        sinks = np.sum(self.TM, axis = 0)
               
        p = self.permutation[:]
        R = np.array((p-np.min(p))/(np.max(p)-np.min(p))*255).astype(int)
        G = np.array((sources-np.min(sources))/(np.max(sources)-np.min(sources))*255).astype(int)
        B = np.array((sinks-np.min(sinks))/(np.max(sinks)-np.min(sinks))*255).astype(int)
        
        for x in range(self.n):
            y_from = s[4*x+0]
            x_from = s[4*x+1]
            
            y_to = y_from + s[4*x+2]
            x_to = x_from + s[4*x+3]
        
            data[int(y_from):int(y_to), int(x_from):int(x_to)] = [R[x], G[x], B[x]]

        return np.array(data, dtype=np.uint8)
    
    def TreeBuilder(self,p,s,o):
        names = {0: 'V', 1: 'H'}
        contains = np.array(p)
        
        W = self.W
        L = self.L
        
        area = W * L
        
        self.STS = Node(name = None, contains = contains, parent = None, area = area, width = W, length = L, upper_left = np.zeros((2,)), lower_right = np.array([W,L]), dtype = float)
        
        for i,r in enumerate(o):         
            name = names[r]        
            cut_after_pos = s[i]
            whats_in_pos = p[cut_after_pos-1]
            
            parent = anytree.search.find(self.STS, lambda node: np.any(node.contains==whats_in_pos))
            parent.name = name
            starting_point = parent.upper_left
            
            cuts = np.split(parent.contains, [np.where(parent.contains == whats_in_pos)[0][0]+1])
            
            for c in cuts:
                area = float(np.sum(self.a[c-1]))              
                length = area/parent.width if name == 'V' else parent.length
                width = area/parent.length if name == 'H' else parent.width
                
                starting_point = starting_point
                
                contains = c 
                
                new_name = None if not len(c)==1 else c[0]
                
                Node(name = new_name, \
                     contains = contains, \
                     parent = parent, \
                     area = area, \
                     width = width, \
                     length = length, \
                     upper_left = starting_point, \
                     lower_right = starting_point + np.array([width, length]), \
                     dtype = float)
                
                starting_point = starting_point + np.array([0, length]) if parent.name == 'V' else starting_point + np.array([width, 0])
                
            parent.contains = None
        self.STS.root.area = np.sum([i.area for i in self.STS.root.children])
        
        s = np.zeros((4*self.n,))
        for l in self.STS.leaves:
            trg = int(l.name)-1
            
            s[4*trg] = l.upper_left[0]
            s[4*trg+1] = l.upper_left[1]
            s[4*trg+2] = l.width
            s[4*trg+3] = l.length
        
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
            i = np.random.randint(0, len(self.permutation)-1)
            j = np.random.randint(0, len(self.permutation)-1)
            
            temp_perm = np.array(self.permutation)
            temp_perm[i], temp_perm[j] = temp_perm[j], temp_perm[i]
            
            self.permutation = np.array(temp_perm)
        
        elif action == 'Slice_Swap':
            i = np.random.randint(0, len(self.slicing)-1)
            j = np.random.randint(0, len(self.slicing)-1)
            
            temp_sli = np.array(self.slicing)
            temp_sli[i], temp_sli[j] = temp_sli[j], temp_sli[i]
            
            self.slicing = np.array(temp_sli)
    
        elif action == 'Shuffle':
            self.slicing = default_rng().choice(range(1,self.n), size=self.n-1, replace=False)
        
        elif action == 'Bit_Swap':
            i = np.random.randint(0, len(self.orientation)-1)
            
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
        return default_rng().choice(range(1,self.n+1),size=self.n, replace=False), \
               default_rng().choice(range(1,self.n), size=self.n-1, replace=False), \
               self.orientation_space.sample()
        
    def close(self):
        None    
 
def getAreaData(df):
    import re
    
    # First check for area data
    if np.any(df.columns.str.contains('Area', na=False, case = False)):
        a = df.filter(regex = re.compile("Area", re.IGNORECASE)).to_numpy()
        #a = np.reshape(a, (a.shape[0],))
        
    else:
        a = None
    
    if np.any(df.columns.str.contains('Length', na=False, case = False)):
        l = df.filter(regex = re.compile("Length", re.IGNORECASE)).to_numpy()
        l = np.reshape(l, (l.shape[0],))
        
    else:
        l = None
    
    if np.any(df.columns.str.contains('Width', na=False, case = False)):
        w = df.filter(regex = re.compile("Width", re.IGNORECASE)).to_numpy()
        w = np.reshape(w, (w.shape[0],))
        
    else:
        w = None

    if np.any(df.columns.str.contains('Aspect', na=False, case = False)):
            ar = df.filter(regex = re.compile("Aspect", re.IGNORECASE)).to_numpy()
            #ar = np.reshape(a, (a.shape[0],))
            
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
        if not l  is None and not w is None:
            a = l * w
        elif not l is None:
            a = l * max(l_min, max(l))
        else:
            a = w * max(l_min, max(w))
    
    if not ar is None and ar.ndim > 1:       
            ar = np.array([np.random.default_rng().uniform(min(ar[i]), max(ar[i])) for i in range(len(ar))])   
    
    if not a is None and a.ndim > 1:
        #a = a[np.where(np.max(np.sum(a, axis = 0))),:]
        a = a[:, 0] # We choose the maximum value here. Can be changed if something else is needed
    
    a = np.reshape(a, (a.shape[0],))    
        
    return ar, l, w, a, l_min

def getDistances(x, y):
    DistanceMatrix = np.zeros((len(x), len(y)))
    
    for i, valx in enumerate(x):
        for j, valy in enumerate(y):
            DistanceMatrix[i][j] = abs(x[j]-valx)+abs(valy-y[i])
                           
    return DistanceMatrix

def divisor(n):
    for i in range(n):
        x = [i for i in range(1,n+1) if not n % i]
        print(i)
    return x

''' 
Friedhof der Code-Schnipsel:
    
1) np.array der Länge X im Bereich A,B mit nur eindeutigen Werten herstellen:
    
    from numpy.random import default_rng
    rng = default_rng()
    numbers = rng.choice(range(A,B), size=X, replace=False)

2) Pygame rendering:
    

pygame.init()

font = pygame.font.SysFont('Arial', 10)
      
# Setting up color objects
color_dict = {
        0: (255, 255, 255), # white
        1: (255, 0, 0),     # red
        2: (0, 255, 0),     # green
        3: (0, 0, 255),      # blue
        4: (0,0,0)} 

pygame.display.set_caption("FBS")

screen = pygame.display.set_mode((SCALE*self.W ,SCALE*self.L))

# Fill background
background = pygame.Surface(screen.get_size())
#background = background.convert()
background.fill((255, 255, 255))

for i in range(len(self.state)):       
    #pygame.draw.rect(screen, color_dict[default_rng().choice(range(len(color_dict)-1))], (SCALE*(self.fac_x[i]-0.5*self.fac_widths[i]), SCALE*(self.fac_y[i]-0.5*self.fac_lengths[i]), SCALE*0.99*(self.fac_widths[i]), SCALE*0.99*self.fac_lengths[i]))
    pygame.draw.rect(screen, color_dict[0], (SCALE*(self.fac_x[i]-0.5*self.fac_widths[i]), SCALE*(self.fac_y[i]-0.5*self.fac_lengths[i]), SCALE*0.99*(self.fac_widths[i]), SCALE*0.99*self.fac_lengths[i]),3)
    screen.blit(font.render(str(self.state[i]), True, color_dict[0]), (SCALE*self.fac_x[i], SCALE*self.fac_y[i]))

pygame.display.update()  

3) Get Bays the hard way
    
def getBays(self, bays): #Deprecated as of 15.10.2020
    temp_bay = np.zeros(len(self.bay))
    temp_state = np.array(0)
        
    for b in bays:
        temp_bay[b[-1]-1] = 1
        temp_state = np.append(temp_state, b)
    
    temp_state = np.delete(temp_state, [0])
    
    return np.array(temp_state), np.array(temp_bay)

'''
