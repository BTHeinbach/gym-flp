# -*- coding: utf-8 -*-
"""
Created on Wed May  4 15:20:31 2022

@author: Shimraz
"""

import math
import os
import pickle
import anytree
import gym
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
from anytree import Node
from gym import spaces
from numpy.random import default_rng
from gym_flp import rewards

class ofpEnvs (gym.Env):
    metadata = {'render.modes': ['rgb_array', 'human']} 
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
        
      2022 update:
      machine class implemented  
      '''    
    
    def __init__(self, mode = None, instance = None, distance = None, aspect_ratio = None, step_size = None, greenfield = None):
        self.mode = mode
        print("Shimraz")
        self.instance = instance 
        self.distance = distance
        self.aspect_ratio = 2 if aspect_ratio is None else aspect_ratio
        self.step_size = 1 if step_size is None else step_size
        self.greenfield = False if greenfield is None else greenfield
        self.num_envs =1  
        __location__ = os.path.realpath(os.path.join(os.getcwd(), os.path.dirname(__file__)))
        
        self.problems, self.FlowMatrices, self.sizes, self.LayoutWidths, self.LayoutLengths = pickle.load(open(os.path.join(__location__,'continual', 'cont_instances.pkl'), 'rb'))
        # print(self.FlowMatrices)
        while not (self.instance in self.FlowMatrices.keys() or self.instance in ['Brewery']):
            print('Available Problem Sets:', self.FlowMatrices.keys())
            self.instance = input('Pick a problem:').strip()
     
        self.F = self.FlowMatrices[self.instance]
        self.n = self.problems[self.instance]
        self.AreaData = self.sizes[self.instance]
        self.beta, self.fac_length_x, self.fac_width_y, self.fac_area, self.min_side_length = getAreaData(self.AreaData) #Investigate available area data and compute missing values if needed
        print(self.beta, self.fac_length_x, self.fac_width_y, self.fac_area, self.min_side_length)
        if self.fac_width_y is None or self.fac_length_x is None:
            self.fac_length_x = np.random.randint(self.min_side_length*self.aspect_ratio, np.min(self.fac_area), size=(self.n, ))
            self.fac_width_y = np.round(self.fac_area/self.fac_length_x)
        # print("FLOW MATRICES",self.F)    
        # Check if there are Layout Dimensions available, if not provide enough (sqrt(a)*1.5)
        if self.instance in self.LayoutWidths.keys() and self.instance in self.LayoutLengths.keys():
            self.plant_X = int(self.LayoutLengths[self.instance]) # We need both values to be integers for converting into image
            self.plant_Y = int(self.LayoutWidths[self.instance]) 
        else:
            self.plant_area = np.sum(self.fac_area)
            # Design a squared plant layout
            self.plant_X = int(round(math.sqrt(self.plant_area),0)) # We want the plant dimensions to be integers to fit them into an image
            self.plant_Y = self.plant_X 
        
        if self.greenfield:
            self.plant_X = 2*self.plant_X
            self.plant_Y = 2*self.plant_Y
            
        # These values need to be set manually, e.g. acc. to data from literature. Following Eq. 1 in Ulutas & Kulturel-Konak (2012), the minimum side length can be determined by assuming the smallest facility will occupy alone. 
        self.aspect_ratio = int(max(self.beta)) if not self.beta is None else self.aspect_ratio
        self.min_side_length = 1
        self.min_width = self.min_side_length * self.aspect_ratio

        # 3. Define the possible actions: 5 for each box [toDo: plus 2 to manipulate sizes] + 1 idle action for each and respective action_space
        # action_set = ['N', 'E', 'S', 'W']
        # self.action_list = [action_set[i] for j in range(self.n) for i in range(len(action_set))]
        # self.action_space = spaces.Discrete(len(self.action_list)) #5 actions for each facility: left, up, down, right, rotate + idle action across all
        
        # action_low = np.array([1,-1,-1])
        # action_high = np.array([self.n, 1,1])
        
        # action space
        action_low = np.zeros(self.n*2)
        action_high =  np.ones(self.n*2)
        
        # action_spaces =  np.ones(self.n*2, np.int8)*3 # expt-3
        # self.action_space = spaces.MultiDiscrete(action_spaces ,dtype= np.int8) # expt-3 & 4 
        
        # print(self.action_space)
        # self.action_space = spaces.Box(action_low,action_high, 
        #                       shape=(1,) ,dtype=np.float32) 
        # 4. Define observation_space for human and rgb_array mode 
        # Formatting for the observation_space:
        # [facility y, facility x, facility width, facility length] --> [self.fac_y, self.fac_x, self.fac_width_y, self.fac_length_x]
        
        if self.mode == "rgb_array":
            if self.plant_Y < 36 or self.plant_X < 36:
                self.plant_Y, self.plant_X = 36, 36
        # print("---fac---",self.fac_width_y, self.plant_Y, self.fac_length_x,self.plant_X)
        #self.action_space = spaces.MultiDiscrete(np.array([self.n,self.plant_X,self.plant_Y]),dtype= np.int8) # expt-1 & 2
        # expt-4
        # lis = [[]] * self.n
        # lis[0].extend([self.plant_X,self.plant_Y])
        
        self.action = np.tile(np.array([self.plant_X,self.plant_Y], np.int8), self.n)
        self.action_spaces = self.action / np.max(self.action)
        # self.action_space = spaces.MultiDiscrete(action_spaces ,dtype= np.int8) # expt- 4 
        
        self.action_space = spaces.Box(low=action_low, high=self.action_spaces, 
                                      dtype = np.float16)
        
        print(self.action_space)
        print(self.action_space.sample())
        self.lower_bounds = {'Y': max(self.fac_width_y)/2,
                             'X': max(self.fac_length_x)/2,
                             'y': min(self.fac_width_y),
                             'x': min(self.fac_length_x)}
        # print(self.lower_bounds)
        self.upper_bounds = {'Y': self.plant_Y - max(self.fac_width_y)/2,
                             'X': self.plant_X - max(self.fac_length_x)/2,
                             'y': max(self.fac_width_y),
                             'x': max(self.fac_length_x)}
        # print(self.upper_bounds)
        observation_low = np.zeros(4* self.n)
        observation_high = np.zeros(4* self.n)
        
        observation_low[0::4] = self.lower_bounds['Y']
        observation_low[1::4] = self.lower_bounds['X']
        observation_low[2::4] = self.lower_bounds['y']
        observation_low[3::4] = self.lower_bounds['x']
        
        observation_high[0::4] = self.upper_bounds['Y']
        observation_high[1::4] = self.upper_bounds['X']
        observation_high[2::4] = self.upper_bounds['y']
        observation_high[3::4] = self.upper_bounds['x'] 
        # print(observation_low)
        #Keep a version of this to sample initial states from in reset()
        self.state_space = spaces.Box(low=observation_low, high=observation_high, 
                                      dtype = np.uint8) 
        
        # print(self.state_space)
        if self.mode == "rgb_array":
            self.observation_space = spaces.Box(low = 0, high = 255, 
                                                shape= (self.plant_Y, self.plant_X, 3), 
                                                dtype = np.uint8) # Image representation, channel-last for PyTorch CNNs

        elif self.mode == "human":
            self.observation_space = spaces.Box(low=observation_low, high=observation_high, 
                                                dtype =  np.uint8) # Vector representation of coordinates
        else:
            print("Nothing correct selected")
        # print(self.observation_space)   
        # 5. Set some starting points
        self.reward = 0
        self.state = None # Variable for state being returned to agent
        self.internal_state = None #Placeholder for state variable for internal manipulation in rgb_array mode
        self.counter = 0
        self.pseudo_stability = 50 #If the reward has not improved in the last 100 steps, terminate the episode
        self.best_reward = None
        self.reset_counter = 0
        self.MHC = rewards.mhc.MHC() 
       
    def reset(self):

        state_prelim = self.state_space.sample()
        # print(state_prelim)
        # print(self.fac_width_y,self.fac_length_x)
        state_prelim[2::4] = self.fac_width_y
        state_prelim[3::4] = self.fac_length_x
        # print(state_prelim)
        i=0
        while self.collision_test(state_prelim) > 0:
            # print(self.collision_test(state_prelim))
            # print(state_prelim)
            state_prelim = self.state_space.sample()
            state_prelim[2::4] = self.fac_width_y
            state_prelim[3::4] = self.fac_length_x
            i += 1
            if i > 1000:
                break
        
        # print(state_prelim)
        # Create fixed positions for reset:
        Y = np.floor(np.outer(np.array([0,0.25,0.5,0.75,1]),self.upper_bounds['Y']))
        X = np.floor(np.outer([0, 1/3, 2/3, 1],self.upper_bounds['X']))
        
        if self.n==12:
            
            y_centroids = np.tile(np.floor([(i+j)/2 for i,j in zip(Y[:,-1], Y[1:,])]).flatten(),3)
            x_centroids = np.tile(np.floor([(i+j)/2 for i,j in zip(X[:,-1], X[1:,])]),4).flatten()
            
            state_prelim[0::4] = y_centroids
            state_prelim[1::4] = x_centroids
        
        elif self.n==6:
            '''
            state_prelim[0]=np.floor(self.upper_bounds['Y'])/2
            state_prelim[1]=np.floor(self.upper_bounds['X'])/2
            state_prelim[4]=np.floor(self.upper_bounds['Y'])-1
            state_prelim[5]=np.floor(self.lower_bounds['X'])+1
            state_prelim[8]=np.floor(self.lower_bounds['Y'])+1
            state_prelim[9]=np.floor(self.lower_bounds['X'])+1
            state_prelim[12]=np.floor(self.upper_bounds['Y'])-1
            state_prelim[13]=np.floor(self.upper_bounds['X'])-1
            state_prelim[16]=np.floor(self.upper_bounds['Y'])/2
            state_prelim[17]=np.floor(self.upper_bounds['X'])-1
            state_prelim[20]=np.floor(self.lower_bounds['Y'])+1
            state_prelim[21]=np.floor(self.upper_bounds['X'])-1
            '''
            
            #Shuffle
            #u.re.
            state_prelim[0]=np.floor(self.upper_bounds['Y'])-1
            state_prelim[1]=np.floor(self.upper_bounds['X'])-1
            
            #o.re.
            state_prelim[4]=np.floor(self.lower_bounds['Y'])+2
            state_prelim[5]=np.floor(self.upper_bounds['X'])-1
            
            #Mitte
            state_prelim[8]=np.floor(self.upper_bounds['Y'])/2
            state_prelim[9]=np.floor(self.upper_bounds['X'])/2
            
            #u.li.
            state_prelim[12]=np.floor(self.upper_bounds['Y'])-1
            state_prelim[13]=np.floor(self.lower_bounds['X'])+2
            
            #Mitte re.
            state_prelim[16]=np.floor(self.upper_bounds['Y'])/2
            state_prelim[17]=np.floor(self.upper_bounds['X'])-1
            
            #o.li.
            state_prelim[20]=np.floor(self.lower_bounds['Y'])+2
            state_prelim[21]=np.floor(self.lower_bounds['X'])+2
        
        self.internal_state = np.array(state_prelim)
        # print('HIiiii',self.internal_state)
        self.state = np.array(self.internal_state) if self.mode == 'human' else self.ConvertCoordinatesToState(self.internal_state)
        self.counter = 0
        
        self.D = getDistances(state_prelim[1::4], state_prelim[0::4])
        mhc, self.TM = self.MHC.compute(self.D, self.F, np.array(range(1,self.n+1)))
        self.last_cost = mhc
        # print(self.state)
        # print("hi")
        return np.array(self.state)
    
    def collision_test(self, state):
        
        y=state[0::4]
        x=state[1::4]
        w=state[2::4]
        l=state[3::4]
        
        collisions = 0
        
        for i in range(0,self.n-1):
            for j in range(i+1, self.n):
                if not (x[i]+0.5*l[i] <= x[j]-0.5*l[j] or 
                        x[i]-0.5*l[i] >= x[j]+0.5*l[j] or
                        y[i]+0.5*w[i] <= y[j]-0.5*w[j] or
                        y[i]-0.5*w[i] >= y[j]+0.5*w[j]):
                    collisions +=1
                    break
        return collisions
    
    def step(self, action):        
        # m = np.int(np.ceil((action+1)/4))   # Facility on which the action is
        # m = int(np.floor(action[0])) # expt 1 and 2
        # print('actionnn',action)
        # print(np.max(self.action) )
        # action = action * np.max(self.action) 
        # print('unnormalized action', action)
        # print(m)
        step_size = self.step_size       
        # print(step_size)
        
        temp_state = np.array(self.internal_state) # Get copy of state to manipulate:
        old_state = np.array(self.internal_state)  # Keep copy of state to restore if boundary condition is met       
        done = False
        # print(temp_state)
        # Do the action expt - 0
        '''
                # if self.action_list[action] == "S":
        #     temp_state[4*(m-1)] += step_size

        # elif self.action_list[action] == "N": 
        #     temp_state[4*(m-1)] -= step_size
                
        # elif self.action_list[action] == "W": 
        #     temp_state[4*(m-1)+1] -= step_size
                      
        # elif self.action_list[action] == "E": 
        #     temp_state[4*(m-1)+1] += step_size 
                                
        # elif self.action_list[action] == "keep":
        #     temp_state = temp_state
        
        # else:
        #     raise ValueError("Received invalid action={} which is not part of the action space".format(action))
        
        # if action[1] == 0:
        #     temp_state = temp_state

        # elif action[1] == 1: #E
        #     temp_state[4*(m-1)+1] += step_size
            
            
        # elif action[1] == 2: #N
        #     temp_state[4*(m-1)] -= step_size
            
        # elif action[1] == 3: #W
        #     temp_state[4*(m-1)+1] -= step_size
            
            
        # elif action[1] == 4: #S
        #     temp_state[4*(m-1)] += step_size                
        # else:
        #     raise ValueError("Received invalid action={} which is not part of the action space".format(action[1]))        
        '''
        # expt 1
        '''
        # if action[1] == 0:
        #     temp_state = temp_state

        # elif action[1] == 1: #N
        #     temp_state[4*(m-1)] -= step_size
            
            
        # elif action[1] == 2: #S
        #     temp_state[4*(m-1)] += step_size
            
        # else:
        #     raise ValueError("Received invalid action={} which is not part of the action space".format(action[1]))
             

        # if action[2] == 0:
        #     temp_state = temp_state
        
        # elif action[2] ==1: # E
        #     temp_state[4*(m-1)+1] += step_size
                      
        # elif action[2] == 2:  # W
        #     temp_state[4*(m-1)+1] -= step_size
        
        # else:
        #     raise ValueError("Received invalid action={} which is not part of the action space".format(action[2]))
        ''' 
        # expt 2
        '''if action[1]>=0:
            temp_state[4*(m-1)+1] = action[1]

        else:
            raise ValueError("Received invalid action={} which is not part of the action space".format(action[1]))
             

        if action[2]>=0:
            
            temp_state[4*(m-1)] = action[2]
        
        else:
            raise ValueError("Received invalid action={} which is not part of the action space".format(action[2]))
         '''
        # expt 3
        '''
        for i in range(len(action)):
            m = np.int(np.ceil((i+1)/2)) # expt 3 and 4
            # print("m = ",m)
            if i % 2 != 0:
                if action[i] == 0:
                    temp_state = temp_state
                
                elif action[i] ==1: # N
                    temp_state[4*(m-1)] -= step_size
                              
                elif action[i] == 2:  # S
                    temp_state[4*(m-1)] += step_size
                
                else:
                    raise ValueError("Received invalid action={} which is not part of the action space".format(action[2]))
             
            else:        
                if action[i] == 0:
                    temp_state = temp_state
                
                elif action[i] ==1: # E
                    temp_state[4*(m-1)+1] += step_size
                              
                elif action[i] == 2:  # W
                    temp_state[4*(m-1)+1] -= step_size
                
                else:
                    raise ValueError("Received invalid action={} which is not part of the action space".format(action[2]))
        '''
        # expt 4
        # print(action)
        # print(temp_state)
        
        for i in range(0, len(action), 2):
            m = np.int(np.ceil((i+1)/2)) # expt 3 and 4
            # print("m = ",m)
            if action[i] >= 0 and action[i+1] >= 0:
                temp_state[4*(m-1)] = action[i]
                temp_state[4*(m-1)+1] = action[i+1]
            else:
                raise ValueError("Received invalid action={} which is not part of the action space".format(action[2]))
        
        # print("temp",temp_state)
        
        '''
        for i in range(len(action)):
            m = i+1 # expt 4
            if action[i][0] >= 0:
                temp_state[4*(m-1)+1] = action[i][0]
    
            else:
                raise ValueError("Received invalid action={} which is not part of the action space".format(action[1]))
                 
    
            if action[i][1] >= 0:
                
                temp_state[4*(m-1)] = action[i][1]
            
            else:
                raise ValueError("Received invalid action={} which is not part of the action space".format(action[2]))
            ''' 
                    
        
        
        # if action[2] < 0:
        #     temp_state[4*(m-1)+1] += action[2]
                      
        # elif action[2] >= 0:
        #     temp_state[4*(m-1)+1] += action[2] 
        
        # else:
        #     raise ValueError("Received invalid action={} which is not part of the action space".format(action[2]))
        # temp_state[4*(m-1)] += step_size
        # print(temp_state)
        
        
        
        
        self.D = getDistances(temp_state[1::4], temp_state[0::4])
        # print(temp_state,temp_state[1::4], temp_state[0::4], self.D)
        mhc, self.TM = self.MHC.compute(self.D, self.F, np.array(range(1,self.n+1)))   
        # print( mhc, self.TM)
        
        if not self.state_space.contains(temp_state):
            done = True
            penalty = -1
            temp_state = np.array(old_state)
        else:
            penalty = 0

        # #2 Test if initial state causing a collision. If yes than initialize a new state until there is no collision
        collisions = self.collision_test(temp_state) # Pass every 4th item starting at 0 (x pos) and 1 (y pos) for checking 
        collision_penalty = -1 if collisions>0 else 0

                # Make new state for observation
        self.internal_state = np.array(temp_state) # Keep a copy of the vector representation for future steps
        self.state = self.ConvertCoordinatesToState(np.array(self.internal_state)) if self.mode == 'rgb_array' else np.array(self.internal_state)
        # print(self.internal_state)
                # Make rewards for observation
        if mhc < self.last_cost:
            self.last_cost = mhc
            self.counter = 0
            cost_penalty = 1
        else:
            self.counter +=1
            cost_penalty =0

        reward = penalty + cost_penalty + collision_penalty
        
        # Check for terminality for observation
        if self.counter >= self.pseudo_stability:
            done = True 
        
        return np.array(self.state), reward, done,  {'mhc': mhc}        
    
    def ConvertCoordinatesToState(self, state_prelim):    
        data = np.zeros((self.plant_Y, self.plant_X, 3),dtype=np.uint8)
        
        sources = np.sum(self.F, axis = 1)
        sinks = np.sum(self.F, axis = 0)
        
        p = np.arange(self.n)
        
        #R = np.array((p-np.min(p))/(np.max(p)-np.min(p))*255).astype(int)
        R = np.ones(shape=(self.n,)).astype(int)*255
        G = np.array((sources-np.min(sources))/(np.max(sources)-np.min(sources))*255).astype(int)
        B = np.array((sinks-np.min(sinks))/(np.max(sinks)-np.min(sinks))*255).astype(int)
       
        for x, p in enumerate(p):
            y_from = state_prelim[4*x+0] -0.5 * state_prelim[4*x+2]
            x_from = state_prelim[4*x+1] -0.5 * state_prelim[4*x+3]
            y_to = state_prelim[4*x+0] + 0.5 * state_prelim[4*x+2]
            x_to = state_prelim[4*x+1] + 0.5 * state_prelim[4*x+3]
        
            data[int(y_from):int(y_to), int(x_from):int(x_to)] = [R[p-1], G[p-1], B[p-1]]
        return np.array(data, dtype=np.uint8)
        
    def render(self, mode = None):       
        return Image.fromarray(self.ConvertCoordinatesToState(self.internal_state), 'RGB') #Convert channel-first back to channel-last for image display
        
    def close(self):
        pass
    
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
    return np.array([[abs(float(x[j])-float(valx))+abs(float(valy)-float(y[i])) for (j, valy) in enumerate(y)] for (i, valx) in enumerate(x)],dtype=float)    
                      
def divisor(n):
    for i in range(n):
        x = [i for i in range(1,n+1) if not n % i]
        print(i)
    return x
