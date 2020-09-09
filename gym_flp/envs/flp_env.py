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


'''
v0.0.2
Significant changes:
    08.09.2020:
    - Dicrete option removed from spaces; only Box allowed 
    - Classes for quadtratic set covering and mixed integer programming (-ish) added
    - Episodic tasks: no more terminal states (exception: max. no. of trials reached)
'''

class qapEnv(gym.Env):
    metadata = {'render.modes': ['rgb_array']}  

      
    
    def __init__(self):
        self.test = "Quadratic Assignment Problem"
        __location__ = os.path.realpath(os.path.join(os.getcwd(), os.path.dirname(__file__)))
        self.DistanceMatrices, self.FlowMatrices = pickle.load(open(os.path.join(__location__,'qap_matrices.pkl'), 'rb'))
        self.transport_intensity = None
        self.instance = None
        
        while not (self.instance in self.DistanceMatrices.keys() or self.instance in self.FlowMatrices.keys() or self.instance in ['Neos-n6', 'Neos-n7', 'Brewery']):
           #print('Available Problem Sets:', self.DistanceMatrices.keys())
            self.instance = input('Pick a problem:')
     
            self.D = self.DistanceMatrices[self.instance]
            self.F = self.FlowMatrices[self.instance]
        
        # Determine problem size relevant for much stuff in here:
        self.n = len(self.D[0])
        
        # Beta: prevent problems smaller than 4 and greater than 10.
        # while self.n>11 or self.n<4:
        #   self.n = int(input('Ouch! Problem size unsuitable. Pick a number greater than 4 and smaller than 10.'))

        
        # Action space has two option:
        # 1) Define as Box with shape (1, 2) and allow values to range from 1 through self.n 
        # 2) Define as Discrete with x = 1+((n^2-n)/2) actions (one half of matrix + 1 value from diagonal) --> Omit "+1" to obtain range from 0 to x!
        # self.action_space = spaces.Box(low=-1, high=6, shape=(1,2), dtype=np.int) # Doubles complexity of the problem as it allows the identical action (1,2) and (2,1)
        self.action_space = spaces.Discrete(int((self.n**2-self.n)*0.5)+1)
                
        # If you are using images as input, the input values must be in [0, 255] as the observation is normalized (dividing by 255 to have values in [0, 1]) when using CNN policies.
        # For problems smaller than n = 10, the program will allow discrete state spaces; for everything above: only boxes 
        
        self.observation_space = spaces.Box(low=1, high = self.n, shape=(self.n,), dtype=np.float32)
        
        self.states = {}    # Create an empty dictonary where states and their respective reward will be stored for future reference
        self.actions = self.actionsMaker(self.n)
        
        # Initialize Environment with empty state and action
        self.action = None
        self.state = None
        
        #Initialize moving target to incredibly high value. To be updated if reward obtained is smaller. 
        self.movingTargetReward = np.inf 
        self.MHC = rewards.mhc.MHC()    # Create an instance of class MHC in module mhc.py from package rewards
        
    def sampler(self):
            return default_rng().choice(range(1,self.n+1), size=self.n, replace=False) 
        
    def actionsMaker(self, x):
        actions = {}
        actions[0] = tuple([1,1])
        cnt = 1
        for idx in range(1,x):
            for idy in range(idx + 1, x+1):
                skraa = tuple([idx, idy])
                actions[cnt] = skraa
                cnt +=1        
        
        # Add idle action to dictionary
        
        return actions
                
                
    def step(self, action):
        # Create new State based on action 
        
        swap = self.actions[action]
        
        fromState = np.array(self.state)
        fromState[swap[0]-1], fromState[swap[1]-1] = fromState[swap[1]-1], fromState[swap[0]-1]
        
        MHC, self.TM = self.MHC.compute(self.D, self.F, fromState)   
        
        self.states[tuple(fromState)] = MHC
        
        if self.movingTargetReward == np.inf:
            self.movingTargetReward = MHC 
     
        reward = self.movingTargetReward - MHC
        self.movingTargetReward = MHC if MHC < self.movingTargetReward else self.movingTargetReward

        newState = np.array(fromState)
        
        #Return the step funtions observation: new State as result of action, reward for taking action on old state, done=False (no terminal state exists), None for no diagostic info
        return newState, reward, False, None
          
    def reset(self):
        return self.sampler()

    def render(self, mode):
        if mode == "rgb_array":
                
            scale = 50  # Scale size of pixels for displayability
            img_h, img_w = scale, (len(self.state))*scale
            data = np.zeros((img_h, img_w, 3), dtype=np.uint8)
            
            sources = np.sum(self.TM, axis = 1)
            sinks = np.sum(self.TM, axis = 0)
            
            R = np.array((self.state-np.min(self.state))/(np.max(self.state)-np.min(self.state))*255).astype(int)
            G = np.array((sources-np.min(sources))/(np.max(sources)-np.min(sources))*255).astype(int)
            B = np.array((sinks-np.min(sinks))/(np.max(sinks)-np.min(sinks))*255).astype(int)
            
            for i, s in enumerate(self.state):
                data[0*50:1*50, i*50:(i+1)*50] = [R[s-1], G[s-1], B[s-1]]
                       
            img = Image.fromarray(data, 'RGB')            
            #img.save('test.png')
            #img.show()
            plt.imshow(img)
            plt.axis('off')
            plt.show()

class qspEnv(gym.Env):
    metadata = {'render.modes': ['rgb_array']}          
    
    def __init__(self):
        self.test = 'Flexible Bay Structure'
        #print('This gym provides an implementation of', self. test)
    
    def step(action):
        ...
    
    def close():
        ...
        
    def render():
        ...
    
    def reset():
        ...
        
class mipEnv(gym.Env):
    metadata = {'render.modes': ['rgb_array']}          
    
    def __init__(self):
        self.test = 'Slicing Tree(-ish)'
        #print('This gym provides an implementation of', self. test)
    
    def step(action):
        ...
    
    def close():
        ...
        
    def render():
        ...
    
    def reset():
        ...

''' 
Friedhof der Code-Schnipsel:
    
1) np.array der Länge X im Bereich A,B mit nur eindeutigen Werten herstellen:
    
    from numpy.random import default_rng
    rng = default_rng()
    numbers = rng.choice(range(A,B), size=X, replace=False)
'''