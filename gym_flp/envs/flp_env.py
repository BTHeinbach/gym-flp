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
import pygame
from IPython.display import display, clear_output

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
        return default_rng().choice(range(1,self.n+1), size=self.n, replace=False) 

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

class fbsEnv(gym.Env):
    metadata = {'render.modes': ['rgb_array', 'human']}          
    
    def __init__(self):
        self.test = 'Flexible Bay Structure'
        self.n = 10 #Set to 6 for testing purposes
        self.W, self.H, self.A, self.chi, self.h, self.w, self.a, self.F = self.getAreaData(self.n) #Pass 6 or 12 in test version
        self.action_space = spaces.Discrete(4)  #Taken from doi:10.1016/j.engappai.2020.103697
        self.actions = {0: 'Randomize', 1: 'Bit Swap', 2: 'Bay Exchange', 3: 'Inverse'}
        self.observation_space = spaces.Box(low=1, high = self.n, shape=(self.n,), dtype=np.float32)
        self.bay_space = spaces.Box(low=0, high = 1, shape=(self.n,), dtype=np.int) # binary vector indicatin bay breaks (i = 1 means last facility in bay)
        self.state = None # Permutation of all n facilities, read from top to bottom
        self.bay = None
        self.aspect_ratio = 5
        self.MHC = rewards.mhc.MHC() 
        
    
    def sampler(self):
            return default_rng().choice(range(1,self.n+1), size=self.n, replace=False)
        
    def step(self, action):
        a = self.actions[action]
        #k = np.count_nonzero(self.bay)
        fromState = np.array(self.state)
        
        # Get lists with a bay positions and facilities in each bay
        facilities = np.where(self.bay==1)[0]
        bay_breaks = np.split(self.bay, facilities[:-1]+1)
        
        # Load indiv. facilities into bay acc. to breaks; omit break on last position to avoid empty array in list.
        bays = np.split(self.state, facilities[:-1]+1)
        
        if a == 'Randomize':
            # Two vector elements randomly chosen are exchanged. Bay vector remains untouched.
            k = default_rng().choice(range(len(self.state-1)), size=1, replace=False)
            l = default_rng().choice(range(len(self.state-1)), size=1, replace=False)
            fromState[k], fromState[l] = fromState[l], fromState[k]
            self.state = np.array(fromState)
            
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
            self.state = np.array(new_state)
            self.bay = np.array(new_bay)
            
        elif a == 'Inverse':
            #Facilities present in a certain bay randomly chosen are inverted.
            q = default_rng().choice(range(len(bays)))
            bays[q] = np.flip(bays[q])
            
            new_bay = np.concatenate(bay_breaks)
            new_state = np.concatenate(bays)
            
            # Make sure state is saved as copy
            self.state = np.array(new_state)
            self.bay = np.array(new_bay)
            
        self.fac_widths, self.fac_lengths, self.fac_x, self.fac_y = self.getCoordinates()
        self.D = self.getDistances(self.fac_x, self.fac_y)
        
        reward, self.TM = self.MHC.compute(self.D, self.F, fromState)   
        
        
        return self.state[:], self.bay[:], reward, None, None
    
    def getBays(self, bays): #Deprecated as of 15.10.2020
        temp_bay = np.zeros(len(self.bay))
        temp_state = np.array(0)
            
        for b in bays:
            temp_bay[b[-1]-1] = 1
            temp_state = np.append(temp_state, b)
        
        temp_state = np.delete(temp_state, [0])
        
        return np.array(temp_state), np.array(temp_bay)
            
    def getDistances(self, x, y):
        DistanceMatrix = np.zeros((len(x), len(y)))
        
        for i, valx in enumerate(x):
            for j, valy in enumerate(y):
                DistanceMatrix[i][j] = abs(x[j]-valx)+abs(valy-y[i])
                
        
        #for ix, valx in enumerate(x):
       #     for jy, valy in enumerate(y):
       #        DistanceMatrix[ix][jy] = Rectangular(valx, x[jy], valy, y[ix])
                
        return DistanceMatrix

    def Rectangular(x,y):
        ...
        
    def close(self):
        self.close()
        
    def getCoordinates(self):
        breaks = np.where(self.bay==1)[0]  #Read all positions with a bay break
        
        fac_widths = np.zeros((len(self.state,)))
        fac_lengths = np.zeros((len(self.state,)))
        fac_x = np.zeros((len(self.state,)))
        fac_y = np.zeros((len(self.state,)))
        
        SCALE = 10
              
        x = 0
        start = 0
        #Get Width of all break facilities
        for b in breaks:
            end = b
            bay = self.state[start:end+1] #Get the facilities that are located in the bay
            
            areas = self.a[bay-1] #Get the area associated with the facilities
            
            fac_widths[start:end+1] = np.sum(areas)/self.H #Calculate all facility widhts in bay acc. to https://doi.org/10.1016/j.eswa.2011.11.046
            fac_lengths[start:end+1] = areas/fac_widths[start:end+1]
            
            fac_x[start:end+1] = fac_widths[start:end+1] * 0.5 + x 
            x += np.sum(areas)/self.H
            
            y = np.ones(len(bay))
            ll = 0
            for idx, l in enumerate(fac_lengths[start:end+1]):
                y[idx] = ll + 0.5*l
                ll += l
            fac_y[start:end+1] = y       
            
            start = end + 1
        
        return fac_widths, fac_lengths, fac_x, fac_y
    
    def render(self, mode = 'rgb_array'):
        
        SCALE = 10 
        #clear_output(wait = False)
        
        if mode == "rgb_array":
            if self.state.ndim==1:
                  
                
                img_w, img_h = SCALE*self.W ,SCALE*self.H
                data = np.zeros((img_h, img_w, 3), dtype=np.uint8)
                
                sources = np.sum(self.TM, axis = 1)
                sinks = np.sum(self.TM, axis = 0)
                
                R = np.array((self.state-np.min(self.state))/(np.max(self.state)-np.min(self.state))*255).astype(int)
                G = np.array((sources-np.min(sources))/(np.max(sources)-np.min(sources))*255).astype(int)
                B = np.array((sinks-np.min(sinks))/(np.max(sinks)-np.min(sinks))*255).astype(int)
                
                for i, s in enumerate(self.state):
                    data[SCALE*int(self.fac_y[i]-0.5*self.fac_lengths[i]):SCALE*int(self.fac_y[i]+0.5*self.fac_lengths[i]), SCALE*int(self.fac_x[i]-0.5*self.fac_widths[i]):SCALE*int(self.fac_x[i]+0.5*self.fac_widths[i])] = [R[s-1], G[s-1], B[s-1]]
                           
                img = Image.fromarray(data, 'RGB')            
                #plt.figure()
                fig = plt.imshow(img)
                plt.axis('off')
                
                display(fig)
                
                
                #plt.show()
                
                #plt.clear()
                plt.close("all")
        else:
            
            #pygame.quit()
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
            
            screen = pygame.display.set_mode((SCALE*self.W ,SCALE*self.H))
            
            # Fill background
            background = pygame.Surface(screen.get_size())
            #background = background.convert()
            background.fill((255, 255, 255))
            
            for i in range(len(self.state)):       
                #pygame.draw.rect(screen, color_dict[default_rng().choice(range(len(color_dict)-1))], (SCALE*(self.fac_x[i]-0.5*self.fac_widths[i]), SCALE*(self.fac_y[i]-0.5*self.fac_lengths[i]), SCALE*0.99*(self.fac_widths[i]), SCALE*0.99*self.fac_lengths[i]))
                pygame.draw.rect(screen, color_dict[0], (SCALE*(self.fac_x[i]-0.5*self.fac_widths[i]), SCALE*(self.fac_y[i]-0.5*self.fac_lengths[i]), SCALE*0.99*(self.fac_widths[i]), SCALE*0.99*self.fac_lengths[i]),3)
                screen.blit(font.render(str(self.state[i]), True, color_dict[0]), (SCALE*self.fac_x[i], SCALE*self.fac_y[i]))
            
            pygame.display.update()   
            
    def reset(self):
        
        new_bay = self.bay_space.sample()
        new_bay[-1] = 1
        self.bay = new_bay
        self.state = self.sampler()
        
        self.fac_widths, self.fac_lengths, self.fac_x, self.fac_y = self.getCoordinates()
        self.D = self.getDistances(self.fac_x, self.fac_y)
        reward, self.TM = self.MHC.compute(self.D, self.F, self.state[:])
        
        return self.state[:], self.bay[:]
    
    def getAreaData(self, n):
        W6 = 30
        W12 = 60
        
        W10 = 51
        H10 = 25
        A10 = H10 * W10
        
        l10_min = 5
        
        
        A6 = W6*W6
        A12 = W12*W12
        
        chi6 = np.array([1,0,0,1,0,1], dtype = np.int)  #Rotation
        w6 = np.array([5,9,6,6,4,5], dtype = np.float32)
        h6 = np.array([4,8,5,4,4,3], dtype = np.float32)
        a6 = w6*h6
        
        chi12 = np.array([1,0,1,0,0,1,1,0,1,0,0,1], dtype = np.int) #Rotation
        w12 = np.array([5,7,6,4,6,5,10,7,6,5,5,6], dtype = np.float32)
        h12 = np.array([4,5,5,4,6,4,7,5,5,5,5,4], dtype = np.float32)
        a12 = w12*h12
        
        a10 = np.array([238,112,160,80,120,80,60,85,221,119])
        
        F6_1 = np.array([[0,12,10,4,0,0],\
                        [0,0,34,30,0,0],\
                        [0,0,0,6,4,18],\
                        [0,0,0,0,0,0],\
                        [0,0,0,0,0,0]], dtype = np.float32)
        
        
        F10_1 = np.array([[0,0,0, 0, 0,218, 0,  0,  0,  0],\
                          [0,0,0, 0, 0,148, 0,  0,296,  0],\
                          [0,0,0,28,70,  0, 0,  0,  0,  0],\
                          [0,0,0, 0, 0, 28,70,140,  0,  0],\
                          [0,0,0, 0, 0,  0, 0,210,  0,  0],\
                          [0,0,0, 0, 0,  0, 0,  0,  0, 20],\
                          [0,0,0, 0, 0,  0, 0,  0,  0, 28],\
                          [0,0,0, 0, 0,  0, 0,  0,  0,888],\
                          [0,0,0, 0, 0,  0, 0,  0,  0, 59.2],\
                          [0,0,0, 0, 0,  0, 0,  0,  0,  0]])    
        
        F12_1 = np.array([[0,18,6,12,2,20,18,10,38,20,26,26],\
                         [0, 0,0, 0,0, 0, 0, 0, 0, 0,18, 0],\
                         [0, 0,0, 0,4, 4,14,30,16,36,32,38],\
                         [0,0,0,0,0,8, 0, 0, 0, 0, 0, 0, 0],\
                         [0,0,0,0,0,0,10, 2,34,30, 6,14,24],\
                         [0,0,0,0,0,0, 0, 0, 0, 0,14, 0, 0],\
                         [0,0,0,0,0,0, 0, 0,36,12,20, 4,28],\
                         [0,0,0,0,0,0, 0, 0, 0, 0, 0, 6, 0],\
                         [0,0,0,0,0,0, 0, 0, 0, 0, 8,22,12],\
                         [0,0,0,0,0,0, 0, 0, 0, 0, 0, 0, 0],\
                         [0,0,0,0,0,0, 0, 0, 0, 0, 0, 0, 6]])
        
        if n == 6:
            return W6, W6, A6,chi6, h6, w6, a6, F6_1
        elif n ==12:
            return W12, W12, A12, chi12, h12, w12, a12, F12_1
        elif n == 10:
            return W10, H10, A10, None, None, None, a10, F10_1
        
class mipEnv(gym.Env):
    metadata = {'render.modes' : ['human']}
          
    
    def __init__(self, grid_size_x = 100, grid_size_y=100):
        
        # initialize distances for reward function
        self.act_d = 0
        self.prev_d = 0
        self.list_d = []
        
        # initialize reward
        self.reward = 0
        
        # Size of the 2D-grid
        self.grid_size_x = grid_size_x
        self.grid_size_y = grid_size_y
        
        self.box_pos_x = []
        self.box_pos_y = []
        self.last_box_pos_x = []
        self.last_box_pos_y = []
        
        '''
        #For Testing
        import numpy as np
        import gym
        from gym import spaces
        grid_size_x = 100
        grid_size_y = 100
        #End Testing
        '''
    
        # Size of the boxes
        self.n = 3 # number of boxes
        n = self.n
        
        #choose size of boxes for testing
        size_of_boxes = 'big'
        if size_of_boxes == 'small':
            box1_w = 10
            box1_h = 8
            box2_w = 5
            box2_h = 4
            box3_w = 7
            box3_h = 7
        elif size_of_boxes == 'medium':
            box1_w = 20
            box1_h = 16
            box2_w = 10
            box2_h = 8
            box3_w = 14
            box3_h = 14   
        elif size_of_boxes == 'big':
            box1_w = 40
            box1_h = 32
            box2_w = 20
            box2_h = 16
            box3_w = 28
            box3_h = 28    
        elif size_of_boxes == 'extreme':
            box1_w = 60
            box1_h = 50
            box2_w = 40
            box2_h = 35
            box3_w = 40
            box3_h = 40    
        
        
        
        box_w = [box1_w, box2_w, box3_w]
        box_h = [box1_h, box2_h, box3_h]
        
        self.box_w = box_w
        self.box_h = box_h
        
        # Calculate boundaries and write it in a list
        self.boundary_x = []
        self.boundary_y = []
        boundary_x = self.boundary_x
        boundary_y = self.boundary_y
        for x in range(1, n+1):
            boundary_x.append(grid_size_x-box_w[x-1])
            boundary_y.append(grid_size_y-box_h[x-1])
            
        action_dict = {}   
        for i in range(1, n+1):
            action_dict[0+(i-1)*5] = "up"
            action_dict[1+(i-1)*5] = "down"
            action_dict[2+(i-1)*5] = "right"
            action_dict[3+(i-1)*5] = "left"
            action_dict[4+(i-1)*5] = "keep"
        self.action_dict = action_dict
                   
        #action space
        self.action_space = spaces.Discrete(5*n) #5 actions for each facility
            
        # observation space low and high boundaries
        observation_low = []
        for i in range(0, n):
            observation_low.append(0) #box_pos_x
            observation_low.append(0) #box_pos_y
            observation_low.append(box_w[i]) #box_w
            observation_low.append(box_h[i]) #box_h
            observation_low.append(0) #Rotation
            
        observation_high = []
        for i in range(0, n):
            observation_high.append(boundary_x[i]) #box_pos_x
            observation_high.append(boundary_y[i]) #box_pos_y
            observation_high.append(box_w[i]) #box_w
            observation_high.append(box_h[i]) #box_h
            observation_high.append(3) #Rotation
            
        #self.observation_space = spaces.Box(observation_low, observation_high)
        self.observation_space = spaces.Box(np.array(observation_low), np.array(observation_high), dtype = int)
        

    def collision_test(self, box_pos_x, box_pos_y):      
        grid_size_x = self.grid_size_x
        grid_size_y = self.grid_size_y
        n = self.n
        box_w = self.box_w
        box_h = self.box_h
        #box_pos_x = self.box_pos_x 
        #box_pos_y = self.box_pos_y
        
        # collision test - part1: create grid and fill it with boxes
        grid = np.zeros((grid_size_y, grid_size_x))
        for i in range (0, n):
            box = np.zeros((box_h[i], box_w[i]))
            box.fill(i+1)
            grid[(box_pos_y[i]):(box_pos_y[i]+box_h[i]), (box_pos_x[i]):(box_pos_x[i]+box_w[i])] = box
        
        # collision test - part2: test if boxes collide
        collision = False #initialize collision for each collision test
        
        for i in range (0, n-1):
            if np.any(grid[(box_pos_y[i]):(box_pos_y[i]+box_h[i]), (box_pos_x[i]):(box_pos_x[i]+box_w[i])] != i+1) == True:
                collision = True
        
        if collision == True:
            print('collision detected')
                
        return collision

    def step(self, action):        
        n = self.n
        #box_pos_x = self.box_pos_x 
        #box_pos_y = self.box_pos_y
        #box_w = self.box_w
        #box_h = self.box_h
        #grid_size_x = self.grid_size_x
        #grid_size_y = self.grid_size_y
        list_d = self.list_d
        prev_d = self.prev_d
        action_dict = self.action_dict
        boundary_x = self.boundary_x
        boundary_y = self.boundary_y
        
        box_pos_x = self.box_pos_x 
        box_pos_y = self.box_pos_y
        last_box_pos_x = self.last_box_pos_x
        last_box_pos_y = self.last_box_pos_y
        

        
        
        m = np.int(np.ceil((action+1)/5))   # Facility on which the action is
        
        box_pos_x = []
        box_pos_y = []
        for i in range(0, n):
            box_pos_x.append(state[5*i+0])
            box_pos_y.append(state[5*i+1])
        
        # Test if initial state causing a collision. If yes than initialize a new state until there is no collision
        collision = mipEnv.collision_test(self, box_pos_x, box_pos_y)        
        while collision == True:
            new_state = mipEnv.reset(self)
            box_pos_x = []
            box_pos_y = []
            for i in range(0, n):
                box_pos_x.append(new_state[5*i+0])
                box_pos_y.append(new_state[5*i+1])
            collision = mipEnv.collision_test(self, box_pos_x, box_pos_y)        
            print('take new state for the beginning')
        
        
        # check if boundary is reached
        left_bound = []
        right_bound = []
        up_bound = []
        down_bound = []
        for i in range(0,n):
            if box_pos_x[i] == 0:
                left_bound.append(True)
            else:
                left_bound.append(False)
            
            if box_pos_x[i] == boundary_x[i]:
                right_bound.append(True)
            else:
                right_bound.append(False)
                
            if box_pos_y[i] == 0:
                down_bound.append(True)
            else:
                down_bound.append(False)
                
            if box_pos_y[i] == boundary_y[i]:
                up_bound.append(True)
            else:
                up_bound.append(False)
                
                
        # if it is on boundary and action takes it over boundary change action to do nothing so that it will take a new random action in next step
        if any(left_bound) == True and action_dict[action] == "left":
            action = 4  # change action to keep, so that there will be no action in this step
            print('Left bound is reached')
        if any(right_bound) == True and action_dict[action] == "right":
            action = 4
            print('Right bound is reached')
        if any(up_bound) == True and action_dict[action] == "up":
            action = 4
            print('Upper bound is reached')
        if any(down_bound) == True and action_dict[action] == "down":
            action = 4
            print('Lower bound is reached')

        if action_dict[action] == "up":
            box_pos_x[m-1] += 0 
            box_pos_y[m-1] += 1

        elif action_dict[action] == "down":
            box_pos_x[m-1] += 0 
            box_pos_y[m-1] -= 1
            
        elif action_dict[action] == "right":
            box_pos_x[m-1] += 1 
            box_pos_y[m-1] += 0
            
        elif action_dict[action] == "left":
            box_pos_x[m-1] -= 1 
            box_pos_y[m-1] += 0
            
        elif action_dict[action] == "keep":
            box_pos_x[m-1] += 0
            box_pos_y[m-1] += 0
            
        else:
            raise ValueError("Received invalid action={} which is not part of the action space".format(action))
        
        collision = mipEnv.collision_test(self, box_pos_x, box_pos_y)
        
        # if there is a collision, take the last position
        if collision is True:
            box_pos_x = last_box_pos_x
            box_pos_y = last_box_pos_y
            collision = False
            print('Took last position')
        
        # calculate distance between boxes for rewards
        act_d = 0
        d_x = 0
        d_y = 0
        for i in range (0, n-1):
            d_x = abs(box_pos_x[i] - box_pos_x[i+1])
            d_y = abs(box_pos_y[i] - box_pos_y[i+1])
            d_i = math.hypot(d_x, d_y)   # pythagoras -> diagonal distanz of x_i and y_i
            act_d += d_i                     # sum of distances between each boxes
                    
        
        # reward
        if collision == True:
            reward = -100    # punishment if there is a collision
        else:
            if act_d > prev_d:
                reward = -10
            elif act_d < prev_d:
                reward = +10
          
        
        #done
        list_d.append(prev_d-act_d)
        eval_n = 10 #Evaluate last 10 distances
        if len(list_d) > (eval_n+1):
            last_d = sum(list_d[-eval_n:])
            if last_d < 0.1:
                done = True
            else:
                done = False
        else:
            done = False

        
        prev_d = act_d  # set actual distance as previous distance for next step
        
        info = {}
        
        for i in range(0, n):
            state[5*i+0] = box_pos_x[i]
            state[5*i+1] = box_pos_y[i]
            
        self.box_pos_x = box_pos_x
        self.box_pos_y = box_pos_y
        
        # save for the next step the last box positions
        self.last_box_pos_x = box_pos_x
        self.last_box_pos_y = box_pos_y
        
        
        # Für Test
        self.left_bound = left_bound
        self.right_bound = right_bound
        self.up_bound = up_bound
        self.down_bound = down_bound
        
            
        return state, reward, done, info


    
    def reset(self):
        return self.observation_space.sample()
        
        
    def render(self, mode = 'rgb_array', close = False):
                
        import pygame as pg
        import numpy as np        

        n = self.n
        box_pos_x = self.box_pos_x 
        box_pos_y = self.box_pos_y
        box_w = self.box_w
        box_h = self.box_h
        grid_size_x = self.grid_size_x
        grid_size_y = self.grid_size_y

        grid = np.zeros((grid_size_x, grid_size_y))
        
        #box_pos_x = [37, 5, 35]
        #box_pos_y = [49, 5, 61]
        
        for i in range (0, n):
            box = np.zeros((box_w[i], box_h[i]))
            box.fill(i+1)
            grid[(box_pos_x[i]):(box_pos_x[i]+box_w[i]), (box_pos_y[i]):(box_pos_y[i]+box_h[i])] = box
        
        
        # color dictionary, represents white, red, green and blue
        color_dict = {
                0: (255, 255, 255), # white
                1: (255, 0, 0),     # red
                2: (0, 255, 0),     # green
                3: (0, 0, 255)      # blue
                } 
        
        
        
        if mode == 'rgb_array':
            #return np.array(...) # return RGB frame suitable for video
        
            #scale = 1  # Scale size of pixels for displayability
            #img_h, img_w = (grid_size_x * scale), (grid_size_y * scale) #Größe des Bildes – sollte der verfügbaren Hallenfläche entsprechen
            img_h, img_w = (grid_size_x * 1), (grid_size_y * 1)
            grid_img = np.zeros((img_h, img_w, 3), dtype=np.uint8) #Leeres Array
            
            '''
            # Die nächsten 5 Zeilen habe ich nur für die Farbgebung geschrieben; du kannst auch alle Felder in Rot oder so machen und diese Zeilen löschen, s. unten
            sources = np.sum(self.transport_intensity, axis = 1)
            sinks = np.sum(self.transport_intensity, axis = 0)
            
            R = np.array((self.state-np.min(self.state))/(np.max(self.state)-np.min(self.state))*255).astype(int)
            G = np.array((sources-np.min(sources))/(np.max(sources)-np.min(sources))*255).astype(int)
            B = np.array((sinks-np.min(sinks))/(np.max(sinks)-np.min(sinks))*255).astype(int)
            '''            
    
            for i in range(grid.shape[0]):
                for j in range(grid.shape[1]):
                    #cells[i][j] = color_dict[random.randrange(3)]
                    if grid[i][j] == 1:
                        grid_img[i][j] = color_dict[1]
                    elif grid[i][j] == 2:
                        grid_img[i][j] = color_dict[2]
                    elif grid[i][j] == 3:
                        grid_img[i][j] = color_dict[3]
                    else:
                        grid_img[i][j] = color_dict[0]
                        
            grid_img = np.transpose(grid_img,(1,0,2))
            '''
            img = Image.fromarray(grid_img, 'RGB')            
            plt.imshow(img)
            plt.axis('off')
            plt.show()
            '''
            
            img = Image.fromarray(grid_img, 'RGB')            
            fig = plt.figure()
            plt.imshow(img)
            plt.axis('off')
            plt.show()
            plt.close(fig)
        
            
        
        
        if mode == 'human':
            # create a 3D array with XxYx3 (the last dimension is for the RGB color)
            cells = np.ndarray((grid_size_x , grid_size_y, 3))
            
            for i in range(grid.shape[0]):
                for j in range(grid.shape[1]):
                    #cells[i][j] = color_dict[random.randrange(3)]
                    if grid[i][j] == 1:
                        cells[i][j] = color_dict[1]
                    elif grid[i][j] == 2:
                        cells[i][j] = color_dict[2]
                    elif grid[i][j] == 3:
                        cells[i][j] = color_dict[3]
                    else:
                        cells[i][j] = color_dict[0]
                    
                    
            # set the size of the screen as multiples of the array
            cellsize = 10
            WIDTH = cells.shape[0] * cellsize
            HEIGHT = cells.shape[1] * cellsize
            
            # initialize pygame
            pg.init()
            screen = pg.display.set_mode((WIDTH, HEIGHT))
            clock = pg.time.Clock()
            
            #create a surface with the size as the array
            surf = pg.Surface((cells.shape[0], cells.shape[1]))
             # draw the array onto the surface
            pg.surfarray.blit_array(surf, cells)
            # transform the surface to screen size
            surf = pg.transform.scale(surf, (WIDTH, HEIGHT))
            
            # game loop
            running = True
            while running:
                clock.tick(60)
                
                for event in pg.event.get():
                    if event.type == pg.QUIT:
                        running = False
                        
                screen.fill((0, 0, 0))           
                # blit the transformed surface onto the screen
                screen.blit(surf, (0, 0))
                
                pg.display.update()
                        
            pg.quit()
            


    def close(self):
        pass

 

''' 
Friedhof der Code-Schnipsel:
    
1) np.array der Länge X im Bereich A,B mit nur eindeutigen Werten herstellen:
    
    from numpy.random import default_rng
    rng = default_rng()
    numbers = rng.choice(range(A,B), size=X, replace=False)
'''
