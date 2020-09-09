import numpy as np

class MHC():

    def __init__(self, shape=None, dtype=np.float32):
        print('Hi, I compute the layout cost.')

        
    def compute(self, D, F, s):
        # Compute reward for taking action on old state:  
        # Calculate permutation matrix for new state
        P = self.permutationMatrix(s)
        
        #Deduct results from known optimal value 
        #reward = self.best if np.array_equal(fromState, self.opt) else self.best - 0.5*np.trace(np.dot(np.dot(self.F,P), np.dot(self.D,P.T))) #313 is optimal for NEOS n6, needs to be replaced with a dict object
        transport_intensity = np.dot(np.dot(F,P), np.dot(D,P.T))
        MHC = 0.5*np.trace(transport_intensity)
                
        return MHC, transport_intensity
    
    def permutationMatrix(self, a):
        P = np.zeros((len(a), len(a)))
        for idx,val in enumerate(a):
            P[idx][val-1]=1
        return P