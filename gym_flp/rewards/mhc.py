import numpy as np

class MHC():

    def __init__(self, shape=None, dtype=np.float32):
        self.shape = shape
        
    def compute(self, D, F, s):
        # Compute reward for taking action on old state:  
        # Calculate permutation matrix for new state
        P = self.permutationMatrix(s)
        
        #Deduct results from known optimal value 
        #reward = self.best if np.array_equal(fromState, self.opt) else self.best - 0.5*np.trace(np.dot(np.dot(self.F,P), np.dot(self.D,P.T))) #313 is optimal for NEOS n6, needs to be replaced with a dict object
        transport_intensity = np.dot(np.dot(D,P), np.dot(F,P.T))
        MHC = np.trace(transport_intensity)
                
        return MHC, transport_intensity
     
    def _compute(self, D, F, s):       
        T = np.zeros((len(s),len(s)))

        for i in range(len(s)):
            for j in range(len(s)):
                if j > i:
                    d = D[i][j]
                    f = F[s[i]-1][s[j]-1]
                    T[i][j] = d*f
                else:
                    T[i][j] = 0 

        return np.sum(T), T
    
    def permutationMatrix(self, a):
        P = np.zeros((len(a), len(a)))
        for idx,val in enumerate(a):
            P[idx][val-1]=1
        return P
