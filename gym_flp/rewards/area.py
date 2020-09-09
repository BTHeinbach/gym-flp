import numpy as np

class area_eff():

    def __init__(self, shape=None, dtype=np.float32):
        print('Hi, I compute the area efficiency.')

        
    def compute(self, X , Y , h , b ):               
        return (h*b)/(X*Y)
