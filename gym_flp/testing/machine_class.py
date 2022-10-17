import numpy as np
from PIL import Image
import matplotlib.pyplot as plt

class machine():

    machineCount = 0
    def __init__(self, x = 0, y = 0, w = 2, h = 2, restrictions = ['clean'], X = 36, Y = 36, color = [255,255,255]):
        self.x = int(x)
        self.y = int(y)
        self.w = int(w)
        self.h = int(h)
        self.restrictions = restrictions
        self.X = int(X)
        self.Y = int(Y)
        self.obs = self.make()
        self.color = color
        machine.machineCount+=1

    def make(self):
        obs = np.zeros((self.Y, self.X, 3),dtype=np.uint8)
        obs[self.x - int(self.w / 2):self.x + int(self.w / 2), self.y - int(self.h / 2):self.y + int(self.h / 2)] = [255,0,0]
        return obs

    def move(self, x, y):
        self.x = self.x + x
        self.y = self.y + y

    def get_center(self):
        return np.array([self.y, self.x])

    def collisions(self, m1, m2):
        return np.sum(m1.obs.astype(int) & m2.obs.astype(int))

def make_state(input):
    machines = list(input)
    return sum(machines)
