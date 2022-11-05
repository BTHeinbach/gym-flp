import numpy as np

from gym import spaces

@staticmethod
def rescale_actions(a: int, b: int, x_min: int, x_max: int, x: float):
    return (x-a)*(x_max-x_min)/(b-a)+x_min

def normalize(self, a: int, b: int, x_min: int, x_max: int, x: float):
    return (b-a)*(x-x_min)/(x_max-x_min)+a

def make_image_from_coordinates(coordinates:np.array, canvas:np.array, flows:np.array) -> np.array:
    sources = np.sum(flows, axis=1)
    sinks = np.sum(flows, axis=0)

    p = np.arange(len(coordinates) / 4)
    R = np.ones(shape=p.shape).astype(int) * 255
    G = np.array((sources - np.min(sources)) / (np.max(sources) - np.min(sources)) * 255).astype(int)
    B = np.array((sinks - np.min(sinks)) / (np.max(sinks) - np.min(sinks)) * 255).astype(int)

    for x, y in enumerate(p):
        y_from = coordinates[4 * x + 0]
        x_from = coordinates[4 * x + 1]
        y_to = coordinates[4 * x + 0] + coordinates[4 * x + 2]
        x_to = coordinates[4 * x + 1] + coordinates[4 * x + 3]

        canvas[int(y_from):int(y_to), int(x_from):int(x_to)] = [R[int(y) - 1], G[int(y) - 1], B[int(y) - 1]]
    return np.array(canvas, dtype=np.uint8)

def build_action_space(space_type, n):
    if space_type == "discrete":
        action_set = ['N', 'E', 'S', 'W']
        action_list = [action_set[i] for j in range(n) for i in range(len(action_set))]
        action_space = spaces.Discrete(len(action_list))

    elif space_type == "multi-discrete":
        action_space = spaces.MultiDiscrete([4 for _ in range(n)])

    elif space_type == "box":
        pass
    else:

        print("No action space selected or selected space not supported")
    return action_space

# self.action_space = spaces.Box(low=np.array([0, 0, 0]), high=np.array([self.n, ]))
