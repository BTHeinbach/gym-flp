import numpy as np

from gym import spaces


@staticmethod
def rescale_actions(a: int, b: int, x_min: int, x_max: int, x: float):
    return (x-a)*(x_max-x_min)/(b-a)+x_min


def normalize(a: int, b: int, x_min: int, x_max: int, x: float):
    return (b-a)*(x-x_min)/(x_max-x_min)+a


def make_image_from_coordinates(coordinates: np.array, canvas: np.array, flows: np.array) -> np.array:
    sources = np.sum(flows, axis=1)
    sinks = np.sum(flows, axis=0)

    p = np.arange(len(coordinates) / 4)
    r = np.ones(shape=p.shape).astype(int) * 255
    g = np.array((sources - np.min(sources)) / (np.max(sources) - np.min(sources)) * 255).astype(int)
    b = np.array((sinks - np.min(sinks)) / (np.max(sinks) - np.min(sinks)) * 255).astype(int)

    for x, y in enumerate(p):
        y_from = coordinates[4 * x + 0]
        x_from = coordinates[4 * x + 1]
        y_to = coordinates[4 * x + 0] + coordinates[4 * x + 2]
        x_to = coordinates[4 * x + 1] + coordinates[4 * x + 3]

        canvas[int(y_from):int(y_to), int(x_from):int(x_to)] = [r[int(y) - 1], g[int(y) - 1], b[int(y) - 1]]
    return np.array(canvas, dtype=np.uint8)


def build_action_space(env, box, multi):
    if not box:
        if not multi:
            action_space = spaces.Discrete(4*env.n+1)
        else:
            action_space = spaces.MultiDiscrete([5 for _ in range(env.n)])

    elif box:
        if multi:
            action_space = spaces.Box(low=np.array([-1.0 for _ in range(env.n*2)]),
                                      high=np.array([1.0 for _ in range(env.n*2)]),
                                      # high=np.tile([env.upper_bounds['Y'], env.upper_bounds['X']], reps=6),
                                      dtype='float32')
        else:
            action_space = spaces.Box(low=np.array([-1.0, -1.0, -1.0]), high=np.array([1.0, 1.0, 1.0]), dtype='float32')
    else:
        print("No action space selected or selected space not supported")
    return action_space

# self.action_space = spaces.Box(low=np.array([0, 0, 0]), high=np.array([self.n, ]))
