import numpy as np

from gym import spaces


def rescale_actions(a: int, b: int, x_min: int, x_max: int, x: float):
    return (x-a)*(x_max-x_min)/(b-a)+x_min


def normalize(a: int, b: int, x_min: int, x_max: int, x: float):
    return (b-a)*(x-x_min)/(x_max-x_min)+a


def make_image_from_coordinates(coordinates: np.array, canvas: np.array, flows: np.array) -> np.array:
    sources = np.sum(flows, axis=1)
    sinks = np.sum(flows, axis=0)

    #coordinates = np.array([12, 20,  2,  7, 15, 17,  4, 12, 15, 12,  5,  5, 19, 17,  3,  6, 14, 4,  8,  5,  6, 17,  6,  9])

    p = np.arange(len(coordinates) / 4)
    r = np.ones(shape=p.shape).astype(int) * 0
    g = np.array((sources - np.min(sources)) / (np.max(sources) - np.min(sources)) * 255).astype(int)
    b = np.array((sinks - np.min(sinks)) / (np.max(sinks) - np.min(sinks)) * 255).astype(int)

    r = normalize(a=40, b=255, x_min=0, x_max=np.max(p), x=p).astype(int)
    g = normalize(a=40, b=255, x_min=np.min(sources), x_max=np.max(sources), x=np.round(sources)).astype(int)
    b = normalize(a=40, b=255, x_min=np.min(sinks), x_max=np.max(sinks), x=np.round(sinks)).astype(int)

    # r = np.zeros(6)
    # b = np.zeros(6)

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
                                      dtype='float32')
        else:
            action_space = spaces.Box(low=np.array([-1.0, -1.0, -1.0]), high=np.array([1.0, 1.0, 1.0]), dtype='float32')
    else:
        print("No action space selected or selected space not supported")
    return action_space


def centroids(state1d):
    top_left_y = state1d[0::4]
    top_left_x = state1d[1::4]
    heights = state1d[2::4]
    widths = state1d[3::4]

    return top_left_y+0.5*heights, top_left_x+0.5*widths

def divisor(n):
    for i in range(n):
        x = [i for i in range(1, n + 1) if not n % i]
    return x
