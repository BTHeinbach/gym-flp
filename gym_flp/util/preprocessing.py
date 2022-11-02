class Spaces():
    def __init__(self):
        self.name = "Test"

    def rescale_actions(self, a: int, b: int, x_min: int, x_max: int, x: float):
        return (x-a)*(x_max-x_min)/(b-a)+x_min

    def normalize(self, a:int, b:int, x_min: int, x_max: int, x:float):
        return (b-a)*(x-x_min)/(x_max-x_min)+a

    def make_image_from_coordinates(self, state):
        data = np.zeros((self.plant_Y, self.plant_X, 3), dtype=np.uint8)

        sources = np.sum(self.F, axis=1)
        sinks = np.sum(self.F, axis=0)

        p = np.arange(len(state) / 4)

        # R = np.array((p-np.min(p))/(np.max(p)-np.min(p))*255).astype(int)
        R = np.ones(shape=(self.n,)).astype(int) * 255
        G = np.array((sources - np.min(sources)) / (np.max(sources) - np.min(sources)) * 255).astype(int)
        B = np.array((sinks - np.min(sinks)) / (np.max(sinks) - np.min(sinks)) * 255).astype(int)

        for x, y in enumerate(p):
            y_from = state[4 * x + 0]
            x_from = state[4 * x + 1]
            y_to = state[4 * x + 0] + state_prelim[4 * x + 2]
            x_to = state[4 * x + 1] + state_prelim[4 * x + 3]

            data[int(y_from):int(y_to), int(x_from):int(x_to)] = [R[int(y) - 1], G[int(y) - 1], B[int(y) - 1]]
        return np.array(data, dtype=np.uint8)
