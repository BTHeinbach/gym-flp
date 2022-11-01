class Spaces():
    def __init__(self):
        self.name = "Test"

    def rescale_actions(self, a: int, b: int, x_min: int, x_max: int, x: float):
        return (x-a)*(x_max-x_min)/(b-a)+x_min
