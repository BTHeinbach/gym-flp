from gym_flp.util.preprocessing import Spaces


def test_rescale():
    x = 0
    xmin = 0
    xmax = 30
    a = -1
    b = 1
    s = Spaces()
    x_ = s.rescale_actions(x=x, x_min=xmin, x_max=xmax, a=a, b=b)

test_rescale()