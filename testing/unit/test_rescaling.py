import pytest
from gym_flp.util.preprocessing import Spaces


@pytest.mark.parametrize(
    "test_input, expected",
    [(0, 15),
     (-1, 0),
     (1, 30),
     (-0.6, 6)]
)
def test_rescale(test_input, expected):
    xmin = 0
    xmax = 30
    a = -1
    b = 1
    x_ = Spaces().rescale_actions(x=test_input, x_min=xmin, x_max=xmax, a=a, b=b)
    assert x_ == expected