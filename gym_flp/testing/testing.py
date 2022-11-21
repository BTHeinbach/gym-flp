<<<<<<< Updated upstream
import gym
import gym_flp
import pytest
import numpy as np
#pytest -l testing.py

@pytest.mark.parametrize(
    "test_action, expected",
    [(0, np.array([22, 20,  4,  5,
                    2, 20,  8,  9,
                   11, 10,  5,  6,
                   21,  2,  4,  6,
                   11, 20,  4,  4,
                    2,  2,  3,  5], dtype=np.uint8)),
     (3, np.array([21, 19,  4,  5,
                    2, 20,  8,  9,
                   11, 10,  5,  6,
                   21,  2,  4,  6,
                   11, 20,  4,  4,
                    2,  2,  3,  5], dtype=np.uint8)),
     (8, np.array([21, 20,  4,  5,
                    2, 20,  8,  9,
                   12, 10,  5,  6,
                   21,  2,  4,  6,
                   11, 20,  4,  4,
                    2,  2,  3,  5], dtype=np.uint8)),
     (9, np.array([21, 20,  4,  5,
                    2, 20,  8,  9,
                   11, 11,  5,  6,
                   21,  2,  4,  6,
                   11, 20,  4,  4,
                    2,  2,  3,  5], dtype=np.uint8)),
     (15, np.array([21, 20,  4,  5,
                    2, 20,  8,  9,
                   11, 10,  5,  6,
                   21,  1,  4,  6,
                   11, 20,  4,  4,
                    2,  2,  3,  5], dtype=np.uint8)),
     (23, np.array([21, 20,  4,  5,
                     2, 20,  8,  9,
                    11, 10,  5,  6,
                    21,  2,  4,  6,
                    11, 20,  4,  4,
                    2,  1,  3,  5], dtype=np.uint8))])
def test_discrete(test_action, expected):
    env = gym.make('ofp-v0', mode='human', instance='P6', aspace='discrete', multi=False)
    env.reset()
    s, r, d, i = env.step(test_action)
    
    assert np.array_equal(s, expected)==True
    assert d == False


@pytest.mark.parametrize(
    "test_action, expected",
    [(np.array([0, 0, 0, 0, 0, 0]), np.array([22, 20,  4,  5,
                                               3, 20,  8,  9,
                                              12, 10,  5,  6,
                                              22,  2,  4,  6,
                                              12, 20,  4,  4,
                                               3,  2,  3,  5], dtype=np.uint8)),
     (np.array([2, 4, 1, 4, 0, 3]), np.array([20, 20,  4,  5,
                                              2, 20,  8,  9,
                                             11, 11,  5,  6,
                                             21,  2,  4,  6,
                                             12, 20,  4,  4,
                                              2,  1,  3,  5], dtype=np.uint8)),
     (np.array([1, 1, 1, 1, 1, 1]), np.array([21, 21,  4,  5,
                                              2, 21,  8,  9,
                                             11, 11,  5,  6,
                                             21,  3,  4,  6,
                                             11, 21,  4,  4,
                                              2,  3,  3,  5], dtype=np.uint8)),
     (np.array([2, 2, 2, 2, 2, 2]), np.array([20, 20,  4,  5,
                                              1, 20,  8,  9,
                                             10, 10,  5,  6,
                                             20,  2,  4,  6,
                                             10, 20,  4,  4,
                                              1,  2,  3,  5], dtype=np.uint8)),
     (np.array([3, 3, 2, 4, 1, 1]), np.array([21, 19,  4,  5,
                                               2, 19,  8,  9,
                                              10, 10,  5,  6,
                                              21,  2,  4,  6,
                                              11, 21,  4,  4,
                                               2,  3,  3,  5], dtype=np.uint8)),
     (np.array([4, 4, 4, 4, 4, 4]), np.array([21, 20,  4,  5,
                                              2, 20,  8,  9,
                                             11, 10,  5,  6,
                                             21,  2,  4,  6,
                                             11, 20,  4,  4,
                                              2,  2,  3,  5], dtype=np.uint8))
             ])
def test_multidiscrete(test_action, expected):
    env = gym.make('ofp-v0', mode='human', instance='P6', aspace='discrete', multi=True)
    env.reset()
    s, r, d, i = env.step(test_action)

    assert np.array_equal(s, expected)==True
    assert d == False

@pytest.mark.parametrize(
    "test_action, expected",
    [(np.array([-1, -1, -1]), np.array([0, 0,  4,  5,
                                              2, 20,  8,  9,
                                             11, 10,  5,  6,
                                             21,  2,  4,  6,
                                             11, 20,  4,  4,
                                              2,  2,  3,  5], dtype=np.uint8)),
     (np.array([-1, 1, 1]), np.array([22, 21,  4,  5,
                                              2, 20,  8,  9,
                                             11, 10,  5,  6,
                                             21,  2,  4,  6,
                                             11, 20,  4,  4,
                                              2,  2,  3,  5], dtype=np.uint8)),
     (np.array([0, 0, 0]), np.array([21, 20,  4,  5,
                                              2, 20,  8,  9,
                                             11, 10,  5,  6,
                                             21,  2,  4,  6,
                                             11, 20,  4,  4,
                                              2,  2,  3,  5], dtype=np.uint8)),
    (np.array([-0.59, 0, 0]), np.array([21, 20,  4,  5,
                                               11, 10,  8,  9,
                                              11, 10,  5,  6,
                                              21,  2,  4,  6,
                                              11, 20,  4,  4,
                                               2,  2,  3,  5], dtype=np.uint8)),
    (np.array([-0.6, 0, 0]), np.array([21, 20,  4,  5,
                                               11, 10,  8,  9,
                                              11, 10,  5,  6,
                                              21,  2,  4,  6,
                                              11, 20,  4,  4,
                                               2,  2,  3,  5], dtype=np.uint8)),
    (np.array([-0.61, 0, 0]), np.array([11, 10,  4,  5,
                                             2, 20,  8,  9,
                                            11, 10,  5,  6,
                                            21,  2,  4,  6,
                                            11, 20,  4,  4,
                                             2,  2,  3,  5], dtype=np.uint8)),
      (np.array([0.2, -0.53, -0.14]), np.array([21, 20,  4,  5,
                                               2, 20,  8,  9,
                                              11, 10,  5,  6,
                                              5,  9,  4,  6,
                                              11, 20,  4,  4,
                                               2,  2,  3,  5], dtype=np.uint8))
             ])
def test_box(test_action, expected):
    env = gym.make('ofp-v0', mode='human', instance='P6', aspace='box', multi=False)
    env.reset()
    s, r, d, i = env.step(test_action)

    assert np.array_equal(s, expected)==True
    assert d == False

@pytest.mark.parametrize(
    "test_action, expected",
    [(np.array([-1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1]),
      np.array([0, 0,  4,  5,
                                              0, 0,  8,  9,
                                             0, 0,  5,  6,
                                             0,  0,  4,  6,
                                             0, 0,  4,  4,
                                              0,  0,  3,  5], dtype=np.uint8)),
     (np.array([1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1]),
      np.array([22, 21,  4,  5,
                                              22, 21,  8,  9,
                                             22, 21,  5,  6,
                                             22,  21,  4,  6,
                                             22, 21,  4,  4,
                                            22,  21,  3,  5], dtype=np.uint8)),
    (np.array([0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]),
          np.array([11, 10,  4,  5,
                    11, 10,  8,  9,
                    11, 10,  5,  6,
                    11, 10,  4,  6,
                    11, 10,  4,  4,
                    11, 10,  3,  5], dtype=np.uint8))

             ])
def test_box_multi(test_action, expected):
    env = gym.make('ofp-v0', mode='human', instance='P6', aspace='box', multi=True)
    env.reset()
    s, r, d, i = env.step(test_action)

    assert np.array_equal(s, expected)==True
=======
import gym
import gym_flp
import pytest
import numpy as np
#pytest -l testing.py

@pytest.mark.parametrize(
    "test_action, expected",
    [(0, np.array([22, 20,  4,  5,
                    2, 20,  8,  9,
                   11, 10,  5,  6,
                   21,  2,  4,  6,
                   11, 20,  4,  4,
                    2,  2,  3,  5], dtype=np.uint8)),
     (3, np.array([21, 19,  4,  5,
                    2, 20,  8,  9,
                   11, 10,  5,  6,
                   21,  2,  4,  6,
                   11, 20,  4,  4,
                    2,  2,  3,  5], dtype=np.uint8)),
     (8, np.array([21, 20,  4,  5,
                    2, 20,  8,  9,
                   12, 10,  5,  6,
                   21,  2,  4,  6,
                   11, 20,  4,  4,
                    2,  2,  3,  5], dtype=np.uint8)),
     (9, np.array([21, 20,  4,  5,
                    2, 20,  8,  9,
                   11, 11,  5,  6,
                   21,  2,  4,  6,
                   11, 20,  4,  4,
                    2,  2,  3,  5], dtype=np.uint8)),
     (15, np.array([21, 20,  4,  5,
                    2, 20,  8,  9,
                   11, 10,  5,  6,
                   21,  1,  4,  6,
                   11, 20,  4,  4,
                    2,  2,  3,  5], dtype=np.uint8)),
     (23, np.array([21, 20,  4,  5,
                     2, 20,  8,  9,
                    11, 10,  5,  6,
                    21,  2,  4,  6,
                    11, 20,  4,  4,
                    2,  1,  3,  5], dtype=np.uint8))])
def test_discrete(test_action, expected):
    env = gym.make('ofp-v0', mode='human', instance='P6', aspace='discrete', multi=False)
    env.reset()
    s, r, d, i = env.step(test_action)
    
    assert np.array_equal(s, expected)==True
    assert d == False


@pytest.mark.parametrize(
    "test_action, expected",
    [(np.array([0, 0, 0, 0, 0, 0]), np.array([22, 20,  4,  5,
                                               3, 20,  8,  9,
                                              12, 10,  5,  6,
                                              22,  2,  4,  6,
                                              12, 20,  4,  4,
                                               3,  2,  3,  5], dtype=np.uint8)),
     (np.array([2, 4, 1, 4, 0, 3]), np.array([20, 20,  4,  5,
                                              2, 20,  8,  9,
                                             11, 11,  5,  6,
                                             21,  2,  4,  6,
                                             12, 20,  4,  4,
                                              2,  1,  3,  5], dtype=np.uint8)),
     (np.array([1, 1, 1, 1, 1, 1]), np.array([21, 21,  4,  5,
                                              2, 21,  8,  9,
                                             11, 11,  5,  6,
                                             21,  3,  4,  6,
                                             11, 21,  4,  4,
                                              2,  3,  3,  5], dtype=np.uint8)),
     (np.array([2, 2, 2, 2, 2, 2]), np.array([20, 20,  4,  5,
                                              1, 20,  8,  9,
                                             10, 10,  5,  6,
                                             20,  2,  4,  6,
                                             10, 20,  4,  4,
                                              1,  2,  3,  5], dtype=np.uint8)),
     (np.array([3, 3, 2, 4, 1, 1]), np.array([21, 19,  4,  5,
                                               2, 19,  8,  9,
                                              10, 10,  5,  6,
                                              21,  2,  4,  6,
                                              11, 21,  4,  4,
                                               2,  3,  3,  5], dtype=np.uint8)),
     (np.array([4, 4, 4, 4, 4, 4]), np.array([21, 20,  4,  5,
                                              2, 20,  8,  9,
                                             11, 10,  5,  6,
                                             21,  2,  4,  6,
                                             11, 20,  4,  4,
                                              2,  2,  3,  5], dtype=np.uint8))
             ])
def test_multidiscrete(test_action, expected):
    env = gym.make('ofp-v0', mode='human', instance='P6', aspace='discrete', multi=True)
    env.reset()
    s, r, d, i = env.step(test_action)

    assert np.array_equal(s, expected)==True
    assert d == False

@pytest.mark.parametrize(
    "test_action, expected",
    [(np.array([-1, -1, -1]), np.array([0, 0,  4,  5,
                                              2, 20,  8,  9,
                                             11, 10,  5,  6,
                                             21,  2,  4,  6,
                                             11, 20,  4,  4,
                                              2,  2,  3,  5], dtype=np.uint8)),
     (np.array([-1, 1, 1]), np.array([22, 21,  4,  5,
                                              2, 20,  8,  9,
                                             11, 10,  5,  6,
                                             21,  2,  4,  6,
                                             11, 20,  4,  4,
                                              2,  2,  3,  5], dtype=np.uint8)),
     (np.array([0, 0, 0]), np.array([21, 20,  4,  5,
                                              2, 20,  8,  9,
                                             11, 10,  5,  6,
                                             21,  2,  4,  6,
                                             11, 20,  4,  4,
                                              2,  2,  3,  5], dtype=np.uint8)),
    (np.array([-0.59, 0, 0]), np.array([21, 20,  4,  5,
                                               11, 10,  8,  9,
                                              11, 10,  5,  6,
                                              21,  2,  4,  6,
                                              11, 20,  4,  4,
                                               2,  2,  3,  5], dtype=np.uint8)),
    (np.array([-0.6, 0, 0]), np.array([21, 20,  4,  5,
                                               11, 10,  8,  9,
                                              11, 10,  5,  6,
                                              21,  2,  4,  6,
                                              11, 20,  4,  4,
                                               2,  2,  3,  5], dtype=np.uint8)),
    (np.array([-0.61, 0, 0]), np.array([11, 10,  4,  5,
                                             2, 20,  8,  9,
                                            11, 10,  5,  6,
                                            21,  2,  4,  6,
                                            11, 20,  4,  4,
                                             2,  2,  3,  5], dtype=np.uint8)),
      (np.array([0.2, -0.53, -0.14]), np.array([21, 20,  4,  5,
                                               2, 20,  8,  9,
                                              11, 10,  5,  6,
                                              5,  9,  4,  6,
                                              11, 20,  4,  4,
                                               2,  2,  3,  5], dtype=np.uint8))
             ])
def test_box(test_action, expected):
    env = gym.make('ofp-v0', mode='human', instance='P6', aspace='box', multi=False)
    env.reset()
    s, r, d, i = env.step(test_action)

    assert np.array_equal(s, expected)==True
    assert d == False

@pytest.mark.parametrize(
    "test_action, expected",
    [(np.array([-1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1]),
      np.array([0, 0,  4,  5,
                                              0, 0,  8,  9,
                                             0, 0,  5,  6,
                                             0,  0,  4,  6,
                                             0, 0,  4,  4,
                                              0,  0,  3,  5], dtype=np.uint8)),
     (np.array([1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1]),
      np.array([22, 21,  4,  5,
                                              22, 21,  8,  9,
                                             22, 21,  5,  6,
                                             22,  21,  4,  6,
                                             22, 21,  4,  4,
                                            22,  21,  3,  5], dtype=np.uint8)),
    (np.array([0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]),
          np.array([11, 10,  4,  5,
                    11, 10,  8,  9,
                    11, 10,  5,  6,
                    11, 10,  4,  6,
                    11, 10,  4,  4,
                    11, 10,  3,  5], dtype=np.uint8))

             ])
def test_box_multi(test_action, expected):
    env = gym.make('ofp-v0', mode='human', instance='P6', aspace='box', multi=True)
    env.reset()
    s, r, d, i = env.step(test_action)

    assert np.array_equal(s, expected)==True
>>>>>>> Stashed changes
    assert d == False