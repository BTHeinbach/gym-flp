from gym_flp.envs import OfpEnv
import numpy as np


def test_collisions():
    env = OfpEnv(instance='P6', mode='rgb_array')
    state = np.array([0, 0, 10, 10, 10, 10, 10, 10], dtype=np.uint8)

    c = env.collision_test(state)

    assert c == 0


test_collisions()
