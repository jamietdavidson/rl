import numpy as np

from rl.buffer import Buffer
from rl.tests.utils import MockSpace

LEFT = 0
RIGHT = 1


def test_buffer():
    """Test is based on cartpole-v1.
    """
    data = [
        # observation, action, policy, reward, value_est, is_done
        (np.random.random((4,)), LEFT, [0.5, 0.5], 1, 0, False),
        (np.random.random((4,)), LEFT, [0.5, 0.5], 1, 0, False),
        (np.random.random((4,)), RIGHT, [0.5, 0.5], 1, 0, False),
        (np.random.random((4,)), RIGHT, [0.5, 0.5], 1, 0, False),
        (np.random.random((4,)), RIGHT, [0.5, 0.5], -100, 0, True),
        # another episode also in the buffer
        (np.random.random((4,)), LEFT, [0.5, 0.5], 2, 0, False),
        (np.random.random((4,)), LEFT, [0.5, 0.5], 2, 0, False),
        (np.random.random((4,)), RIGHT, [0.5, 0.5], 2, 0, False),
        (np.random.random((4,)), RIGHT, [0.5, 0.5], 2, 0, False),
        (np.random.random((4,)), RIGHT, [0.5, 0.5], -100, 0, True),
    ]
    # mock the space objects from openai gym
    act_space = MockSpace(shape=(), n=2)
    buffer = Buffer(obs_shape=(4,), act_space=act_space, size=10)

    for sample in data[:9]:
        buffer.store(*sample)

    # after a terminal state is encountered (as is the case with observation at
    # index 4, the buffer should finalize the information and continue storing).
    assert buffer.path_start_idx == 5
    assert np.allclose(buffer.ret_buf[:5], [-92.1192, -94.0598, -96.02, -98.0, -100.0])

    buffer.store(*data[-1])
    assert np.allclose(
        buffer.ret_buf[5:10], [-88.1788, -91.0897, -94.03, -97.0, -100.0]
    )
