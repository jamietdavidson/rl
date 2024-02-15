import numpy as np
import tensorflow as tf
from gym import spaces

from rl import utils


class Buffer:
    """A buffer for storing trajectories experienced by a PPO agent interacting
    with the environment, and using Generalized Advantage Estimation (GAE-Lambda)
    for calculating the advantages of state-action pairs.
    """

    def __init__(
        self,
        obs_shape,
        act_space,
        gamma=0.99,
        lam=0.99,
        size=50000,
        scale_rewards=False,
    ):
        """Create a buffer for storing PPO runtime information.
        
        Parameters
        ----------
        `obs_shape`: `tuple`.
            Of the observation dimension.
        
        `act_space`: `tuple`.
            Of actions.
        
        `gamma`: `float`, optional.
            Our trust in the value.
        
        `lam`: `float`, optional.
            Assigns credit to recent actions.
        
        `size`: `int`, optional.
            The size of the buffer.
        
        `scale_rewards`: `bool`, optional.
            If `True`, all rewards will be scaled in range `(-1, 1)`.
        """
        # fmt: off
        self.obs_buf = np.zeros(utils.combined_shape(size, obs_shape), dtype=np.float32)
        self.act_buf = np.zeros(utils.combined_shape(size, act_space.shape), dtype=np.int)
        self.pol_buf = np.zeros(utils.combined_shape(size, act_space.n), dtype=np.float32)
        self.adv_buf = np.zeros(size, dtype=np.float32)
        self.rew_buf = np.zeros(size, dtype=np.float32)
        self.ret_buf = np.zeros(size, dtype=np.float32)
        self.val_buf = np.zeros(size, dtype=np.float32)
        self.done_buf = np.zeros(size, dtype=np.int)
        # fmt: on

        self.gamma = gamma
        self.lam = lam
        self.scale_rewards = scale_rewards
        self.batch_size = 512

        self.idx = 0
        self.path_start_idx = 0

    @classmethod
    def from_collections(cls, collections):
        """Create a buffer object from collections of other buffers. Useful for
        multiprocessing.
        
        Parameters
        ----------
        `collections`: `list(np.ndarray, ...)`.
            That will be chained into one buffer and returned.
        
        Returns
        -------
        `Buffer`
        """
        buffer = cls(
            act_space=spaces.Discrete(3),
            obs_shape=np.shape(collections[0][0])[1:],
            size=max(50000, sum([len(collection[0]) for collection in collections])),
        )
        start_idx = 0
        for collection in collections:
            size = len(collection[0])
            stop_idx = start_idx + size
            buffer.obs_buf[start_idx:stop_idx] = collection[0]
            buffer.act_buf[start_idx:stop_idx] = collection[1]
            buffer.pol_buf[start_idx:stop_idx] = collection[2]
            buffer.adv_buf[start_idx:stop_idx] = collection[3]
            buffer.rew_buf[start_idx:stop_idx] = collection[4]
            buffer.ret_buf[start_idx:stop_idx] = collection[5]
            buffer.val_buf[start_idx:stop_idx] = collection[6]
            buffer.done_buf[start_idx:stop_idx] = collection[7]
            start_idx = stop_idx

        buffer.idx = stop_idx
        buffer.path_start_idx = stop_idx
        return buffer

    def as_collection(self):
        """Convert all of the data into a list of `np.ndarray`s that can later
        be reassembled (useful for multiprocessing using the `ray` module).
        
        Returns
        -------
        `list(np.ndarray, ...)`
        """
        self._finish()
        collection = (
            self.obs_buf[: self.idx],
            self.act_buf[: self.idx],
            self.pol_buf[: self.idx],
            self.adv_buf[: self.idx],
            self.rew_buf[: self.idx],
            self.ret_buf[: self.idx],
            self.val_buf[: self.idx],
            self.done_buf[: self.idx],
        )
        # clear the buffer so that old policy samples aren't used in future updates
        self.idx = 0
        self.path_start_idx = 0
        return collection

    def as_dataset(self):
        """Convert the buffer into a dataset that is optionally batched.

        Returns
        -------
        `tf.data.Dataset`
        """
        collection = self.as_collection()
        collection = list(collection)

        # remove "rew" from the collection, not needed for training
        collection.pop(4)
        collection = tuple(collection)

        # cast as dataset
        dataset = tf.data.Dataset.from_tensor_slices(collection)
        dataset = dataset.batch(self.batch_size)
        return dataset

    def store(self, observation, action, policy, reward, value_est, is_done):
        """Append one timestep of agent-environment interaction to the buffer.
        
        Parameters
        ----------
        `observation`: `np.ndarray`.
            An observation at t.
        
        `action`: `int`.
            The action taken after observing `observation`.
        
        `policy`: `np.float32`.
            The action distribution under `observation`.
        
        `reward`: `np.float32`.
            The reward observed for taking `action`.
        
        `value_est`: `np.float32`.
            The value prediction of `observation`.
        
        `is_done`: `np.bool`.
            Whether `action` led to a terminal state or not. 
        """
        self.obs_buf[self.idx] = observation
        self.act_buf[self.idx] = action
        self.pol_buf[self.idx] = policy
        self.rew_buf[self.idx] = reward
        self.val_buf[self.idx] = value_est
        self.done_buf[self.idx] = int(bool(is_done))

        self.idx += 1
        if is_done:
            self._finish()

    def _finish(self):
        """Compute the GAE-Lambda advantage calculation. This computes a rolling
        advantage estimate over all states.
        """
        if self.path_start_idx == self.idx:
            return

        last_val = self.val_buf[self.idx - 1] if self.done_buf[self.idx - 1] else 0
        path_slice = slice(self.path_start_idx, self.idx)
        rews = np.append(self.rew_buf[path_slice], last_val)
        vals = np.append(self.val_buf[path_slice], last_val)

        if self.scale_rewards:
            rews = rews / (np.max(np.abs(rews)) + 1e-5)

        # the next two lines implement GAE-Lambda advantage calculation
        deltas = rews[:-1] + self.gamma * vals[1:] - vals[:-1]
        self.adv_buf[path_slice] = utils.discount_cumsum(deltas, self.gamma * self.lam)

        # the next line computes rewards-to-go, to be targets for the value function
        self.ret_buf[path_slice] = utils.discount_cumsum(rews, self.gamma)[:-1]
        self.path_start_idx = self.idx

        # normalized advantage trick
        self.adv_buf[path_slice] = (
            self.adv_buf[path_slice] - np.mean(self.adv_buf[path_slice])
        ) / (np.std(self.adv_buf[path_slice]) + 1e-5)
