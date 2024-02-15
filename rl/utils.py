import numpy as np
import ray
import scipy.signal
import tensorflow as tf

from framework.utils import *


def combined_shape(shape_1, shape_2):
    """
    Parameters
    ----------
    `shape_1`: `tuple`.
        To be prepended to `shape_2`.
        
    `shape_2`: `tuple`.
        To be appended to `shape_1
        
    Returns
    -------
    `tuple`
    """
    if not isinstance(shape_1, tuple):
        shape_1 = (shape_1,)
    if not isinstance(shape_2, tuple):
        shape_2 = (shape_2,)
    return shape_1 + shape_2


def discount_cumsum(x, discount):
    """Magic from rllab for computing discounted cumulative sums of vectors.
    
    Parameters
    ----------
    `x`: `np.ndarray`.
    
    `discount`: `float`.
        The discount factor.
    
    Returns
    -------
    `np.ndarray`
    
    Example
    -------
    ```
    >>> r = discount_cumsum([x0, x1, x2])
    >>> r
    [
        x0 + discount^1 * x1 + discount^2 * x2,
        x1 + discount^1 * x2, 
        x2,
    ]
    ```
    """
    return scipy.signal.lfilter([1], [1, float(-discount)], x[::-1], axis=0)[::-1]


def kl(self, a, b):
    """Compute the Kullback Leibler divergence between two categorical 
    probability distributions.
    
    Parameters
    ----------
    `a`: `np.ndarray`, or `tf.Tensor`.
        Such that `axis=-1` sums up to 1.
        
    `b`: `np.ndarray`, or `tf.Tensor`.
        Such that `axis=-1` sums up to 1.
        
    Returns
    -------
    `tf.float32`
    """
    A = tf.distributions.Categorical(probs=a)
    B = tf.distributions.Categorical(probs=b)
    return tf.distributions.kl_divergence(A, B)


class Loop:
    """For single process execution.
    """

    def __init__(self, agent, env, obs, buffer=None, progress=None):
        """
        Parameters
        ----------
        `agent`: `Agent`.
            That is being ran.
        
        `env`: `Environment`.
            That the agent is acting in.

        `obs`: `np.ndarray`.
            The latest observation in the environment.
            
        `buffer`: `Buffer`, optional.
            To record agent - env interactions with.
            
        `progress`: `tqdm`, optional.
            To track the progress of `call`.
        """
        self.agent = agent
        self.env = env
        self.obs = obs
        self.buffer = buffer
        self.progress = progress

    def call(self, steps=1):
        """
        Parameters
        ----------
        `steps`: `int`.
            The amount of agent - env interactions.
        
        Returns
        -------
        `bool, int`
        """
        for step in range(steps):
            action, policy, value_est = self.agent.step(self.obs)
            next_obs, reward, is_done, info = self.env.step(action)

            if self.buffer:
                self.buffer.store(
                    observation=self.obs,
                    action=action,
                    policy=policy,
                    reward=reward,
                    value_est=value_est,
                    is_done=is_done,
                )
            self.obs = next_obs
            if self.progress:
                self.progress.update()
            if is_done:
                break
        return is_done, steps - step - 1


@ray.remote
class DistributedLoop(Loop):
    """For multiprocess execution.
    """

    def __init__(self, model_name, env, obs):
        from rl.agent import Agent

        agent = Agent(
            state_space=env.observation_space,
            action_space=env.action_space,
            model_name=model_name,
        )
        buffer = agent.get_buffer()
        super().__init__(agent=agent, env=env, buffer=buffer, obs=obs)

    def call(self, steps):
        remaining = steps
        while remaining > 0:
            is_done, remaining = super().call(steps=remaining)
            if is_done:
                self.obs = self.env.reset()
        return self.buffer.as_collection()

    def set_weights(self, weights):
        self.agent.model.set_weights(weights)
