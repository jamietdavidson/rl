from pathlib import Path

import numpy as np
import tensorflow as tf
from tensorflow import keras

from rl import utils
from rl.buffer import Buffer


class Agent:
    """An agent for interacting with a broker to make trades.
    
    The agent specifies to the broker how confident he feels about a particular
    commodit(ies), and the broker translates that confidence into buy or sell
    orders whilst taking in the current status of the agent.
    """

    def __init__(self, *, state_space, action_space, model=None, model_name=None):
        """Create an `ActorValueNetwork`. When passing in state information, do
        not include the batch size.
        
        Parameters
        ----------
        `state_space`: `gym.Space`.
            The state space.
        
        `action_space`: `gym.Space`.
            The action space.
        
        `models`: `tf.keras.Model`, optional.
            The Actor Critic Model.
        
        `model_name`: `Path` or `str`, optional.
            The saved location of the agent.
        
        Note
        ----
        One of `model` or `model_name` must be provided.
        """
        if model is None:
            model = keras.models.load_model(
                Path(f"models/{model_name}", "model.h5"), compile=False
            )

        self.state_space = state_space
        self.action_space = action_space
        self.model = model
        self.model_name = model_name

        self.metrics = None
        self.gamma = 0.99
        self.entropy_coeff = 0.01
        self.clip = 0.2
        self.gradient_clip = True

        self.optimizer = keras.optimizers.Adam(learning_rate=0.002)
        # self.model.summary()

    def get_buffer(self, **kwargs):
        """Get a `Buffer` that is initialized to work with this agent effectively.

        Returns
        -------
        `Buffer`.
        """
        return Buffer(
            obs_shape=self.state_space.shape,
            act_space=self.action_space,
            gamma=self.gamma,
            lam=0.95,
            **kwargs,
        )

    def get_model(self):
        """Retrieve the model. Useful for multiprocessing with `ray`, where 
        accessing class attributes directly is not allowed.
        
        Returns
        -------
        `tf.keras.Model`
        """
        return self.model

    def step(self, state):
        """Determine an action to take in `state`.
        
        Parameters
        ----------
        `state`: `np.ndarray`.
            Of data to get an action for.
        
        Returns
        -------
        `action`, `sampled_action_prob`, `value_est`.
        """
        logits, value_est = self._call_model(state, single=True)
        policy = tf.nn.softmax(logits).numpy()
        action = np.random.choice(len(policy), p=policy)
        return (action, policy, value_est)

    # @tf.function
    def optimize(self, dataset, n_iters):
        """Optimize this policy using samples from batch.

        Parameters
        ----------
        `dataset`: `tf.data.Dataset`.
            That stores information from interacting with the environment.

        `n_iters`: 'int`.
            Number of times to iterate over the batch and perform gradient updates.

        Returns
        -------
        `dict`. Containing useful information for training visualization.
        """
        # fmt: off
        for idx in tf.range(n_iters):
            for states, actions, policies, advantages, rewards, values, dones in dataset:
                with tf.GradientTape() as tape:
                    # compute values needed for gradient
                    logits, values = self._call_model(states)

                    # get actor loss
                    ratios, kl_div = self._get_ratios(
                        actions=actions, policies=policies, logits=logits
                    )
                    actor_loss = self._get_actor_loss(
                        ratios=ratios, adv=advantages, policies=policies
                    )
                    # get critic loss
                    critic_loss = self._get_critic_loss(rewards=rewards, values=values)

                    # sum them up :)
                    loss = actor_loss + 0.5 * critic_loss
                    loss = tf.reduce_mean(loss)

                if kl_div < 0.01:
                    # gather trainable variables
                    variables = self.model.trainable_variables
                    gradients = tape.gradient(loss, variables)

                    # normalize gradients otherwise they will explode
                    if self.gradient_clip:
                        gradients, _ = tf.clip_by_global_norm(gradients, 5)

                    self.optimizer.apply_gradients(zip(gradients, variables))
        # fmt: on

    def _call_model(self, states, single=False):
        """Call the model to take care of adjusting for the batch dimension.

        Parameters
        ----------
        `state`: `np.ndarray`.
            Inputs to the model.
        """
        if single:
            # tf.print(states)
            states = tf.expand_dims(states, 0)
            logits, value_est = self.model(states)

            # squeeze values
            logits = tf.squeeze(logits)
            value_est = tf.squeeze(value_est)
        else:
            logits, value_est = self.model(states)
        return logits, value_est

    def _get_ratios(self, actions, policies, logits):
        """Compute r_t(theta), or more simply, the ratio of probabilities of taking
        action in `state` under the sampled policy (old), vs the current policy
        (new).

        Parameters
        ----------
        `actions`: `int`.
            The action chosen.

        `policies`: `np.ndarray`.
            The sampled softmax output of the policy in `state`.

        `logits`: `tf.Tensor`.
            From applying `states` through the network.

        Returns
        -------
        ratios, kl_div
        """
        old_log_policy = tf.math.log(policies)
        new_log_policy = tf.nn.log_softmax(logits, axis=1)

        # get log action probabilities
        old_log_action_probs = (
            tf.one_hot(actions, depth=logits.shape[-1]) * old_log_policy
        )
        new_log_action_probs = (
            tf.one_hot(actions, depth=logits.shape[-1]) * new_log_policy
        )

        # reduce sum so that one-hot is now a scalar
        old_log_sampled_policy = tf.reduce_sum(old_log_action_probs, axis=1)
        new_log_sampled_policy = tf.reduce_sum(new_log_action_probs, axis=1)

        ratios = tf.exp(new_log_sampled_policy - old_log_sampled_policy)

        kl_div = tf.reduce_sum(policies * (old_log_policy - new_log_policy), axis=1)
        kl_div = tf.reduce_max(kl_div)

        # record ratio for insights on policy updates
        return ratios, kl_div

    def _get_actor_loss(self, ratios, adv, policies):
        """Computes loss as defined by the PPO paper by Schulman et al 2017.

        Parameters
        ----------
        `ratios`: `tf.Tensor`.
            The ratios of new policy probabilities over the old.

        `adv`: `np.ndarray`.
            The advantage of actions computed with the old policy.

        `policies`: `np.ndarray`.
            The policies.
        """
        G = tf.where(adv > 0, (1 + self.clip) * adv, (1 - self.clip) * adv)

        # see the shortcut formula referenced by spinningup - PPO
        loss = tf.minimum(ratios * adv, G)

        # add in the entropy term to policy (read: actor network) only
        H = -tf.reduce_mean(policies * tf.math.log(policies), axis=1)
        loss = -loss + self.entropy_coeff * H

        # record metrics
        if self.metrics:
            self.metrics.record_epoch(
                abs_adv=tf.math.abs(adv),
                actor_loss=loss,
                entropy=H,
                ratio_max=tf.reduce_max(ratios),
                ratio_min=tf.reduce_min(ratios),
            )
        return loss

    def _get_critic_loss(self, rewards, values):
        """Performs MSE.

        Parameters
        ----------
        `rewards`: `tf.Tensor`.
            The reward observed by the environment.

        `values`: `tf.Tensor`.
            The predicted value from the critic network output.
        """
        loss = tf.reduce_mean((rewards - values) ** 2, axis=1)
        if self.metrics:
            self.metrics.record_epoch(critic_loss=loss)
        return loss
