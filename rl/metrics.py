import os
import shutil

# so that tests will not fail to be discovered
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

import tensorflow as tf
from tensorflow import keras


class Metrics:
    """Class for recording epoch & episode information that can be viewed with
    TensorBoard.
    """

    def __init__(self, model_name):
        """
        Parameters
        ----------
        `model_name`: `str`.
            To save the logs under.
        """
        self.epoch_metrics = {}
        self.episode_metrics = {}

        shutil.rmtree(f"models/{model_name}/logs/", ignore_errors=True)
        self.writer = tf.summary.create_file_writer(f"models/{model_name}/logs/")

    def record_epoch(self, **kwargs):
        """
        Parameters
        ----------
        `**kwargs`: `dict`.
            Whose values will be averaged over its collected epoch.
        """
        for metric, value in kwargs.items():
            if metric not in self.epoch_metrics:
                self.epoch_metrics[metric] = keras.metrics.Mean(
                    metric, dtype=tf.float32
                )
            self.epoch_metrics[metric](value)

    def record_episode(self, **kwargs):
        """
        Parameters
        ----------
        `**kwargs`: `dict`.
            Whose values will be averaged over its collected episode.
        """
        for metric, value in kwargs.items():
            if metric not in self.episode_metrics:
                self.episode_metrics[metric] = keras.metrics.Mean(
                    metric, dtype=tf.float32
                )
            self.episode_metrics[metric](value)

    def summarize_epoch(self, epoch):
        """
        Parameters
        ----------
        `epoch`: `int`.
            Used for writing to tensorboard.
        """
        with self.writer.as_default():
            for metric, value in self.epoch_metrics.items():
                tf.summary.scalar(metric, value.result(), step=epoch)

    def summarize_episode(self, episode):
        """
        Parameters
        ----------
        `episode`: `int`.
            Used for writing to tensorboard.
        """
        with self.writer.as_default():
            for metric, value in self.episode_metrics.items():
                tf.summary.scalar(metric, value, step=episode)
