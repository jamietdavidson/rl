import json
import os
import shutil
from pathlib import Path
from zipfile import ZipFile

import gym
import numpy as np
import ray
import tensorflow as tf
from dateutil.parser import parse as parse_datetime
from google.cloud import storage
from tensorflow import keras
from tqdm import tqdm

from framework import (
    Currencies,
    Environment,
    FrameworkError,
    LiveExchange,
    MockExchange,
    Time,
    Wallet,
)
from framework import utils as framework_utils
from rl import utils
from rl.agent import Agent
from rl.buffer import Buffer
from rl.metrics import Metrics


class Runner:
    """Runner for the algorithm. For training, testing, and serving the agent.
    """

    def __init__(self, agent, env, model_name):
        """
        Parameters
        ----------
        `agent`: `Agent`.
            That is being ran.
        
        `env`: `Environment`.
            That the agent is acting in.
        
        `model_name`: `str`.
            To save this runner under.
        """
        self.agent = agent
        self.env = env
        self.model_name = model_name
        self.metrics = Metrics(model_name=self.model_name)
        self.agent.metrics = self.metrics

    @classmethod
    def from_save(cls, model_name, on_cloud=False):
        """Restore a model.
        
        Parameters
        ----------
        `model_name`: `Path` or `str`.
            The model name specified on the command line.
        
        `on_cloud`: `bool`, optional.
            Whether or not the model resides on the cloud.
        
        Note
        ----
        Right now, the environment returned will include a `MockExchange`. In the
        future this will be exchangable for a `LiveExchange`.If `on_cloud` is 
        `True`, then the environment variable `GCP_BUCKET` must be set.
        Returns
        -------
        `Runner`
        """
        path = Path(f"models/", model_name)
        path.mkdir(parents=True, exist_ok=True)

        if on_cloud:
            client = storage.Client()
            model_path = f"models/{model_name}.zip"
            bucket = client.bucket(os.environ["GCP_BUCKET"])
            blob = storage.Blob(model_path, bucket)
            with open(model_path, "wb") as f:
                blob.download_to_file(f)
            with ZipFile(model_path, "r") as f:
                f.extractall(f"models/{model_name}")
            os.remove(model_path)

        exchange_name = "binance"
        with open(Path(path, "data.json"), "r") as f:
            data = json.load(f)

        # load time info
        start = parse_datetime(data["time"]["start"])
        stop = parse_datetime(data["time"]["stop"])
        lookback = data["time"]["lookback"]
        interval = framework_utils.str_to_timedelta(data["time"]["interval"])
        time = Time(start=start, stop=stop, interval=interval, lookback=lookback)

        # load currencies
        purchase_currency = Currencies.from_ticker(
            data["exchange"]["purchase_currency"]
        )
        trading_currency = Currencies.from_ticker(
            data["environment"]["trading_currency"]
        )
        aux_currencies = [
            Currencies.from_ticker(ticker)
            for ticker in data["environment"]["aux_currencies"]
        ]

        # gather dataframes
        dfs = {
            currency: framework_utils.get_filename(
                folder=f"data/{exchange_name}",
                purchase_currency=purchase_currency,
                currency=currency,
                interval=interval,
            )
            for currency in [trading_currency] + aux_currencies
        }
        # load env
        env = Environment(
            exchange=MockExchange(
                dfs=dfs,
                folder=f"data/{exchange_name}",
                wallet=Wallet(
                    purchase_currency(amount=100),
                    trading_currency(amount=0),
                    *[currency(amount=0) for currency in aux_currencies],
                ),
                purchase_currency=purchase_currency,
                interpolate=True,
            ),
            aux_currencies=aux_currencies,
            trading_currency=trading_currency,
            time=time,
        )
        # load agent
        model = keras.models.load_model(Path(path, "model.h5"), compile=False)
        agent = Agent(
            state_space=env.observation_space,
            action_space=env.action_space,
            model=model,
        )
        return cls(agent=agent, env=env, model_name=model_name)

    def train(self, epochs=10, steps=4096, iters=10, workers=None, test=None):
        """Train the agent.

        Parameters
        ----------
        `epochs`: `int`, optional.
            The amount of epochs to train for.
        
        `steps`: `int`, optional.
            The amount of agent - env interactions.
        
        `iters`: `int`, optional.
            The amount of time to train collected samples against the network.

        `workers`: `int`, optional.
            How many workers to spaw, If `None`, no extra process(s) will be
            spawned.
        
        `test`: `int` or `None`, optional.
            How often (in terms of epochs) to test the agent being trained. If 
            `None`, the agent will not be tested.
        """
        agent = self.agent
        env = self.env
        epoch = 0
        env.mode = "train"
        obs = env.reset()

        # spawn multiple processes for training
        if workers:
            ray.init()

            # create distributed copies of the agent & env
            loops = [
                utils.DistributedLoop.remote(
                    model_name=self.model_name, env=env, obs=obs
                )
                for worker in range(workers)
            ]
            progress = tqdm(total=epochs)
            while epoch < epochs:
                # collect and join samples from each loop
                collections = ray.get([loop.call.remote(steps=steps) for loop in loops])
                buffer = Buffer.from_collections(collections=collections)

                self.metrics.record_epoch(rewards=buffer.rew_buf[: buffer.idx])

                # update agent
                agent.optimize(dataset=buffer.as_dataset(), n_iters=tf.constant(iters))

                # after optimization, update loops with new model weights
                weights = agent.model.get_weights()
                ray.get([loop.set_weights.remote(weights) for loop in loops])

                # misc
                self.metrics.summarize_epoch(epoch=epoch)
                progress.update()
                epoch += 1
            progress.close()

        # run locally in one process, slow but useful for debugging
        else:
            buffer = agent.get_buffer()
            progress = tqdm(total=epochs * steps)
            loop = utils.Loop(
                agent=agent, env=env, obs=obs, buffer=buffer, progress=progress
            )
            remaining = steps
            while epoch < epochs:
                is_done, remaining = loop.call(steps=remaining)
                if is_done:
                    if test and epoch % test == 0 and False:
                        self.test(verbose=False)
                        env.mode = "train"
                    loop.obs = env.reset()

                if remaining:
                    continue
                else:
                    remaining = steps

                self.metrics.record_epoch(rewards=buffer.rew_buf[: buffer.idx])

                # update agent
                agent.optimize(dataset=buffer.as_dataset(), n_iters=tf.constant(iters))

                # misc
                self.metrics.summarize_epoch(epoch=epoch)
                epoch += 1
            progress.close()

    def test(self, epochs=1, verbose=True):
        """Train the agent.

        Parameters
        ----------
        `epochs`: `int`, optional.
            The amount of epochs to test for.
        
        `verbose`: `bool`, optional.
            If `True`, a live plot will display training data.
        """
        agent = self.agent
        env = self.env
        epoch = 0
        is_done = False
        env.mode = "test"

        while epoch < epochs:
            obs = env.reset()
            while not is_done:
                # step agent, then environment
                action, policy, value_est = agent.step(obs)
                obs, reward, is_done, info = env.step(action)

            if verbose:
                env.render()

            is_done = False
            epoch += 1

    def save(self, on_cloud=False):
        """Save an agent and an environment so that they can be loaded later and 
        restored for training/testing/inference.

        `on_cloud`: `bool`, optional.
            Whether or not the model resides on the cloud.
        
        Note
        ----
        This will also save everything needed to reinstantiate `Time`. If 
        `on_cloud` is `True`, then the environment variable `GCP_BUCKET` must be
        set.
        """
        agent = self.agent
        env = self.env
        data = dict(
            time=dict(
                start=str(env.time.index[0]),
                stop=str(env.time.index[-1]),
                interval=framework_utils.timedelta_to_str(env.time.interval),
                lookback=env.time.lookback,
            ),
            exchange=dict(purchase_currency=env.exchange.purchase_currency.ticker()),
            environment=dict(
                trading_currency=env.trading_currency.ticker(),
                aux_currencies=[currency.ticker() for currency in env.aux_currencies],
            ),
        )
        path = Path(f"models/{self.model_name}/")
        path.mkdir(parents=True, exist_ok=True)

        with open(Path(path, "data.json"), "w") as f:
            json.dump(data, f, indent=4)
        agent.model.save(Path(path, "model.h5"), save_format="tf")
        agent.path = path

        if on_cloud:
            model_path = f"models/{self.model_name}"
            shutil.make_archive(model_path, "zip", model_path)
            client = storage.Client()
            bucket = client.bucket(os.environ["GCP_BUCKET"])
            blob = storage.Blob(f"{model_path}.zip", bucket)
            with open(f"{model_path}.zip", "rb") as f:
                blob.upload_from_file(f)
