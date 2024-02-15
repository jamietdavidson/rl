import os
from pathlib import Path

from framework import utils as framework_utils
from framework import (
    Currencies,
    Environment,
    FrameworkError,
    LiveExchange,
    MockExchange,
    Time,
    Wallet,
)


def get_environment(
    start,
    stop,
    interval,
    lookback,
    trading_ticker,
    aux_tickers,
    exchange_name="binance",
    interpolate=False,
    live=False,
):
    """Return the environment, setup with a `MockExchange`, loaded with the 
    specified data.
    
    Parameters
    ----------
    `start`: `datetime`.
        To start running from.
    
    `stop`: `datetime`.
        To stop running from.
    
    `interval`: `timedelta`.
        The trading frequency.
    
    `lookback`: `int`.
        How far back to look for environment observations.
    
    `trading_ticker`: `str`.
        The primary ticker that will be bought and sold against `purchase_currency`.
    
    `aux_tickers`: `list(str, ...)`.
        Of tickers whose data we are interested in learning from as well.
    
    `exchange_name`: `str`, optional.
        The exchange to perform the trading against.
    
    `interpolate`: `bool`, optional.
        If `True`, missing data will be interpolated. If `False`, an exception
        will be raised.
        
    `live`: `bool`, optional.
        To run the agent in a live exchange or not.
    
    Note
    ----
    `purchase_currency` is set at the exchange, as some exchanges promote their 
    own coin in return for cheaper commission.
    
    Returns
    -------
    `Environment`
    """
    if live:
        raise NotImplementedError

    # currencies
    aux_currencies = [Currencies.from_ticker(t) for t in sorted(aux_tickers)]
    trading_currency = Currencies.from_ticker(trading_ticker)
    purchase_currency = Currencies.BNB

    dfs = {}
    exchange = LiveExchange.from_id(exchange_name, purchase_currency=purchase_currency)

    # ensure all dataframes are stored locally, or download them through ccxt
    for currency in [trading_currency] + aux_currencies:
        # if the environment variables for the exchange are not set, and the data
        # exists locally, the exchange will not make a live call. If this is not
        # True, an exception will be raised when the api keys are set as env variables
        exchange.download(
            currencies=currency,
            start=start,
            stop=stop,
            interval=interval,
            raise_exception=False,
        )

        # get the filename for the mock exchange, this same function is used
        # to save the files in `download`
        dfs[currency] = framework_utils.get_filename(
            folder=f"data/{exchange_name}",
            purchase_currency=purchase_currency,
            currency=currency,
            interval=interval,
        )

    # preload mock exchange with data for training
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
            interpolate=interpolate,
        ),
        aux_currencies=aux_currencies,
        trading_currency=trading_currency,
        time=Time(start=start, stop=stop, interval=interval, lookback=lookback),
    )
    return env


def get_network(state_space, action_space):
    """Create a simple Actor Critic model that can take in time series data and 
    produce logits and a value estimation.
    
    Parameters
    ----------
    `state_space`: `gym.spaces.Space`.
        Of an observation.
    
    `action_space`: `gym.spaces.Space`.
        Of the available actions.
    
    Returns
    -------
    `keras.models.Model`
    """
    # keep these imports in here so that basic `python go.py` usage isn't
    # slow due to imports
    import tensorflow as tf
    from tensorflow import keras

    n_timesteps, n_features = state_space.shape

    heads = []
    inputs = keras.layers.Input(shape=state_space.shape, dtype="float32")

    # create a multi-headed model (with one input)
    for k in [4, 8]:
        # fmt: off
        x = keras.layers.Conv1D(filters=16, kernel_size=k, activation="relu", dtype="float32")(inputs)
        k = k // 2
        x = keras.layers.Conv1D(filters=32, kernel_size=k, activation="relu", dtype="float32")(x)
        x = keras.layers.Flatten()(x)
        # fmt: on
        heads.append(x)

    # join 1D conv heads
    x = keras.layers.Concatenate()(heads)
    x = keras.layers.Dense(64, activation="relu")(x)
    x = keras.layers.BatchNormalization()(x)
    x = keras.layers.Dense(64, activation="relu")(x)
    trunk = keras.layers.BatchNormalization()(x)

    x1 = keras.layers.Dense(128, activation="relu")(trunk)
    logits = keras.layers.Dense(action_space.n, activation="linear")(x1)

    x2 = keras.layers.Dense(128, activation="relu")(trunk)
    value_est = keras.layers.Dense(1, activation="linear")(x2)

    # compose model
    model = keras.Model(inputs=inputs, outputs=[logits, value_est])
    return model


def get_runner(
    model_name,
    start,
    stop,
    interval,
    lookback,
    trading_ticker,
    aux_tickers,
    exchange_name="binance",
    interpolate=False,
):
    """Get the runner, loaded up to train, test, or serve from.
    
    Parameters
    ----------
    `model_name`: `str`.
        The name of the model. If it exists, it will be loaded, otherwise it 
        will be created.
    
    `start`: `datetime`.
        To start running from.
    
    `stop`: `datetime`.
        To stop running from.
    
    `interval`: `str`.
        In the format `"1m`", `"5m"`, `"1h"`, etc.
    
    `lookback`: `int`.
        How far back to look for environment observations.
    
    `trading_ticker`: `str`.
        The primary ticker that will be bought and sold against `purchase_currency`.
    
    `aux_tickers`: `list(str, ...)`.
        Of tickers whose data we are interested in learning from as well.
    
    `exchange_name`: `str`, optional.
        The exchange to perform the trading against.
    
    `interpolate`: `bool`, optional.
        If `True`, missing data will be interpolated. If `False`, and there is 
        missing data, an exception will be raised.
    
    Note
    ----
    `purchase_currency`  s set at the exchange, as some exchanges promote their 
    own coin in return for cheaper commission.
    
    Returns
    -------
    `Runner`
    """
    from rl.runner import Runner
    from rl.agent import Agent

    # checked later
    interval = framework_utils.str_to_timedelta(interval)

    path = Path(f"models/{model_name}")
    if path.exists() and not any(path.iterdir()):
        runner = Runner.from_save(model_name)
    else:
        env = get_environment(
            start=start,
            stop=stop,
            interval=interval,
            lookback=lookback,
            trading_ticker=trading_ticker,
            aux_tickers=aux_tickers,
            exchange_name=exchange_name,
            interpolate=interpolate,
        )
        agent = Agent(
            state_space=env.observation_space,
            action_space=env.action_space,
            model=get_network(
                state_space=env.observation_space, action_space=env.action_space
            ),
        )
        runner = Runner(agent=agent, env=env, model_name=model_name)
        runner.env.time.start = start
        runner.env.time.stop = stop

    runner.env.reset()  # BUG: abstract exchange appending to cached data, causing unneccesary reload
    return runner
