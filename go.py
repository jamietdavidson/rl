from datetime import datetime, timedelta
from multiprocessing import Process

import click
from click import argument, command, group, option

import config
from framework import utils as framework_utils
from framework import Currencies

# Options that are shared among various subcommands
datetime_options = [
    option("--start", type=framework_utils.parse_datetime, required=True),
    option("--stop", type=framework_utils.parse_datetime, required=True),
    option(
        "--interval", type=click.Choice(["1m", "5m", "15m", "1h", "6h"]), required=True
    ),
    option("--lookback", type=int, required=True),
]

ticker_options = [
    option("--trading_ticker", required=True),
    option("--aux_tickers", "-a", multiple=True, required=False),
]

# Helper function for adding options to subcommands
def add_options(options):
    """Helper function for click to decorate subcommands with the same group
    of parameters. 
    
    See this thread: https://github.com/pallets/click/issues/108
    """

    def _add_options(func):
        for option in reversed(options):
            func = option(func)
        return func

    return _add_options


@group()
def cli():
    """Entrypoint to the DLT trading algorithm.
    """


@cli.command()
@add_options(datetime_options)
@add_options(ticker_options)
@option("--model", required=True)
@option("--steps", type=click.IntRange(0, 10000), required=False, default=10)
@option("--epochs", type=click.IntRange(0, 1000000), required=False, default=100)
@option("--workers", type=click.IntRange(0, 64), required=False, default=None)
@option("--test_every", type=click.IntRange(0, 10000), required=False, default=None)
@option("--interpolate", type=click.BOOL, required=False, default=False)
def train(
    start,
    stop,
    interval,
    lookback,
    trading_ticker,
    aux_tickers,
    model,
    steps,
    epochs,
    workers,
    test_every,
    interpolate,
):
    """Train. Define a model name - if it exists, it will be loaded. If not, it 
    will be created. Specify workers to run training in parallel processes.
    
    Once training has completed, the model will be saved locally under 
    "models/{model}".
    """
    from rl.runner import Runner

    runner = config.get_runner(
        model_name=model,
        start=start,
        stop=stop,
        interval=interval,
        lookback=lookback,
        trading_ticker=trading_ticker,
        aux_tickers=aux_tickers,
        interpolate=interpolate,
    )
    runner.train(epochs=epochs, steps=steps, workers=workers, test=test_every)
    runner.test(epochs=1, verbose=True)
    runner.save()


if __name__ == "__main__":
    cli()
