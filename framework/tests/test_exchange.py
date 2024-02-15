import pytest

from framework import utils
from framework.core import Currencies


def test_reset(exchange):
    exchange.reset(Currencies.BTC)


def test_trades(exchange):
    exchange.commission = 0
    time = exchange.time
    time.lookback = 0
    time.reset()
    trading_currency = Currencies.BTC
    purchase_currency = exchange.purchase_currency

    currencies = utils.to_iterable([trading_currency, purchase_currency])

    # prices are a sin wave, so selling high then buying
    # low should increase our networth
    networth1 = utils.calculate_networth(
        prices=exchange.get_prices(currencies=currencies),
        wallet=exchange.get_wallet(currencies=currencies),
        purchase_currency=purchase_currency,
        trading_currencies=trading_currency,
    )

    # sell all bitcoins
    exchange.trade({Currencies.BTC: -10.0})

    # simulation is over 1 day with 1 hour
    # steps, here we step to the bottom of wave
    [time.step() for _ in range(12)]

    # purchase all bitcoins
    exchange.trade({Currencies.BTC: 16.25})

    # step back up to top of wave
    [time.step() for _ in range(12)]

    # and calculate networth again
    networth2 = utils.calculate_networth(
        prices=exchange.get_prices(currencies=currencies),
        wallet=exchange.get_wallet(currencies=currencies),
        purchase_currency=purchase_currency,
        trading_currencies=trading_currency,
    )
    assert networth1 < networth2
