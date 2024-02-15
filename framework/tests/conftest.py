from datetime import datetime, timedelta

import pytest

from framework.core import Currencies, Time, Wallet
from framework.environment import Environment
from framework.exchanges import MockExchange
from framework.processor import Processor


@pytest.fixture
def time():
    time = Time(
        start=datetime(2020, 1, 1),
        stop=datetime(2020, 1, 2),
        interval=timedelta(hours=1),
    )
    time.reset()
    return time


@pytest.fixture
def exchange(time):
    exchange = MockExchange(
        folder="framework/tests",
        dfs={Currencies.BTC: "framework/tests/data/BTC-BNB-1:00:00.csv"},
        wallet=Wallet(Currencies.BTC(10), Currencies.BNB(10)),
        purchase_currency=Currencies.BNB,
    )
    exchange.time = time
    time.reset()
    exchange.reset(Currencies.BTC)
    return exchange


@pytest.fixture
def environment(time):
    return Environment(time=time)
