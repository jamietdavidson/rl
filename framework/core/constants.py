import math

from framework import utils
from framework.core.errors import FrameworkError


class Operations:
    HOLD = 0
    BUY = 1
    SELL = 2


class Currencies:
    class Currency:
        precision = 10
        H = 0

        @classmethod
        def ticker(cls):
            return cls.__name__

        def __init__(self, amount=0):
            self.amount = amount

        def __repr__(self):
            return f"{round(self.amount, self.precision)} {self.__class__.__name__}"

        def __str__(self):
            return self.__repr__()

        def __hash__(self):
            return hash(self.H)

        def __eq__(self, other):
            return math.fabs(self.amount - other.amount) < 1e-6

        def __add__(self, other):
            amount = self.amount + utils.check(other, type(self), "other").amount
            return type(self)(amount=amount)

        def __sub__(self, other):
            amount = self.amount - utils.check(other, type(self), "other").amount
            return type(self)(amount=amount)

        def __iadd__(self, other):
            amount = utils.check(other, type(self), "other").amount
            self.amount += amount
            return self

        def __isub__(self, other):
            amount = utils.check(other, type(self), "other").amount
            self.amount -= amount
            return self

    class USD(Currency):
        H = 1

    class CAD(Currency):
        H = 2

    class BTC(Currency):
        H = 3

    class ETH(Currency):
        H = 4

    class XRP(Currency):
        H = 5

    class IOTA(Currency):
        H = 6

    class BNB(Currency):
        H = 7

    class USDT(Currency):
        H = 8

    class LTC(Currency):
        H = 9

    @staticmethod
    def from_ticker(ticker: str, raise_exception=False):
        if isinstance(ticker, Currencies.Currency):
            return ticker
        for currency in Currencies.Currency.__subclasses__():
            if currency.ticker() == ticker.upper():
                return currency
        if raise_exception:
            raise FrameworkError.from_runtime(message=f"Ticker: {ticker} not found.")
