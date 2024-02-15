import itertools
import os
from datetime import datetime, timedelta
from decimal import Decimal
from pprint import pprint
from time import timezone

import ccxt
import numpy as np
import pandas as pd
import pytz
from ccxt.base.errors import BadSymbol

from framework import utils
from framework.core import Currencies, Time, Wallet
from framework.core.errors import FrameworkError
from framework.exchanges.abstract import AbstractExchange


class LiveExchange(AbstractExchange):
    """A generic class for trading at an exchange, and gathering training 
    data. Based off of the github repository `CCXTExchange`.
    """

    def __init__(self, exchange_id, purchase_currency=Currencies.USD):
        """
        Parameters
        ----------
        `exchange_id`: `str`.
            The name of the exchange (I.e., `binance`, etc).
            
        `purchase_currency`: `Currency`.
            That is used in all currency pair requests.
        """
        self._inverted = set()
        self._exchange = getattr(ccxt, exchange_id.lower())(
            {
                "apiKey": os.environ.get(f"API_KEY_{exchange_id.upper()}", None),
                "secret": os.environ.get(f"API_SECRET_{exchange_id.upper()}", None),
                "enableRateLimit": True,
                "timeout": 10000,
                "rateLimit": 2000,
            }
        )
        if not self._exchange.has["fetchOHLCV"]:
            raise FrameworkError.from_attribute(
                obj=self._exchange, attribute="fetchOHLCV"
            )
        super().__init__(
            maker_commission=0.001,
            taker_commission=0.001,
            folder=f"data/{exchange_id}",
            purchase_currency=purchase_currency,
        )

    @classmethod
    def from_id(cls, exchange_id, **kwargs):
        """Get a derived instance of this class from an exchange ID.
        
        If a subclass exists matching exchange_id, this will return an
        instance of that subclass. Otherwise, it will instantiate a `LiveExchange` 
        instance.
        
        Note
        ----
        `LiveExchange` is built on top of the `ccxt` module, which can be 
        limiting under some circumstances.
        
        Parameters
        ----------
        `exchange_id`: `str`.
            The name of the exchange. Must exist in the `ccxt` module.

        `**kwargs`.
            See constructor arguments.
        """
        exchange_id = exchange_id.lower()

        # see if a direct subclass exists that we can use instead.
        found = []
        for subclass in cls.__subclasses__():
            if exchange_id in subclass.__name__.lower():
                found.append(subclass)

        # make sure we haven't found more than one
        if len(found) > 1:
            raise FrameworkError.from_value(
                field="exchange_id",
                expected=f"to only find one matching exchange",
                received=found,
            )

        # exchange specific subclass exists
        elif found:
            subclass = found[0]
            return subclass(**kwargs)

        # return a default LiveExchange instance
        else:
            return cls(exchange_id=exchange_id, **kwargs)

    def _get_df(
        self, currency, start, stop, interval, start_key=None, stop_key=None, func=None
    ):
        """Return a Pandas DataFrame for `currency`, from `start`to
        `stop`, at every `interval`.

        Parameters
        ----------
        `currency`: `Currency`
            To get data for.

        `start`: `datetime`.
            To start the dataframe from.

        `stop`: `datetime`.
            To finish the dataframe at.
        
        `interval`: `timedelta`.
            The frequency to get data at.

        `start_key`: `str`, default: `None`.
            That is exchange specific, to add to `params`.
        
        `stop_key`: `str`, default: `None`.
            That is exchange specific, to add to `params`.
            
        `func`: `callable(datetime)`, default: `None`.
            To process the `start` and `stop` datetimes.

        Note
        ----
        As per the `ccxt` docs, the return format looks like this.
        ```
        [
            [
                1504541580000, # UTC timestamp in milliseconds, integer
                4235.4,        # (O)pen price, float
                4240.6,        # (H)ighest price, float
                4230.0,        # (L)owest price, float
                4230.7,        # (C)losing price, float
                37.72941911    # (V)olume (in terms of the base currency), float
            ],
            ...
        ]
        ```
        Returns
        -------
        `pd.DataFrame`
        """
        if func is None:
            func = lambda x: x

        params = {}
        if start_key and stop_key:
            params = {
                start_key: func(start),
                stop_key: func(stop),
            }
        elif bool(start_key) != bool(stop_key):
            raise FrameworkError.from_value(
                field=["start_key", "stop_key"],
                expected="that neither or both be provided",
                received={"start_key": start_key, "stop_key": stop_key},
            )

        # ccxt is stupid, and doesn't accept timedeltas
        interval_str = utils.timedelta_to_str(interval)
        last_timestamp = start
        last = None
        prev = last
        data = []

        if currency in self._inverted:
            quote_currency = self.purchase_currency
            purchase_currency = currency
        else:
            quote_currency = currency
            purchase_currency = self.purchase_currency

        symbol = utils.make_symbol(
            quote_currency=quote_currency, purchase_currency=purchase_currency, sep="/"
        )

        utils.print_bar()
        print(f"Downloading data for {currency.ticker()} from {start} to {stop}.")
        while True:
            # we can either use params to iterate over the data, or just keep
            # calling with the since argument
            try:
                if params:
                    iter_data = self._exchange.fetch_ohlcv(
                        symbol, timeframe=interval_str, params=params
                    )
                else:
                    iter_data = self._exchange.fetch_ohlcv(
                        symbol=symbol,
                        timeframe=interval_str,
                        since=(last_timestamp + interval).timestamp() * 1000,
                    )
            except BadSymbol as e:
                symbol = self._resolve_bad_symbol(e, currency)
                continue

            # data may not exist at a certain timestep,
            if iter_data == []:
                last_timestamp += 10 * interval
                continue

            data.append(iter_data)
            last = iter_data[-1][0]  # timestamp in milliseconds

            # get last datetime from
            tz = pytz.timezone("UTC")
            last_timestamp = tz.localize(
                datetime.utcfromtimestamp(int(last / 1000)), is_dst=None
            )

            if prev == last or last_timestamp >= stop:
                break
            if params:
                params[start_key] = func(last_timestamp + interval)
            prev = last

        # flatten data so its as if it was made in one API call
        data = itertools.chain.from_iterable(data)

        # generate df
        df = pd.DataFrame(
            data=data, columns=["Timestamp", "Open", "High", "Low", "Close", "Volume",],
        )
        df = df.set_index("Timestamp")
        if currency in self._inverted:
            for column in ["Open", "High", "Low", "Close"]:
                df[column] = 1.0 / df[column]
        df.index = pd.to_datetime(df.index, unit="ms")
        return df

    def _trade(self, currency, amount):
        """Handle market buys that require cost. Optional limit order override 
        boolean.
        """
        assert False  # for now, just in case we call this by accident

    def _resolve_bad_symbol(self, e, currency):
        """Helper method for common exception handling done in this class.
        """
        if not isinstance(type(currency), type):
            raise FrameworkError.from_type(
                field="currency", expected=type, received=type(currency)
            )
        if currency in self._inverted:
            raise BadSymbol(*e.args, self._inverted)
        # invert symbol and try again
        symbol = utils.make_symbol(
            quote_currency=self.purchase_currency, purchase_currency=currency, sep="/",
        )
        self._inverted.add(currency)
        return symbol

    def _get_wallet(self):
        """
        Returns
        -------
        `Wallet`.
        """
        raise NotImplementedError


class BinanceExchange(LiveExchange):
    """Override the `_get_df` function to allow for setting of interval times.
    """

    def __init__(self, **kwargs):
        if "purchase_currency" not in kwargs:
            kwargs["purchase_currency"] = Currencies.BNB
        super().__init__(exchange_id="binance", **kwargs)

    def _get_df(self, currency, start, stop, interval):
        """Return a Pandas DataFrame for `currency`, from `start`to
        `stop`, at every `interval`.
        """
        return super()._get_df(
            currency=currency,
            start=start,
            stop=stop,
            interval=interval,
            start_key="startTime",
            stop_key="endTime",
            func=lambda x: str(utils.to_milliseconds(x)),
        )
