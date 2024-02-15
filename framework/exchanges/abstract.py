import os
from abc import ABCMeta, abstractmethod
from collections import OrderedDict
from datetime import datetime, timedelta
from decimal import Decimal
from pathlib import Path

import numpy as np
import pandas as pd

from framework import utils
from framework.core import Wallet
from framework.core import returns as return_funcs
from framework.processor import Processor
from framework.utils import FrameworkError


class AbstractExchange(metaclass=ABCMeta):
    """Abstract class for all exchanges.
    """

    def __init__(
        self,
        purchase_currency,
        folder,
        key="close",
        processor=Processor(),
        interpolate=False,
        maker_commission=0.005,
        taker_commission=0.005,
    ):
        """
        Parameters
        ----------
        `purchase_currency`: `Currency`.
            That prices will be compared against, and the exchange will used
            to make trades.
        
        `folder`: `str`.
            Where to load/save data for this exchange.
        
        `key`: `str`, optional.
            Used as the dataframe field for evaluating prices, networth, etc.
        
        `processor`: `callable`.
            Used to process exchange data.
        
        `interpolate`: `bool`, optional.
            To missing interpolate data from `.get_df()`.
        
        `maker_commission`: `float`.
            How much commission is taken during a 'maker' trade.

        `taker_commission`: `float`.
            How much commission is taken during a 'taker' trade.
        """
        self.key = key
        self.time = None  # gets set externally
        self.folder = folder
        self.processor = processor
        self.interpolate = interpolate
        self.commission = max(maker_commission, taker_commission)
        self.purchase_currency = purchase_currency

        self._cache = dict(
            time=set(),
            runtime_df=None,
            individual_dfs=dict(),
            processed_ndarray=None,
            runtime_currencies=set(),
        )

    def download(self, currencies, start, stop, interval, *, raise_exception=False):
        """
        Parameters
        ----------
        `currencies`: `Currency` or `tuple(Currency, ...)`
            To get data for.
        
        `start`: `datetime`.
            To start downloading from.

        `stop`: `datetime`.
            To finishing downloading from.

        `interval`: `timedelta`.
            The period to download data between.
        
        `raise_exception`: `bool`, optional.
            Whether or not to raise an exception if the downloaded data is 
            inconsistent with the request.
    
        Raises
        ------
        `TypeError`, `ValueError`
        """
        return self.get_df(
            currencies=currencies,
            start=start,
            stop=stop,
            interval=interval,
            save=True,
            raise_exception=raise_exception,
        )

    def get_data(self, currencies):
        """Get a `np.ndarray` of observations for all `trading_currencies`, with
        a window size of `self.time.lookback`.

        Parameters
        ----------
        `currencies`: `tuple(Currency, ...)`.
            Of currencies to return in the `observation`.

        Returns
        -------
        `np.ndarray`
        """
        data = self._cache["processed_ndarray"]

        # observation is cached
        if data is not None and len(data) == len(self.time):
            idx = self.time.current_index
            return data[idx - self.time.lookback : idx]

        # need to recompute it
        else:
            df = self.get_df(
                currencies=currencies,
                start=self.time.current_datetime
                - self.time.interval * self.time.lookback,
                stop=self.time.current_datetime,
                interval=self.time.interval,
            )
            data = self.processor(df)
            self._cache["processed_ndarray"] = data
            return data[-self.time.lookback :]

    def get_df(
        self,
        currencies,
        start,
        stop,
        interval,
        *,
        save=False,
        raise_exception=False,
        as_multiindex=False,
    ):
        """Get data for `currency`. If a file already exists matching the 
        parameters, it will load it into memory and gather the remaining data.

        Parameters
        ----------
        `currencies`: `Currency` or `tuple(Currency, ...)`
            To get data for.
        
        `start`: `datetime`.
            To start the dataframe from.

        `stop`: `datetime`.
            To finish the datafrom at.

        `interval`: `timedelta`.
            The frequency to get data at.

        `validate`: `bool`, optional.
            Check the dataframe afterwards.

        `save`: `bool`, optional.
            To save the resulting dataframe or not.

        `raise_exception`: `bool`, optional.
            Whether or not to raise an exception if the saved data is 
            inconsistent with the request.

        `as_multiindex`: `bool`, optional.
            Return a `pd.DataFrame` with a `pd.MultiIndex` regardless of how 
            many currencies are passed in.

        Raises
        ------
        `TypeError`, `ValueError`
        
        Note
        ----
        The returned `pd.DataFrame` will have a `pd.MultiIndex` if `currencies` 
        contains more than one `Currency`, or if ` as_multiindex` is `True`.
        
        Returns
        -------
        `pd.DataFrame`
        """
        currencies = utils.to_iterable(currencies)

        if (
            self._cache["runtime_df"] is not None
            and start >= self._cache["runtime_df"].index[0]
            and stop <= self._cache["runtime_df"].index[-1]
            and set(currencies) == self._cache["runtime_currencies"]
        ):
            # we already have this dataframe, happens during training
            df = self._cache["runtime_df"]
            return df[(df.index >= start) & (df.index <= stop)]

        dfs = {}
        for currency in currencies:
            filename = utils.get_filename(
                folder=self.folder,
                purchase_currency=self.purchase_currency,
                currency=currency,
                interval=interval,
            )
            beginning, end = None, None

            if currency not in self._cache["individual_dfs"]:
                # df could still be saved as a file
                if not os.path.exists(filename):
                    df = self._get_df(currency, start, stop, interval)
                else:
                    # df is saved as a file, open it
                    df = pd.read_csv(filename, index_col=0, parse_dates=True)
                    df.index = pd.to_datetime(df.index, utc=True)
                    beginning = df.index[0] - interval
                    end = df.index[-1] + interval
            else:
                df = self._cache["individual_dfs"][currency]
                beginning = df.index[0] - interval
                end = df.index[-1] + interval

            # download missing head data, if any
            if beginning and beginning >= start:
                pre_df = self._get_df(currency, start, beginning, interval)
                df = pd.concat([pre_df, df], axis=0)

            # download missing tail data, if any
            if end and end <= stop:
                post_df = self._get_df(currency, end, stop, interval)
                df = pd.concat([df, post_df], axis=0)

            # cast to utc for consistency
            df.index = pd.to_datetime(df.index, utc=True)
            if save:
                path = Path(filename.rsplit("/", 1)[0])
                path.mkdir(parents=True, exist_ok=True)
                df = utils.process_df(df, interval, raise_exception=raise_exception)
                df.to_csv(filename, float_format="%.6f")

            dfs[currency] = df

        if len(currencies) > 1 or as_multiindex:
            df = utils.join_dfs(dfs)
            if self.interpolate:
                df = utils.process_df(df=df, interval=interval, interpolate=True)
            self._cache["runtime_df"] = df
            self._cache["individual_dfs"].update(dfs)
            self._cache["runtime_currencies"] = set(currencies)

        # no need to validate start and end dates in index, as an
        # exception will happen at runtime if they do not exist
        return df[(df.index >= start) & (df.index <= stop)]

    def get_wallet(self, currencies):
        """Get the current wallet.
        
        Returns
        -------
        `Wallet`
        """
        wallet = self._get_wallet()

        # add in purchase currency
        currencies = set(currencies)
        currencies.add(self.purchase_currency)

        if set(wallet.currencies) != currencies:
            raise FrameworkError.from_value(
                field="get_wallet",
                expected=tuple(currencies),
                received=tuple(wallet.currencies),
            )
        return wallet

    def get_prices(self, currencies, index=None):
        """Get the current prices of all `trading_currencies`. 

        Parameters
        ----------
        `index` : `int`, default : None
            The index of time to retrieve the prices. If not specified, the
            current index of `self.time` will be used (the most up to date 
            prices).

        Returns
        -------
        `dict(Currency=float, ...)`
        """
        df = self.get_df(
            currencies=currencies,
            start=self.time.start,
            stop=self.time.current_datetime,
            interval=self.time.interval,
            as_multiindex=True,
        )
        # reduce the multi-index df to a singular dataframe
        df = df.xs(self.key, axis=1, level=1)

        # extract the values as a dictionary
        index = index if index else self.time.relative_index

        try:
            prices = df.iloc[[index]].to_dict("records")[0]
        except IndexError as e:
            raise FrameworkError.from_runtime(
                f"Attempted to access {index} at {self.time}.\n\n"
                f"DataFrame index:\n\n{df.index}"
            )

        # map the purchase currency
        prices[self.purchase_currency] = 1.0
        return prices

    def reset(self, currencies):
        """Reset the exchange. This also ensures that the model is setup before
        execution.

        Parameters
        ----------
        `currencies`: `tuple(Currency, ...)`.
            To reset this exchange with.

        Note
        ----
        This function returns `None`. In order to get an observation, call
        `get_data()` after this.
        """
        time = set([self.time.start, self.time.stop])
        if self._cache["time"] != time:
            currencies = utils.to_iterable(currencies)
            df = self.get_df(
                currencies=currencies,
                start=self.time.start,
                stop=self.time.stop,
                interval=self.time.interval,
                as_multiindex=True,
            )
            self.time.register(df)

            # cache the processor output
            self._cache["processed_ndarray"] = self.processor(df)
            self._cache["time"] == time

    def trade(self, trades):
        """Make a series of trades on the exchange.
        
        Parameters
        ----------
        `trades`: `dict(Currency=0.1, ...)`.
            Dictionary with how much of each currency to purchase. Negative
            values indicate a sell.

        Returns
        -------
        `list(obj, ...)`
        """
        # FIXME: returned objects incorrect
        obj = None
        for currency, amount in trades.items():
            obj = self._trade(currency=currency, amount=amount)
        return obj

    @abstractmethod
    def _get_df(self):
        """
        """

    @abstractmethod
    def _get_wallet(self):
        """
        """

    @abstractmethod
    def _trade(self, amount, currency):
        """
        """
