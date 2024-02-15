import os
from abc import ABCMeta, abstractmethod
from collections import OrderedDict
from copy import deepcopy
from datetime import datetime, timedelta
from decimal import Decimal
from os.path import abspath
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from gym import spaces

from framework import utils
from framework.core import Currencies, FrameworkError, Operations, Time, Wallet
from framework.core import returns as return_funcs


class Environment:
    """A trading environment.
    """

    def __init__(self, exchange, time, trading_currency, aux_currencies=None):
        """
        Parameters
        ----------
        `exchange`: `Exchange`.
            That this broker will be trading on.
        
        `time`: `Time`.
            The time object that will be used to keep all exchanges in sync.
        
        `trading_currency`: `Currency`.
            That we are trading `purchase_currency` for.

        `aux_currencies`: `list(Currency, ...)`, optional.
            Other currencies to include in the observations.
        """
        self.exchange = exchange
        self.trading_currency = trading_currency
        self.aux_currencies = utils.to_iterable(aux_currencies)

        self.time = time
        self.exchange.time = time
        self.summary = dict(networth=[], trades=[])

        # for logging env
        currencies = self._get_currencies()
        self.exchange.reset(currencies=currencies)
        self.init_networth = self._calculate_networth()
        self.prev_networth = self.init_networth

        # calculate space information for agent
        self.action_space = spaces.Discrete(3)  # buy, sell, hold
        # feed through processor
        obs = self._get_observation()
        self.observation_space = spaces.Box(
            low=np.float32(-1),
            high=np.float32(1),
            shape=np.shape(obs),
            dtype=np.float32,
        )

    @property
    def mode(self):
        """Retrieve the current mode.
        
        Returns
        -------
        `str`
        """
        return self.time.mode

    @mode.setter
    def mode(self, x):
        """Set the mode of time. Can be one of `"train"` or `"test"`.
        
        Parameters
        ----------
        `x`: `str`.
            To configure the environment to.
        """
        self.time.mode = x

    def step(self, action):
        """
        Parameters
        ----------
        `action`: `int`.
            One of [`0`, `1`, `2`].
        
        Returns
        -------
        `np.ndarray`, `np.float32`, `np.bool`, `dict`.
        """
        ratio = 0.3
        currencies = self._get_currencies()
        prices = self.exchange.get_prices(currencies=currencies)
        wallet = self.exchange.get_wallet(currencies=currencies)
        purchase_currency = self.exchange.purchase_currency

        # process action
        if action != Operations.HOLD:
            if action == Operations.BUY:
                amount = (
                    wallet[purchase_currency] / prices[self.trading_currency] * ratio
                )
            elif action == Operations.SELL:
                amount = -wallet[self.trading_currency] * ratio
            # execute trade
            if amount:
                trade_info = self.exchange.trade(trades={self.trading_currency: amount})
                trade_info["timestamp"] = self.time.current_datetime
                self.summary["trades"].append(trade_info)

        # wait ...
        self.time.step()

        # calculate reward
        networth = self._calculate_networth()
        reward = networth - self.prev_networth
        self.prev_networth = networth
        self.summary["networth"].append(networth)

        # get state
        state = self._get_observation()

        # check if we are going broke
        done = networth < 0.5 * self.init_networth or self.time.done()
        return state, reward, done, {}

    def reset(self):
        """Reset the broker for more trading.
        """
        self.summary = dict(networth=[], trades=[])
        self.time.reset()
        self.exchange.reset(self._get_currencies())
        self.init_networth = self._calculate_networth()
        self.prev_networth = self.init_networth
        self.summary["networth"].append(self.init_networth)
        return self._get_observation()

    def render(self):
        """Render the environment for visualization & training insight.
        """
        exchange_df = self.exchange.get_df(
            currencies=self.trading_currency,
            start=self.time.start + self.time.interval * self.time.lookback,
            stop=self.time.stop,
            interval=self.time.interval,
        )
        if not self.summary["trades"]:
            print("No trades made")
            return
        exchange_df = utils.process_df(
            exchange_df, interval=self.time.interval, interpolate=True
        )
        trades_df = pd.DataFrame(data=self.summary["trades"])
        trades_df = trades_df.set_index("timestamp")
        trades_df.index = pd.to_datetime(trades_df.index, utc=True)
        exchange_df["networth"] = self.summary["networth"][
            -len(exchange_df.index) :
        ]  # TODO: why is this neccesary

        exchange_df["buy"] = np.nan
        exchange_df["sell"] = np.nan

        buy_mask = trades_df.index[trades_df["amount"] > 0]
        sell_mask = trades_df.index[trades_df["amount"] < 0]

        missing_indices = set(buy_mask).union(sell_mask).difference(exchange_df.index)
        if missing_indices:
            raise FrameworkError.from_runtime(
                'The following indices exist in the "trades" summary but '
                f"are not available in the exchange data:\n\n{list(sorted(missing_indices))}\n\n"
                f"Exchange index: {exchange_df.index}"
            )

        exchange_df.loc[buy_mask, "buy"] = exchange_df.loc[buy_mask]["close"]
        exchange_df.loc[sell_mask, "sell"] = exchange_df.loc[sell_mask]["close"]
        index = exchange_df.index

        fig = go.Figure()
        fig.add_trace(
            go.Scatter(
                x=index,
                y=exchange_df["close"],
                name="Close",
                line_color="#CC5500",
                mode="lines",
            )
        )
        fig.add_trace(
            go.Scatter(
                x=index,
                y=exchange_df["buy"],
                name="Buy",
                line_color="#42B3D5",
                mode="markers",
            )
        )
        fig.add_trace(
            go.Scatter(
                x=index,
                y=exchange_df["sell"],
                name="Sell",
                line_color="#1A237E",
                mode="markers",
            )
        )
        fig.add_trace(
            go.Scatter(
                x=index,
                y=exchange_df["networth"],
                name="Networth",
                yaxis="y2",
                line_color="#E1AD01",
                mode="lines",
            )
        )
        fig.update_traces(
            marker=dict(size=12, line=dict(width=2, color="DarkSlateGrey"))
        )
        fig.update_layout(
            title_text="Trading summary",
            xaxis=dict(rangeslider=dict(visible=True)),
            yaxis=dict(
                title=f"Closing price of {self.trading_currency.ticker()}", side="left"
            ),
            yaxis2=dict(
                title=f"Networth w/ respect to {self.exchange.purchase_currency.ticker()}",
                overlaying="y",
                side="right",
            ),
        )
        fig.show()

    def _calculate_networth(self):
        """Shortcut for calling the `calculate_networth` utility function.
        
        Returns
        -------
        `float`
        """
        currencies = self._get_currencies()
        networth = utils.calculate_networth(
            prices=self.exchange.get_prices(currencies=currencies),
            wallet=self.exchange.get_wallet(currencies=currencies),
            purchase_currency=self.exchange.purchase_currency,
            trading_currencies=self.trading_currency,
        )
        return networth

    def _get_currencies(self):
        """Return all of the currencies needed to make calls into the exchange.
        """
        return (self.trading_currency,) + self.aux_currencies

    def _get_observation(self):
        """Return a processed observation.
        """
        currencies = self._get_currencies()
        obs = self.exchange.get_data(currencies=currencies)
        return obs
