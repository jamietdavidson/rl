from copy import deepcopy

import numpy as np
import pandas as pd
from gym.spaces import Space

from framework import utils
from framework.core import Currencies, Time, Wallet
from framework.exchanges import AbstractExchange


class MockExchange(AbstractExchange):
    """
    """

    def __init__(self, dfs, wallet, *args, **kwargs):
        """
        """
        self.dfs = dfs
        self.wallet = None
        self.init_wallet = wallet
        if "purchase_currency" not in kwargs:
            kwargs["purchase_currency"] = Currencies.USD
        super().__init__(*args, **kwargs)

    def reset(self, *args, **kwargs):
        """Reset the initial state of the wallet, then `super().reset()`.
        """
        self.wallet = None
        return super().reset(*args, **kwargs)

    def _get_df(self, currency, *args):
        """
        """
        df = pd.read_csv(self.dfs[currency], index_col=0)
        df.index = pd.to_datetime(df.index, utc=True)

        if self.interpolate:
            df = utils.process_df(
                df=df,
                interval=self.time.interval,
                interpolate=True,
                raise_exception=True,
            )
        return df

    def _get_wallet(self):
        """
        """
        if self.wallet is None:
            self.wallet = self.init_wallet.copy()
        return self.wallet

    def _trade(self, amount, currency):
        """Spoof trade on a mock exchange. What this means is that we need to 
        adjust our wallet accordingly. 
        """
        prices = self.get_prices(currencies=currency)
        adj_amount, adj_price, fee = utils.calculate_commission(
            amount=amount, unit_price=prices[currency], commission=self.commission,
        )
        # save the information
        self.wallet -= self.purchase_currency(adj_price)
        self.wallet += currency(adj_amount)
        return dict(
            amount=amount,
            adj_amount=adj_amount,
            unit_price=prices[currency],
            adj_price=adj_price,
            fee=fee,
        )
