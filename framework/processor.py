import numpy as np
import pandas as pd
from talib import ADX, BBANDS, DX, MACD, RSI, SAR

pd.options.mode.chained_assignment = None


class Processor:
    """Process observations into a format well suited for a Neural Network.
    """

    def __call__(self, df):
        """
        Parameters
        ----------
        `df`: `pd.DataFrame`.
            A multi indexed dataframe that is processed into an `np.ndarray`.
        """
        processed = {}
        currencies = df.columns.unique(level=0)
        dfs = [df.xs(key=currency, axis=1, level=0) for currency in currencies]

        # main loop
        for currency, df in zip(currencies, dfs):
            # save and continue
            processed[currency] = self.process(df)

        df = pd.concat(processed, axis=1)
        X = df.to_numpy()
        X = np.nan_to_num(X)
        return X

    @classmethod  # easier to run tests against
    def process(cls, df):
        """Process a single dataframe.
        
        Parameters
        ----------
        `df`: `pd.DataFrame`.
            Containing columns: `'close'`, `'low'`, and `'high'`.
        
        Returns
        -------
        `pd.DataFrame`
        """
        _, df["close_norm"], _ = BBANDS(df["close"])

        # normalize between 0 - 1
        df["adx_norm"] = ADX(df["high"], df["low"], df["close"]) / 100.0
        df["rsi_norm"] = RSI(df["close"]) / 100.0
        df["macd_norm"], _, _ = MACD(df["close"])
        df["macd_norm"] = df["macd_norm"] / 100.0

        # normalize between -1 - 1
        df["dmi_norm"] = DX(df["high"], df["low"], df["close"])
        df["dmi_norm"] = df["dmi_norm"] / df["dmi_norm"].abs().max()

        df["sar_norm"] = SAR(df["high"], df["low"])
        df["sar_norm"] = df["close"] - df["sar_norm"]
        df["sar_norm"] = df["sar_norm"] / df["sar_norm"].abs().max()

        df["close_norm"] = np.log(df["close"]) - np.log(df["close_norm"])
        df["close_norm"] = df["close_norm"] / df["close_norm"].abs().max()

        # only keep processed columns
        df = df.filter(
            ["adx_norm", "rsi_norm", "dmi_norm", "sar_norm", "macd_norm", "close_norm"]
        )
        return df
