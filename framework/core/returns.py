import numpy as np


class Ratio:
    """Basic simple returns calculations. 
    """

    def __call__(self, *args, **kwargs):
        """
        """
        return self.call(*args, **kwargs)

    def call(self, returns):
        """
        """
        returns = np.log(returns)
        returns = np.diff(returns)

        # filter diff, or floats will cause an issue when calculating returns
        mask_float_err = ~(np.absolute(returns) > 1e-4)
        mask_not_zero = returns != 0
        mask = mask_float_err & mask_not_zero
        returns[mask] = 0.0

        ratio = self._calculate(returns)
        if np.isnan(ratio) or np.isinf(ratio):
            return 0.0
        return ratio

    def _calculate(self, returns):
        """
        """
        return returns[-1]


class SharpeRatio(Ratio):
    """
    """

    def _calculate(self, returns):
        """
        """
        return returns.mean() / returns.std()


class SortinoRatio(Ratio):
    """The following formula shows calculation of Sortino ratio:

    Sortino Ratio =	(Portfolio Return − Risk Free Rate) / 
                    Portfolio Downside Standard Deviation
    
    The numerator of Sortino ratio equals Jensen's alpha. Portfolio return equals
    the weighted-average return of the whole portfolio of investments. It is 
    calculated as the sum of product of investment weights and individual return. 
    Risk-free rate equals the yield on long-term government bonds.

    Downside standard deviation is calculated as follows:

    Step 1:
        Identify the reference point below which the return is considered bad, 
        let us call it minimum acceptable return (MVR), it might be the mean 
        return, the risk-free rate or 0.

    Step 2: 
        Find deviation of each return value from the minimum acceptable return, 
        if the value is above MVR, ignore it and if the value is below MVR, 
        square it.

    Step 3: 
        Sum up all the squared values in Step 2.

    Step 4: 
        Divide values obtained in Step 3 by n, i.e. the total number of 
        observations.

    Step 5: 
        Take square root of the value in Step 4.
        
    Parameters
    ----------
    `returns` : `array like`.
        Containing a diff of timeseries data to evaluate.
        
    Returns
    -------
    `float` : The Sortino Ratio.
    """

    def _calculate(self, returns):
        """
        """
        # setup
        downside = np.empty_like(returns)
        downside[:] = np.nan

        # step 1
        mask = returns < 0

        # step 2
        downside[mask] = returns[mask] ** 2

        # step 3
        downside_sum = np.nansum(downside)

        # step 4
        downside_normalized = downside_sum / len(downside)

        # step 5 (final step)
        downside_std = np.sqrt(downside_normalized)

        mean_return = returns.mean()

        ratio = (mean_return - 0) / downside_std
        return ratio


# alias's
percent = Ratio()
sharpe = SharpeRatio()
sortino = SortinoRatio()
