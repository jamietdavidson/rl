import shutil
from datetime import datetime, timedelta
from decimal import Decimal
from pprint import pformat, pprint

import numpy as np
import pandas as pd
import pytz
from dateutil.parser import parse

from framework.core.errors import FrameworkError


def calculate_commission(amount, unit_price, commission):
    """Calculate adjusted amount/ price depending on whether or not a buy 
    or sell order is being executed.

    When buying, we often specify how much of something we want to buy based on
    the price we see it advertised at. What we don't take into consideration is
    the commission associated with buying. This function translates how much we expected
    to buy at a certain price into how much we need to buy in order to keep the 
    price the same.

    When selling, it is the opposite. We specify how much we want to sell, but
    we often don't take into account the commission we have to pay. This function 
    translates how much we expect to make during a sale into how much we will
    actually make during a sale.

    We also return the fee, which is effectively how much the exchange made 
    from us.

    Parameters
    ----------
    `amount`: (`Decimal`, `float`).
        The quantity of the currency to buy/sell.

    `unit_price`: (`Decimal`, `float`).
        The cost to buy 1 unit of the currency.

    `commission`: `float`.
        The percentage of commission the exchange takes.

    Returns
    -------
    (`amount`, `price`, `fee`).
        Adjusted accordingly.

    Note
    ----
    When `amount` is negative, `price` will come back negative as well.

    Examples
    --------
    ```
    >>> from framework import utils
    >>> amount, price, fee = utils.calculate_commission(
    ...    amount=1, 
    ...    unit_price=1, 
    ...    commission=0.5
    ... )
    >>> print(amount, price, fee)
    0.995 1 .005
    >>> amount, price, fee = utils.calculate_commission(
    ...    amount=-1, 
    ...    unit_price=1, 
    ...    commission=0.5
    ...)
    >>> print(amount, price, fee)
    -1 -0.995 .005
    ```
    """
    price = unit_price * amount
    commission = commission / 100.0

    # buying, implies commission is taken "off the top" of `amount`.
    if amount >= 0:
        adj_amount = amount * (1 - commission)
        adj_price = price
        fee = (amount - adj_amount) * unit_price

    # selling, implies commission is taken from expected `price`
    elif amount < 0:
        adj_amount = amount
        adj_price = price * (1 - commission)
        fee = adj_price - price

    assert fee >= 0
    return (adj_amount, adj_price, fee)


def calculate_networth(prices, wallet, purchase_currency, trading_currencies):
    """Calculate the networth of a wallet with respect to `purchase_currency`.
    
    Parameters
    ----------
    `prices`: `dict(Currency=float, ...)`.
        With respect to `purchase_currency`.
    
    `wallet`: `Wallet`.
        A wallet instance.
    
    `purchase_currency`: `Currency`.
        Being used as a reference point.
    
    `trading_currencies`: `Currency` or `list(Currency, ...)`.
        Whose amount(s) we want translated into `purchase_currency`.
    
    Returns
    -------
    `float`
    """
    trading_currencies = to_iterable(trading_currencies)
    networth = wallet[purchase_currency]
    for trading_currency in trading_currencies:
        networth += wallet[trading_currency] * prices[trading_currency]
    return networth


def check(obj, required_type, name="obj", context=None):
    """Checks and returns `value` if it is the specified type.

    Parameters
    ----------
    `obj`: `object`.
        Whose values will be checked against `required_type`.

    `required_type`: `type` (or tuple of types).
        Used to validate the type of 'obj'.

    `name`: `str`, optional.
        To make the error handling better.

    `context`: `object`, optional.
        To describe the calling code better.

    Raises
    ------
    `TypeError`

    Example
    -------
    ```
    from framework.utils import check
    amount = check('30.00', (int, Decimal)) # raises TypeError
    amount = check(30.00, (int, Decimal)) # assigns amount = 30.00
    ```
    """
    if not isinstance(obj, required_type):
        # format the error to make it human readable
        formatted_obj = obj if not isinstance(obj, dict) else pformat(obj)
        raise FrameworkError.from_type(
            field=formatted_obj, expected=required_type, received=obj,
        )
    return obj


def check_numeric(num, name="num"):
    """Checks to see if `num` is numeric.

    Parameters
    ----------
    `num`: (`int`, `Decimal`).
        Whose type will be checked against (`int`, `Decimal`).

    `name`: `str`, optional.
        To make the error handling better.

    Raises
    ------
    `TypeError`

    Example
    -------
    ```
    from framework.utils import check_numeric
    amount = check_numeric('30.00') # raises TypeError
    amount = check_numeric(30.00) # assigns amount = 30.00
    ```
    """
    num = check(num, (int, float, Decimal), name=name)
    if isinstance(num, float):
        num = Decimal(num)
    if isinstance(num, Decimal):
        num = round(num, 2)
    return num


def check_range(num, lb, ub, name="num"):
    """Checks and returns `num` if within range [`lb`, `ub`]
    
    Parameters
    ----------
    `num`: (`int`, `Decimal`).
        whose value will be compared against `lb` and `ub`.

    `lb`: (`int`, `Decimal`).
        used to validate the lower bound of `num`.

    `ub`: (`int`, `Decimal`).
        used to validate the upper bound of `num`.

    `name`: `str`, optional.
        to make the error handling better.

    Raises
    ------
    `ValueError`

    Note
    ----
    Endpoints of range are inclusive

    Example
    -------
    ```
    from framework.utils import check_range
    amount = check_range(30.01, 10, 30) # raises ValueError
    amount = check_range(30.00, 10, 30) # assigns amount = 30.00
    ```
    """
    if lb <= num <= ub:
        return num
    else:
        raise ValueError(f"'{name}' expected to be <= {ub} and >= {lb}, found: {num}")


def check_str_fits(string, size):
    """Checks and returns string iff len(string) <= size, else raises
    a `ValueError`.

    Parameters
    ----------
    `string`: `str`.
        Whose length will be compared against `size`.

    `size`: `int`.
        Used to validate the length of `string`.

    Raises
    ------
    `ValueError`

    Example
    -------
    ```
    from framework.utils import check_str_fits
    string = check_str_fits('hey there', 5) # raises ValueError
    string = check_str_fits('hey there', 10) # assigns string = 'hey there'
    ```
    """
    check(size, int, "size")
    if len(string) <= size:
        return string
    else:
        raise ValueError("len('{}') <= {}".format(string, size))


def check_not_null(obj, name="obj"):
    """Checks and returns 'num' if within range [lb, ub]

    Parameters
    ----------
    `obj`: `object`.
        That will be checked if null.

    `name`: `str`, optional.
        To make the error handling better.

    Raises
    ------
    `TypeError`

    Example
    -------
    ```
    from framework.utils import check_not_null
    amount = check_not_null(None) # raises TypeError
    amount = check_not_null(30.00) # assigns amount = 30.00
    ```
    """
    if obj is None:
        raise TypeError(f"Expected {name} to not be {None}")
    return obj


def get_filename(folder, purchase_currency, currency, interval):
    """Get the filename used for file saving.
    
    Parameters
    ----------
    `folder`: `str`.
        Whose name this data will be saved under.
    
    `purchase_currency`: `Currency`.
        The currency that is being used to purchase `currency`. Also the first
        ticker in the filename.
    
    `currency`: `Currency`.
        That is being purchased. The second ticker in the filename.

    `interval`: `timedelta`.
        Used for naming.
    
    Returns
    -------
    `str`
    """
    folder = folder if folder[-1] != "/" else folder[:-1]
    filename = (
        f"{folder}/{currency.ticker()}-"
        f"{purchase_currency.ticker()}-{timedelta_to_str(interval)}.csv"
    )
    return filename


def make_symbol(*, quote_currency, purchase_currency, sep=""):
    """Join the tickers of two currencies.
    
    Parameters
    ----------
    `quote_currency`: `Currency`.
        The currency that is being bought.
    
    `purchase_currency`: `Currency`.
        The currency that is `quote_currency` is being bought with.
    
    `sep`: `str`, optional.
        Used to separate the two symbols.
    
    Returns
    -------
    `str`
    
    Example
    -------
    ```
    from framework import utils
    
    sym = utils.make_symbol(
        currency=BTC, 
        purchase_currency=BNB,
    )
    print(sym) # 'BTCBNB'
    ```
    """
    return f"{quote_currency.ticker()}{sep}{purchase_currency.ticker()}"


def join_dfs(dfs, check=False):
    """Check that all columns are the same, and lowercase all columns.

    Parameters
    ----------
    `dfs`: `{Currency : pd.DataFrame, ...}`.
        To check the columns of.

    `check`: `bool`, optional.
        Check that all columns are the same. Happens after lowercasing.
    
    Returns
    -------
    `pd.DataFrame` with `MultiIndex`.
    """
    for df in dfs.values():
        df.columns = map(str.lower, df.columns)
    if check:
        columns_same = len(set([tuple(df.columns) for df in dfs.values()])) == 1
        if not columns_same:
            raise FrameworkError.from_value(
                field="dfs",
                expected="for each df to contain the same columns",
                recieved="different columns among the dfs",
            )
    dfs = pd.concat(dfs, axis=1)
    return dfs


def to_iterable(obj):
    """Cast an obj into a tuple if its not already a list or tuple.
    
    Parameters
    ----------
    `obj`: `object`.
        To wrap into a tuple.
    
    Returns
    -------
    `(object, )`
    """
    if obj is None:
        return tuple()
    if not isinstance(obj, (list, tuple)):
        return (obj,)
    else:
        return tuple(obj)


def to_milliseconds(dt):
    """Round a `datetime` to the nearest second and return its epoch time in 
    milliseconds.
    
    Parameters
    ----------
    `dt`: `datetime`.
        To convert into milliseconds.
    
    Returns
    -------
    `int`
    """
    return int(dt.timestamp()) * 1000


def parse_datetime(dt_str):
    """Convert a string representation into a timezone aware datetime object.
    
    Parameters
    ----------
    `dt_str`: `str`.
        To convert into a UTC datetime.
    
    Returns
    -------
    `datetime`
    """
    if dt_str == "now":
        dt = datetime.utcnow().replace(microsecond=0, second=0, minute=0)
    else:
        dt = parse(dt_str)
    return dt.replace(tzinfo=pytz.UTC)


def parse_interval(interval_str):
    """Convert a string representation of an interval into a timedelta. It is 
    expected that the value for `interval_str` is an integer representing 
    minutes.
    
    Parameters
    ----------
    `interval_str`: `str`.
        To convert into a timedelta. Must be an integer.
    
    Returns
    -------
    `timedelta`
    """
    return timedelta(minutes=int(interval_str))


def pct_change(x1, x2):
    """Compare the percent between two values.
    
    `x1`: One of (`float`, `np.ndarray`).
        As the first value.
    
    `x2`: One of (`float`, `np.ndarray`).
        That is being compared against `z1`.
    
    Returns
    -------
    (`float`, `np.ndarray`). Depending on what was passed in.
    """
    num = x2 - x1
    den = np.abs(x1)
    val = num / (den + 1e-4)
    return val * 100.0


def process_df(
    df, interval, start=None, stop=None, raise_exception=False, interpolate=False
):
    """Process a dataframe so that it contains no missing indices.
    
    Parameters
    ----------
    `df`: `pd.DataFrame`.
        To process.

    `interval`: `timedelta`.
        To check the consistency of `df.index` against.
    
    `start`: `datetime`, optional.
        To check that first index value against.

    `stop`: `datetime`, optional.
        To check the last index value against.

    `raise_exception`: `bool`, optional.
        Whether or not to raise an exception if `df.index` is not consistent.

    `interpolate`: `bool`, optional.
        If `True`, missing data will be interpolated.

    Note
    ----
    `raise_exception` and `interpolate` must both be `False`, or be mutually 
    exclusive.

    Returns
    -------
    `pd.DataFrame`

    Raises
    ------
    `FrameworkError`
    """
    correct_interval = interval

    # make sure all of the intervals are the same on the dataframe
    index = df.index.to_series()[1:]
    deltas = df.index.to_series().diff()[1:]
    intervals = sorted(set(deltas))

    if len(intervals) != 1:
        if correct_interval in intervals:
            intervals.remove(correct_interval)
        mapping = {}
        for interval in intervals:
            mapping[interval] = index[deltas == interval].tolist()

        # generate missing rows, assign them a value of NaN
        missing = []
        for interval, timestamps in mapping.items():
            for timestamp in timestamps:
                initial = timestamp - interval + correct_interval
                while initial < timestamp:
                    missing.append(initial)
                    initial += correct_interval

        if interpolate:
            missing_df = pd.DataFrame(index=missing, columns=df.columns, dtype=np.float)
            df = pd.concat([missing_df, df], axis=0)
            df = df.sort_index()

            # interpolate NaN values
            df = df.interpolate()
            assert len(missing) + len(index) == len(df) - 1  # -1 cause of `diff()`

        elif raise_exception:
            raise FrameworkError.from_value(
                field="df.index",
                expected="a consistent timedelta",
                received=pformat(mapping),
            )
        else:
            print("WARN: Inconsistent dataframe index", pformat(mapping))

    if start is not None and start < df.index[0]:
        raise FrameworkError.from_value(
            field="df.index[0]",
            expected=f"for it to be <= start ({start})",
            received=df.index[0],
        )
    if stop is not None and stop > df.index[-1]:
        raise FrameworkError.from_value(
            field="df.index[-1]",
            expected=f"a value > stop ({stop})",
            received=df.index[-1],
        )
    return df


def print_bar(char="="):
    """Print a bar the width of the terminal.
    
    Parameters
    ----------
    `char`: `str`.
        Of length one, to be printed across the console.
    """
    width = shutil.get_terminal_size()[0]
    print(char * width)


def nested_to_ndarray(self, objs):
    """
    Parameters
    ----------
    `objs`: `list`.
        Of nested objects to be cast into `np.ndarray`.
    
    Returns
    -------
    `list(np.ndarray, ...)`
    """
    cast_objs = []
    for obj in objs:
        cast_objs.append(np.asarray(obj))
    return cast_objs


def str_to_timedelta(s):
    """
    Parameters
    ----------
    `s`: `str`.
        In the format `"1m`", `"5m"`, `"1h"`, etc.
    
    Returns
    -------
    `timedelta`
    """
    time, units = int(s[:-1]), s[-1]
    if units == "m":
        seconds = time * 60
    elif units == "h":
        seconds = time * 3600
    return timedelta(seconds=seconds)


def timedelta_to_str(td):
    """
    Parameters
    ----------
    `td`: `timedelta`.
         To cast in a str in the format `"1m`", `"5m"`, `"1h"`, etc.
    
    Returns
    -------
    `str`
    """
    total_seconds = td.total_seconds()
    if total_seconds <= 60:
        return "1m"
    elif total_seconds <= 60 * 5:
        return "5m"
    elif total_seconds <= 60 * 15:
        return "15m"
    elif total_seconds <= 60 * 60:
        return "1h"
    elif total_seconds <= 60 * 60 * 6:
        return "6h"
    else:
        raise FrameworkError.from_value(
            field="td", expected="a timedelta less than 6 hours in length", received=td
        )
