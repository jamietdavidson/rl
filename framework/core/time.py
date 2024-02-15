from datetime import datetime, timedelta

import pytz
from pandas import DatetimeIndex, Index

from framework.core.errors import FrameworkError
from framework import utils


class Time:
    """A special object used to keep time-dependant objects in sync.
    """

    def __init__(self, start, stop, interval, lookback=32):
        """
        Parameters
        ----------
        `start`: `datetime`.
            To start stepping time through at.
        
        `stop`: `datetime`.
            To stop stepping time at.
        
        `interval`: `timedelta`.
            At what frequency time should be stepped at.
        
        `lookback`: `int`, optional.
            How far back to look back when performing timestep calculations.
        """
        self._lookback = lookback
        self._start_index = 0
        self._current_index = 0 + lookback
        self._stop_index = None
        self._interval = None
        self._mode = None
        self._start = None
        self._stop = None

        self.interval = interval
        self.start = start
        self.stop = stop

    def __str__(self):
        return f"Time({self._start}, {self._stop}, {self.current_datetime})"

    def __repr__(self):
        return f"Time({self._start}, {self._stop})"

    def __len__(self):
        return self._stop_index - self._start_index

    # properties

    @property
    def current_datetime(self):
        """Retrieve the current datetime.
        
        Returns
        -------
        `datetime`
        """
        dt = self._index[self._current_index]
        return dt

    @property
    def current_index(self):
        """Retrieve the current index.
        
        Returns
        -------
        `int`
        """
        return self._current_index

    @property
    def lookback(self):
        """Retrieve the lookback value.
        
        Returns
        -------
        `int`
        """
        return self._lookback

    @property
    def mode(self):
        """Retrieve the current mode.
        
        Returns
        -------
        `str`
        """
        return self._mode

    @property
    def relative_index(self):
        """Retrieve the index relative to the current mode.
        
        Returns
        -------
        `int`
        """
        return self._current_index - self._start_index

    @property
    def start(self):
        """Retrieve the start date.
        
        Returns
        -------
        `datetime`
        """
        return self._start

    @property
    def steps(self):
        """Retrieve the amount of steps in the index.
        
        Note
        ----
        This includes the lookback window.
        
        Returns
        -------
        `int`
        """
        return len(self._index)

    @property
    def stop(self):
        """Retrieve the stop date.
        
        Returns
        -------
        `datetime`
        """
        return self._stop

    @property
    def index(self):
        """Retrieve the underlying `DatetimeIndex`.
        
        Returns
        -------
        `DatetimeIndex`
        """
        return self._index

    @property
    def interval(self):
        """Retrieve the interval.
        
        Returns
        -------
        `timedelta`
        """
        return self._interval

    # setters

    @mode.setter
    def mode(self, x):
        """Set the mode of time. Can be one of `"train"` or `"test"`.
        
        Parameters
        ----------
        `x`: `str`.
            To conigure the time to.
        """
        if x is None:
            if self._mode is not None:
                self.start = self._index[0]
                self.stop = self._index[-1]
            self._mode = None
            return

        allowed_values = ("train", "test")
        if x not in allowed_values:
            raise FrameworkError.from_value(
                field="mode", expected=f"one of {allowed_values}", received=x
            )

        split = 0.8
        idx = int(len(self._index) * split)
        available_lookback = len(self._index) - idx
        if available_lookback <= self._lookback:
            raise FrameworkError.from_value(
                field="mode",
                expected=f"a test split lookback size > {self._lookback}",
                received=available_lookback,
            )

        if x == "train":
            self._start = self._index[0]
            self._stop = self._index[idx]
            self._start_index = 0
            self._stop_index = idx
        elif x == "test":
            self._start = self._index[idx]
            self._stop = self._index[-1]
            self._start_index = idx
            self._stop_index = len(self._index) - 1

    @start.setter
    def start(self, x):
        """Set the start date.

        Parameters
        ----------
        `x`: `datetime`.
            To assign the start date to.
            
        Note
        ----
        Setting `start` also sets `stop` to `None`.
        """
        utils.check(x, datetime, "start")
        x = x.replace(tzinfo=pytz.UTC)
        self._start = x
        self._stop = None
        self._build_index()

    @stop.setter
    def stop(self, x):
        """Set the stop date.

        Parameters
        ----------
        `x`: `datetime`.
            To assign the stop date to.
        """
        utils.check(x, datetime, "stop")
        x = x.replace(tzinfo=pytz.UTC)
        self._stop = x
        self._build_index()

    @interval.setter
    def interval(self, x):
        """Set the interval.

        Parameters
        ----------
        `x`: `timedelta`.
            To assign the interval to.
        """
        utils.check(x, timedelta, "interval")
        self._interval = x

    @lookback.setter
    def lookback(self, x):
        """Set the lookback.

        Parameters
        ----------
        `x`: `int`.
            To assign the lookback to.
        """
        self._lookback = x

    def step(self):
        """Step the time object forwards once.
        """
        if self._current_index + 1 > len(self._index):
            raise FrameworkError.from_runtime("Index stepped out of bounds")
        self._current_index += 1

    def reset(self):
        """Reset the time object back to the beginning of the index.
        """
        self._current_index = self._start_index + self._lookback

    def register(self, df):
        """Register a datetime and check its index against other registered
        dataframes.
        
        Parameters
        ----------
        `df`: `pd.DataFrame`.
            To be registered and have it's index checked.
        """
        # check the time series domains in the dataframe
        utils.process_df(
            df=df,
            interval=self._interval,
            start=self._start,
            stop=self._stop,
            raise_exception=True,
        )

    def done(self):
        """Determine wether there is any steps remaining in the index.
        
        Returns
        -------
        `bool`
        """
        return self._current_index == self._stop_index

    def _build_index(self):
        """Build index from start and stop dates.
        """
        # build datetime index
        if not self._start or not self._stop:
            return
        x = self.start
        index = []
        while x <= self.stop:
            index.append(x)
            x += self.interval
        self._index = DatetimeIndex(index)
        self._stop_index = len(self._index) - 1
