from copy import copy, deepcopy
from decimal import Decimal

from framework import utils
from framework.core import Currencies


class Wallet:
    def __init__(self, *currencies):
        self.currencies = {}
        for currency in currencies:
            self._check(currency)
            self.currencies[type(currency)] += currency

    def __eq__(self, other):
        return set(self.currencies.values()) == set(
            other.currencies.values()
        )  # TODO: fixthis

    def __contains__(self, currency):
        return currency in set(self.currencies.keys())

    def __iter__(self):
        self._n = 0
        self.iter_values = [
            (key, value.amount) for key, value in self.currencies.items()
        ]
        return self

    def __next__(self):
        if self._n < len(self.iter_values):
            result = self.iter_values[self._n]
            self._n += 1
            return result
        else:
            raise StopIteration

    def __getitem__(self, currency):
        assert issubclass(currency, Currencies.Currency)
        return self.currencies[currency].amount

    def __repr__(self):
        c = ", ".join([str(x) for x in self.currencies.values()])
        return f"{self.__class__.__name__}({c})"

    def __iadd__(self, currency):
        """
        """
        utils.check(currency, Currencies.Currency, "currency")
        self._check(currency)
        self.currencies[type(currency)] += currency
        return self

    def __isub__(self, currency):
        """
        """
        utils.check(currency, Currencies.Currency, "currency")
        self._check(currency)
        self.currencies[type(currency)] -= currency
        if self.currencies[type(currency)].amount < 0:
            raise ValueError(
                f"{self.__class__.__name__} has {self.currencies[type(currency)]}"
            )
        return self

    def __len__(self):
        """
        """
        return len(self.currencies)

    def _check(self, currency):
        if type(currency) not in self.currencies:
            self.currencies[type(currency)] = type(currency)(amount=0)

    def copy(self):
        """Copying a wallet is inherently tricky. We specify a more explicit
        method here.
        """
        currencies = []
        for currency, value in self.currencies.items():
            currencies.append(currency(value.amount))
        new_wallet = Wallet(*currencies)
        return new_wallet
