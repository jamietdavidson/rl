import pytest

from framework.core import Currencies, Wallet
from framework.core.errors import FrameworkError


def test_initialization():
    wallet = Wallet()


def test_add():
    pass


def test_subtract():
    pass


def test_subtract_into_negative_fails():
    wallet = Wallet(Currencies.BTC(1.0), Currencies.ETH(1.0))
    with pytest.raises(ValueError):
        wallet -= Currencies.BTC(2)


def test_access_wallet_by_currency():
    wallet = Wallet(Currencies.BTC(1.0), Currencies.ETH(1.0))
    assert wallet[Currencies.BTC] == Currencies.BTC(1.0).amount


def test_wallet_copy():
    wallet = Wallet(Currencies.BTC(1.0), Currencies.ETH(1.0))
    copied_wallet = wallet.copy()
    assert wallet is not copied_wallet
    assert wallet == copied_wallet


def test_initialization():
    currency = Currencies.BTC(1)


def test_add():
    assert Currencies.BTC(1) + Currencies.BTC(2) == Currencies.BTC(3)


def test_subtract():
    assert Currencies.BTC(3) - Currencies.BTC(2) == Currencies.BTC(1)


def test_mismatched_operation_fails():
    with pytest.raises(FrameworkError):
        Currencies.BTC(1) + Currencies.ETH(2)
