"""Transformer factories for use with :func:`bchomp.compose.compose`."""

from functools import partial
from typing import TYPE_CHECKING

from . import parser as p

if TYPE_CHECKING:
    from collections.abc import Callable


# TODO: Fix loss of typing


def at(offset: int) -> Callable:
    """Return a combinator that runs a parser at ``offset``.

    Example: ``compose(at(100), p.bytes_n(4))``.
    """
    return partial(p.at, offset)


def take(n: int) -> Callable:
    """Return a combinator that limits a parser to the next ``n`` bytes.

    Example: ``compose(take(16), some_parser)``.
    """
    return partial(p.take, n)
