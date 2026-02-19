"""Parser composition."""

from functools import reduce
from typing import TYPE_CHECKING, overload

if TYPE_CHECKING:
    from collections.abc import Callable

    from bchomp.parser import BlockingParser


# 1-arg
@overload
def compose[T1](p1: BlockingParser[T1], /) -> BlockingParser[T1]: ...


# 2-arg
@overload
def compose[P, T1](
    p: BlockingParser[P], p1: Callable[[BlockingParser[P]], BlockingParser[T1]], /
) -> BlockingParser[T1]: ...


# 3-arg
@overload
def compose[P, T1, T2](
    p: BlockingParser[P],
    p1: Callable[[BlockingParser[T1]], BlockingParser[T2]],
    p2: Callable[[BlockingParser[P]], BlockingParser[T1]],
    /,
) -> BlockingParser[T2]: ...


# 4-arg
@overload
def compose[P, T1, T2, T3](
    p: BlockingParser[P],
    p1: Callable[[BlockingParser[T2]], BlockingParser[T3]],
    p2: Callable[[BlockingParser[T1]], BlockingParser[T2]],
    p3: Callable[[BlockingParser[P]], BlockingParser[T1]],
    /,
) -> BlockingParser[T3]: ...


def compose(*parsers):
    """Compose a parser and transformers from right to left."""
    result, *rest = parsers
    return reduce(lambda acc, f: f(acc), reversed(rest), result)
