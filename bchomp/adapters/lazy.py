"""Lazy evaluation combinators and wrappers for bchomp parsing."""

from typing import TYPE_CHECKING, Any

import bchomp.parser as p

if TYPE_CHECKING:
    from collections.abc import Callable, Generator


class Lazy[T]:
    """A wrapper for a value that is computed lazily.

    The thunk is a function that takes no arguments and returns the value.
    """

    def __init__(self, thunk: Callable[[], T]) -> None:
        self._thunk = thunk
        self._value: T | None = None

    @property
    def value(self) -> T:
        """Compute and return the value, caching it for future access."""
        if self._value is None:
            self._value = self._thunk()
        return self._value

    def __repr__(self) -> str:
        if self._value is None:
            return "<Lazy (not yet computed)>"
        return f"<Lazy value={self._value!r}>"


@p.do
def lazy_unbounded[T](parser: p.BlockingParser[T]) -> Generator[p.BlockingParser, Any, Lazy[T]]:
    """Create a lazy-evaluated value from the current stream position.

    Captures the current parser state and returns a `Lazy` object immediately.
    The actual parsing is deferred until `.value` is accessed for the first time.
    This combinator does *not* advance the stream — the caller is responsible
    for any position management.
    """
    lazy_content_state: p.ParserState = yield p.get_state()

    def lazy_thunk() -> T:
        result = parser(lazy_content_state)
        if isinstance(result, p.Failure):
            msg = f"Failed to parse lazy content: {result.message}"
            raise ValueError(msg)  # noqa: TRY004
        return result.value

    return Lazy(lazy_thunk)
