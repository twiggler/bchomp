"""Adapters for using dissect.cstruct types with bchomp parsers.

This module provides a single combinator `from_cstruct` which takes a
dissect.cstruct type (or a callable that reads from a file-like object)
and returns a `bchomp` parser that runs the cstruct reader on the
current parser position and advances the `bchomp` state by the number
of bytes consumed.
"""

from __future__ import annotations

import io
from dataclasses import replace
from typing import TYPE_CHECKING

import bchomp.parser as p

if TYPE_CHECKING:
    from collections.abc import Callable


class CStructStream(io.BufferedIOBase):
    """Bridge a `bchomp.Readable` to a file-like `BufferedIOBase`.

    This maps the `Readable.read(n, offset)` API to the standard
    file-like methods expected by cstruct readers.
    """

    def __init__(self, readable: p.Readable, start: int, total_len: int) -> None:
        self._readable = readable
        self._start = start
        self._total_len = total_len
        self._pos = 0

    def read(self, n: int | None = -1) -> bytes:
        """Read up to ``n`` bytes from the stream view.

        This forwards to ``self._readable.read(n, absolute_offset)`` where
        ``absolute_offset`` is ``self._start + self._pos``. Returns fewer
        bytes at EOF.
        """
        if n is None or n < 0:
            n = self._total_len - self._start - self._pos
        n = max(0, min(n, self._total_len - self._start - self._pos))
        if n == 0:
            return b""
        data = self._readable.read(n, self._start + self._pos)
        self._pos += len(data)
        return data

    def seek(self, offset: int, whence: int = io.SEEK_SET) -> int:
        """Move the stream position and return the new position.

        The resulting position is relative to the start of the bridge
        view and is clamped to the available range. Seek semantics are
        identical to a regular file (``SEEK_SET``, ``SEEK_CUR``,
        ``SEEK_END``), but note that actual reads are forwarded to the
        underlying `Readable` using absolute offsets computed from the
        view start.
        """
        if whence == io.SEEK_SET:
            new = offset
        elif whence == io.SEEK_CUR:
            new = self._pos + offset
        elif whence == io.SEEK_END:
            new = (self._total_len - self._start) + offset
        else:
            msg = f"invalid whence: {whence}"
            raise ValueError(msg)
        if new < 0:
            msg = "negative seek"
            raise ValueError(msg)
        self._pos = min(new, max(0, self._total_len - self._start))
        return self._pos

    def tell(self) -> int:
        """Return the current stream position relative to the view start."""
        return self._pos

    def readable(self) -> bool:
        """Return True: the stream supports reading."""
        return True

    def seekable(self) -> bool:
        """Return True: the stream supports seeking."""
        return True

    def writable(self) -> bool:
        """Return False: the stream is read-only."""
        return False


def from_cstruct[T](c_parser: Callable[[io.BufferedIOBase], T]) -> p.BlockingParser[T]:
    """Create a `bchomp` parser from a cstruct-style reader.

    The returned parser will present a small file-like wrapper backed
    by the current `ParserState.readable` at `state.pos`. The underlying
    readable is not mutated; after the cstruct reader finishes the
    parser advances the `bchomp` position by the number of bytes the
    cstruct reader consumed.
    """

    def _parser(state: p.ParserState) -> p.BlockingResult[T]:
        start = state.pos
        readable = state.readable
        total_len = len(state)

        fh = CStructStream(readable, start, total_len)

        try:
            value = c_parser(fh)
        except Exception as exc:  # noqa: BLE001
            return p.Failure(f"cstruct reader error: {exc}", state)

        consumed = fh.tell()  # Alternatively, use `value.size`? This seems more solid though.
        new_state = replace(state, pos=state.pos + consumed)

        return p.Success(value, new_state)

    return _parser
