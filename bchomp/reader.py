"""Readable protocol and concrete reader implementations for bchomp.

The central design decision in this module is the ``Readable`` protocol's
signature::

    def read(self, n: int, offset: int) -> bytes: ...

Offset and length are passed together in a single call, making each read
stateless — there is no mutable cursor to manage.  The alternative, a
separate ``seek`` / ``read`` pair, creates implicit shared state that every
caller must coordinate correctly.  Forgetting to seek before reading, or two
 logically independent reads accidentally overwriting each other's position,
are representative bugs that this interface makes structurally impossible.

The parser's own notion of "current position" (``ParserState.pos``) is kept
separate and explicit, so backtracking and sub-region scoping (``SubReader``,
``take``, ``with_relocation``) can be implemented purely in terms of values
rather than side-effects on shared state.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from typing import TYPE_CHECKING, BinaryIO, Protocol

if TYPE_CHECKING:
    from collections.abc import Iterable


class Readable(Protocol):
    """A stateless, randomly-accessible data source.

    Each call to ``read`` is self-contained: the caller supplies both the
    offset and the byte count, so the protocol carries no mutable cursor.
    Implementations are therefore free to be used concurrently or
    re-entrantly without any additional coordination.

    This decouples the parsers from a specific data backend (bytes, mmap,
    file objects, …).  Concrete adapters are provided in this module.
    """

    def read(self, n: int, offset: int) -> bytes:
        """Return up to *n* bytes starting at *offset*.

        Fewer than *n* bytes may be returned when *offset* + *n* exceeds the
        length of the source; an empty ``bytes`` object is returned when
        *offset* is at or beyond the end.
        """
        ...

    def __len__(self) -> int: ...


class BytesReader:
    """An implementation of the Readable protocol for an in-memory bytes object."""

    def __init__(self, data: bytes) -> None:
        self._data = data
        self._len = len(data)

    def read(self, n: int, offset: int) -> bytes:
        """Read n bytes from the given offset."""
        return self._data[offset : offset + n]

    def __len__(self) -> int:
        return self._len


class BinaryIOReader:
    """An implementation of the Readable protocol for a file-like object."""

    def __init__(self, f: BinaryIO) -> None:
        self._f = f
        # Get the total size by seeking to the end and back.
        self._f.seek(0, os.SEEK_END)
        self._len = self._f.tell()
        self._f.seek(0)

    def read(self, n: int, offset: int) -> bytes:
        """Read n bytes from the given offset."""
        self._f.seek(offset)
        return self._f.read(n)

    def __len__(self) -> int:
        return self._len


class SubReader:
    """A Readable that presents a rebased, bounded view of another Readable.

    Offsets passed to ``read`` are local (0 = first byte of the sub-region).
    """

    def __init__(self, base_reader: Readable, base_offset: int, length: int) -> None:
        self._base_reader = base_reader
        self._base_offset = base_offset
        self._len = length

    def read(self, n: int, offset: int) -> bytes:
        """Read n bytes at local offset (0 = start of the sub-region)."""
        if offset >= self._len:
            return b""
        n = min(n, self._len - offset)
        return self._base_reader.read(n, self._base_offset + offset)

    def __len__(self) -> int:
        return self._len


@dataclass(frozen=True)
class _Chunk:
    reader: Readable
    start: int  # absolute start offset of this chunk within the ChainReader

    @property
    def end_offset(self) -> int:
        return self.start + len(self.reader)


class ChainReader:
    """A Readable that lazily chains multiple readers into a single contiguous view.

    Chunks are pulled from the supplied iterator on demand as reads advance
    into them, so the entire sequence never needs to be materialised up front.
    Once pulled, a chunk is retained in memory to support arbitrary backward seeks.
    """

    def __init__(self, readers: Iterable[Readable]) -> None:
        self._iter = iter(readers)
        self._loaded: list[_Chunk] = []
        self._total_len: int | None = None

    def _load_up_to(self, offset: int | None = None) -> None:
        """Pull chunks until *offset* is covered; pass None to exhaust the iterator."""
        if self._total_len is not None:
            return
        if offset is not None and self._loaded and self._loaded[-1].end_offset > offset:
            return
        for chunk in self._iter:
            current_end = self._loaded[-1].end_offset if self._loaded else 0
            self._loaded.append(_Chunk(chunk, current_end))
            if offset is not None and self._loaded[-1].end_offset > offset:
                return
        self._total_len = self._loaded[-1].end_offset if self._loaded else 0

    def _find_chunk_index(self, offset: int) -> int:
        """Binary-search for the index of the chunk containing *offset*.

        Returns ``len(self._loaded)`` when *offset* is not covered by any loaded chunk.
        """
        if not self._loaded:
            return 0
        lo, hi = 0, len(self._loaded) - 1
        while lo <= hi:
            mid = (lo + hi) // 2
            chunk = self._loaded[mid]
            if offset < chunk.start:
                hi = mid - 1
            elif offset >= chunk.end_offset:
                lo = mid + 1
            else:
                return mid
        return len(self._loaded)

    def read(self, n: int, offset: int) -> bytes:
        """Read *n* bytes starting at *offset*, spanning chunk boundaries as needed."""
        if n <= 0:
            return b""
        self._load_up_to(offset + n - 1)

        result = bytearray()
        remaining = n
        pos = offset
        i = self._find_chunk_index(pos)
        while remaining > 0 and i < len(self._loaded):
            chunk = self._loaded[i]
            local_offset = pos - chunk.start
            to_read = min(remaining, chunk.end_offset - pos)
            part = chunk.reader.read(to_read, local_offset)
            result.extend(part)
            remaining -= len(part)
            pos += len(part)
            if len(part) < to_read:
                break  # chunk returned fewer bytes than requested; stop early
            i += 1

        return bytes(result)

    def __len__(self) -> int:
        self._load_up_to()
        return self._total_len  # type: ignore[return-value]
