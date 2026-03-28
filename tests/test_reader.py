"""Unit tests for bchomp.reader."""

from __future__ import annotations

import io

from bchomp.reader import BinaryIOReader, BytesReader, ChainReader, SubReader


class TestBytesReader:
    """Tests for BytesReader."""

    def test_len(self) -> None:
        """Length matches the underlying bytes object."""
        r = BytesReader(b"hello")
        assert len(r) == 5

    def test_read_all(self) -> None:
        """Reading from offset 0 returns all bytes."""
        r = BytesReader(b"hello")
        assert r.read(5, 0) == b"hello"

    def test_read_at_offset(self) -> None:
        """Reading at a non-zero offset returns the correct slice."""
        r = BytesReader(b"hello world")
        assert r.read(5, 6) == b"world"

    def test_read_partial_at_end(self) -> None:
        """Reading past the end returns only the remaining bytes."""
        r = BytesReader(b"hi")
        assert r.read(10, 0) == b"hi"

    def test_read_zero_bytes(self) -> None:
        """Reading 0 bytes always returns an empty bytes object."""
        r = BytesReader(b"hello")
        assert r.read(0, 2) == b""

    def test_read_empty_source(self) -> None:
        """Reading from an empty source returns empty bytes."""
        r = BytesReader(b"")
        assert r.read(1, 0) == b""


class TestBinaryIOReader:
    """Tests for BinaryIOReader."""

    def test_len(self) -> None:
        """Length matches the size of the underlying file-like object."""
        f = io.BytesIO(b"hello")
        r = BinaryIOReader(f)
        assert len(r) == 5

    def test_read_all(self) -> None:
        """Reading from offset 0 returns all bytes."""
        f = io.BytesIO(b"hello")
        r = BinaryIOReader(f)
        assert r.read(5, 0) == b"hello"

    def test_read_at_offset(self) -> None:
        """Reading at a non-zero offset returns the correct bytes."""
        f = io.BytesIO(b"hello world")
        r = BinaryIOReader(f)
        assert r.read(5, 6) == b"world"

    def test_read_partial_at_end(self) -> None:
        """Reading past the end returns only the remaining bytes."""
        f = io.BytesIO(b"hi")
        r = BinaryIOReader(f)
        assert r.read(10, 0) == b"hi"

    def test_constructor_leaves_position_at_start(self) -> None:
        """Constructor resets the file position to the beginning."""
        f = io.BytesIO(b"hello")
        f.seek(3)  # move to a non-zero position before constructing
        r = BinaryIOReader(f)
        assert len(r) == 5
        assert r.read(5, 0) == b"hello"


class TestSubReader:
    """Tests for SubReader."""

    def test_len(self) -> None:
        """Length reflects the declared sub-region size."""
        base = BytesReader(b"hello world")
        r = SubReader(base, 6, 5)
        assert len(r) == 5

    def test_read_full_region(self) -> None:
        """Reading the whole sub-region returns the correct bytes."""
        base = BytesReader(b"hello world")
        r = SubReader(base, 6, 5)
        assert r.read(5, 0) == b"world"

    def test_read_at_local_offset(self) -> None:
        """Local offsets are correctly translated to base offsets."""
        base = BytesReader(b"abcdef")
        r = SubReader(base, 2, 3)  # region is b"cde"
        assert r.read(2, 1) == b"de"

    def test_read_clamped_at_boundary(self) -> None:
        """A read that extends past the region boundary is clamped."""
        base = BytesReader(b"hello world")
        r = SubReader(base, 6, 5)
        assert r.read(10, 0) == b"world"

    def test_read_at_or_beyond_length_returns_empty(self) -> None:
        """A read starting at or past the region end returns empty bytes."""
        base = BytesReader(b"hello world")
        r = SubReader(base, 6, 5)
        assert r.read(1, 5) == b""
        assert r.read(1, 100) == b""


class TestChainReader:
    """Tests for ChainReader."""

    def test_len_single_chunk(self) -> None:
        """Length of a single-chunk chain equals the chunk size."""
        r = ChainReader([BytesReader(b"hello")])
        assert len(r) == 5

    def test_len_multiple_chunks(self) -> None:
        """Length is the sum of all chunk sizes."""
        r = ChainReader([BytesReader(b"hel"), BytesReader(b"lo")])
        assert len(r) == 5

    def test_len_empty(self) -> None:
        """Length of an empty chain is zero."""
        r = ChainReader([])
        assert len(r) == 0

    def test_read_single_chunk(self) -> None:
        """Reading from a single chunk works like a plain reader."""
        r = ChainReader([BytesReader(b"hello")])
        assert r.read(5, 0) == b"hello"

    def test_read_spanning_chunks(self) -> None:
        """A read that spans a chunk boundary concatenates both halves."""
        r = ChainReader([BytesReader(b"hel"), BytesReader(b"lo")])
        assert r.read(5, 0) == b"hello"

    def test_read_at_offset_within_second_chunk(self) -> None:
        """Reading with an offset that falls in the second chunk is correct."""
        r = ChainReader([BytesReader(b"abc"), BytesReader(b"def")])
        assert r.read(2, 4) == b"ef"

    def test_read_partial_at_end(self) -> None:
        """Reading past the total length returns only the remaining bytes."""
        r = ChainReader([BytesReader(b"hi")])
        assert r.read(10, 0) == b"hi"

    def test_read_zero_bytes(self) -> None:
        """Reading 0 bytes returns empty bytes without advancing."""
        r = ChainReader([BytesReader(b"hello")])
        assert r.read(0, 0) == b""

    def test_lazy_loading(self) -> None:
        """Chunks are only loaded as needed — requesting fewer bytes loads fewer chunks."""
        loaded: list[int] = []

        def make_reader(idx: int, data: bytes) -> BytesReader:
            loaded.append(idx)
            return BytesReader(data)

        # Build readers eagerly for this test, but track construction order
        readers = [make_reader(i, bytes([i] * 4)) for i in range(3)]
        loaded.clear()  # reset; we only want to track reads, not construction

        r = ChainReader(iter(readers))
        # Reading 4 bytes from offset 0 should only need the first chunk.
        r.read(4, 0)
        assert len(r._loaded) == 1  # noqa: SLF001

    def test_backward_seek_after_forward_read(self) -> None:
        """Already-loaded chunks support backward seeks without re-loading."""
        r = ChainReader([BytesReader(b"abc"), BytesReader(b"def")])
        assert r.read(3, 3) == b"def"
        assert r.read(3, 0) == b"abc"  # re-read the first chunk via loaded cache
