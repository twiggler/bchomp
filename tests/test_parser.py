"""Unit tests for core combinators in bchomp.parser."""

from __future__ import annotations

from unittest.mock import MagicMock

import bchomp.parser as p
from bchomp.reader import BytesReader


def _state(data: bytes) -> p.ParserState:
    return p.initial_state(BytesReader(data))


class TestMapP:
    """Tests for map_p."""

    def test_successful_parse_transforms_value_and_advances_position(self) -> None:
        """The mapping function is applied to the parsed value and position advances."""
        parser = p.map_p(p.uint8, lambda n: n * 2)
        result = parser(_state(b"\x03\xff"))
        assert isinstance(result, p.Success)
        assert result.value == 6
        assert result.state.pos == 1

    def test_propagates_failure(self) -> None:
        """A failure from the wrapped parser is passed through unchanged."""
        parser = p.map_p(p.bytes_n(4), lambda b: b)
        result = parser(_state(b"\x01"))
        assert isinstance(result, p.Failure)

    def test_identity_map(self) -> None:
        """Mapping with the identity function returns the same value."""
        parser = p.map_p(p.bytes_n(3), lambda b: b)
        result = parser(_state(b"abc"))
        assert isinstance(result, p.Success)
        assert result.value == b"abc"


class TestBindP:
    """Tests for bind_p."""

    def test_chains_parsers_and_advances_position(self) -> None:
        """The second parser receives the first result and both positions are consumed."""
        # Parse a length byte then that many bytes.
        parser = p.bind_p(p.uint8, p.bytes_n)
        result = parser(_state(b"\x03abc"))
        assert isinstance(result, p.Success)
        assert result.value == b"abc"
        assert result.state.pos == 4

    def test_failure_in_first_parser_propagates(self) -> None:
        """A failure in the first parser stops execution."""
        continuation = MagicMock(return_value=p.uint8)
        parser = p.bind_p(p.bytes_n(4), continuation)
        result = parser(_state(b"\x01"))
        assert isinstance(result, p.Failure)
        continuation.assert_not_called()

    def test_failure_in_second_parser_propagates(self) -> None:
        """A failure produced by the second parser is returned."""
        # Parse a length byte, then try to read more bytes than are available.
        parser = p.bind_p(p.uint8, p.bytes_n)
        result = parser(_state(b"\x05ab"))  # asks for 5 bytes but only 2 remain
        assert isinstance(result, p.Failure)

    def test_constant_second_parser(self) -> None:
        """bind_p can be used to discard the first result (like then_p)."""
        parser = p.bind_p(p.uint8, lambda _: p.uint8)
        result = parser(_state(b"\x01\x02"))
        assert isinstance(result, p.Success)
        assert result.value == 2
