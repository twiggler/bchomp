"""A monadic parser combinator library for binary data."""

import dataclasses
import os
import struct
from collections.abc import Callable, Generator
from dataclasses import dataclass, replace
from enum import IntEnum
from functools import wraps
from typing import (
    Any,
    BinaryIO,
    Generic,
    Protocol,
    TypeVar,
    get_args,
)


class Readable(Protocol):
    """A protocol for a readable data source that allows random access.

    This decouples the parsers from a specific data backend like bytes or mmap.
    """

    def read(self, n: int, offset: int) -> bytes:
        """Read n bytes from the given offset."""
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
    """A Readable that presents a limited view (a slice) of another Readable."""

    def __init__(self, base_reader: Readable, base_offset: int, length: int) -> None:
        self._base_reader = base_reader
        self._base_offset = base_offset
        self._len = length
        self._end_offset = base_offset + length

    def read(self, n: int, offset: int) -> bytes:
        """Read from the sub-reader."""
        if offset < self._base_offset or offset >= self._end_offset:
            return b""  # Reading outside our slice returns nothing.

        # Clamp the number of bytes to not read past our end.
        n = min(n, self._end_offset - offset)

        # The read is forwarded to the base reader with the absolute offset.
        return self._base_reader.read(n, offset)

    def __len__(self) -> int:
        return self._len


@dataclass(frozen=True)
class ParserState:
    """The state of the parser."""

    readable: Readable
    """The data source to parse."""
    pos: int = 0
    """The current absolute position in the data source."""
    anchors: tuple[int, ...] = ()
    """A stack of positions for relative seeking."""

    def __len__(self) -> int:
        return len(self.readable)


# Make the parser/result type covariant (some checkers are conservative).
T_co = TypeVar("T_co", covariant=True)


@dataclass(frozen=True)
class Success(Generic[T_co]):  # noqa: UP046
    """Represents a successful parse."""

    value: T_co
    state: ParserState

    def __repr__(self) -> str:
        return f"Success(value={self.value}, pos={self.state.pos})"


@dataclass(frozen=True)
class Failure:
    """Represents a failed parse."""

    message: str
    state: ParserState

    def __repr__(self) -> str:
        return f"Failure(message='{self.message}', pos={self.state.pos})"


type Result[T_co] = Success[T_co] | Failure
type Parser[T_co] = Callable[[ParserState], Result[T_co]]


def run_parser[T](parser: Parser[T], data: Readable) -> Result[T]:
    """Run a parser on a readable data source."""
    # Initialize with a root anchor at position 0. This allows `seek` to
    # function as an absolute seek from the start of the file when used
    # at the top level.
    state = ParserState(data, anchors=(0,))
    return parser(state)


def bytes_n(n: int) -> Parser[bytes]:
    """Parse n bytes."""

    def _bytes_n(state: ParserState) -> Result[bytes]:
        read_bytes = state.readable.read(n, state.pos)
        if len(read_bytes) < n:
            return Failure(
                f"End of stream: expected {n} bytes, but only got {len(read_bytes)}",
                state,
            )
        return Success(read_bytes, replace(state, pos=state.pos + n))

    return _bytes_n


def map_p[T, R](fn: Callable[[T], R], p: Parser[T]) -> Parser[R]:
    """Map a function over the result of a parser."""

    def _map_p(state: ParserState) -> Result[R]:
        result = p(state)
        if isinstance(result, Failure):
            return result
        return Success(fn(result.value), result.state)

    return _map_p


any_byte = map_p(lambda b: b[0], bytes_n(1))


def seek(offset: int) -> Parser[None]:
    """Move the stream to a position relative to the current anchor.

    If no anchor is set, seeks relative to the start of the stream.
    This operation is safe, composable, and allows for backtracking.
    The check for validity is deferred to the next read operation.
    """

    def _seek_rel(state: ParserState) -> Result[None]:
        base_pos = state.anchors[-1]
        new_pos = base_pos + offset
        return Success(None, replace(state, pos=new_pos))

    return _seek_rel


def with_relocation[T](p: Parser[T]) -> Parser[T]:
    """Create a new local frame of reference for seeking.

    It runs a parser `p` within an "relocated" context. Any `seek` calls
    inside `p` will be relative to the stream position where the anchor
    was created. This allows for composable, relocatable parsers that can
    perform seeks without breaking backtracking.
    """

    def _relocatable(state: ParserState) -> Result[T]:
        # Create a new stream with the current position added to the anchor stack.
        anchored_state = replace(state, anchors=(*state.anchors, state.pos))

        # Run the wrapped parser in this new anchored context.
        result = p(anchored_state)

        if isinstance(result, Failure):
            # On failure, we return the failure but with the *original* stream,
            # preserving the backtracking contract.
            return Failure(result.message, state)

        # On success, calculate bytes consumed inside the anchor and advance
        # the outer stream by that amount, discarding the anchor.
        bytes_consumed = result.state.pos - state.pos
        final_state = replace(state, pos=state.pos + bytes_consumed)
        return Success(result.value, final_state)

    return _relocatable


def relocatable[T](parser_producer: Callable[..., Parser[T]]) -> Callable[..., Parser[T]]:
    """Make a parser-producing function create a relocatable parser.

    This automatically wraps the returned parser in `with_relocation`.
    """

    @wraps(parser_producer)
    def wrapper(*args: Any, **kwargs: Any) -> Parser[T]:  # noqa: ANN401
        parser = parser_producer(*args, **kwargs)
        return with_relocation(parser)

    return wrapper


def satisfy(predicate: Callable[[int], bool]) -> Parser[int]:
    """Parse a single byte and check if it satisfies a predicate."""

    def _satisfy(state: ParserState) -> Result[int]:
        result = any_byte(state)
        if isinstance(result, Failure):
            return result
        if predicate(result.value):
            return result
        return Failure(f"Predicate failed for {result.value}", state)

    return _satisfy


def byte(b: int) -> Parser[int]:
    """Parse a specific byte."""
    return satisfy(lambda x: x == b)


def sequence(*parsers: Parser) -> Parser[tuple]:
    """Run a sequence of parsers and return their results as a tuple.

    Note: This uses a simplified type signature because of a limitation in
    type checkers like Pyright when handling `TypeVarTuple` with generic
    classes like `Parser`. The ideal signature would be:
    `def sequence[*Ts](*parsers: Parser[*Ts]) -> Parser[tuple[*Ts]]`
    but this is not supported.

    See: https://github.com/microsoft/pyright/issues/3187
    """

    def _sequence(state: ParserState) -> Result[tuple]:
        results = []
        current_state = state
        for p in parsers:
            result = p(current_state)
            if isinstance(result, Failure):
                return result
            results.append(result.value)
            current_state = result.state
        return Success(tuple(results), current_state)

    return _sequence


def many[T](p: Parser[T]) -> Parser[list[T]]:
    """Run a parser zero or more times and return the results as a list."""

    def _many(state: ParserState) -> Result[list[T]]:
        results = []
        current_state = state
        while True:
            result = p(current_state)
            if isinstance(result, Failure):
                break
            results.append(result.value)
            current_state = result.state
        return Success(results, current_state)

    return _many


def count[T](n: int, parser: Parser[T]) -> Parser[list[T]]:
    """Run a parser n times and return the results as a list."""
    return map_p(list, sequence(*([parser] * n)))  # type: ignore[return-value]


def bind_p[T, R](p: Parser[T], f: Callable[[T], Parser[R]]) -> Parser[R]:
    """Apply a function to the result of a parser (monadic bind).

    Runs parser `p`, and if it succeeds, passes its result to function `f`.
    `f` must return a new parser, which is then run on the stream.
    """

    def _bind_p(state: ParserState) -> Result[R]:
        initial_result = p(state)
        if isinstance(initial_result, Failure):
            return initial_result

        new_parser = f(initial_result.value)
        return new_parser(initial_result.state)

    return _bind_p


def string(s: str) -> Parser[str]:
    """Parse a specific string."""
    return map_p(lambda x: x.decode("utf-8"), bytes_n(len(s.encode("utf-8"))))


def position() -> Parser[int]:
    """Return the current position in the stream.

    This is useful for calculations that depend on the
    size of a parsed block.
    """

    def _position(state: ParserState) -> Result[int]:
        return Success(state.pos, state)

    return _position


def get_state() -> Parser[ParserState]:
    """Return the current stream object.

    This is useful within a `do` block to get access to the stream.
    """

    def _get_stream(state: ParserState) -> Result[ParserState]:
        return Success(state, state)

    return _get_stream


def failure(message: str) -> Parser[Any]:
    """Create a parser that always fails with the given message.

    This is useful for reporting custom error messages within a `do` block
    or other complex parsers.
    """

    def _failure(state: ParserState) -> Failure:
        return Failure(message, state)

    return _failure


def pure[T](value: T) -> Parser[T]:
    """Return a parser that succeeds with `value` without consuming input."""

    def _pure(state: ParserState) -> Result[T]:
        return Success(value, state)

    return _pure


def peek[T](p: Parser[T]) -> Parser[T]:
    """Run a parser `p` without consuming any input.

    It returns the result of `p`, but the stream position is reset to
    where it was before `p` was run. This is useful for looking ahead
    in the stream to decide which parser to use next.
    """

    def _peek(state: ParserState) -> Result[T]:
        result = p(state)
        if isinstance(result, Failure):
            return result
        # On success, return the value but with the original, unmodified stream.
        return Success(result.value, state)

    return _peek


def take[T](n: int, p: Parser[T]) -> Parser[T]:
    """Create a parser that runs another parser within a limited byte context.

    This combinator is implemented by creating a sub-parser that is first
    anchored, then reads `n` bytes, and runs `p` on that isolated block.
    This demonstrates how more complex combinators can be built from simpler ones.
    """

    @do
    def _take_impl() -> Generator[Parser, Any, T]:
        # Get the current stream to calculate the sub-reader's offset and data source.
        current_state = yield get_state()

        # Create a SubReader for the next `n` bytes.
        sub_reader = SubReader(
            base_reader=current_state.readable,
            base_offset=current_state.pos,
            length=n,
        )
        # The new state uses the SubReader, but `pos` is still absolute.
        scoped_state = replace(current_state, readable=sub_reader)

        # Run the provided parser `p` on the scoped state.
        # The SubReader will enforce the boundary.
        result = p(scoped_state)

        if isinstance(result, Failure):
            yield failure(f"Parser failed within take({n}): {result.message}")
            return  # type: ignore[return-value]

        # If the inner parser succeeds, we advance the outer stream's position by `n` bytes.
        yield skip(n)

        return result.value

    return _take_impl()


def create_parser_from_dataclass(dc: type) -> Parser:
    """Automatically create a parser for a dataclass from Annotated metadata.

    It inspects the dataclass's type hints. For fields annotated as
    Annotated[<type>, <parser>], it extracts the parser, sequences them,
    and maps the result to the dataclass constructor.
    """
    field_parsers = []
    for f in dataclasses.fields(dc):
        # get_args returns the arguments of a generic type.
        # For Annotated[int, p.uint8], it returns (int, p.uint8)
        args = get_args(f.type)
        if args and isinstance(args[1], Callable):
            # The parser is the second argument in our Annotated type
            field_parsers.append(args[1])

    return map_p(lambda results: dc(*results), sequence(*field_parsers))


def do[R](fn: Callable[..., Generator[Parser, Any, R]]) -> Callable[..., Parser[R]]:
    """Enable monadic comprehensions (do-notation) for parsers.

    This allows writing complex, sequential parsers in a
    clean, imperative style, avoiding nested `bind_p` calls.

    The decorated function must be a generator that yields parsers. The
    result of each yielded parser is sent back into the generator. The
    generator's final return value becomes the success value of the
    overall parser.
    """

    @wraps(fn)
    def wrapper(*args: Any, **kwargs: Any) -> Parser[R]:  # noqa: ANN401
        def _do(state: ParserState) -> Result[R]:
            gen = fn(*args, **kwargs)

            def step(
                g: Generator[Parser, Any, R],
                state: ParserState,
                value: Any | None = None,  # noqa: ANN401
            ) -> Result[R]:
                try:
                    p = g.send(value)
                    result = p(state)
                    if isinstance(result, Failure):
                        return result
                    return step(g, result.state, result.value)
                except StopIteration as e:
                    # The generator has finished, its return value is the final result.
                    return Success(e.value, state)

            return step(gen, state)

        return _do

    return wrapper


@do
def with_bytes_read[T](parser: Parser[T]) -> Generator[Parser, Any, tuple[T, int]]:
    """Return the result of a parser and the number of bytes it consumed."""
    start_pos = yield position()
    result = yield parser
    end_pos = yield position()
    bytes_read = end_pos - start_pos
    return result, bytes_read


def skip(n: int) -> Parser[None]:
    """Skip n bytes, failing if there are not enough bytes."""

    def _parser(state: ParserState) -> Result[None]:
        if n < 0 or n > len(state):
            return Failure(message="Not enough bytes to skip", state=state)

        new_pos = state.pos + n
        return Success(value=None, state=replace(state, pos=new_pos))

    return _parser


def uint_be(n: int) -> Parser[int]:
    """Parse a big-endian unsigned integer of `n` bytes."""
    return map_p(lambda b: int.from_bytes(b, "big", signed=False), bytes_n(n))


uint8 = uint_be(1)
uint16_be = uint_be(2)
uint32_be = uint_be(4)


def int_be(n: int) -> Parser[int]:
    """Parse a big-endian signed integer of `n` bytes."""
    return map_p(lambda b: int.from_bytes(b, "big", signed=True), bytes_n(n))


int8 = int_be(1)
int16_be = int_be(2)
int24_be = int_be(3)
int32_be = int_be(4)
int48_be = int_be(6)
int64_be = int_be(8)


def float_be(n: int) -> Parser[float]:
    """Parse a big-endian float of `n` bytes."""
    return map_p(lambda b: struct.unpack(">d", b)[0], bytes_n(n))


@do
def enum[E: IntEnum](p_int: Parser[int], enum_type: type[E]) -> Generator[Parser, Any, E]:
    """Parse an int and convert it to `enum_type`.

    On invalid values, yield a failing parser so the overall parser
    returns a `Failure` (with the original state) and can backtrack.
    """
    val = yield p_int
    try:
        return enum_type(val)
    except ValueError:
        yield failure(f"Invalid {enum_type.__name__} value: {val}")
        return  # type: ignore[return-value]
