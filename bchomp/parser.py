"""A monadic parser combinator library for binary data."""

from __future__ import annotations

import dataclasses
import os
import struct
from collections.abc import Callable, Generator, Iterator
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
    overload,
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
U_co = TypeVar("U_co", covariant=True)
Y_co = TypeVar("Y_co", covariant=True)


@dataclass(frozen=True)
class Success(Generic[T_co]):
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


@dataclass(frozen=True)
class Suspend(Generic[T_co, Y_co]):
    """Represents a suspended parser for streaming."""

    value: Y_co
    resume: Parser[T_co, Y_co]
    state: ParserState

    def __repr__(self) -> str:
        return f"Suspend(state={self.state.pos})"


type Result[T_co, Y_co] = Success[T_co] | Failure | Suspend[T_co, Y_co]
type BlockingResult[T_co] = Success[T_co] | Failure
type Parser[T_co, Y_co] = Callable[[ParserState], Result[T_co, Y_co]]
type BlockingParser[T_co] = Callable[[ParserState], BlockingResult[T_co]]
type StreamingParser[Y_co] = Callable[[ParserState], Result[None, Y_co]]


def run_parser[T_co](parser: BlockingParser[T_co], data: Readable) -> BlockingResult[T_co]:
    """Run a parser on a readable data source."""
    # Initialize with a root anchor at position 0. This allows `seek` to
    # function as an absolute seek from the start of the file when used
    # at the top level.
    state = ParserState(data, anchors=(0,))
    return parser(state)


def stream(parser: StreamingParser[Y_co], data: Readable) -> Iterator[Y_co]:
    """Run a streaming parser on a readable data source, yielding emitted values."""
    state = ParserState(data, anchors=(0,))
    result = parser(state)
    while isinstance(result, Suspend):
        yield result.value
        result = result.resume(result.state)
    if isinstance(result, Failure):
        msg = f"Parsing failed: {result.message} at pos {result.state.pos}"
        raise ValueError(msg)  # noqa: TRY004


def bytes_n(n: int) -> BlockingParser[bytes]:
    """Parse n bytes."""

    def _bytes_n(state: ParserState) -> BlockingResult[bytes]:
        read_bytes = state.readable.read(n, state.pos)
        if len(read_bytes) < n:
            return Failure(
                f"End of stream: expected {n} bytes, but only got {len(read_bytes)}",
                state,
            )
        return Success(read_bytes, replace(state, pos=state.pos + n))

    return _bytes_n


def chain(
    result: Result[T_co, Y_co], on_success: Callable[[Success[T_co]], Result[U_co, Y_co]]
) -> Result[U_co, Y_co]:
    """Handle the result of a parser and determine the next step.

    This function acts as a combinator for parser results, enabling chaining and control flow:
    - If the result is a Suspend, it wraps the resume function recursively, allowing streaming
      parsers to emit values and continue parsing.
    - If the result is a Success, it invokes the provided on_success callback with the successful
      value, enabling further parsing steps.
    - If the result is a Failure, it propagates the failure unchanged, allowing backtracking
      or error handling.
    """
    # TODO: Optimize using trampoline because Python has no tail-call elimination.

    # On Suspend, we wrap it (Recursive step)
    if isinstance(result, Suspend):

        def resume_wrapper(s: ParserState) -> Result[U_co, Y_co]:
            return chain(result.resume(s), on_success)

        return Suspend(result.value, resume_wrapper, result.state)

    # On Success, we run the next step
    if isinstance(result, Success):
        return on_success(result)

    # 3. Failure passes through
    return result


def bind_p[T, Y, R](p: Parser[T, Y], f: Callable[[T], Parser[R, Y]]) -> Parser[R, Y]:
    """Apply a function to the result of a parser (monadic bind).

    Runs parser `p`, and if it succeeds, passes its result to function `f`.
    `f` must return a new parser, which is then run on the stream.
    """

    def _bind_p(state: ParserState) -> Result[R, Y]:
        initial_result = p(state)
        return chain(initial_result, lambda success: f(success.value)(success.state))

    return _bind_p


def map_p(fn: Callable[[T_co], U_co], p: BlockingParser[T_co]) -> BlockingParser[U_co]:
    """Map a function over the result of a parser."""

    def _map_p(state: ParserState) -> BlockingResult[U_co]:
        result = p(state)
        if isinstance(result, Failure):
            return result
        return Success(fn(result.value), result.state)

    return _map_p


any_byte = map_p(lambda b: b[0], bytes_n(1))


def seek(offset: int) -> BlockingParser[None]:
    """Move the stream to a position relative to the current anchor.

    If no anchor is set, seeks relative to the start of the stream.
    This operation is safe, composable, and allows for backtracking.
    The check for validity is deferred to the next read operation.
    """

    def _seek_rel(state: ParserState) -> BlockingResult[None]:
        base_pos = state.anchors[-1]
        new_pos = base_pos + offset
        return Success(None, replace(state, pos=new_pos))

    return _seek_rel


def with_relocation(p: Parser[T_co, Y_co]) -> Parser[T_co, Y_co]:
    """Create a new local frame of reference for seeking.

    It runs a parser `p` within an "relocated" context. Any `seek` calls
    inside `p` will be relative to the stream position where the anchor
    was created. This allows for composable, relocatable parsers that can
    perform seeks without breaking backtracking.
    """

    def _relocatable(state: ParserState) -> Result[T_co, Y_co]:
        # Create a new stream with the current position added to the anchor stack.
        anchored_state = replace(state, anchors=(*state.anchors, state.pos))

        # Run the wrapped parser in this new anchored context.
        result = p(anchored_state)

        # On success, calculate bytes consumed inside the anchor and advance
        # the outer stream by that amount, discarding the anchor.
        def on_success(result: Success[T_co]) -> Result[T_co, Y_co]:
            bytes_consumed = result.state.pos - state.pos
            new_state = replace(state, pos=state.pos + bytes_consumed)
            return Success(result.value, new_state)

        return chain(result, on_success)

    return _relocatable


def relocatable(
    parser_producer: Callable[..., Parser[T_co, Y_co]],
) -> Callable[..., Parser[T_co, Y_co]]:
    """Make a parser-producing function create a relocatable parser.

    This automatically wraps the returned parser in `with_relocation`.
    """

    @wraps(parser_producer)
    def wrapper(*args: Any, **kwargs: Any) -> Parser[T_co, Y_co]:  # noqa: ANN401
        parser = parser_producer(*args, **kwargs)
        return with_relocation(parser)

    return wrapper


def satisfy(predicate: Callable[[int], bool]) -> BlockingParser[int]:
    """Parse a single byte and check if it satisfies a predicate."""

    def _satisfy(state: ParserState) -> BlockingResult[int]:
        result = any_byte(state)
        if isinstance(result, Failure):
            return result
        if predicate(result.value):
            return result
        return Failure(f"Predicate failed for {result.value}", state)

    return _satisfy


def byte(b: int) -> BlockingParser[int]:
    """Parse a specific byte."""
    return satisfy(lambda x: x == b)


def sequence(*parsers: BlockingParser) -> BlockingParser[tuple]:
    """Run a sequence of parsers and return their results as a tuple.

    Note: This uses a simplified type signature because of a limitation in
    type checkers like Pyright when handling `TypeVarTuple` with generic
    classes like `Parser`. The ideal signature would be:
    `def sequence[*Ts](*parsers: Parser[*Ts]) -> Parser[tuple[*Ts]]`
    but this is not supported.

    See: https://github.com/microsoft/pyright/issues/3187
    """

    def _sequence(state: ParserState) -> BlockingResult[tuple]:
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


def many[T](p: BlockingParser[T]) -> BlockingParser[list[T]]:
    """Run a parser zero or more times and return the results as a list."""

    def _many(state: ParserState) -> BlockingResult[list[T]]:
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


def count[T](n: int, parser: BlockingParser[T]) -> BlockingParser[list[T]]:
    """Run a parser n times and return the results as a list."""
    return map_p(list, sequence(*([parser] * n)))


def string(s: str) -> BlockingParser[str]:
    """Parse a specific string."""
    return map_p(lambda x: x.decode("utf-8"), bytes_n(len(s.encode("utf-8"))))


def get_state() -> BlockingParser[ParserState]:
    """Return the current stream object.

    This is useful within a `do` block to get access to the stream.
    """

    def _get_state(state: ParserState) -> BlockingResult[ParserState]:
        return Success(state, state)

    return _get_state


def position() -> BlockingParser[int]:
    """Return the current position in the stream.

    This is useful for calculations that depend on the size of a parsed block.
    """
    return map_p(lambda s: s.pos, get_state())


def failure(message: str) -> BlockingParser[T_co]:
    """Create a parser that always fails with the given message.

    This is useful for reporting custom error messages within a `do` block
    or other complex parsers.
    """

    def _failure(state: ParserState) -> Failure:
        return Failure(message, state)

    return _failure


def pure[T](value: T) -> BlockingParser[T]:
    """Return a parser that succeeds with `value` without consuming input."""

    def _pure(state: ParserState) -> BlockingResult[T]:
        return Success(value, state)

    return _pure


def peek[T](p: BlockingParser[T]) -> BlockingParser[T]:
    """Run a parser `p` without consuming any input.

    It returns the result of `p`, but the stream position is reset to
    where it was before `p` was run. This is useful for looking ahead
    in the stream to decide which parser to use next.
    """

    def _peek(state: ParserState) -> BlockingResult[T]:
        result = p(state)
        if isinstance(result, Failure):
            return result
        # On success, return the value but with the original, unmodified stream.
        return Success(result.value, state)

    return _peek


@overload
def take(n: int, p: BlockingParser[T_co]) -> BlockingParser[T_co]: ...


@overload
def take(n: int, p: Parser[T_co, Y_co]) -> Parser[T_co, Y_co]: ...


def take(n, p) -> Parser[T_co, Y_co]:
    """Create a parser that runs another parser within a limited byte context.

    This combinator is implemented by creating a sub-parser that is first
    anchored, then reads `n` bytes, and runs `p` on that isolated block.
    This demonstrates how more complex combinators can be built from simpler ones.
    """

    def _take_impl(current_state: ParserState) -> Result[T_co, Y_co]:
        # Get the current stream to calculate the sub-reader's offset and data source.
        # Create a SubReader for the next `n` bytes.
        sub_reader = SubReader(
            base_reader=current_state.readable,
            base_offset=current_state.pos,
            length=n,
        )

        # The new state uses the SubReader, but `pos` is still absolute.
        scoped_state = replace(current_state, readable=sub_reader)

        return chain(
            p(scoped_state),
            lambda success: Success(
                success.value, replace(current_state, pos=current_state.pos + n)
            ),
        )

    return _take_impl


def create_parser_from_dataclass(dc: type) -> BlockingParser:
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


@overload
def do(
    fn: Callable[..., Generator[BlockingParser, Any, T_co]],
) -> Callable[..., BlockingParser[T_co]]: ...


@overload
def do(fn: Callable[..., Generator[Parser, Any]]) -> Callable[..., Parser]: ...


def do(fn) -> Callable[..., Parser]:
    """Enable monadic comprehensions (do-notation) for parsers.

    This allows writing complex, sequential parsers in a
    clean, imperative style, avoiding nested `bind_p` calls.

    The decorated function must be a generator that yields parsers. The
    result of each yielded parser is sent back into the generator. The
    generator's final return value becomes the success value of the
    overall parser.
    """

    @wraps(fn)
    def wrapper(*args: Any, **kwargs: Any) -> Parser[T_co, Y_co]:  # noqa: ANN401
        def _do(state: ParserState) -> Result[T_co, Y_co]:
            gen = fn(*args, **kwargs)

            def step(
                g: Generator[Parser, Any, T_co],
                state: ParserState,
                value: Any | None = None,  # noqa: ANN401
            ) -> Result[T_co, Y_co]:
                try:
                    p = g.send(value)
                    result = p(state)
                    return chain(result, lambda r: step(g, r.state, r.value))
                except StopIteration as e:
                    # The generator has finished, its return value is the final result.
                    return Success(e.value, state)

            return step(gen, state)

        return _do

    return wrapper


def emit[Y_co](value: Y_co) -> StreamingParser[Y_co]:
    """Create a streaming parser that emits a value without consuming input.

    This is useful for producing intermediate results in a streaming context.
    """

    def _emit(state: ParserState) -> Result[None, Y_co]:
        return Suspend(value=value, resume=lambda s: Success(None, s), state=state)

    return _emit


@do
def with_bytes_read[T](parser: BlockingParser[T]) -> Generator[BlockingParser, Any, tuple[T, int]]:
    """Return the result of a parser and the number of bytes it consumed."""
    start_pos = yield position()
    result = yield parser
    end_pos = yield position()
    bytes_read = end_pos - start_pos
    return result, bytes_read


def uint_be(n: int) -> BlockingParser[int]:
    """Parse a big-endian unsigned integer of `n` bytes."""
    return map_p(lambda b: int.from_bytes(b, "big", signed=False), bytes_n(n))


uint8 = uint_be(1)
uint16_be = uint_be(2)
uint32_be = uint_be(4)


def int_be(n: int) -> BlockingParser[int]:
    """Parse a big-endian signed integer of `n` bytes."""
    return map_p(lambda b: int.from_bytes(b, "big", signed=True), bytes_n(n))


int8 = int_be(1)
int16_be = int_be(2)
int24_be = int_be(3)
int32_be = int_be(4)
int48_be = int_be(6)
int64_be = int_be(8)


def float_be(n: int) -> BlockingParser[float]:
    """Parse a big-endian float of `n` bytes."""
    return map_p(lambda b: struct.unpack(">d", b)[0], bytes_n(n))


@do
def enum[E: IntEnum](
    p_int: BlockingParser[int], enum_type: type[E]
) -> Generator[BlockingParser, Any, E]:
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
