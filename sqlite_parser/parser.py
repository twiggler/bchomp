from typing import Any, Callable, Generator, Generic, Self, TypeVar, TypeVarTuple, Union, Optional, Protocol, IO, get_args, TYPE_CHECKING
from functools import wraps
import struct
import os
import dataclasses
from dataclasses import dataclass, field, replace

T = TypeVar("T")
R = TypeVar("R")

class Lazy(Generic[T]):
    """
    A wrapper for a value that is computed lazily.
    The thunk is a function that takes no arguments and returns the value.
    """
    def __init__(self, thunk: Callable[[], T]):
        self._thunk = thunk
        self._value: Optional[T] = None

    @property
    def value(self) -> T:
        if self._value is None:
            self._value = self._thunk()
        return self._value

    def __repr__(self) -> str:
        if self._value is None:
            return "<Lazy (not yet computed)>"
        return f"<Lazy value={self._value!r}>"


class Readable(Protocol):
    """
    A protocol for a readable data source that allows random access.
    This decouples the parsers from a specific data backend like bytes or mmap.
    """
    def read(self, n: int, offset: int) -> bytes:
        ...

    def __len__(self) -> int:
        ...


class BytesReader:
    """An implementation of the Readable protocol for an in-memory bytes object."""
    def __init__(self, data: bytes):
        self._data = data
        self._len = len(data)

    def read(self, n: int, offset: int) -> bytes:
        return self._data[offset:offset + n]

    def __len__(self) -> int:
        return self._len


class BinaryIOReader:
    """
    An implementation of the Readable protocol for a file-like object
    opened in binary mode (IO[bytes]).
    """
    def __init__(self, f: IO[bytes]):
        self._f = f
        # Get the total size by seeking to the end and back.
        self._f.seek(0, os.SEEK_END)
        self._len = self._f.tell()
        self._f.seek(0)

    def read(self, n: int, offset: int) -> bytes:
        # This is where the side-effect of seeking happens, encapsulated
        # away from the pure parsers.
        self._f.seek(offset)
        return self._f.read(n)

    def __len__(self) -> int:
        return self._len

class SubReader:
    """
    A Readable that presents a limited view (a slice) of another Readable.
    """
    def __init__(self, base_reader: Readable, base_offset: int, length: int):
        self._base_reader = base_reader
        self._base_offset = base_offset
        self._len = length
        self._end_offset = base_offset + length

    def read(self, n: int, offset: int) -> bytes:
        if offset < self._base_offset or offset >= self._end_offset:
            return b'' # Reading outside our slice returns nothing.

        # Clamp the number of bytes to not read past our end.
        n = min(n, self._end_offset - offset)
        
        # The read is forwarded to the base reader with the absolute offset.
        return self._base_reader.read(n, offset)

    def __len__(self) -> int:
        # The length of a subreader is the length of its slice.
        return self._len


@dataclass(frozen=True)
class ParserState:
    readable: Readable
    pos: int = 0
    anchors: tuple[int, ...] = ()

    def __len__(self):
        return len(self.readable)


class Success(Generic[T]):
    def __init__(self, value: T, state: ParserState):
        self.value = value
        self.state = state

    def __repr__(self):
        return f"Success(value={self.value}, pos={self.state.pos})"


class Failure:
    def __init__(self, message: str, state: ParserState):
        self.message = message
        self.state = state

    def __repr__(self):
        return f"Failure(message='{self.message}', pos={self.state.pos})"


Result = Union[Success[T], Failure]
Parser = Callable[[ParserState], Result[T]]


def run_parser(parser: Parser[T], data: Readable) -> Result[T]:
    state = ParserState(data)
    return parser(state)


def bytes_n(n: int) -> Parser[bytes]:
    def _bytes_n(state: ParserState) -> Result[bytes]:
        read_bytes = state.readable.read(n, state.pos)
        if len(read_bytes) < n:
            return Failure(
                f"End of stream: expected {n} bytes, but only got {len(read_bytes)}",
                state,
            )
        return Success(read_bytes, replace(state, pos=state.pos + n))

    return _bytes_n


def map_p(fn: Callable[[T], R], p: Parser[T]) -> Parser[R]:
    def _map_p(state: ParserState) -> Result[R]:
        result = p(state)
        if isinstance(result, Failure):
            return result
        return Success(fn(result.value), result.state)
    return _map_p


any_byte = map_p(lambda b: b[0], bytes_n(1))


def seek_absolute(pos: int) -> Parser[None]:
    """
    Moves the stream to an absolute position.

    Warning: This is a low-level and dangerous operation that breaks parser
    composability. A parser that uses `seek_absolute` cannot be safely
    used inside other combinators like `choice` or `many`, as it makes
    backtracking impossible.

    This should only be used for top-level parsing tasks, such as jumping
    to an initial offset from the start of a file. For all other purposes,
    use the composable `anchor` and `seek` combinators.
    """
    def _seek(state: ParserState) -> Result[None]:
        if 0 <= pos < len(state):
            return Success(None, replace(state, pos=pos))
        return Failure(f"Cannot seek to absolute position {pos}", state)

    return _seek


def seek(offset: int) -> Parser[None]:
    """
    Moves the stream to a position relative to the current anchor.
    If no anchor is set, seeks relative to the start of the stream.
    This operation is safe, composable, and allows for backtracking.
    The check for validity is deferred to the next read operation.
    """
    def _seek_rel(state: ParserState) -> Result[None]:
        base_pos = state.anchors[-1] if state.anchors else 0
        new_pos = base_pos + offset
        return Success(None, replace(state, pos=new_pos))

    return _seek_rel


def with_relocation(p: Parser[T]) -> Parser[T]:
    """
    A combinator that creates a new local frame of reference for seeking.

    It runs a parser `p` within an "relocated" context. Any `seek` calls
    inside `p` will be relative to the stream position where the anchor
    was created. This allows for composable, relocatable parsers that can
    perform seeks without breaking backtracking.
    """
    def _relocatable(state: ParserState) -> Result[T]:
        # 1. Create a new stream with the current position added to the anchor stack.
        anchored_state = replace(state, anchors=state.anchors + (state.pos,))

        # 2. Run the wrapped parser in this new anchored context.
        result = p(anchored_state)

        if isinstance(result, Failure):
            # On failure, we return the failure but with the *original* stream,
            # preserving the backtracking contract.
            return Failure(result.message, state)

        # 3. On success, calculate bytes consumed inside the anchor and advance
        # the outer stream by that amount, discarding the anchor.
        bytes_consumed = result.state.pos - state.pos
        final_state = replace(state, pos=state.pos + bytes_consumed)
        return Success(result.value, final_state)

    return _relocatable


def relocatable(parser_producer: Callable[..., Parser[T]]) -> Callable[..., Parser[T]]:
    """
    A decorator that makes a parser-producing function create a relocatable parser.
    This automatically wraps the returned parser in `with_relocation`.
    """

    @wraps(parser_producer)
    def wrapper(*args: Any, **kwargs: Any) -> Parser[T]:
        parser = parser_producer(*args, **kwargs)
        return with_relocation(parser)

    return wrapper

def satisfy(predicate: Callable[[int], bool]) -> Parser[int]:
    def _satisfy(state: ParserState) -> Result[int]:
        result = any_byte(state)
        if isinstance(result, Failure):
            return result
        if predicate(result.value):
            return result
        return Failure(f"Predicate failed for {result.value}", state)

    return _satisfy


def byte(b: int) -> Parser[int]:
    return satisfy(lambda x: x == b)

def sequence(*parsers: Parser) -> Parser[list]:
    def _sequence(state: ParserState) -> Result[list]:
        results = []
        current_state = state
        for p in parsers:
            result = p(current_state)
            if isinstance(result, Failure):
                return result
            results.append(result.value)
            current_state = result.state
        return Success(results, current_state)

    return _sequence

def many(p: Parser[T]) -> Parser[list[T]]:
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

def uint_be(n: int) -> Parser[int]:
    """
    A generic parser for big-endian unsigned integers of `n` bytes.
    """
    def _uint_be(state: ParserState) -> Result[int]:
        result = bytes_n(n)(state)
        if isinstance(result, Failure):
            return result
        value = int.from_bytes(result.value, 'big', signed=False)
        return Success(value, result.state)
    return _uint_be


uint8 = uint_be(1)
uint16_be = uint_be(2)
uint32_be = uint_be(4)

def count(n: int, p: Parser[T]) -> Parser[list[T]]:
    """Runs a parser `p` exactly `n` times."""
    return sequence(*([p] * n))


def bind_p(p: Parser[T], f: Callable[[T], Parser[R]]) -> Parser[R]:
    """
    The 'bind' monadic operator for parsers.
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
    """Parses a specific string."""
    return map_p(
        lambda x: x.decode("utf-8"),
        bytes_n(len(s.encode("utf-8")))
    )


def position() -> Parser[int]:
    """
    A parser that consumes no input and returns the current position
    in the stream. This is useful for calculations that depend on the
    size of a parsed block.
    """
    def _position(state: ParserState) -> Result[int]:
        return Success(state.pos, state)
    return _position


def get_state() -> Parser[ParserState]:
    """
    A parser that consumes no input and returns the current stream object.
    This is useful within a `do` block to get access to the stream.
    """
    def _get_stream(state: ParserState) -> Result[ParserState]:
        return Success(state, state)
    return _get_stream


def failure(message: str) -> Parser[Any]:
    """
    A parser that always fails with the given message.
    This is useful for reporting custom error messages within a `do` block
    or other complex parsers.
    """
    def _failure(state: ParserState) -> Failure:
        return Failure(message, state)
    return _failure


def peek(p: Parser[T]) -> Parser[T]:
    """
    Runs a parser `p` without consuming any input.
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


def take(n: int, p: Parser[T]) -> Parser[T]:
    """
    Creates a parser that consumes a fixed number of bytes and runs another
    parser within that limited context.

    This combinator is implemented by creating a sub-parser that is first
    anchored, then reads `n` bytes, and runs `p` on that isolated block.
    This demonstrates how more complex combinators can be built from simpler ones.
    """
    @do
    def _take_impl() -> Generator[Parser, Any, T]:
        # 1. Get the current stream to calculate the sub-reader's offset and data source.
        current_state = yield get_state()
        
        # 2. Create a SubReader for the next `n` bytes.
        sub_reader = SubReader(
            base_reader=current_state.readable,
            base_offset=current_state.pos,
            length=n
        )
        # The new state uses the SubReader, but `pos` is still absolute.
        scoped_state = replace(current_state, readable=sub_reader)

        # 3. Run the provided parser `p` on the scoped state.
        #    The SubReader will enforce the boundary.
        result = p(scoped_state)

        if isinstance(result, Failure):
            yield failure(f"Parser failed within take({n}): {result.message}")
            return  # type: ignore

        # 4. If the inner parser succeeds, we advance the outer stream's
        #    position by `n` bytes.
        yield skip(n)
        
        return result.value

    return _take_impl()


def create_parser_from_dataclass(dc: type) -> Parser:
    """
    Automatically creates a parser for a dataclass from Annotated metadata.

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

    return map_p(
        lambda results: dc(*results),
        sequence(*field_parsers)
    )

def do(fn: Callable[..., Generator[Parser, Any, R]]) -> Callable[..., Parser[R]]:
    """
    A decorator that enables monadic comprehensions (do-notation) for parsers
    using generators. This allows writing complex, sequential parsers in a
    clean, imperative style, avoiding nested `bind_p` calls.

    The decorated function must be a generator that yields parsers. The
    result of each yielded parser is sent back into the generator. The
    generator's final return value becomes the success value of the
    overall parser.
    """
    @wraps(fn)
    def wrapper(*args, **kwargs) -> Parser[R]:
        def _do(state: ParserState) -> Result[R]:
            gen = fn(*args, **kwargs)
            
            def step(g: Generator[Parser[Any], Any, R], state: ParserState, value: Any = None) -> Result[R]:
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
def lazy(
    size: int,
    parser: Callable[[int], Parser[T]],
) -> Generator[Parser, Any, Lazy[T]]:
    """
    Creates a lazy-evaluated value by parsing a block of a given size.

    This combinator is essential for performance. It immediately skips the
    main stream forward by `size` bytes, while returning a `Lazy` object.
    The actual parsing of the content block is deferred until the `.value`
    of the `Lazy` object is accessed for the first time.

    This uses the `take` combinator internally to provide a safe, isolated
    stream for the deferred parsing.
    """
  
    # `take` will run `get_stream` on an isolated sub-stream of `size` bytes.
    # The result, `lazy_content_stream`, will be a Stream object whose data
    # is only the `size` bytes we skipped over.

    lazy_content_stream = yield take(size, get_state())

    def lazy_thunk() -> T:
        # When the thunk is finally called, it runs the real parser on the
        # captured, isolated sub-stream.
        result = parser(size)(lazy_content_stream)
        if isinstance(result, Failure):
            # If parsing fails, raise an exception to signal the problem.
            raise ValueError(f"Failed to parse lazy content: {result.message}")
        return result.value

    # Return a Lazy object containing the thunk. The main stream has already
    # been advanced by `take`.
    return Lazy(lazy_thunk)


@do
def with_bytes_read(parser: Parser[T]) -> Generator[Parser, Any, tuple[T, int]]:
    """
    A combinator that returns the result of a parser along with the number of bytes it consumed.
    """
    
    start_pos = yield position()
    result = yield parser
    end_pos = yield position()
    bytes_read = end_pos - start_pos
    return result, bytes_read


def skip(n: int) -> Parser[None]:
    """
    A parser that skips n bytes, failing if there are not enough bytes.
    """

    def _parser(state: ParserState) -> Result[None]:
        if n < 0 or n > len(state):
            return Failure(message="Not enough bytes to skip", state = state)

        new_pos = state.pos + n
        return Success(value=None, state=replace(state, pos=new_pos))

    return _parser

