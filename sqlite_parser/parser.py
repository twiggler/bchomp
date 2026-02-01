from typing import Any, Callable, Generator, Generic, TypeVar, TypeVarTuple, Union, Optional, Protocol, IO, get_args, TYPE_CHECKING
from functools import wraps
import struct
import os
import dataclasses

T = TypeVar("T")
R = TypeVar("R")
Ts = TypeVarTuple("Ts")

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


class Stream:
    def __init__(self, data: Readable, pos: int = 0):
        self.data = data
        self.pos = pos

    def __len__(self):
        return len(self.data)


class Success(Generic[T]):
    def __init__(self, value: T, stream: Stream):
        self.value = value
        self.stream = stream

    def __repr__(self):
        return f"Success(value={self.value}, pos={self.stream.pos})"


class Failure:
    def __init__(self, message: str, stream: Stream):
        self.message = message
        self.stream = stream

    def __repr__(self):
        return f"Failure(message='{self.message}', pos={self.stream.pos})"


Result = Union[Success[T], Failure]
Parser = Callable[[Stream], Result[T]]


def parser(fn: Callable[..., Parser[R]]) -> Callable[..., Parser[R]]:
    @wraps(fn)
    def wrapper(*args, **kwargs) -> Parser[R]:
        return fn(*args, **kwargs)

    return wrapper


def run_parser(parser: Parser[T], data: Readable) -> Result[T]:
    stream = Stream(data)
    return parser(stream)


def any_byte(stream: Stream) -> Result[int]:
    if stream.pos < len(stream.data):
        read_bytes = stream.data.read(1, stream.pos)
        if not read_bytes:
            return Failure("End of stream", stream)
        return Success(read_bytes[0], Stream(stream.data, stream.pos + 1))
    return Failure("End of stream", stream)


@parser
def seek(pos: int) -> Parser[None]:
    def _seek(stream: Stream) -> Result[None]:
        if pos < len(stream.data):
            return Success(None, Stream(stream.data, pos))
        return Failure(f"Cannot seek to {pos}", stream)

    return _seek


@parser
def seek_relative(offset: int) -> Parser[None]:
    """Advance the stream position by `offset` (can be negative).

    This is similar to file-like relative seek semantics but restricted
    to the in-memory `Stream`. If the resulting position would be out
    of bounds, returns a `Failure`.
    """
    def _seek_rel(stream: Stream) -> Result[None]:
        new_pos = stream.pos + offset
        if 0 <= new_pos < len(stream.data):
            return Success(None, Stream(stream.data, new_pos))
        return Failure(f"Cannot seek relative by {offset} from {stream.pos}", stream)

    return _seek_rel


@parser
def satisfy(predicate: Callable[[int], bool]) -> Parser[int]:
    def _satisfy(stream: Stream) -> Result[int]:
        result = any_byte(stream)
        if isinstance(result, Failure):
            return result
        if predicate(result.value):
            return result
        return Failure(f"Predicate failed for {result.value}", stream)

    return _satisfy


@parser
def byte(b: int) -> Parser[int]:
    return satisfy(lambda x: x == b)


@parser
def bytes_n(n: int) -> Parser[bytes]:
    def _bytes_n(stream: Stream) -> Result[bytes]:
        start = stream.pos
        end = start + n
        if end > len(stream.data):
            return Failure("Not enough bytes", stream)
        return Success(stream.data.read(n, start), Stream(stream.data, end))

    return _bytes_n


def sequence(*parsers: Parser) -> Parser[list]:
    def _sequence(stream: Stream) -> Result[list]:
        results = []
        current_stream = stream
        for p in parsers:
            result = p(current_stream)
            if isinstance(result, Failure):
                return result
            results.append(result.value)
            current_stream = result.stream
        return Success(results, current_stream)

    return _sequence


def choice(*parsers: Parser) -> Parser:
    def _choice(stream: Stream) -> Result:
        for p in parsers:
            result = p(stream)
            if isinstance(result, Success):
                return result
        return Failure("All choices failed", stream)

    return _choice


def many(p: Parser[T]) -> Parser[list[T]]:
    def _many(stream: Stream) -> Result[list[T]]:
        results = []
        current_stream = stream
        while True:
            result = p(current_stream)
            if isinstance(result, Failure):
                break
            results.append(result.value)
            current_stream = result.stream
        return Success(results, current_stream)

    return _many


def map_p(fn: Callable[[T], R], p: Parser[T]) -> Parser[R]:
    def _map_p(stream: Stream) -> Result[R]:
        result = p(stream)
        if isinstance(result, Failure):
            return result
        return Success(fn(result.value), result.stream)

    return _map_p


def uint16_be(stream: Stream) -> Result[int]:
    result = bytes_n(2)(stream)
    if isinstance(result, Failure):
        return result
    value = struct.unpack(">H", result.value)[0]
    return Success(value, result.stream)

def uint32_be(stream: Stream) -> Result[int]:
    result = bytes_n(4)(stream)
    if isinstance(result, Failure):
        return result
    value = struct.unpack(">I", result.value)[0]
    return Success(value, result.stream)

def uint8(stream: Stream) -> Result[int]:
    result = bytes_n(1)(stream)
    if isinstance(result, Failure):
        return result
    return Success(result.value[0], result.stream)


def count(n: int, p: Parser[T]) -> Parser[list[T]]:
    """Runs a parser `p` exactly `n` times."""
    return sequence(*([p] * n))


def bind_p(p: Parser[T], f: Callable[[T], Parser[R]]) -> Parser[R]:
    """
    The 'bind' monadic operator for parsers.
    Runs parser `p`, and if it succeeds, passes its result to function `f`.
    `f` must return a new parser, which is then run on the stream.
    """

    def _bind_p(stream: Stream) -> Result[R]:
        initial_result = p(stream)
        if isinstance(initial_result, Failure):
            return initial_result

        new_parser = f(initial_result.value)
        return new_parser(initial_result.stream)

    return _bind_p

def string(s: str) -> Parser[str]:
    """Parses a specific string."""
    return map_p(
        lambda x: x.decode("utf-8"),
        bytes_n(len(s.encode("utf-8")))
    )

def make_lazy(proto: type, lazy_fields: list[str]) -> type:
    """
    Dynamically creates a dataclass that implements a given protocol,
    but with specified fields converted to lazy-loading properties.

    Args:
        proto: A class (ideally a Protocol) defining the desired fields and types.
        lazy_fields: A list of field names from the protocol that should be lazy.

    Returns:
        A new dataclass type with lazy-loading capabilities.
    """
    # Get the annotations from the protocol. 
    attrs = getattr(proto, '__annotations__', {})

    # Prepare fields for the new dataclass
    new_class_fields = []
    # Store info for creating properties later
    property_map = {}

    for name, field_type in attrs.items():
        if name in lazy_fields:
            # For lazy fields, the internal storage will be a Lazy[T]
            # The field name in the dataclass is prefixed to avoid clashes
            internal_name = f"_{name}_lazy"
            new_class_fields.append((internal_name, Lazy[field_type]))

            # The property will provide transparent access to the lazy value
            def make_prop(internal_name):
                return property(lambda self: getattr(self, internal_name).value)

            property_map[name] = make_prop(internal_name)
        else:
            # Regular fields are added as-is
            new_class_fields.append((name, field_type))

    # Create the new dataclass
    new_class = dataclasses.make_dataclass(
        f"{proto.__name__}Implementation",
        fields=new_class_fields,
    )

    # Add the lazy properties to the new class
    for name, prop in property_map.items():
        setattr(new_class, name, prop)

    # This helps type checkers understand the relationship
    if TYPE_CHECKING:
        # In a type-checking context, we can assert it implements the protocol
        assert issubclass(new_class, proto)

    return new_class


@parser
def position() -> Parser[int]:
    """
    A parser that consumes no input and returns the current position
    in the stream. This is useful for calculations that depend on the
    size of a parsed block.
    """
    def _position(stream: Stream) -> Result[int]:
        return Success(stream.pos, stream)
    return _position


@parser
def get_stream() -> Parser[Stream]:
    """
    A parser that consumes no input and returns the current stream object.
    This is useful within a `do` block to get access to the stream.
    """
    def _get_stream(stream: Stream) -> Result[Stream]:
        return Success(stream, stream)
    return _get_stream


@parser
def peek(p: Parser[T]) -> Parser[T]:
    """
    Runs a parser `p` without consuming any input.
    It returns the result of `p`, but the stream position is reset to
    where it was before `p` was run. This is useful for looking ahead
    in the stream to decide which parser to use next.
    """
    def _peek(stream: Stream) -> Result[T]:
        result = p(stream)
        if isinstance(result, Failure):
            return result
        # On success, return the value but with the original, unmodified stream.
        return Success(result.value, stream)
    return _peek


def take(n: int, p: Parser[T]) -> Parser[T]:
    """
    Creates a parser that consumes a fixed number of bytes, and runs
    another parser within that limited context.

    This is a powerful combinator for parsing nested structures that are
    prefixed with their total size.

    Args:
        n: The number of bytes to consume from the main stream.
        p: The parser to run on the sub-stream of `n` bytes.

    Returns:
        A new parser that, if successful, returns the result of `p` but
        advances the original stream by `n` bytes.
    """
    def _take(stream: Stream) -> Result[T]:
        # 1. Read the `n` bytes for the sub-parser from the current stream.
        sub_bytes_result = bytes_n(n)(stream)
        if isinstance(sub_bytes_result, Failure):
            return sub_bytes_result # Not enough bytes in the main stream

        sub_bytes = sub_bytes_result.value
        
        # 2. Create a completely new, temporary parsing context for the sub-problem.
        sub_reader = BytesReader(sub_bytes)
        sub_stream = Stream(sub_reader)

        # 3. Run the provided parser `p` on the temporary stream.
        result = p(sub_stream)

        if isinstance(result, Failure):
            # The sub-parser failed. We report the failure but position the error
            # at the start of the block in the *original* stream for better context.
            return Failure(f"Parser failed within take({n}): {result.message}", stream)

        # 4. The sub-parser succeeded. We return its value, but the new stream
        # is the *original* stream, advanced by `n` bytes.
        return Success(result.value, sub_bytes_result.stream)

    return _take


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
        def _do(stream: Stream) -> Result[R]:
            gen = fn(*args, **kwargs)
            
            def step(g: Generator[Parser[Any], Any, R], stream: Stream, value: Any = None) -> Result[R]:
                try:
                    p = g.send(value)
                    result = p(stream)
                    if isinstance(result, Failure):
                        return result
                    return step(g, result.stream, result.value)
                except StopIteration as e:
                    # The generator has finished, its return value is the final result.
                    return Success(e.value, stream)

            return step(gen, stream)
        return _do
    return wrapper

@do
def lazy(
    size: int,
    parser: Callable[[int], Parser[T]],
) -> Generator[Parser, Any, Lazy[T]]:
    """
    Parses a sequence where a block of data is evaluated lazily.
  
    The main stream is advanced past the entire content block, while the lazy thunk
    gets a separate stream positioned at the start of the content block.
    """
    # The size_parser tells us the size of the content that follows it.
    
    lazy_content_stream = yield get_stream()
   
    def lazy_thunk():
        # When the thunk is called, it creates a new stream at the correct
        # starting position for the lazy content.
        result = parser(size)(lazy_content_stream)
        if isinstance(result, Failure):
            raise ValueError(f"Failed to parse lazy content: {result}")
        return result.value

    lazy_value = Lazy(lazy_thunk)

    # Advance the main stream past the entire content block.
    yield seek_relative(size)

    return lazy_value
