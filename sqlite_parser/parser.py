from typing import Any, Callable, Generator, Generic, Self, TypeVar, TypeVarTuple, Union, Optional, Protocol, IO, get_args, TYPE_CHECKING
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


class SubReader:
    """
    A lazy implementation of the Readable protocol for a sub-section of a
    larger Readable object. It does not copy the data, but rather holds a
    reference to the original data source and an offset.
    """
    def __init__(self, original_reader: Readable, base_offset: int, length: int):
        self._reader = original_reader
        self._offset = base_offset
        self._len = length

    def read(self, n: int, offset: int) -> bytes:
        """Reads from the original reader, adjusting for the sub-stream's offset."""
        if offset + n > self._len:
            n = self._len - offset
        if n < 0:
            n = 0
        
        # The read happens from the original reader at the base offset + sub-stream offset.
        return self._reader.read(n, self._offset + offset)

    def __len__(self) -> int:
        return self._len


class ParserState:
    def __init__(self, readable: Readable, pos: int = 0, anchors: list[int] = []):
        self.readable = readable
        self.pos = pos
        self.anchors = anchors

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


def any_byte(state: ParserState) -> Result[int]:
    if state.pos < len(state.readable):
        read_bytes = state.readable.read(1, state.pos)
        if not read_bytes:
            return Failure("End of stream", state)
        return Success(read_bytes[0], ParserState(state.readable, state.pos + 1))
    return Failure("End of stream", state)


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
        if 0 <= pos < len(state.readable):
            return Success(None, ParserState(state.readable, pos, anchors=state.anchors))
        return Failure(f"Cannot seek to absolute position {pos}", state)

    return _seek


def seek(offset: int) -> Parser[None]:
    """
    Moves the stream to a position relative to the current anchor.
    If no anchor is set, seeks relative to the start of the stream.
    This operation is safe, composable, and allows for backtracking.
    """
    def _seek_rel(state: ParserState) -> Result[None]:
        base_pos = state.anchors[-1] if state.anchors else 0
        new_pos = base_pos + offset

        if 0 <= new_pos < len(state.readable):
            return Success(None, ParserState(state.readable, new_pos, anchors=state.anchors))
        
        msg = f"Cannot seek to offset {offset} from anchor {base_pos}"
        return Failure(msg, state)

    return _seek_rel


def anchor(p: Parser[T]) -> Parser[T]:
    """
    A combinator that creates a new local frame of reference for seeking.

    It runs a parser `p` within an "anchored" context. Any `seek` calls
    inside `p` will be relative to the stream position where the anchor
    was created. This allows for composable, relocatable parsers that can
    perform seeks without breaking backtracking.
    """
    def _anchor(state: ParserState) -> Result[T]:
        # 1. Create a new stream with the current position added to the anchor stack.
        anchored_state = ParserState(state.readable, state.pos, anchors=[*state.anchors, state.pos])

        # 2. Run the wrapped parser in this new anchored context.
        result = p(anchored_state)

        if isinstance(result, Failure):
            # On failure, we return the failure but with the *original* stream,
            # preserving the backtracking contract.
            return Failure(result.message, state)

        # 3. On success, calculate bytes consumed inside the anchor and advance
        # the outer stream by that amount, discarding the anchor.
        bytes_consumed = result.state.pos - state.pos
        final_state = ParserState(state.readable, state.pos + bytes_consumed, anchors=state.anchors)
        return Success(result.value, final_state)

    return _anchor


class AnchorContextManager:
    """
    A parser-aware context manager for creating temporary anchor points.

    This is designed to be used within a `do`-block. Yielding this object
    returns a context manager that can be used with a `with` statement.
    The `__enter__` and `__exit__` methods of the returned manager are themselves
    parsers that handle the anchor stack.
    """
    def __enter__(self) -> Parser[None]:
        """
        Returns a parser that, when yielded, pushes the current stream
        position onto the anchor stack.
        """
        def _enter(state: ParserState) -> Result[None]:
            anchored_state = ParserState(state.readable, state.pos, anchors=[*state.anchors, state.pos])
            return Success(None, anchored_state)
        return _enter

    def __exit__(self, exc_type, exc_val, exc_tb) -> Parser[None]:
        """
        Returns a parser that, when yielded, pops the most recent anchor
        from the stack, preserving the stream position.
        """
        def _exit(state: ParserState) -> Result[None]:
            if not state.anchors:
                return Failure("Cannot exit anchor context: no anchor on the stack.", state)
            
            # The stream position is preserved, only the anchor stack is modified.
            final_state = ParserState(state.readable, state.pos, anchors=state.anchors[:-1])
            return Success(None, final_state)
        return _exit

    def __call__(self) -> Parser[Self]:
        return lambda state: Success(self, state)

anchor_cm = AnchorContextManager()


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


def bytes_n(n: int) -> Parser[bytes]:
    def _bytes_n(state: ParserState) -> Result[bytes]:
        start = state.pos
        end = start + n
        if end > len(state.readable):
            return Failure("Not enough bytes", state)
        return Success(state.readable.read(n, start), ParserState(state.readable, end))

    return _bytes_n


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


def choice(*parsers: Parser) -> Parser:
    def _choice(state: ParserState) -> Result:
        for p in parsers:
            result = p(state)
            if isinstance(result, Success):
                return result
        return Failure("All choices failed", state)

    return _choice


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


def map_p(fn: Callable[[T], R], p: Parser[T]) -> Parser[R]:
    def _map_p(state: ParserState) -> Result[R]:
        result = p(state)
        if isinstance(result, Failure):
            return result
        return Success(fn(result.value), result.state)
    return _map_p


def uint16_be(state: ParserState) -> Result[int]:
    result = bytes_n(2)(state)
    if isinstance(result, Failure):
        return result
    value = struct.unpack(">H", result.value)[0]
    return Success(value, result.state)

def uint32_be(state: ParserState) -> Result[int]:
    result = bytes_n(4)(state)
    if isinstance(result, Failure):
        return result
    value = struct.unpack(">I", result.value)[0]
    return Success(value, result.state)

def uint8(state: ParserState) -> Result[int]:
    result = bytes_n(1)(state)
    if isinstance(result, Failure):
        return result
    return Success(result.value[0], result.state)

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

def make_lazy(proto: type, lazy_fields: list[str]) -> type:
    """
    Dynamically creates a plain class that implements a protocol with some
    fields being lazy. The constructor accepts public field names.
    """
    if not hasattr(proto, "__annotations__"):
        raise TypeError(f"{proto.__name__} is not a valid protocol for lazy loading.")

    all_fields = list(proto.__annotations__.keys())

    def __init__(self, **kwargs):
        """
        Initializes the object, mapping public lazy field names (e.g., 'payload')
        to their internal storage (e.g., '_payload_lazy').
        """
        for key, value in kwargs.items():
            if key in lazy_fields:
                # Store the Lazy object in the internal attribute.
                setattr(self, f"_{key}_lazy", value)
            else:
                # Set regular attributes directly.
                setattr(self, key, value)

    # The `attrs` dictionary will form the body of our new class.
    attrs: dict[str, Any] = {"__init__": __init__}

    # For each lazy field, create a property that evaluates the lazy value on access.
    for field in lazy_fields:
        internal_name = f"_{field}_lazy"
        
        # The property uses the `.value` attribute of our `Lazy` class,
        # which already handles the caching internally.
        attrs[field] = property(lambda self, name=internal_name: getattr(self, name).value)

    # Add a __repr__ for better debugging output.
    def __repr__(self):
        parts = []
        for name in all_fields:
            if name in lazy_fields:
                val = getattr(self, f"_{name}_lazy")
                parts.append(f"{name}={val!r}")
            else:
                val = getattr(self, name)
                parts.append(f"{name}={val!r}")
        return f"Lazy{proto.__name__}({', '.join(parts)})"

    attrs["__repr__"] = __repr__

    # Create the new class dynamically using type().
    new_class_name = f"Lazy{proto.__name__}"
    return type(new_class_name, (), attrs)


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
        
        # 2. Create a lazy SubReader for the next `n` bytes. This does not
        #    perform any reading yet.
        sub_reader = SubReader(current_state.readable, current_state.pos, n)
        sub_stream = ParserState(sub_reader)

        # 3. Run the provided parser `p` on the temporary, lazy stream.
        #    The actual reads will happen inside `p`.
        result = p(sub_stream)

        if isinstance(result, Failure):
            yield failure(f"Parser failed within take({n}): {result.message}")
            return  # type: ignore

        # 4. If the inner parser succeeds, we must advance the outer stream
        #    by `n` bytes to reflect the bytes that were "taken".
        yield seek(current_state.pos + n)
        
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
    Parses a sequence where a block of data is evaluated lazily.
  
    The main stream is advanced past the entire content block, while the lazy thunk
    gets a separate stream positioned at the start of the content block.
    """
    # The size_parser tells us the size of the content that follows it.
    
    lazy_content_stream = yield get_state()
   
    def lazy_thunk():
        # When the thunk is called, it creates a new stream at the correct
        # starting position for the lazy content.
        result = parser(size)(lazy_content_stream)
        if isinstance(result, Failure):
            raise ValueError(f"Failed to parse lazy content: {result}")
        return result.value

    lazy_value = Lazy(lazy_thunk)

    # Advance the main stream past the entire content block.
    yield seek(size)

    return lazy_value
