"""Lazy evaluation combinators and wrappers for bchomp parsing."""

from typing import TYPE_CHECKING, Any, get_type_hints

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
def lazy[T](
    size: int, parser: Callable[[int], p.BlockingParser[T]]
) -> Generator[p.BlockingParser, Any, Lazy[T]]:
    """Create a lazy-evaluated value by parsing a block of a given size.

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

    lazy_content_stream: p.ParserState = yield p.take(size, p.get_state())

    def lazy_thunk() -> T:
        # When the thunk is finally called, it runs the real parser on the
        # captured, isolated sub-stream.
        result = parser(size)(lazy_content_stream)
        if isinstance(result, p.Failure):
            # If parsing fails, raise an exception to signal the problem.
            msg = f"Failed to parse lazy content: {result.message}"
            raise ValueError(msg)  # noqa: TRY004
        return result.value

    # Return a Lazy object containing the thunk. The main stream has already
    # been advanced by `take`.
    return Lazy(lazy_thunk)


def make_lazy(proto: type, lazy_fields: list[str]) -> type:
    """Dynamically creates a plain class that implements a protocol lazily.

    Resolves the protocol's annotated fields with ``get_type_hints`` so
    forward-references are handled correctly at runtime.
    """
    hints = get_type_hints(proto, include_extras=False)

    all_fields = list(hints.keys())

    def _init(self: object, **kwargs: dict[str, Any]) -> None:
        """Initialize the object, mapping public lazy field names."""
        for key, value in kwargs.items():
            if key in lazy_fields:
                # Store the Lazy object in the internal attribute.
                setattr(self, f"_{key}_lazy", value)
            else:
                # Set regular attributes directly.
                setattr(self, key, value)

    # Add a __repr__ for better debugging output.
    def _repr(self: object) -> str:
        parts = []
        for name in all_fields:
            if name in lazy_fields:
                val = getattr(self, f"_{name}_lazy")
                parts.append(f"{name}={val!r}")
            else:
                val = getattr(self, name)
                parts.append(f"{name}={val!r}")
        return f"Lazy{proto.__name__}({', '.join(parts)})"

    # Create a dictionary of attributes for the new class, starting with __init__ and __repr__.
    attrs: dict[str, Any] = {"__init__": _init, "__repr__": _repr}

    # For each lazy field, create a property that evaluates the lazy value on access.
    for field in lazy_fields:
        internal_name = f"_{field}_lazy"

        # The property uses the `.value` attribute of our `Lazy` class,
        # which already handles the caching internally.
        attrs[field] = property(lambda self, name=internal_name: getattr(self, name).value)

    # Create the new class dynamically using type().
    new_class_name = f"Lazy{proto.__name__}"
    return type(new_class_name, (), attrs)
