"""Parsers for the SQLite database file format structures."""

from dataclasses import dataclass
from enum import IntEnum
from functools import reduce
from typing import TYPE_CHECKING, Annotated, Any, Protocol

import bchomp.parser as p
import bchomp.transformers as t
from bchomp.adapters.lazy import lazy, make_lazy
from bchomp.compose import compose

if TYPE_CHECKING:
    from collections.abc import Generator


HEADER_SIZE = 100


class PageType(IntEnum):
    """An enumeration of B-Tree page types."""

    INDEX_INTERIOR = 2
    TABLE_INTERIOR = 5
    INDEX_LEAF = 10
    TABLE_LEAF = 13


class SerialType(IntEnum):
    """An enumeration of serial types for record values."""

    NULL = 0
    INT8 = 1
    INT16 = 2
    INT24 = 3
    INT32 = 4
    INT48 = 5
    INT64 = 6
    FLOAT64 = 7
    ZERO = 8
    ONE = 9
    RESERVED = 10
    RESERVED_11 = 11
    BLOB = 12
    TEXT = 13


@dataclass(frozen=True)
class Blob:
    """Represents a BLOB value with a specified size in bytes."""

    size: int


@dataclass(frozen=True)
class Text:
    """Represents a TEXT value with a specified size in bytes."""

    size: int


SerialKind = SerialType | Blob | Text


@dataclass
class TableLeafCell:
    """A cell in a table leaf page."""

    rowid: int
    payload: Record


class LeafPage(Protocol):
    """A leaf page in a B-Tree, containing a header and cells."""

    header: LeafPageHeader
    cells: list[TableLeafCell]


ColumnValue = int | None | bytes | str | float


def read_varint(state: p.ParserState) -> p.BlockingResult[int]:
    """Parse a variable-length integer (varint)."""

    def process_bytes(parts: tuple[list[int], int]) -> int:
        head, tail = parts
        initial_value = reduce(lambda acc, byte: (acc << 7) | (byte & 0x7F), head, 0)
        return (initial_value << 7) | (tail & 0x7F)

    continuation_byte = p.satisfy(lambda b: (b & 0x80) != 0)
    final_byte = p.satisfy(lambda b: (b & 0x80) == 0)
    parser = p.sequence(p.many(continuation_byte), final_byte)

    # TODO: check case where 9 bytes are read
    return p.map_p(process_bytes, parser)(state)


parse_page_type = p.enum(p.uint8, PageType)


@p.do
def read_serial_type() -> Generator[p.BlockingParser, Any, SerialKind]:
    """Parse a varint and return either a `SerialType`, `Blob`, or `Text`.

    For serial-type values >= 12, the value encodes a BLOB or TEXT with a
    length: even values => BLOB, odd values => TEXT.
    """
    v = yield read_varint
    match v:
        case x if x >= SerialType.BLOB and x % 2 == 0:
            return Blob((x - SerialType.BLOB) // 2)
        case x if x >= SerialType.TEXT:
            return Text((x - SerialType.TEXT) // 2)
        case x:
            try:
                return SerialType(x)
            except ValueError as exc:
                msg = f"Unknown serial type: {x}"
                raise p.ParseError(msg) from exc


@dataclass
class PageHeaderBase:
    """The base header for a B-Tree page, common to all page types.

    This structure contains metadata about the page, such as its type, the
    number of cells it contains, and information about free space. The first
    100 bytes of the database file are the database header, but for all other
    pages, the B-Tree page header is at the beginning of the page.
    """

    page_type: Annotated[int, p.uint8]
    freeblock_start: Annotated[int, p.uint16_be]
    cell_count: Annotated[int, p.uint16_be]
    cell_start: Annotated[int, p.uint16_be]
    fragmented_bytes: Annotated[int, p.uint8]


@dataclass
class InteriorPageHeader(PageHeaderBase):
    """The header for a B-Tree interior page, extending the base header.

    Interior pages (both for tables and indexes) contain a `right_most_pointer`
    which points to a child page.
    """

    right_most_pointer: Annotated[int, p.uint32_be]


LeafPageHeader = PageHeaderBase


read_interior_page_header: p.BlockingParser[InteriorPageHeader] = p.create_parser_from_dataclass(
    InteriorPageHeader,
)
read_leaf_page_header: p.BlockingParser[LeafPageHeader] = p.create_parser_from_dataclass(
    LeafPageHeader,
)


@dataclass
class Record:
    """Represents a data record from a table, corresponding to a single row.

    In SQLite, the payload of a table B-tree leaf cell is a record. This
    class holds the parsed content of such a record, which includes a list of
    serial types that describe the data, and the actual column values.
    """

    serial_types: list[SerialKind]
    values: list[ColumnValue]


def read_cell_pointer_array(cell_count: int) -> p.BlockingParser[list[int]]:
    """Parse an array of cell pointers."""
    return p.count(cell_count, p.uint16_be)


@p.do
def read_record_header() -> p.BlockingScript[tuple[int, list[SerialKind]]]:
    """Parse the record header using do-notation for clarity.

    Returns:
        A tuple containing:
        - The total size of the header in bytes (including the size varint).
        - A list of the serial types.

    """
    total_header_size, varint_size = yield p.with_bytes_read(read_varint)

    header_content_size = total_header_size - varint_size

    header_content = yield p.take(header_content_size, p.many(read_serial_type()))

    return total_header_size, header_content


@p.do
def read_record(payload_size: int) -> p.BlockingScript[Record]:
    """Parse a record of a given size."""
    header_size, header_content = yield read_record_header()

    body_size = payload_size - header_size

    body_parser = read_parse_record_body(header_content)
    values = yield p.take(body_size, body_parser)

    return Record(serial_types=header_content, values=values)


@p.do
def read_table_interior_cell() -> p.BlockingScript[int]:
    """Parse a table interior cell to find the left-child pointer.

    An interior cell format is:
    - 4-byte page number (the left child pointer)
    - A varint for the integer key.

    We only need the page number for our traversal.
    """
    page_number = yield p.uint32_be
    _ = yield read_varint  # We don't need the key, but we must parse it to advance the stream.
    return page_number


@p.do
def read_table_leaf_cell() -> p.BlockingScript[TableLeafCell]:
    """Parse a table leaf cell."""
    payload_size = yield read_varint
    row_id = yield read_varint

    # The payload of a table leaf cell is a record.
    # We can parse it directly since we know its size.
    payload = yield p.take(payload_size, read_record(payload_size))

    return TableLeafCell(rowid=row_id, payload=payload)


def read_column_value(st: SerialKind) -> p.BlockingParser[ColumnValue]:  # noqa: C901, PLR0911, PLR0912
    """Return a parser for a single column value described by the serial kind."""
    match st:
        case Blob(size=size):
            return p.bytes_n(size)
        case Text(size=size):
            return p.map_p(lambda b: b.decode(), p.bytes_n(size))
        case SerialType.NULL:
            return p.map_p(lambda _: None, p.bytes_n(0))
        case SerialType.INT8:
            return p.int8
        case SerialType.INT16:
            return p.int16_be
        case SerialType.INT24:
            return p.int24_be
        case SerialType.INT32:
            return p.int32_be
        case SerialType.INT48:
            return p.int48_be
        case SerialType.INT64:
            return p.int64_be
        case SerialType.FLOAT64:
            return p.float_be(8)
        case SerialType.ZERO:
            return p.pure(0)
        case SerialType.ONE:
            return p.pure(1)
        case _:
            return p.failure(f"Unsupported serial kind: {st}")


def read_parse_record_body(
    serial_types: list[SerialKind],
) -> p.BlockingParser[tuple[ColumnValue, ...]]:
    """Create a parser for a record's body based on its serial types.

    `serial_types` is a list of `SerialKind` items (either a `SerialType`
    enum member, or a sized `Blob`/`Text`). This function builds a sequence
    parser composed from the appropriate primitive parsers for each
    serial-kind.
    """
    parsers = [read_column_value(st) for st in serial_types]
    return p.sequence(*parsers)


def read_parse_table_leaf_pages(
    page_num: int,
    page_size: int,
) -> p.StreamingParser[LeafPage]:
    """Recursively traverse a B-Tree and return a list of all parsed leaf pages."""

    @p.do
    def _traverse(
        current_page_num: int,
    ) -> Generator[p.Parser, Any]:
        page_start = (current_page_num - 1) * page_size
        yield p.seek(page_start)
        offset = HEADER_SIZE if current_page_num == 1 else 0

        page_type = yield compose(parse_page_type, p.with_relocation, p.peek, t.at(offset))
        if page_type == PageType.TABLE_LEAF:
            # This is a table leaf page, parse it and return.
            leaf_page = yield compose(
                read_leaf_page(page_size), p.with_relocation, t.take(page_size), t.at(offset)
            )
            yield p.emit(leaf_page)

        elif page_type == PageType.TABLE_INTERIOR:
            # This is a table interior page. Get the child pointers to recurse.
            child_page_pointers = yield compose(
                read_interior_page_child_pointers(),
                p.with_relocation,
                t.take(page_size),
                t.at(offset),
            )
            for child_page_num in child_page_pointers:
                yield _traverse(child_page_num)

    return _traverse(page_num)


@p.do
def read_interior_page_child_pointers() -> p.BlockingScript[list[int]]:
    """Parse an interior page to extract all child page pointers."""
    header = yield read_interior_page_header

    cell_pointers = yield read_cell_pointer_array(header.cell_count)

    child_pages = []
    for cell_ptr_offset in cell_pointers:
        yield p.seek(cell_ptr_offset)
        child_page_num = yield read_table_interior_cell()
        child_pages.append(child_page_num)

    # For interior pages, we also need to include the right-most pointer.
    child_pages.append(header.right_most_pointer)

    return child_pages


@p.do
def read_leaf_page(
    page_size: int,
) -> p.BlockingScript[LeafPage]:
    """Parse a leaf page, returning a `LeafPage` object with lazy-evaluated cells."""

    @p.do
    def parse_leaf_page_cells(
        cell_count: int,
    ) -> p.BlockingScript[list[TableLeafCell]]:
        cell_pointers = yield read_cell_pointer_array(cell_count)

        cells = []
        for cell_ptr_offset in cell_pointers:
            yield p.seek(cell_ptr_offset)
            cell = yield read_table_leaf_cell()
            cells.append(cell)

        return cells

    header, header_size = yield p.with_bytes_read(read_leaf_page_header)

    lazy_leaf_page = make_lazy(LeafPage, lazy_fields=["cells"])
    lazy_cells = yield lazy(
        page_size - header_size,
        lambda _: parse_leaf_page_cells(header.cell_count),
    )

    return lazy_leaf_page(header=header, cells=lazy_cells)
