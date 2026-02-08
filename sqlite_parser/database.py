"""Parsers for the SQLite database file format structures."""

from dataclasses import dataclass
from enum import IntEnum
from functools import reduce
from typing import TYPE_CHECKING, Annotated, Any

from sqlite_parser import parser as p

if TYPE_CHECKING:
    from collections.abc import Generator, Iterable

# Constants for serial types
SERIAL_TYPE_NULL = 0
SERIAL_TYPE_INT8 = 1
SERIAL_TYPE_INT16 = 2
SERIAL_TYPE_INT24 = 3
SERIAL_TYPE_INT32 = 4
SERIAL_TYPE_INT48 = 5
SERIAL_TYPE_FLOAT64 = 7
SERIAL_TYPE_ZERO = 8
SERIAL_TYPE_ONE = 9
SERIAL_TYPE_BLOB_MIN = 13
SERIAL_TYPE_TEXT_MIN = 12


class PageType(IntEnum):
    """An enumeration of B-Tree page types."""

    INDEX_INTERIOR = 2
    TABLE_INTERIOR = 5
    INDEX_LEAF = 10
    TABLE_LEAF = 13


@dataclass
class BTreePageHeaderBase:
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
class BTreeInteriorPageHeader(BTreePageHeaderBase):
    """The header for a B-Tree interior page, extending the base header.

    Interior pages (both for tables and indexes) contain a `right_most_pointer`
    which points to a child page.
    """

    right_most_pointer: Annotated[int, p.uint32_be]


BTreeLeafPageHeader = BTreePageHeaderBase


b_tree_interior_page_header: p.Parser[BTreeInteriorPageHeader] = p.create_parser_from_dataclass(
    BTreeInteriorPageHeader,
)
b_tree_leaf_page_header: p.Parser[BTreeLeafPageHeader] = p.create_parser_from_dataclass(
    BTreeLeafPageHeader,
)


def cell_pointer_array(cell_count: int) -> p.Parser[list[int]]:
    """Parse an array of cell pointers."""
    return p.count(cell_count, p.uint16_be)


def varint(state: p.ParserState) -> p.Result[int]:
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


@dataclass
class Record:
    """Represents a data record from a table, corresponding to a single row.

    In SQLite, the payload of a table B-tree leaf cell is a record. This
    class holds the parsed content of such a record, which includes a list of
    serial types that describe the data, and the actual column values.
    """

    serial_types: list[int]
    values: list[ColumnValue]


@p.do
def record_header() -> Generator[p.Parser, Any, tuple[int, list[int]]]:
    """Parse the record header using do-notation for clarity.

    Returns:
        A tuple containing:
        - The total size of the header in bytes (including the size varint).
        - A list of the serial types.

    """
    total_header_size, varint_size = yield p.with_bytes_read(varint)

    header_content_size = total_header_size - varint_size

    header_content = yield p.take(header_content_size, p.many(varint))

    return total_header_size, header_content


@p.do
def record(payload_size: int) -> Generator[p.Parser, Any, Record]:
    """Parse a record of a given size."""
    header_size, header_content = yield record_header()

    body_size = payload_size - header_size

    body_parser = parse_record_body(header_content)
    values = yield p.take(body_size, body_parser)

    return Record(serial_types=header_content, values=values)


@p.do
def table_interior_cell() -> Generator[p.Parser, Any, int]:
    """Parse a table interior cell to find the left-child pointer.

    An interior cell format is:
    - 4-byte page number (the left child pointer)
    - A varint for the integer key.

    We only need the page number for our traversal.
    """
    page_number = yield p.uint32_be
    _ = yield varint  # We don't need the key, but we must parse it to advance the stream.
    return page_number


@dataclass
class TableLeafCell:
    """A cell in a table leaf page."""

    rowid: int
    payload: Record


@p.do
def table_leaf_cell() -> Generator[p.Parser, Any, TableLeafCell]:
    """Parse a table leaf cell."""
    payload_size = yield varint
    row_id = yield varint

    # The payload of a table leaf cell is a record.
    # We can parse it directly since we know its size.
    payload = yield p.take(payload_size, record(payload_size))

    return TableLeafCell(rowid=row_id, payload=payload)


ColumnValue = int | None | bytes | str

# Mappings from serial type to a parser for that type.
# https://www.sqlite.org/fileformat.html#record_format
serial_type_parsers: dict[int, p.Parser[ColumnValue]] = {
    SERIAL_TYPE_NULL: p.map_p(lambda _: None, p.bytes_n(0)),  # NULL
    SERIAL_TYPE_INT8: p.map_p(
        lambda b: int.from_bytes(b, "big", signed=True),
        p.bytes_n(1),
    ),  # 8-bit integer
    SERIAL_TYPE_INT16: p.map_p(
        lambda b: int.from_bytes(b, "big", signed=True),
        p.bytes_n(2),
    ),  # 16-bit integer
    SERIAL_TYPE_INT24: p.map_p(
        lambda b: int.from_bytes(b, "big", signed=True),
        p.bytes_n(3),
    ),  # 24-bit integer
    SERIAL_TYPE_INT32: p.map_p(
        lambda b: int.from_bytes(b, "big", signed=True),
        p.bytes_n(4),
    ),  # 32-bit integer
    SERIAL_TYPE_INT48: p.map_p(
        lambda b: int.from_bytes(b, "big", signed=True),
        p.bytes_n(6),
    ),  # 48-bit integer
}


def parse_record_body(serial_types: list[int]) -> p.Parser[tuple[ColumnValue, ...]]:
    """Create a parser for a record's body based on its serial types."""
    parsers = []
    for st in serial_types:
        if st in serial_type_parsers:
            parsers.append(serial_type_parsers[st])
        elif st >= SERIAL_TYPE_BLOB_MIN and st % 2 != 0:
            # BLOB
            blob_len = (st - SERIAL_TYPE_BLOB_MIN) // 2
            parsers.append(p.bytes_n(blob_len))
        elif st >= SERIAL_TYPE_TEXT_MIN and st % 2 == 0:
            # TEXT
            text_len = (st - SERIAL_TYPE_TEXT_MIN) // 2
            parsers.append(p.map_p(lambda b: b.decode(), p.bytes_n(text_len)))
        else:
            return p.failure(f"Unknown serial type: {st}")

    return p.sequence(*parsers)


def parse_table_leaf_pages(
    page_num: int,
    page_size: int,
) -> p.Parser[Iterable[LeafPage]]:
    """Recursively traverse a B-Tree and return a list of all parsed leaf pages."""

    @p.do
    def _traverse(
        current_page_num: int,
    ) -> Generator[p.Parser, Any, Iterable[LeafPage]]:
        page_start = (current_page_num - 1) * page_size
        offset = 100 if current_page_num == 1 else 0
        yield p.seek(page_start + offset)

        page_type_val = yield p.peek(p.uint8)
        page_type = PageType(page_type_val)  # TODO: Create IntEnum parser

        if page_type == PageType.TABLE_LEAF:
            # This is a table leaf page, parse it and return.
            leaf_page = yield parse_leaf_page(offset, page_size)
            return [leaf_page]

        elif page_type == PageType.TABLE_INTERIOR:
            # This is a table interior page. Get the child pointers to recurse.
            child_page_pointers = yield parse_interior_page_child_pointers(offset)
            found_pages = []
            for child_page_num in child_page_pointers:
                child_results = yield _traverse(child_page_num)
                found_pages.extend(child_results)

            return found_pages
        else:
            # This is an index page or other type we don't care about.
            # Stop traversing this branch by returning an empty list.
            return []

    return _traverse(page_num)


@p.relocatable
@p.do
def parse_interior_page_child_pointers(
    offset: int,
) -> Generator[p.Parser, Any, list[int]]:
    """Parse an interior page to extract all child page pointers."""
    header = yield b_tree_interior_page_header

    cell_pointers = yield cell_pointer_array(header.cell_count)

    child_pages = []
    for cell_ptr_offset in cell_pointers:
        yield p.seek(cell_ptr_offset - offset)
        child_page_num = yield table_interior_cell()
        child_pages.append(child_page_num)

    # For interior pages, we also need to include the right-most pointer.
    child_pages.append(header.right_most_pointer)

    return child_pages


@dataclass
class LeafPage:
    """A leaf page in a B-Tree, containing a header and cells."""

    header: BTreeLeafPageHeader
    cells: p.Lazy[list[TableLeafCell]]


@p.relocatable
@p.do
def parse_leaf_page(
    offset: int,
    page_size: int,
) -> Generator[p.Parser, Any, LeafPage]:
    """Parse a leaf page, returning a `LeafPage` object with lazy-evaluated cells."""

    @p.do
    def parse_leaf_page_cells(
        cell_count: int,
    ) -> Generator[p.Parser, Any, list[TableLeafCell]]:
        cell_pointers = yield cell_pointer_array(cell_count)

        cells = []
        for cell_ptr_offset in cell_pointers:
            yield p.seek(cell_ptr_offset - offset)
            cell = yield table_leaf_cell()
            cells.append(cell)

        return cells

    header, header_size = yield p.with_bytes_read(b_tree_leaf_page_header)

    lazy_cells = yield p.lazy(
        page_size - header_size,
        lambda _: parse_leaf_page_cells(header.cell_count),
    )

    return LeafPage(header=header, cells=lazy_cells)
