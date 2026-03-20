"""Parsers for the SQLite database file format structures."""

from dataclasses import dataclass
from enum import IntEnum
from functools import reduce
from typing import TYPE_CHECKING, Annotated, Any

import bchomp.parser as p
import bchomp.transformers as t
from bchomp.adapters.lazy import Lazy, lazy_unbounded
from bchomp.compose import compose

if TYPE_CHECKING:
    from collections.abc import Generator, Iterator


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
class Header:
    """Minimal representation required for parsing pages."""

    page_size: int


class SQLite:
    """Provide a facade for parsing SQLite database files."""

    _reader: p.Readable
    header: Header

    def __init__(self, reader: p.Readable) -> None:
        self._reader = reader
        self.header = p.run_parser(read_header(), self._reader)

    def read_table(self, start_page_num: int) -> Iterator[Page]:
        """Return an iterator over all pages reachable from the given start page number."""
        table_parser = read_table(start_page_num, self.header.page_size)
        return p.stream(table_parser, self._reader)


@dataclass
class TableLeafCell:
    """A cell in a table leaf page."""

    rowid: int
    payload: Record


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


@dataclass
class InteriorPage:
    """An interior page in a B-Tree, containing a header and child page pointers."""

    header: InteriorPageHeader
    child_page_numbers: list[int]


@dataclass
class LeafPage:
    """A leaf page in a B-Tree, containing a header and cells."""

    header: LeafPageHeader
    _lazy_cells: Lazy[list[TableLeafCell]]

    @property
    def cells(self) -> list[TableLeafCell]:
        """Access the cells of the leaf page, triggering lazy parsing if necessary."""
        return self._lazy_cells.value


Page = InteriorPage | LeafPage


@p.do
def read_header() -> p.BlockingScript[Header]:
    """Parse minimal SQLite database header."""
    yield p.string("SQLite format 3\0")
    page_size = yield p.map_p(lambda v: v if v != 1 else 65536, p.uint16_be)
    return Header(page_size=page_size)


@p.do
def read_interior_page() -> p.BlockingScript[InteriorPage]:
    """Parse an interior page, returning an `InteriorPage` object with child page numbers."""
    header: InteriorPageHeader = yield read_interior_page_header
    cell_pointers = yield read_cell_pointer_array(header.cell_count)
    child_pages = yield p.gather(cell_pointers, read_table_interior_cell())
    return InteriorPage(header, child_pages)


@p.do
def read_leaf_page(
    page_size: int,
) -> p.BlockingScript[LeafPage]:
    """Parse a leaf page, returning a `LeafPage` object with lazy-evaluated cells."""

    @p.do
    def read_leaf_page_cells(
        cell_count: int,
    ) -> p.BlockingScript[list[TableLeafCell]]:
        cell_pointers = yield read_cell_pointer_array(cell_count)
        cells = yield p.gather(cell_pointers, read_table_leaf_cell(page_size))
        return cells

    header: LeafPageHeader = yield read_leaf_page_header
    lazy_cells = yield lazy_unbounded(read_leaf_page_cells(header.cell_count))

    return LeafPage(header=header, _lazy_cells=lazy_cells)


@p.do
def read_page(page_num: int, page_size: int) -> p.BlockingScript[Page]:
    """Parse a B-Tree page at a given index (1-based)."""
    page_start = (page_num - 1) * page_size
    yield p.seek(page_start)
    offset = HEADER_SIZE if page_num == 1 else 0

    # Shift parser if this is the first page, to skip the database header.
    page_type = yield compose(parse_page_type, p.with_relocation, p.peek, t.at(offset))
    match page_type:
        case PageType.TABLE_LEAF:
            # No take: leaf cells may span overflow pages, so the full database
            # readable must remain visible inside read_leaf_page.
            page = yield compose(read_leaf_page(page_size), p.with_relocation, t.at(offset))
        case PageType.TABLE_INTERIOR:
            page = yield compose(read_interior_page(), t.take(page_size), t.at(offset))
        case _:
            msg = f"Unsupported page type: {page_type}"
            raise p.ParseError(msg)

    return page


ColumnValue = int | None | bytes | str | float


def read_varint(state: p.ParserState) -> p.BlockingResult[int]:
    """Parse a variable-length integer (varint)."""

    def process_bytes(parts: tuple[list[int], int]) -> int:
        head, tail = parts
        initial_value = reduce(lambda acc, byte: (acc << 7) | (byte & 0x7F), head, 0)
        return (initial_value << 7) | (tail & 0x7F)

    continuation_byte = p.satisfy(lambda b: (b & 0x80) != 0)
    final_byte = p.satisfy(lambda b: (b & 0x80) == 0)
    parser = p.sequence((p.many(continuation_byte), final_byte))

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

    values = yield p.take(body_size, read_record_body(header_content))

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
def read_fragments(
    payload_size: int,
    page_size: int,
) -> p.Script[None]:
    """Yield (offset, size) tuples covering the full payload, following overflow pages.

    Obtains the current stream position and computes the local payload size
    internally.
    """
    local_start = yield p.position()
    local = local_payload_size(payload_size, PageType.TABLE_LEAF, page_size) or payload_size
    yield p.emit((local_start, local))
    remaining = payload_size - local
    if remaining == 0:
        return
    yield p.seek(local_start + local)
    next_page_num = yield p.uint32_be
    while next_page_num != 0:
        data_offset = (next_page_num - 1) * page_size + _OVERFLOW_PAGE_HEADER
        data_size = min(page_size - _OVERFLOW_PAGE_HEADER, remaining)
        yield p.emit((data_offset, data_size))
        remaining -= data_size
        if remaining == 0:
            break
        yield p.seek((next_page_num - 1) * page_size)
        next_page_num = yield p.uint32_be


@p.do
def read_table_leaf_cell(
    page_size: int,
) -> p.BlockingScript[TableLeafCell]:
    """Parse a table leaf cell, following overflow pages when the payload spills."""
    payload_size = yield read_varint
    row_id = yield read_varint
    payload = yield p.linearize(
        p.absolute(read_fragments(payload_size, page_size)),
        read_record(payload_size),
    )
    return TableLeafCell(rowid=row_id, payload=payload)


# Structural overhead constants (SQLite file format spec §2.3.3).
_OVERFLOW_PAGE_HEADER = 4  # next-page pointer at the start of each overflow page
_PAGE_HEADER = 8  # b-tree page header bytes
_MIN_LOCAL_BYTES = 3  # SQLite requires at least 3 bytes stored on the b-tree page
_CELL_POINTER = 2  # cell-pointer array entry
_MAX_VARINT = 9  # maximum bytes in a varint

# Bytes subtracted from U before applying the 32/255 or 64/255 fill fraction.
_PAGE_FORMULA_OVERHEAD = _PAGE_HEADER + _OVERFLOW_PAGE_HEADER  # 12

# Maximum per-cell overhead for a table-leaf (payload + rowid varints,
# cell pointer, overflow pointer, minimum local bytes).
_TABLE_LEAF_CELL_OVERHEAD = (
    _PAGE_HEADER + _CELL_POINTER + 2 * _MAX_VARINT + _OVERFLOW_PAGE_HEADER + _MIN_LOCAL_BYTES
)  # 35

# Per-cell overhead used in the index / shared min_local formula.
_INDEX_CELL_OVERHEAD = _CELL_POINTER + 2 * _MAX_VARINT + _MIN_LOCAL_BYTES  # 23


def local_payload_size(
    payload_size: int,
    page_type: PageType,
    usable_size: int,
) -> int | None:
    """Return the local payload byte count when a cell overflows, or None if no overflow.

    SQLite stores as much payload as possible on the b-tree page and spills
    the rest to a chain of overflow pages.  The number of bytes kept locally
    depends on the page type and the usable page size (page size minus the
    reserved region).

    Returns:
        None  — the entire payload fits on the page; no overflow pointer follows.
        int   — number of bytes stored locally; a 4-byte overflow page number
                immediately follows the local payload.

    """
    min_local = ((usable_size - _PAGE_FORMULA_OVERHEAD) * 32 // 255) - _INDEX_CELL_OVERHEAD
    match page_type:
        case PageType.TABLE_LEAF:
            max_local = usable_size - _TABLE_LEAF_CELL_OVERHEAD
        case PageType.INDEX_LEAF | PageType.INDEX_INTERIOR:
            max_local = ((usable_size - _PAGE_FORMULA_OVERHEAD) * 64 // 255) - _INDEX_CELL_OVERHEAD
        case _:
            msg = f"Page type {page_type} does not carry payload"
            raise ValueError(msg)

    if payload_size <= max_local:
        return None

    local = min_local + ((payload_size - min_local) % (usable_size - _OVERFLOW_PAGE_HEADER))
    return min_local if local > max_local else local


def read_column_value(st: SerialKind) -> p.BlockingParser[ColumnValue]:  # noqa: C901, PLR0911, PLR0912
    """Return a parser for a single column value described by the serial kind."""
    match st:
        case Blob(size=size):
            return p.bytes_n(size)
        case Text(size=size):
            return p.map_p(lambda b: b.decode(), p.bytes_n(size))
        case SerialType.NULL:
            return p.pure(None)
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


def read_record_body(
    serial_types: list[SerialKind],
) -> p.BlockingParser[tuple[ColumnValue, ...]]:
    """Create a parser for a record's body based on its serial types.

    `serial_types` is a list of `SerialKind` items (either a `SerialType`
    enum member, or a sized `Blob`/`Text`). This function builds a sequence
    parser composed from the appropriate primitive parsers for each
    serial-kind.
    """
    parsers = [read_column_value(st) for st in serial_types]
    return p.sequence(parsers)


def read_table(start_page_num: int, page_size: int) -> p.StreamingParser[Page]:
    """Recursively traverse the B-Tree and stream all pages reachable from the given start page."""

    @p.do
    def _traverse(current_page_num: int) -> p.Script:
        page: Page = yield read_page(current_page_num, page_size)
        yield p.emit(page)
        if isinstance(page, InteriorPage):
            for child_page_num in page.child_page_numbers:
                yield _traverse(child_page_num)
            yield _traverse(page.header.right_most_pointer)

    return _traverse(start_page_num)
