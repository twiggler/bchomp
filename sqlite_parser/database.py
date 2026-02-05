from dataclasses import dataclass
from sqlite_parser import parser as p
from functools import reduce
from typing import Any, Callable, Annotated, Generator, Iterable, Union, Iterator
from enum import IntEnum


class PageType(IntEnum):
    INDEX_INTERIOR = 2
    TABLE_INTERIOR = 5
    INDEX_LEAF = 10
    TABLE_LEAF = 13


@dataclass
class BTreePageHeaderBase:
    page_type: Annotated[int, p.uint8]
    freeblock_start: Annotated[int, p.uint16_be]
    cell_count: Annotated[int, p.uint16_be]
    cell_start: Annotated[int, p.uint16_be]
    fragmented_bytes: Annotated[int, p.uint8]


@dataclass
class BTreeInteriorPageHeader(BTreePageHeaderBase):
    right_most_pointer: Annotated[int, p.uint32_be]


BTreeLeafPageHeader = BTreePageHeaderBase


b_tree_interior_page_header: p.Parser[BTreeInteriorPageHeader] = p.create_parser_from_dataclass(BTreeInteriorPageHeader)
b_tree_leaf_page_header: p.Parser[BTreeLeafPageHeader] = p.create_parser_from_dataclass(BTreeLeafPageHeader)


def cell_pointer_array(cell_count: int) -> p.Parser[list[int]]:
    return p.count(cell_count, p.uint16_be)


def varint(state: p.ParserState) -> p.Result[int]:
    def process_bytes(parts):
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
    serial_types: list[int]
    values: list['ColumnValue']


@p.do
def record_header() -> Generator[p.Parser, Any, tuple[int, list[int]]]:
    """
    Parses the record header using do-notation for clarity.

    Returns a tuple containing:
    - The total size of the header in bytes (including the size varint).
    - A list of the serial types.
    """
    
    total_header_size, varint_size = yield p.with_bytes_read(varint)

    header_content_size = total_header_size - varint_size

    header_content = yield p.take(header_content_size, p.many(varint))

    return total_header_size, header_content


@p.do
def record(payload_size: int) -> Generator[p.Parser, Any, Record]:
    header_size, header_content = yield record_header()

    body_size = payload_size - header_size
    
    body_parser = parse_record_body(header_content)
    values = yield p.take(body_size, body_parser)
    
    return Record(serial_types=header_content, values=values)


@p.do
def table_interior_cell() -> Generator[p.Parser, Any, int]:
    """
    Parses a table interior cell to find the left-child pointer.
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
    rowid: int
    payload: Record

@p.do
def table_leaf_cell() -> Generator[p.Parser, Any, TableLeafCell]:
    payload_size = yield varint
    row_id = yield varint
    
    # The payload of a table leaf cell is a record.
    # We can parse it directly since we know its size.
    payload = yield p.take(payload_size, record(payload_size))
    
    return TableLeafCell(rowid=row_id, payload=payload)


ColumnValue = Union[int, None, bytes, str]

# Mappings from serial type to a parser for that type.
# https://www.sqlite.org/fileformat.html#record_format
serial_type_parsers: dict[int, p.Parser[ColumnValue]] = {
    0: p.map_p(lambda _: None, p.bytes_n(0)),  # NULL
    1: p.map_p(lambda b: int.from_bytes(b, 'big', signed=True), p.bytes_n(1)), # 8-bit integer
    2: p.map_p(lambda b: int.from_bytes(b, 'big', signed=True), p.bytes_n(2)), # 16-bit integer
    3: p.map_p(lambda b: int.from_bytes(b, 'big', signed=True), p.bytes_n(3)), # 24-bit integer
    4: p.map_p(lambda b: int.from_bytes(b, 'big', signed=True), p.bytes_n(4)), # 32-bit integer
    5: p.map_p(lambda b: int.from_bytes(b, 'big', signed=True), p.bytes_n(6)), # 48-bit integer
}


def parse_record_body(serial_types: list[int]) -> p.Parser[list[ColumnValue]]:
    """
    Creates a parser for a record's body based on its serial types.
    """
    parsers = []
    for st in serial_types:
        if st in serial_type_parsers:
            parsers.append(serial_type_parsers[st])
        elif st >= 13 and st % 2 != 0:
            # BLOB
            blob_len = (st - 13) // 2
            parsers.append(p.bytes_n(blob_len))
        elif st >= 12 and st % 2 == 0:
            # TEXT
            text_len = (st - 12) // 2
            parsers.append(p.map_p(lambda b: b.decode(), p.bytes_n(text_len)))
        else:
            return p.failure(f"Unknown serial type: {st}")

    return p.sequence(*parsers)

def parse_table_leaf_pages(page_num: int, page_size: int) -> p.Parser[Iterable[LeafPage]]:
    """
    A recursive parser that traverses a B-Tree and returns a list of
    all parsed TABLE_LEAF pages, ignoring index pages.
    """
    
    @p.do
    def _traverse(current_page_num: int) -> Generator[p.Parser, Any, Iterable[LeafPage]]:
        page_start = (current_page_num - 1) * page_size
        offset = 100 if current_page_num == 1 else 0
        yield p.seek(page_start + offset)

        page_type_val = yield p.peek(p.uint8)
        page_type = PageType(page_type_val) # TODO: Create IntEnum parser

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
def parse_interior_page_child_pointers(offset : int) -> Generator[p.Parser, Any, list[int]]:
    """
    Parses an interior page to extract all child page pointers.
    """
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
    header: BTreeLeafPageHeader
    cells: p.Lazy[list[TableLeafCell]]


@p.relocatable
@p.do
def parse_leaf_page(offset: int, page_size: int) -> Generator[p.Parser, Any, LeafPage]:
    """
    Parses a leaf page, returning a `LeafPage` object with a lazily-evaluated
    list of cells. This is a relocatable parser.
    """

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
