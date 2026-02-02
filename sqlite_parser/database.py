from dataclasses import dataclass
from sqlite_parser import parser as p
from functools import reduce
from typing import Any, Protocol, Callable, Annotated, Generator, Union

@dataclass
class BTreeInteriorPageHeader:
    page_type: Annotated[int, p.uint8]
    freeblock_start: Annotated[int, p.uint16_be]
    cell_count: Annotated[int, p.uint16_be]
    cell_start: Annotated[int, p.uint16_be]
    fragmented_bytes: Annotated[int, p.uint8]
    right_most_pointer: Annotated[int, p.uint32_be]

@dataclass
class BTreeLeafPageHeader:
    page_type: Annotated[int, p.uint8]
    freeblock_start: Annotated[int, p.uint16_be]
    cell_count: Annotated[int, p.uint16_be]
    cell_start: Annotated[int, p.uint16_be]
    fragmented_bytes: Annotated[int, p.uint8]

BTreePageHeader = Union[BTreeInteriorPageHeader, BTreeLeafPageHeader]

b_tree_interior_page_header: p.Parser[BTreeInteriorPageHeader] = p.create_parser_from_dataclass(BTreeInteriorPageHeader)
b_tree_leaf_page_header: p.Parser[BTreeLeafPageHeader] = p.create_parser_from_dataclass(BTreeLeafPageHeader)

def b_tree_page_header(stream: p.Stream) -> p.Result[BTreePageHeader]:
    """
    Parses a B-Tree page header by first peeking at the page type
    and then choosing the appropriate parser.
    """
    
    @p.do
    def _parse_header() -> Generator[p.Parser, Any, BTreePageHeader]:
        page_type = yield p.peek(p.uint8)

        if page_type == 5:
            header = yield b_tree_interior_page_header
            return header
        else:
            # Assume leaf page for any other type for now
            header = yield b_tree_leaf_page_header
            return header

    return _parse_header()(stream)


def cell_pointer_array(cell_count: int) -> p.Parser[list[int]]:
    return p.count(cell_count, p.uint16_be)

def varint(stream: p.Stream) -> p.Result[int]:
    def process_bytes(parts):
        head, tail = parts
        initial_value = reduce(lambda acc, byte: (acc << 7) | (byte & 0x7F), head, 0)
        return (initial_value << 7) | (tail & 0x7F)

    continuation_byte = p.satisfy(lambda b: (b & 0x80) != 0)
    final_byte = p.satisfy(lambda b: (b & 0x80) == 0)
    parser = p.sequence(p.many(continuation_byte), final_byte)

    # TODO: check case where 9 bytes are read
    return p.map_p(process_bytes, parser)(stream)

@dataclass
class Record:
    serial_types: list[int]
    values: list[ColumnValue]

@p.do
def record_header() -> Generator[p.Parser, Any, tuple[int, list[int]]]:
    """
    Parses the record header using do-notation for clarity.

    Returns a tuple containing:
    - The total size of the header in bytes (including the size varint).
    - A list of the serial types.
    """
    start_pos = yield p.position()
    total_header_size = yield varint
    end_pos = yield p.position()

    varint_size = end_pos - start_pos
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
    return page_number


class TableLeafCell(Protocol):
    rowid: int
    payload: Record

def create_table_leaf_cell_parser() -> Callable[[], p.Parser[TableLeafCell]]:
    LazyTableCell = p.make_lazy(TableLeafCell, lazy_fields=["payload"])

    @p.do
    def table_leaf_cell() -> Generator[p.Parser, Any, TableLeafCell]:
        size = yield varint

        row_id = yield varint
        
        lazy_cell_payload_value = yield p.lazy(size, record)
        
        return LazyTableCell(rowid=row_id, _payload_lazy=lazy_cell_payload_value)
    
    return table_leaf_cell

table_leaf_cell = create_table_leaf_cell_parser()


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
            raise ValueError(f"Unknown serial type: {st}")

    return p.sequence(*parsers)

def get_page_offset(page_num: int, page_size: int) -> int:
    """Calculates the absolute byte offset of a given page number."""
    if page_num <= 0:
        raise ValueError(f"Invalid page number: {page_num}")
    offset = (page_num - 1) * page_size
    if page_num == 1:
        offset += 100   # TODO: Replace with header size constant
    return offset


def find_first_leaf_page(page_num: int, page_size: int) -> p.Parser[int]:
    """
    A recursive, composable parser that traverses a B-Tree to find the page
    number of the first leaf page.
    """
    
    @p.do
    def _traverse() -> Generator[p.Parser, Any, int]:
        page_start = get_page_offset(page_num, page_size)
        
        yield p.seek(page_start)

        # All operations for parsing this page are now done inside a *new*
        # nested anchor, making them relative to the start of the page.
        with (yield p.anchor_cm()):
            result = yield _parse_page_for_leaf(page_size, page_num)
            return result

    return _traverse()


@p.do
def _parse_page_for_leaf(page_size: int, page_num: int) -> Generator[p.Parser, Any, int]:
    """
    Helper parser that operates within a nested anchor set to the start of a page.
    All seeks are relative to the start of the current page.
    Returns the page number of a leaf page.
    """
    page_type = yield p.peek(p.uint8)

    if page_type == 10: # TODO: Make Table Leaf index constant
        return page_num
    elif page_type == 5: # TODO: Make Table Interior index constant
        # This is an interior page, so we parse its header to find the
        # left-most child pointer and continue traversal.
        header = yield b_tree_interior_page_header
        
        cell_pointers = yield cell_pointer_array(header.cell_count)
        
        # The first cell pointer gives the location of the left-most child cell,
        # relative to the start of the page (our current anchor).
        first_cell_ptr_offset = cell_pointers[0]
        
        yield p.seek(first_cell_ptr_offset)
        child_page_num = yield table_interior_cell()
        
        # Recursively call the traversal parser. Since we are no longer in the
        # nested page anchor, this will operate relative to the file anchor.
        result = yield find_first_leaf_page(child_page_num, page_size)
        return result
    else:
        # We are assuming we only encounter table leaf or interior pages.
        # Other page types like index pages would need to be handled here.
        raise TypeError(f"Unsupported page type for traversal: {page_type}")


@p.do
def parse_leaf_page(page_num: int, page_size: int) -> Generator[p.Parser, Any, tuple[BTreeLeafPageHeader, list[int]]]:
    """
    Parses a given leaf page, returning its header and cell pointers.
    Assumes it is running in a file-level anchor context.
    """
    page_start = get_page_offset(page_num, page_size)
    yield p.seek(page_start)

    with (yield p.anchor_cm()):
        header = yield b_tree_leaf_page_header
        cell_pointers = yield cell_pointer_array(header.cell_count)
        return header, cell_pointers
