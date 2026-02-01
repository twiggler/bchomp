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


@p.parser
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

@dataclass
class BTreePage:
    """Represents a parsed B-Tree page, containing its header and starting position."""
    header: BTreePageHeader
    page_start: int

def find_first_leaf_page(page_num: int, page_size: int) -> p.Parser[BTreePage]:
    """
    A recursive parser that traverses a B-Tree to find the first leaf page.

    Starting from the given `page_num`, it checks the page type. If it's an
    interior page, it finds the left-most child pointer and recurses. If it's
    a leaf page, it returns the parsed page.
    """
    
    @p.do
    def _traverse() -> Generator[p.Parser, Any, BTreePage]:
        if page_num <= 0:
            raise ValueError(f"Invalid page number: {page_num}")

        page_start = (page_num - 1) * page_size
        offset = 100 if page_num == 1 else 0
        
        # TODO: allowing absolute seeks breaks composability.
        # Having to store page_start is in BTreePage is also a workaround.
        # Support a stack of anchors to create a local frame of reference for seeks? 
        yield p.seek(page_start + offset)
        
        header = yield b_tree_page_header

        if isinstance(header, BTreeLeafPageHeader):
            return BTreePage(header=header, page_start=page_start)

        elif isinstance(header, BTreeInteriorPageHeader):
            # This is an interior page, we need to go deeper.
            # The cell pointers are right after the header.
            cell_pointers = yield cell_pointer_array(header.cell_count)
            
            # The first cell pointer gives the location of the left-most child cell.
            first_cell_ptr_offset = cell_pointers[0]
            
            # Seek to that cell's location to parse the child page number.
            yield p.seek(page_start + first_cell_ptr_offset)
            child_page_num = yield table_interior_cell()
            
            # Recursively call the traversal parser on the child page.
            result = yield find_first_leaf_page(child_page_num, page_size)
            return result
        else:
            # This should not happen with the current header parsing logic
            raise TypeError(f"Unknown header type: {type(header)}")

    return _traverse()
