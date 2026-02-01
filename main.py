from typing import Any, Generator
import sqlite_parser.parser as p
from sqlite_parser.database import (
    cell_pointer_array,
    table_leaf_cell,
    find_first_leaf_page,
)


@p.do
def example() -> Generator[p.Parser, Any, None]:
    """
    A test parser that demonstrates finding and parsing the first leaf page
    of a table in a SQLite database (messy).
    """
    # Parse the initial 16-byte database header.
    magic = yield p.string("SQLite format 3\\0")
    print(f"Magic string: {magic}")

    page_size = yield p.uint16_be
    page_size = page_size if page_size != 1 else 65536  # TODO: move this to a parser
    print(f"Page size: {page_size}")

    # Start at page 1 (the root) and traverse the B-Tree to find the
    # first leaf page.
    print("\\n--- Traversing B-Tree to find first leaf page ---")
    leaf_page = yield find_first_leaf_page(page_num=1, page_size=page_size)
    print(f"Found leaf page! Header: {leaf_page.header}")

    cell_pointers = yield cell_pointer_array(leaf_page.header.cell_count)
    print(f"Leaf page cell pointers: {cell_pointers}")

    # Seek to the first cell's location and parse it.
    first_cell_offset = leaf_page.page_start + cell_pointers[0]
    yield p.seek(first_cell_offset)

    first_cell = yield table_leaf_cell()
    print(f"First cell (before evaluation): {first_cell}")

    # Trigger the lazy evaluation of the payload.
    print("\\n--- Evaluating lazy payload ---")
    evaluated_payload = first_cell.payload
    print(f"Evaluated Payload: {evaluated_payload}")
    print(f"Record Values: {evaluated_payload.values}")


def main():
    with open("chinook.db", "rb") as f:
        reader = p.BinaryIOReader(f)
        result = p.run_parser(example(), reader)
        if isinstance(result, p.Failure):
            print(f"\\nParser failed: {result}")

if __name__ == "__main__":
    main()
