from typing import Any, Generator
import sqlite_parser.parser as p
from sqlite_parser.database import (
    table_leaf_cell,
    find_first_leaf_page,
    parse_leaf_page,
    get_page_offset,
)


@p.do
def example() -> Generator[p.Parser, Any, None]:
    """
    A test parser that demonstrates finding and parsing the first leaf page
    of a table in a SQLite database.
    """
    # Parse the initial 16-byte database header.
    magic = yield p.string("SQLite format 3\\0")
    print(f"Magic string: {magic}")

    page_size = yield p.uint16_be
    page_size = page_size if page_size != 1 else 65536
    print(f"Page size: {page_size}")

    # Start at page 1 (the root) and traverse the B-Tree to find the
    # page number of the first leaf page.
    print("\\n--- Traversing B-Tree to find first leaf page ---")
    yield p.seek_absolute(0)  # Ensure we start at the beginning of the file
    leaf_page_num = yield p.anchor(find_first_leaf_page(page_num=1, page_size=page_size))
    print(f"Found leaf page number: {leaf_page_num}")

    # Now that we have the leaf page number, parse its header and cell pointers.
    header, cell_pointers = yield parse_leaf_page(leaf_page_num, page_size)
    print(f"Leaf page header: {header}")
    print(f"Leaf page cell pointers: {cell_pointers}")

    # We need to re-seek to the start of the page and then anchor to parse the cell
    # Workaround for now until we refactor parse_leaf_page to return a parser for cells.
    leaf_page_start = get_page_offset(leaf_page_num, page_size)
    yield p.seek_absolute(leaf_page_start + cell_pointers[0])

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
