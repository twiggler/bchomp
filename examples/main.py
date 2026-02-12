"""A simple SQLite database parser."""

from pathlib import Path
from typing import TYPE_CHECKING, Any

import bchomp.parser as p
from bchomp.adapters.cstruct import from_cstruct
from examples.c_header import c_header
from examples.sqlite import (
    LeafPage,
    read_parse_table_leaf_pages,
)

if TYPE_CHECKING:
    from collections.abc import Generator, Iterable


@p.do
def parse_schema_table() -> Generator[p.Parser, Any, Iterable[LeafPage]]:
    """Traverse the B-Tree of a SQLite database.

    Returns:
        A list of all leaf pages from the master table.

    """
    # The first 16 bytes of the database file is the magic string
    yield p.string("SQLite format 3\0")
    # The next 2 bytes are the page size
    page_size_val = yield p.uint16_be
    # A page size of 1 means 65536
    page_size = page_size_val if page_size_val != 1 else 65536

    # Alternatively, we can use cstruct to parse the header:
    yield p.seek(0)
    header = yield from_cstruct(c_header.header)
    if header.schema_format_number != 1:
        yield p.failure(f"Unsupported schema format number: {header.schema_format_number}")

    # The schema is on page 1, so we start there.
    yield p.seek(0)

    leaf_pages = yield read_parse_table_leaf_pages(page_num=1, page_size=page_size)
    return leaf_pages


def print_schema_results(leaf_pages: Iterable[LeafPage]) -> None:
    """Iterate through parsed leaf pages and prints their contents."""
    for leaf_page in leaf_pages:
        print(f"Found leaf page with header: {leaf_page.header}")

        # The cells are lazy, so accessing them will trigger parsing.
        for cell in leaf_page.cells:
            # The payload of the cell is a Record object.
            # For the sqlite_schema table, the columns are:
            # type, name, tbl_name, rootpage, sql
            record = cell.payload

            # We can now inspect the values
            if record.values:
                schema_type, name, tbl_name, rootpage, sql = record.values
                print(
                    f"    - Schema Object: type={schema_type}, name={name}, "
                    f"tbl_name={tbl_name}, rootpage={rootpage}",
                )
                print(f"      SQL: {sql}")


def main() -> None:
    """Parse the schema of a SQLite database and print the results."""
    db_path = Path(__file__).parent / "data" / "chinook.db"
    with db_path.open("rb") as f:
        reader = p.BinaryIOReader(f)
        result = p.run_parser(parse_schema_table(), reader)
        if isinstance(result, p.Failure):
            print(f"Parsing failed: {result.message} at pos {result.state.pos}")
        else:
            # Pass the successful result to the printing function
            print_schema_results(result.value)


if __name__ == "__main__":
    main()
