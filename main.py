from typing import Any, Generator
import sqlite_parser.parser as p
from sqlite_parser.database import (
    traverse_and_parse_leaf_pages,
)


@p.do
def example() -> Generator[p.Parser, Any, None]:
    """
    A test parser that demonstrates finding and parsing all leaf pages
    of the schema table in a SQLite database.
    """
    # Parse the initial 16-byte database header.
    magic_string = yield p.string("SQLite format 3\0")
    print(f"Magic string: {magic_string}")

    page_size = yield p.uint16_be
    page_size = page_size if page_size != 1 else 65536
    print(f"Page size: {page_size}")

    # Start at page 1 (the root) and traverse the B-Tree to find all leaf pages.
    print("\\n--- Traversing B-Tree to find and parse all leaf pages ---")
    yield p.seek_absolute(0)
    
    leaf_pages = yield traverse_and_parse_leaf_pages(page_num=1, page_size=page_size)

    for leaf_page in leaf_pages:
        print(f"Found leaf page with header: {leaf_page.header}")
        
        # The cells are lazy, so accessing them will trigger parsing.
        print("  --- Evaluating lazy cells ---")
        for cell in leaf_page.cells.value:
            # The payload of the cell is a Record object.
            # For the sqlite_schema table, the columns are:
            # type, name, tbl_name, rootpage, sql
            record = cell.payload
            
            # We can now inspect the values
            if record.values:
                schema_type, name, tbl_name, rootpage, sql = record.values
                print(f"    - Schema Object: type={schema_type}, name={name}, tbl_name={tbl_name}, rootpage={rootpage}")
                print(f"      SQL: {sql}")

    return None


def main():
    with open("chinook.db", "rb") as f:
        reader = p.BinaryIOReader(f)
        result = p.run_parser(example(), reader)
        if isinstance(result, p.Failure):
            print(f"Parsing failed: {result.message} at pos {result.state.pos}")
        else:
            print("\\nParsing successful.")

if __name__ == "__main__":
    main()
