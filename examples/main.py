"""A simple SQLite database parser."""

from pathlib import Path

from bchomp.reader import BinaryIOReader
from examples.sqlite import (
    LeafPage,
    SQLite,
)


def dump_leaf_page(leaf_page: LeafPage) -> None:
    """Print the contents of a parsed leaf page from the SQLite schema table."""
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
        database = SQLite(BinaryIOReader(f))
        for page in database.read_table(start_page_num=1):
            if isinstance(page, LeafPage):
                dump_leaf_page(page)


if __name__ == "__main__":
    main()
