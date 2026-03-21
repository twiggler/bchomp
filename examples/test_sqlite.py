"""Integration tests for the SQLite example parser."""

import itertools
import sqlite3
import string

import bchomp.parser as p
from examples.sqlite import LeafPage, SQLite, TableLeafCell

# With page_size=512, max_local = 512 - 35 = 477 bytes.
# A TEXT value of _LONG_SIZE chars produces a payload of ~503 bytes → overflow.
# A TEXT value of _SHORT_SIZE chars produces a payload of ~103 bytes → no overflow.
_PAGE_SIZE = 512
_SHORT_SIZE = 100
_LONG_SIZE = 500


def _make_db(*values: str) -> SQLite:
    """Create an in-memory SQLite DB with page_size=512 and one row per value."""
    conn = sqlite3.connect(":memory:")
    conn.execute(f"PRAGMA page_size = {_PAGE_SIZE}")
    conn.execute("CREATE TABLE t (v TEXT)")
    for value in values:
        conn.execute("INSERT INTO t VALUES (?)", (value,))
    conn.commit()
    data = bytes(conn.serialize())
    conn.close()
    return SQLite(p.BytesReader(data))


def _root_page(db: SQLite, table: str) -> int:
    """Return the root page number of *table* from sqlite_schema."""
    for page in db.read_table(1):
        if isinstance(page, LeafPage):
            for cell in page.cells:
                _, name, _, rootpage, _ = cell.payload.values
                if name == table:
                    return int(rootpage)  # type: ignore[arg-type]
    msg = f"{table!r} not found in sqlite_schema"
    raise AssertionError(msg)


def _collect_cells(db: SQLite, root_page: int) -> list[TableLeafCell]:
    """Collect all leaf cells reachable from *root_page*."""
    return [
        cell
        for page in db.read_table(root_page)
        if isinstance(page, LeafPage)
        for cell in page.cells
    ]


def _varied(n: int) -> str:
    """Return a string of length *n* cycling through ascii lowercase."""
    return "".join(itertools.islice(itertools.cycle(string.ascii_lowercase), n))


def test_no_overflow() -> None:
    """Payload fits on a single page — no overflow chain is followed."""
    value = _varied(_SHORT_SIZE)
    db = _make_db(value)
    cells = _collect_cells(db, _root_page(db, "t"))
    assert len(cells) == 1
    assert cells[0].payload.values[0] == value


def test_overflow() -> None:
    """Payload exceeds max_local for page_size=512 — overflow chain is followed."""
    value = _varied(_LONG_SIZE)
    db = _make_db(value)
    cells = _collect_cells(db, _root_page(db, "t"))
    assert len(cells) == 1
    assert cells[0].payload.values[0] == value


def test_mixed() -> None:
    """Both a non-overflowing and an overflowing cell are parsed correctly."""
    short = _varied(_SHORT_SIZE)
    long_ = _varied(_LONG_SIZE)
    db = _make_db(short, long_)
    cells = _collect_cells(db, _root_page(db, "t"))
    values = [cell.payload.values[0] for cell in cells]
    assert short in values
    assert long_ in values
