# bchomp

![bchomp logo](logo800x436.png)

## Project Overview

`bchomp` is a lightweight parser-combinator library for declarative binary parsing in Python. Its primary goal is to make it easy to build small, composable parsers that are readable and testable for large binary formats. The library emphasizes:

- composability of primitive parsers into higher-level grammars,
- relocatability so parsers can be reused without global seeks,
- and lazy parsing so expensive payloads are only decoded when needed.

The SQLite file parser at `examples/sqlite.py` is included as a non-core example that demonstrates how to apply `bchomp` to a real-world binary format: B-Tree traversal, record decoding, and on-demand payload parsing. The example is valuable as a usage demonstration, but the library itself is intended to be format-agnostic and reusable for any binary format.

### Key Features & Concepts

- **Parser-Combinator Library (`bchomp/parser.py`):** Core of the project — a small, functional toolkit of primitives and combinators (e.g., `sequence`, `choice`, `many`, `map_p`, `satisfy`) and helpers for working with stream positions (`seek`, `position`).
- **Lazy Evaluation:** Built-in `lazy`/deferred parsers allow expensive decoding to be postponed until the value is actually needed, enabling fast initial scans of large files.
- **Relocatable Parsers:** Support for `with_relocation` and the `@relocatable` decorator makes parsers composable without relying on global file offsets.
- **Declarative Example (`examples/sqlite.py`):** A comprehensive example showing how `bchomp` can be used to parse SQLite files; it demonstrates B-Tree traversal, serial-type handling, and record payload decoding, but it is presented as an application of the library rather than the library itself.

## Analysis

### Pros
- **Educational:** An excellent tool for learning about parser design, functional programming concepts, and the internals of the SQLite file format.
- **Declarative & Readable:** The grammar is easy to read and maintain because the code structure directly maps to the data structure being parsed. It communicates the *process* of parsing, not just the *layout* of the data.
- **Efficient Lazy Loading:** Avoids parsing data until it is needed, which is a sophisticated and efficient approach for handling large data files.

### Cons
- **Performance:** As a pure Python implementation running on CPython, it will be significantly slower than production-grade parsers written in systems languages like C or Rust. This is expected for a project of this nature.
- **Error Reporting:** Error messages are currently basic. A production-ready parser would require more sophisticated error reporting to provide better context on failures.

## Roadmap

- **`dissect.cstruct` Adapter (implemented):** A small adapter has been added to bridge `dissect.cstruct` types to `bchomp` parsers. See `bchomp/adapters/cstruct.py` for `from_cstruct(...)`, which wraps a cstruct-style reader as a `bchomp` parser. This lets you reuse compiled cstruct readers (zero-copy where possible) while keeping the combinator-based control flow.

## Quickstart

First, ensure you have Python 3.14+ installed.

1.  **Set up a virtual environment (optional but recommended):**
    ```bash
    python3 -m venv .venv
    source .venv/bin/activate
    ```

2.  **Run the parser:**
    The main entrypoint is located at `examples/main.py` and reads the example database at `examples/data/chinook.db`. Run it as a module so imports resolve correctly:
    ```bash
    python3 -m examples.main
    ```

## Files of Interest

- `examples/main.py`: The main entrypoint that runs the parser.
- `bchomp/parser.py`: The core parser-combinator library.
- `examples/sqlite.py`: The SQLite grammar example that uses `bchomp`.
- `examples/data/chinook.db`: A sample SQLite database file for parsing.
- `bchomp/adapters/cstruct.py`: Adapter that converts cstruct readers into `bchomp` parsers (`from_cstruct`).