# SQLite Parser

A toy SQLite database file parser written in Python using a custom parser-combinator library. This project is an exploration of binary parsing, lazy evaluation, and the elegance of functional parsing techniques.

## Project Overview

This parser is designed to read a SQLite database file (like the included `chinook.db`) and traverse its B-Tree structure to find and parse records. It is built from scratch with a custom parser-combinator library.

### Key Features & Concepts

- **Parser-Combinator Library (`parser.py`):** A small, functional library for building complex parsers from simple ones. It includes basic combinators like `sequence`, `choice`, `many`, and `map_p`, as well as file-positioning helpers like `seek` and `position`.
- **Lazy Evaluation:** The parser uses a `lazy` combinator to defer the parsing of expensive table record payloads. The payload is only parsed from the byte stream when its value is explicitly requested, making initial loading very fast.
- **Declarative Grammar (`database.py`):** The SQLite file format grammar is defined declaratively using the combinator library. This makes the code highly readable and closely mirrors the official SQLite documentation.
- **B-Tree Traversal:** The parser can walk the database's B-Tree structure, starting from the root page, to locate the first leaf page where table data is stored.
- **Composable, Relocatable Parsers:** To solve the problem of absolute seeks breaking parser composability, the library uses a relocation system. The `with_relocation` combinator establishes a new local frame of reference, allowing all subsequent `seek` calls to be relative to that point. Alternatively, parsers can be decorated with `@relocatable`, which does the same.

## Analysis

### Pros
- **Educational:** An excellent tool for learning about parser design, functional programming concepts, and the internals of the SQLite file format.
- **Declarative & Readable:** The grammar is easy to read and maintain because the code structure directly maps to the data structure being parsed. It communicates the *process* of parsing, not just the *layout* of the data.
- **Efficient Lazy Loading:** Avoids parsing data until it is needed, which is a sophisticated and efficient approach for handling large data files.

### Cons
- **Performance:** As a pure Python implementation running on CPython, it will be significantly slower than production-grade parsers written in systems languages like C or Rust. This is expected for a project of this nature.
- **Error Reporting:** Error messages are currently basic. A production-ready parser would require more sophisticated error reporting to provide better context on failures.

## Roadmap

- **`dissect.cstruct` Adapters:** To dramatically improve performance, we plan to introduce adapters that leverage the `dissect.cstruct` library for parsing binary data. This will allow us to replace pure Python parsing functions with highly optimized implementations where possible, while still retaining the declarative, high-level structure of the parser-combinator library.

## Quickstart

First, ensure you have Python 3.10+ installed.

1.  **Set up a virtual environment (optional but recommended):**
    ```bash
    python3 -m venv .venv
    source .venv/bin/activate
    ```

2.  **Run the parser:**
    The main entrypoint reads the `chinook.db` file and demonstrates traversing the B-Tree to find and parse the first record in the first leaf page.
    ```bash
    python3 main.py
    ```

## Files of Interest

- `main.py`: The main entrypoint that runs the parser.
- `sqlite_parser/parser.py`: The core parser-combinator library.
- `sqlite_parser/database.py`: The grammar definition for the SQLite file format.
- `chinook.db`: A sample SQLite database file for parsing.