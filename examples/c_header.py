"""Parse and define the SQLite3 file header structure using dissect.cstruct."""

from dissect.cstruct import cstruct

sqlite3_def = """
struct header {
    char    magic[16];
    uint16  page_size;
    uint8   write_version;
    uint8   read_version;
    uint8   reserved_size;
    uint8   max_embedded_payload_fraction;
    uint8   min_embedded_payload_fraction;
    uint8   leaf_payload_fraction;
    uint32  change_counter;
    uint32  page_count;
    uint32  first_freelist_page;
    uint32  freelist_page_count;
    uint32  schema_cookie;
    uint32  schema_format_number;
    uint32  page_cache_size;
    uint32  largest_root_btree_page;
    uint32  text_encoding;
    uint32  user_version;
    uint32  incremental_vacuum_mode;
    uint32  application_id;
    char    reserved1[20];
    uint32  version_valid_for_number;
    uint32  sqlite_version_number;
};
"""

c_header = cstruct(endian=">").load(sqlite3_def)
