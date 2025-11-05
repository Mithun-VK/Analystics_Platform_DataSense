"""
Utilities Package.

Common utility functions and helpers used across the application.
Provides file operations, validation, formatting, and general-purpose helpers.
"""

from app.utils.helpers import (
    format_bytes,
    format_number,
    format_percentage,
    truncate_string,
    generate_random_string,
    sanitize_input,
    parse_date,
    is_valid_email,
    is_valid_url,
    calculate_age,
    get_client_ip,
    timing_decorator,
    retry_decorator,
    validate_file_extension,
    get_mime_type,
)

from app.utils.file_utils import (
    ensure_directory,
    get_file_size,
    get_file_hash,
    delete_file,
    delete_directory,
    copy_file,
    move_file,
    list_files,
    compress_file,
    decompress_file,
    read_json_file,
    write_json_file,
    read_csv_chunks,
    save_uploaded_file,
    cleanup_old_files,
)


__all__ = [
    # Formatting helpers
    "format_bytes",
    "format_number",
    "format_percentage",
    "truncate_string",
    
    # String utilities
    "generate_random_string",
    "sanitize_input",
    
    # Validation
    "is_valid_email",
    "is_valid_url",
    "validate_file_extension",
    
    # Date utilities
    "parse_date",
    "calculate_age",
    
    # Request utilities
    "get_client_ip",
    "get_mime_type",
    
    # Decorators
    "timing_decorator",
    "retry_decorator",
    
    # File operations
    "ensure_directory",
    "get_file_size",
    "get_file_hash",
    "delete_file",
    "delete_directory",
    "copy_file",
    "move_file",
    "list_files",
    "compress_file",
    "decompress_file",
    "read_json_file",
    "write_json_file",
    "read_csv_chunks",
    "save_uploaded_file",
    "cleanup_old_files",
]

# Package version
__version__ = "1.0.0"
