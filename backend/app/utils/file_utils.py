"""
File Operation Utilities.

Functions for file system operations, file management, and I/O.
"""

import os
import shutil
import hashlib
import json
import gzip
import logging
from pathlib import Path
from typing import Optional, List, Iterator, Any
from datetime import datetime, timedelta

import pandas as pd
from fastapi import UploadFile


logger = logging.getLogger(__name__)


# ============================================================
# DIRECTORY OPERATIONS
# ============================================================

def ensure_directory(path: str | Path) -> Path:
    """
    Ensure directory exists, create if it doesn't.
    
    Args:
        path: Directory path
        
    Returns:
        Path object
        
    Example:
        >>> ensure_directory("/path/to/dir")
        PosixPath('/path/to/dir')
    """
    path = Path(path)
    path.mkdir(parents=True, exist_ok=True)
    return path


def delete_directory(path: str | Path, ignore_errors: bool = False) -> bool:
    """
    Delete directory and all contents.
    
    Args:
        path: Directory path
        ignore_errors: Whether to ignore deletion errors
        
    Returns:
        True if successful, False otherwise
        
    Example:
        >>> delete_directory("/path/to/dir")
        True
    """
    try:
        shutil.rmtree(path, ignore_errors=ignore_errors)
        logger.info(f"Deleted directory: {path}")
        return True
    except Exception as e:
        logger.error(f"Failed to delete directory {path}: {str(e)}")
        return False


def list_files(
    directory: str | Path,
    pattern: str = "*",
    recursive: bool = False
) -> List[Path]:
    """
    List files in directory matching pattern.
    
    Args:
        directory: Directory path
        pattern: Glob pattern (e.g., "*.csv")
        recursive: Search recursively
        
    Returns:
        List of Path objects
        
    Example:
        >>> list_files("/path/to/dir", "*.csv")
        [PosixPath('/path/to/dir/file1.csv'), ...]
    """
    directory = Path(directory)
    
    if not directory.exists():
        return []
    
    if recursive:
        return list(directory.rglob(pattern))
    else:
        return list(directory.glob(pattern))


# ============================================================
# FILE OPERATIONS
# ============================================================

def get_file_size(filepath: str | Path) -> int:
    """
    Get file size in bytes.
    
    Args:
        filepath: File path
        
    Returns:
        File size in bytes
        
    Example:
        >>> get_file_size("/path/to/file.csv")
        1048576
    """
    return os.path.getsize(filepath)


def get_file_hash(filepath: str | Path, algorithm: str = "sha256") -> str:
    """
    Calculate file hash.
    
    Args:
        filepath: File path
        algorithm: Hash algorithm (md5, sha1, sha256)
        
    Returns:
        Hex digest of file hash
        
    Example:
        >>> get_file_hash("/path/to/file.csv")
        'a1b2c3d4...'
    """
    hash_func = hashlib.new(algorithm)
    
    with open(filepath, 'rb') as f:
        # Read in chunks for memory efficiency
        while chunk := f.read(8192):
            hash_func.update(chunk)
    
    return hash_func.hexdigest()


def delete_file(filepath: str | Path) -> bool:
    """
    Delete a file.
    
    Args:
        filepath: File path
        
    Returns:
        True if successful, False otherwise
        
    Example:
        >>> delete_file("/path/to/file.csv")
        True
    """
    try:
        os.remove(filepath)
        logger.info(f"Deleted file: {filepath}")
        return True
    except Exception as e:
        logger.error(f"Failed to delete file {filepath}: {str(e)}")
        return False


def copy_file(src: str | Path, dst: str | Path) -> bool:
    """
    Copy file from source to destination.
    
    Args:
        src: Source path
        dst: Destination path
        
    Returns:
        True if successful, False otherwise
        
    Example:
        >>> copy_file("/src/file.csv", "/dst/file.csv")
        True
    """
    try:
        shutil.copy2(src, dst)
        logger.info(f"Copied file: {src} -> {dst}")
        return True
    except Exception as e:
        logger.error(f"Failed to copy file: {str(e)}")
        return False


def move_file(src: str | Path, dst: str | Path) -> bool:
    """
    Move file from source to destination.
    
    Args:
        src: Source path
        dst: Destination path
        
    Returns:
        True if successful, False otherwise
        
    Example:
        >>> move_file("/src/file.csv", "/dst/file.csv")
        True
    """
    try:
        shutil.move(src, dst)
        logger.info(f"Moved file: {src} -> {dst}")
        return True
    except Exception as e:
        logger.error(f"Failed to move file: {str(e)}")
        return False


# ============================================================
# COMPRESSION
# ============================================================

def compress_file(filepath: str | Path, output_path: Optional[str | Path] = None) -> str:
    """
    Compress file using gzip.
    
    Args:
        filepath: File to compress
        output_path: Output path (default: filepath.gz)
        
    Returns:
        Path to compressed file
        
    Example:
        >>> compress_file("/path/to/file.csv")
        '/path/to/file.csv.gz'
    """
    if output_path is None:
        output_path = f"{filepath}.gz"
    
    with open(filepath, 'rb') as f_in:
        with gzip.open(output_path, 'wb') as f_out:
            shutil.copyfileobj(f_in, f_out)
    
    logger.info(f"Compressed file: {filepath} -> {output_path}")
    return str(output_path)


def decompress_file(filepath: str | Path, output_path: Optional[str | Path] = None) -> str:
    """
    Decompress gzip file.
    
    Args:
        filepath: Compressed file
        output_path: Output path (default: remove .gz extension)
        
    Returns:
        Path to decompressed file
        
    Example:
        >>> decompress_file("/path/to/file.csv.gz")
        '/path/to/file.csv'
    """
    if output_path is None:
        output_path = str(filepath).replace('.gz', '')
    
    with gzip.open(filepath, 'rb') as f_in:
        with open(output_path, 'wb') as f_out:
            shutil.copyfileobj(f_in, f_out)
    
    logger.info(f"Decompressed file: {filepath} -> {output_path}")
    return str(output_path)


# ============================================================
# JSON OPERATIONS
# ============================================================

def read_json_file(filepath: str | Path) -> Any:
    """
    Read JSON file.
    
    Args:
        filepath: JSON file path
        
    Returns:
        Parsed JSON data
        
    Example:
        >>> read_json_file("/path/to/data.json")
        {'key': 'value'}
    """
    with open(filepath, 'r', encoding='utf-8') as f:
        return json.load(f)


def write_json_file(filepath: str | Path, data: Any, indent: int = 2) -> None:
    """
    Write data to JSON file.
    
    Args:
        filepath: Output file path
        data: Data to write
        indent: JSON indentation
        
    Example:
        >>> write_json_file("/path/to/data.json", {"key": "value"})
    """
    with open(filepath, 'w', encoding='utf-8') as f:
        json.dump(data, f, indent=indent, ensure_ascii=False)


# ============================================================
# CSV OPERATIONS
# ============================================================

def read_csv_chunks(
    filepath: str | Path,
    chunk_size: int = 10000
) -> Iterator[pd.DataFrame]:
    """
    Read CSV file in chunks for memory efficiency.
    
    Args:
        filepath: CSV file path
        chunk_size: Number of rows per chunk
        
    Yields:
        DataFrame chunks
        
    Example:
        >>> for chunk in read_csv_chunks("/path/to/large.csv"):
        ...     process(chunk)
    """
    for chunk in pd.read_csv(filepath, chunksize=chunk_size):
        yield chunk


# ============================================================
# UPLOAD OPERATIONS
# ============================================================

async def save_uploaded_file(
    upload_file: UploadFile,
    destination: str | Path
) -> str:
    """
    Save uploaded file to destination.
    
    Args:
        upload_file: FastAPI UploadFile object
        destination: Destination path
        
    Returns:
        Path to saved file
        
    Example:
        >>> await save_uploaded_file(file, "/path/to/save/file.csv")
        '/path/to/save/file.csv'
    """
    destination = Path(destination)
    
    # Ensure parent directory exists
    destination.parent.mkdir(parents=True, exist_ok=True)
    
    # Save file
    with open(destination, 'wb') as f:
        # Read and write in chunks
        while chunk := await upload_file.read(8192):
            f.write(chunk)
    
    logger.info(f"Saved uploaded file to: {destination}")
    return str(destination)


# ============================================================
# CLEANUP OPERATIONS
# ============================================================

def cleanup_old_files(
    directory: str | Path,
    age_days: int = 30,
    pattern: str = "*",
    dry_run: bool = False
) -> int:
    """
    Delete files older than specified age.
    
    Args:
        directory: Directory to clean
        age_days: Age threshold in days
        pattern: File pattern to match
        dry_run: If True, only log what would be deleted
        
    Returns:
        Number of files deleted (or would be deleted if dry_run)
        
    Example:
        >>> cleanup_old_files("/temp", age_days=7)
        15
    """
    directory = Path(directory)
    cutoff_time = datetime.now() - timedelta(days=age_days)
    deleted_count = 0
    
    for filepath in directory.glob(pattern):
        if not filepath.is_file():
            continue
        
        # Check file modification time
        file_mtime = datetime.fromtimestamp(filepath.stat().st_mtime)
        
        if file_mtime < cutoff_time:
            if dry_run:
                logger.info(f"Would delete: {filepath}")
            else:
                try:
                    filepath.unlink()
                    logger.info(f"Deleted old file: {filepath}")
                except Exception as e:
                    logger.error(f"Failed to delete {filepath}: {str(e)}")
                    continue
            
            deleted_count += 1
    
    logger.info(
        f"Cleanup complete: {deleted_count} files "
        f"{'would be' if dry_run else ''} deleted from {directory}"
    )
    
    return deleted_count
