#!/usr/bin/env python
"""
ADK Function Tool for extracting various archive file formats.
"""

import os
import pathlib
import zipfile
import tarfile
import gzip
import bz2
import lzma
import shutil
from typing import Dict, List, Optional, Tuple, Any
from google.adk.tools import FunctionTool
from dotenv import load_dotenv

load_dotenv()

def _get_file_list(directory: str) -> List[Dict[str, Any]]:
    """
    Get a list of files and directories in the specified directory and its subdirectories.

    Args:
        directory: The directory path to scan.

    Returns:
        List of dictionaries with file information ('path', 'type', 'size').
    """
    result = []
    try:
        base_path = pathlib.Path(directory)
        for path in base_path.rglob('*'):
            try:
                is_dir = path.is_dir()
                size = path.stat().st_size if path.is_file() else 0
                # Store relative path for cleaner output if desired, or keep absolute
                # relative_path_str = str(path.relative_to(base_path))
                absolute_path_str = str(path.resolve()) # Use absolute path for clarity
                file_info = {
                    "path": absolute_path_str,
                    "type": "directory" if is_dir else "file",
                    "size": size
                }
                result.append(file_info)
            except OSError as stat_err:
                 print(f"Warning: Could not stat file {path}: {stat_err}")
            except Exception as item_err:
                 print(f"Warning: Error processing item {path}: {item_err}")

    except Exception as e:
        print(f"Error scanning directory {directory}: {str(e)}") # Log error but might return partial list

    return result


def _extract_archive(archive_path: str, extract_to: str) -> Tuple[bool, Optional[str]]:
    """
    Extract various archive formats to the specified directory.
    Handles zip, tar (gz, bz2, xz), and single-file gz, bz2, xz.

    Args:
        archive_path: Path to the archive file.
        extract_to: Directory to extract contents into.

    Returns:
        Tuple of (success: bool, error_message: Optional[str]).
    """
    archive_path_lower = archive_path.lower()
    try:
        if zipfile.is_zipfile(archive_path):
            with zipfile.ZipFile(archive_path, 'r') as zip_ref:
                zip_ref.extractall(extract_to)
            return True, None
        elif tarfile.is_tarfile(archive_path):
            # 'r:*' automatically handles compression formats (gz, bz2, xz)
            with tarfile.open(archive_path, 'r:*') as tar_ref:
                # Add errorlevel=1 to ignore certain non-fatal extraction errors if needed
                # tar_ref.extractall(extract_to, errorlevel=1)
                tar_ref.extractall(extract_to)
            return True, None
        elif archive_path_lower.endswith('.gz') and not archive_path_lower.endswith('.tar.gz'):
            output_path = pathlib.Path(extract_to) / pathlib.Path(archive_path).stem # Use stem to remove .gz
            with gzip.open(archive_path, 'rb') as f_in, open(output_path, 'wb') as f_out:
                shutil.copyfileobj(f_in, f_out)
            return True, None
        elif archive_path_lower.endswith('.bz2') and not archive_path_lower.endswith('.tar.bz2'):
            output_path = pathlib.Path(extract_to) / pathlib.Path(archive_path).stem # Use stem to remove .bz2
            with bz2.open(archive_path, 'rb') as f_in, open(output_path, 'wb') as f_out:
                shutil.copyfileobj(f_in, f_out)
            return True, None
        elif archive_path_lower.endswith('.xz') and not archive_path_lower.endswith('.tar.xz'):
            output_path = pathlib.Path(extract_to) / pathlib.Path(archive_path).stem # Use stem to remove .xz
            with lzma.open(archive_path, 'rb') as f_in, open(output_path, 'wb') as f_out:
                shutil.copyfileobj(f_in, f_out)
            return True, None
        else:
            # Attempt tar extraction again just in case is_tarfile missed it (e.g., unusual naming)
            try:
                 with tarfile.open(archive_path, 'r:*') as tar_ref:
                      tar_ref.extractall(extract_to)
                 print(f"Warning: Extracted {archive_path} as tar after initial check failed.")
                 return True, None
            except tarfile.TarError:
                 return False, f"Unsupported or corrupted archive format: {archive_path}"
            except Exception as fallback_e:
                 return False, f"Extraction failed during fallback attempt: {str(fallback_e)}"

    except (zipfile.BadZipFile, tarfile.TarError, EOFError, gzip.BadGzipFile, lzma.LZMAError) as archive_err:
         return False, f"Extraction failed due to archive error: {str(archive_err)}"
    except Exception as e:
        # Catch other potential errors like permissions, disk space, etc.
        return False, f"Extraction failed due to unexpected error: {str(e)}"


async def unzip_file(file_path: str) -> Dict[str, Any]:
    """
    Extract an archive file (zip, tar, tar.gz, tar.bz2, tar.xz, gz, bz2, xz)
    to a subdirectory within the directory specified by the UNZIP_PATH environment variable.

    Requires the UNZIP_PATH environment variable to be set to the base directory for extractions.

    Args:
        file_path: Absolute or relative path to the archive file to extract.

    Returns:
        A dictionary containing:
        - 'status': 'success' or 'error'.
        - 'extract_dir': The absolute path to the directory where files were extracted (if successful).
        - 'extracted_files': A list of dictionaries detailing extracted items (path, type, size) (if successful).
        - 'message': A status message indicating success or detailing the error.
    """
    unzip_base_path = os.environ.get("UNZIP_PATH")
    if not unzip_base_path:
        return {"status": "error", "extracted_files": [], "message": "Error: UNZIP_PATH environment variable is not set."}

    p_file_path = pathlib.Path(file_path).resolve() # Ensure absolute path

    if not p_file_path.exists():
        return {"status": "error", "extracted_files": [], "message": f"Error: The file {file_path} does not exist."}
    if not p_file_path.is_file():
        return {"status": "error", "extracted_files": [], "message": f"Error: {file_path} is not a file."}

    # Create a unique extraction directory based on the filename (without extension)
    file_name_stem = p_file_path.stem
    # If it was a compressed tar, stem might include .tar, remove it too
    if file_name_stem.lower().endswith('.tar'):
        file_name_stem = pathlib.Path(file_name_stem).stem

    extract_dir = pathlib.Path(unzip_base_path).resolve() / file_name_stem
    extract_dir_str = str(extract_dir)

    extracted_files_list = []

    # Check if already extracted (idempotency)
    if extract_dir.exists() and extract_dir.is_dir():
        # Verify it wasn't just an empty dir creation failure before
        if any(extract_dir.iterdir()):
            extracted_files_list = _get_file_list(extract_dir_str)
            return {
                "status": "success",
                "extract_dir": extract_dir_str,
                "extracted_files": extracted_files_list,
                "message": f"Archive '{file_path}' appears to be already extracted to '{extract_dir_str}'."
            }
        else:
             print(f"Warning: Existing empty extraction directory found: {extract_dir_str}. Attempting extraction again.")
             # Proceed with extraction below

    try:
        # Create extraction directory
        extract_dir.mkdir(parents=True, exist_ok=True)
    except OSError as mkdir_err:
         return {"status": "error", "extracted_files": [], "message": f"Error creating extraction directory '{extract_dir_str}': {mkdir_err}"}


    # Extract the archive
    success, error = _extract_archive(str(p_file_path), extract_dir_str)

    if not success:
        # Clean up potentially partially created directory if extraction failed
        try:
            if extract_dir.exists():
                 shutil.rmtree(extract_dir)
        except Exception as cleanup_err:
             print(f"Warning: Failed to clean up directory '{extract_dir_str}' after failed extraction: {cleanup_err}")
        return {"status": "error", "extracted_files": [], "message": f"Failed to extract {file_path}: {error}"}

    # List the extracted files
    extracted_files_list = _get_file_list(extract_dir_str)

    return {
        "status": "success",
        "extract_dir": extract_dir_str,
        "extracted_files": extracted_files_list,
        "message": f"Successfully extracted {file_path} to {extract_dir_str}. Found {len(extracted_files_list)} items."
    }

# Wrap the function with FunctionTool
unzip_tool = FunctionTool(func=unzip_file)
