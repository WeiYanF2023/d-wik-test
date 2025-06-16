#!/usr/bin/env python
"""
ADK Function Tool for executing Python code in an isolated environment.
"""

import subprocess
import tempfile
import shutil
import sys
import pathlib
from typing import List, Optional, Dict, Any
from google.adk.tools import FunctionTool

def _get_venv_executable_path(venv_path: pathlib.Path, executable_name: str) -> pathlib.Path:
    """Gets the platform-specific path for an executable within the venv."""
    if sys.platform == "win32":
        return venv_path / "Scripts" / f"{executable_name}.exe"
    else:
        return venv_path / "bin" / executable_name

async def execute_python_code(
    python_code: str,
    required_packages: Optional[List[str]] = None,
    external_path: Optional[str] = None,
    return_file: Optional[str] = None
) -> Dict[str, Any]:
    """
    Executes Python code in an isolated temporary virtual environment and returns the result.

    Creates a venv, installs optional packages, copies optional external files/dirs,
    runs the code, and returns stdout/stderr or specified file content.

    Args:
        python_code: The Python code string to execute.
        required_packages: Optional list of package names (e.g., ["requests", "numpy"])
                           to install via pip before running the code.
        external_path: Optional path to a file or directory. Its contents will be copied
                       into the execution environment's working directory.
        return_file: Optional filename. If provided, returns the content of this file
                     from the execution environment instead of stdout/stderr.

    Returns:
        A dictionary containing:
        - 'status': 'success' or 'error'.
        - 'output': The standard output and standard error (if successful and not returning file),
                    or the content of the specified return_file (if successful and file returned),
                    or None (if error).
        - 'message': An error message (if status is 'error'), or details about execution outcome.
                     Includes stdout/stderr context on execution errors.
    """
    result_status = "error"
    result_output = None
    result_message = ""

    try:
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_dir_path = pathlib.Path(temp_dir)
            venv_path = temp_dir_path / ".venv"

            # --- 1. Create Virtual Environment ---
            try:
                subprocess.run(
                    [sys.executable, "-m", "venv", str(venv_path)],
                    check=True, capture_output=True, text=True, timeout=30, encoding='utf-8'
                )
            except (subprocess.CalledProcessError, subprocess.TimeoutExpired) as e:
                stderr = e.stderr if hasattr(e, 'stderr') else 'Timeout during venv creation.'
                result_message = f"Error creating virtual environment: {stderr}"
                return {"status": result_status, "output": result_output, "message": result_message}
            except Exception as e:
                 result_message = f"Unexpected error creating virtual environment: {e}"
                 return {"status": result_status, "output": result_output, "message": result_message}

            python_executable = _get_venv_executable_path(venv_path, "python")
            pip_executable = _get_venv_executable_path(venv_path, "pip")

            # --- 2. Copy External Path ---
            if external_path:
                source_path = pathlib.Path(external_path)
                if not source_path.exists():
                    result_message = f"Error: external_path '{external_path}' not found."
                    return {"status": result_status, "output": result_output, "message": result_message}
                try:
                    dest_path = temp_dir_path / source_path.name
                    if source_path.is_dir():
                        shutil.copytree(source_path, dest_path, dirs_exist_ok=True)
                    else:
                        shutil.copy(source_path, dest_path)
                except Exception as e:
                    result_message = f"Error copying external_path '{external_path}': {e}"
                    return {"status": result_status, "output": result_output, "message": result_message}

            # --- 3. Install Packages ---
            if required_packages:
                install_command = [str(pip_executable), "install"] + required_packages
                try:
                    subprocess.run(
                        install_command, check=True, capture_output=True, text=True,
                        cwd=str(temp_dir_path), timeout=180, encoding='utf-8'
                    )
                except (subprocess.CalledProcessError, subprocess.TimeoutExpired) as e:
                    stderr = e.stderr if hasattr(e, 'stderr') else 'Timeout during package installation.'
                    stdout = e.stdout if hasattr(e, 'stdout') else ''
                    result_message = f"Error installing packages:\n--- STDERR ---\n{stderr}\n--- STDOUT ---\n{stdout}"
                    return {"status": result_status, "output": result_output, "message": result_message}
                except Exception as e:
                    result_message = f"Unexpected error installing packages: {e}"
                    return {"status": result_status, "output": result_output, "message": result_message}

            # --- 4. Write Code to File ---
            script_path = temp_dir_path / "script.py"
            try:
                script_path.write_text(python_code, encoding='utf-8')
            except Exception as e:
                result_message = f"Error writing script file: {e}"
                return {"status": result_status, "output": result_output, "message": result_message}

            # --- 5. Execute Code ---
            exec_command = [str(python_executable), script_path.name]
            return_file_content = None
            return_file_error = None
            try:
                exec_result = subprocess.run(
                    exec_command, capture_output=True, text=True,
                    cwd=str(temp_dir_path), timeout=60, encoding='utf-8', errors='replace'
                )

                # --- 6. Handle Return File (if specified) ---
                if return_file:
                    return_file_path = temp_dir_path / return_file
                    if return_file_path.is_file():
                        try:
                            return_file_content = return_file_path.read_text(encoding='utf-8', errors='replace')
                        except Exception as e:
                            return_file_error = f"Error reading return_file '{return_file}': {e}"
                    else:
                        return_file_error = f"Specified return_file '{return_file}' not found after execution."

                # --- 7. Check Execution Result ---
                if exec_result.returncode != 0:
                    result_status = "error"
                    result_message = (
                        f"Execution Error (Exit Code {exec_result.returncode}):\n"
                        f"--- STDOUT ---\n{exec_result.stdout}\n"
                        f"--- STDERR ---\n{exec_result.stderr}"
                    )
                    if return_file and return_file_error:
                         result_message += f"\nAdditionally: {return_file_error}"
                    elif return_file and return_file_content is not None:
                         result_message += f"\nNote: return_file '{return_file}' was created but execution failed."

                else:
                    # Execution succeeded
                    result_status = "success"
                    if return_file:
                        if return_file_content is not None:
                            result_output = return_file_content
                            result_message = f"Execution Successful. Content of '{return_file}' returned."
                        else:
                            result_status = "error" # File expected but not found/readable
                            result_message = return_file_error or f"Error: return_file '{return_file}' not found or could not be read after successful execution."
                    else:
                        # Default: return combined stdout/stderr
                        result_output = f"--- STDOUT ---\n{exec_result.stdout}"
                        if exec_result.stderr:
                             result_output += f"\n--- STDERR ---\n{exec_result.stderr}"
                        result_message = "Execution Successful. Standard output/error returned."


            except subprocess.TimeoutExpired:
                result_status = "error"
                result_message = "Error: Code execution timed out after 60 seconds."
            except Exception as e:
                result_status = "error"
                result_message = f"Error during script execution: {e}"

    except Exception as e:
        # Catch errors during temp dir creation/cleanup
        result_status = "error"
        result_message = f"Error setting up execution environment: {e}"

    return {"status": result_status, "output": result_output, "message": result_message}


# Wrap the function with FunctionTool
python_execution_tool = FunctionTool(func=execute_python_code)
