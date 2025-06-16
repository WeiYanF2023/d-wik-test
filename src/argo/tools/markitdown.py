#!/usr/bin/env python
"""
ADK Function Tools for converting content from URLs or local paths to Markdown.

Uses MarkItDown library, Mistral AI OCR, and Playwright for comprehensive conversion.
"""

import json
import os
import random
import mimetypes
import requests
import tempfile
import pathlib
from typing import Dict, Any

from markitdown import MarkItDown
from openai import OpenAI
from mistralai import Mistral, File  # Corrected import
from playwright.async_api import async_playwright
from dotenv import load_dotenv
from google.adk.tools import FunctionTool

load_dotenv()

# Initialize MarkItDown with sensible defaults
# Consider if plugins should be enabled based on use case
md_converter = MarkItDown(enable_plugins=False)

# Initialize LLM clients - handle potential missing keys gracefully
llm_client = None
mistral_client = None

try:
    llm_api_key = os.getenv("LLM_API_KEY")
    if llm_api_key:
        llm_client = OpenAI(
            base_url=os.getenv("LLM_BASE_URL", "https://api.openai.com/v1"),
            api_key=llm_api_key,
        )
except Exception as e:
    print(f"Warning: Failed to initialize OpenAI client: {e}")

try:
    mistral_api_key = os.getenv("MISTRAL_API_KEY")
    if mistral_api_key:
        mistral_client = Mistral(api_key=mistral_api_key)  # Use MistralClient
except Exception as e:
    print(f"Warning: Failed to initialize Mistral client: {e}")

# --- Helper Functions ---


async def _better_pdf_to_markdown(pdf_url: str) -> Dict[str, Any] | str:
    """
    Convert a PDF from a URL to Markdown using Mistral OCR.

    Requires MISTRAL_API_KEY environment variable.

    Args:
        pdf_url: The URL of the PDF file to convert.

    Returns:
        The OCR result dictionary from Mistral or an error string.
    """
    if not mistral_client:
        return "Error: Mistral client not initialized. Check MISTRAL_API_KEY."
    try:
        # Use process_document_async for async operation
        ocr_response = await mistral_client.ocr.process_async(
            model="mistral-ocr-latest",
            document={
                "type": "document_url",
                "document_url": pdf_url,
                "include_image_base64": False,  # Usually not needed for text extraction
            },
        )
        # Assuming the response structure is directly usable or needs minimal parsing
        # The original code used json.loads(ocr_response.model_dump_json()), adapt if needed
        # Based on potential Mistral client changes, this might return a Pydantic model directly
        # Let's assume it returns a model with a method like .to_dict() or similar
        if hasattr(ocr_response, "model_dump"):
            return ocr_response.model_dump()  # Use model_dump if available
        else:
            # Fallback or adjust based on actual Mistral client library structure
            # This might need refinement based on the exact library version/behavior
            return json.loads(ocr_response.model_dump_json())

    except Exception as e:
        return f"Error converting PDF URL via Mistral OCR: {str(e)}"


async def _better_pdf_to_markdown_local(pdf_path: str) -> Dict[str, Any] | str:
    """
    Convert a local PDF file to Markdown using Mistral OCR.

    Requires MISTRAL_API_KEY environment variable.

    Args:
        pdf_path: The path to the local PDF file to convert.

    Returns:
        The OCR result dictionary from Mistral or an error string.
    """
    if not mistral_client:
        return "Error: Mistral client not initialized. Check MISTRAL_API_KEY."
    try:
        with open(pdf_path, "rb") as pdf_file:
            # Use files.create for uploading
            uploaded_pdf = await mistral_client.files.upload_async(
                file=File(
                    file_name=os.path.basename(pdf_path),
                    content=pdf_file,
                    content_type="application/pdf",
                ),
                purpose="ocr",
            )
        signed_url_response = await mistral_client.files.get_signed_url_async(
            file_id=uploaded_pdf.id
        )
        # Use documents.process for processing uploaded file
        ocr_response = await mistral_client.ocr.process_async(
            model="mistral-ocr-latest",
            document={
                "type": "document_url",
                "document_url": signed_url_response.url,
                "include_image_base64": False,  # Usually not needed for text extraction
            },
        )
        # Similar handling as the URL version for response structure
        if hasattr(ocr_response, "model_dump"):
            return ocr_response.model_dump()
        else:
            return json.loads(ocr_response.model_dump_json())

    except Exception as e:
        return f"Error converting local PDF via Mistral OCR: {str(e)}"


async def save_page_as_pdf(url: str, pdf_path: str) -> None:
    """
    Renders a web page using Playwright and saves it as a PDF.

    Requires BROWSER_DATA_PATH and ADBLOCK_ID environment variables for persistent context and adblocking.

    Args:
        url: The URL of the web page.
        pdf_path: The path where the PDF should be saved.

    Raises:
        Exception: If Playwright fails to render or save the PDF.
    """
    browser_data_path = os.getenv("BROWSER_DATA_PATH", "")
    adblock_id = os.getenv("ADBLOCK_ID", "")
    extension_path = (
        os.path.join(browser_data_path, "extensions", adblock_id)
        if browser_data_path and adblock_id
        else None
    )

    async with async_playwright() as p:
        browser_context = None
        try:
            launch_options = {
                "headless": True,  # Usually run headless for automation
                "user_agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Version/18.3.1 Safari/605.1.15",
                "args": [],
                "viewport": {"width": 1280, "height": 720},
                "locale": "en-US",
            }
            if browser_data_path:
                launch_options["user_data_dir"] = os.path.join(
                    browser_data_path, ".user_data"
                )
            if extension_path and os.path.exists(extension_path):
                launch_options["args"].extend(
                    [
                        f"--disable-extensions-except={extension_path}",
                        f"--load-extension={extension_path}",
                    ]
                )

            # Use launch_persistent_context if user_data_dir is set, otherwise launch
            if "user_data_dir" in launch_options:
                browser_context = await p.chromium.launch_persistent_context(
                    **launch_options
                )
                page = await browser_context.new_page()
            else:
                browser = await p.chromium.launch(**launch_options)
                browser_context = await browser.new_context(
                    **launch_options
                )  # Apply options to context
                page = await browser_context.new_page()

            await page.goto(
                url, wait_until="domcontentloaded", timeout=60000
            )  # Increased timeout

            # Scroll randomly to trigger lazy-loaded content
            for _ in range(random.randint(2, 5)):
                scroll_distance = random.randint(300, 900)
                await page.mouse.wheel(0, scroll_distance)
                await page.wait_for_timeout(
                    random.randint(500, 1500)
                )  # Wait a bit after scroll

            await page.wait_for_timeout(3000)  # Final wait

            await page.pdf(
                path=pdf_path,
                format="Letter",
                print_background=True,
                display_header_footer=False,
                timeout=60000,
            )  # PDF timeout

        finally:
            if browser_context:
                await browser_context.close()
            # If not using persistent context, browser object needs closing if it exists
            # This part depends on whether launch() or launch_persistent_context() was used
            # Simplified: context close should handle associated browser resources in most cases.


# --- ADK Function Tools ---


async def get_content_from_url(url: str) -> Dict[str, Any]:
    """
    Get content from a URL and convert it to Markdown format.

    Downloads content from the URL, attempts conversion using MarkItDown library,
    Mistral OCR for PDFs, or Playwright rendering fallback.
    Preserves structure like headings, lists, tables, links.
    Supports HTML, PDF, DOCX, YouTube, and other web content via MarkItDown or OCR/rendering.

    Requires environment variables:
    - Optional: LLM_API_KEY, LLM_BASE_URL (for MarkItDown LLM features)
    - Optional: MISTRAL_API_KEY (for PDF OCR)
    - Optional: DOCINTEL_ENDPOINT (for MarkItDown DocIntel features)
    - Optional: BROWSER_DATA_PATH, ADBLOCK_ID (for Playwright fallback)

    Args:
        url: The URL of the content to convert (must be http or https).

    Returns:
        A dictionary containing:
        - 'status': 'success' or 'error'.
        - 'markdown': The converted Markdown content (if successful).
        - 'message': An error message (if status is 'error').
    """
    if not url.startswith(("http://", "https://")):
        return {"status": "error", "message": "URL must start with http:// or https://"}

    final_markdown = None
    error_message = None

    try:
        # 1. Check if PDF via HEAD request (quick check)
        is_pdf = False
        try:
            response = requests.head(url, allow_redirects=True, timeout=10)
            response.raise_for_status()
            content_type = response.headers.get("Content-Type", "").lower()
            if (
                url.endswith(".pdf")
                or "application/pdf" in content_type
                or "pdf" in content_type
            ):
                is_pdf = True
        except requests.exceptions.RequestException as head_err:
            print(
                f"Warning: HEAD request failed for {url}: {head_err}. Proceeding with other methods."
            )
            # Cannot determine type via HEAD, will try other methods

        # 2. If PDF, try Mistral OCR
        if is_pdf and mistral_client:
            ocr_results = await _better_pdf_to_markdown(pdf_url=url)
            if isinstance(ocr_results, dict):
                page_markdown = [
                    page.get("markdown", "") for page in ocr_results.get("pages", [])
                ]
                final_markdown = "\n\n".join(filter(None, page_markdown)).strip()
                if final_markdown:
                    return {"status": "success", "markdown": final_markdown}
                else:
                    error_message = (
                        "Mistral OCR succeeded but extracted no markdown content."
                    )
            else:  # It's an error string
                error_message = f"Mistral OCR failed: {ocr_results}"
                print(f"Mistral OCR failed for URL {url}: {ocr_results}")
        elif is_pdf and not mistral_client:
            print(
                f"URL {url} detected as PDF, but Mistral client not available for OCR."
            )
            error_message = "Detected PDF, but OCR tool (Mistral) is not configured."

        # 3. Try MarkItDown library (if not PDF or OCR failed/unavailable)
        if final_markdown is None:
            try:
                # Pass LLM client only if available
                kwargs = {}
                if llm_client:
                    kwargs["llm_client"] = llm_client
                    kwargs["llm_model"] = os.getenv(
                        "LLM_MODEL", "gpt-4o"
                    )  # Use env var or default
                docintel_endpoint = os.getenv("DOCINTEL_ENDPOINT")
                if docintel_endpoint:
                    kwargs["docintel_endpoint"] = docintel_endpoint

                result = md_converter.convert(source=url, **kwargs)
                final_markdown = result.markdown
                if final_markdown:
                    return {"status": "success", "markdown": final_markdown}
                else:
                    # MarkItDown might return empty if it can't handle it, prepare for fallback
                    print(f"MarkItDown returned empty markdown for URL: {url}")
                    if not error_message:  # Don't overwrite OCR error
                        error_message = (
                            "MarkItDown conversion resulted in empty content."
                        )

            except Exception as md_err:
                print(f"MarkItDown failed for URL {url}: {md_err}")
                if not error_message:  # Don't overwrite OCR error
                    error_message = f"MarkItDown conversion failed: {str(md_err)}"

        # 4. Fallback: Render with Playwright and OCR the PDF (if other methods failed)
        if final_markdown is None:
            print(f"Attempting Playwright rendering fallback for URL: {url}")
            if not mistral_client:
                return {
                    "status": "error",
                    "message": f"All conversion methods failed. Playwright fallback requires Mistral client for OCR. Last error: {error_message or 'Unknown failure'}",
                }

            try:
                with tempfile.TemporaryDirectory(delete=True) as temp_dir:
                    temp_pdf_path = os.path.join(temp_dir, "rendered.pdf")
                    await save_page_as_pdf(url, temp_pdf_path)
                    ocr_results = await _better_pdf_to_markdown_local(
                        pdf_path=temp_pdf_path
                    )

                    if isinstance(ocr_results, dict):
                        page_markdown = [
                            page.get("markdown", "")
                            for page in ocr_results.get("pages", [])
                        ]
                        final_markdown = "\n\n".join(
                            filter(None, page_markdown)
                        ).strip()
                        if final_markdown:
                            return {"status": "success", "markdown": final_markdown}
                        else:
                            error_message = "Playwright rendering + OCR succeeded but extracted no markdown."
                            return {"status": "error", "message": error_message}
                    else:  # It's an error string
                        error_message = f"Playwright fallback OCR failed: {ocr_results}"
                        return {"status": "error", "message": error_message}

            except Exception as render_err:
                error_message = (
                    f"Playwright rendering fallback failed: {str(render_err)}"
                )
                return {"status": "error", "message": error_message}

        # Should not be reached if logic is correct, but as a safeguard:
        return {
            "status": "error",
            "message": f"Conversion failed. Last error: {error_message or 'Unknown failure'}",
        }

    except Exception as e:
        # Catch-all for unexpected issues
        return {
            "status": "error",
            "message": f"Unexpected error converting URL content: {str(e)}",
        }


async def get_content_from_path(file_path: str) -> Dict[str, Any]:
    """
    Get content from a local file path and convert it to Markdown format.

    Reads content from the file path, attempts conversion using MarkItDown library or Mistral OCR for PDFs.
    Supports PDF, DOCX, PPTX, Excel, HTML, and other formats via MarkItDown or OCR.

    Requires environment variables:
    - Optional: LLM_API_KEY, LLM_BASE_URL (for MarkItDown LLM features)
    - Optional: MISTRAL_API_KEY (for PDF OCR)
    - Optional: DOCINTEL_ENDPOINT (for MarkItDown DocIntel features)

    Args:
        file_path: The path to the local file to convert.

    Returns:
        A dictionary containing:
        - 'status': 'success' or 'error'.
        - 'markdown': The converted Markdown content (if successful).
        - 'message': An error message (if status is 'error').
    """
    try:
        p_file_path = pathlib.Path(file_path)
        if not p_file_path.exists():
            return {
                "status": "error",
                "message": f"File not found at path: {file_path}",
            }
        if not p_file_path.is_file():
            return {"status": "error", "message": f"Path is not a file: {file_path}"}
        if not os.access(file_path, os.R_OK):
            return {
                "status": "error",
                "message": f"No permission to read file at: {file_path}",
            }

        final_markdown = None
        error_message = None

        # 1. Check if PDF
        is_pdf = (
            file_path.lower().endswith(".pdf")
            or mimetypes.guess_type(file_path)[0] == "application/pdf"
        )

        # 2. If PDF, try Mistral OCR
        if is_pdf and mistral_client:
            ocr_results = await _better_pdf_to_markdown_local(pdf_path=file_path)
            if isinstance(ocr_results, dict):
                page_markdown = [
                    page.get("markdown", "") for page in ocr_results.get("pages", [])
                ]
                final_markdown = "\n\n".join(filter(None, page_markdown)).strip()
                if final_markdown:
                    return {"status": "success", "markdown": final_markdown}
                else:
                    error_message = (
                        "Mistral OCR succeeded but extracted no markdown content."
                    )
            else:  # It's an error string
                error_message = f"Mistral OCR failed for local file: {ocr_results}"
                print(f"Mistral OCR failed for local file {file_path}: {ocr_results}")
        elif is_pdf and not mistral_client:
            print(
                f"File {file_path} detected as PDF, but Mistral client not available for OCR."
            )
            error_message = "Detected PDF, but OCR tool (Mistral) is not configured."

        # 3. Try MarkItDown library (if not PDF or OCR failed/unavailable)
        if final_markdown is None:
            try:
                # Pass LLM client only if available
                kwargs = {}
                if llm_client:
                    kwargs["llm_client"] = llm_client
                    kwargs["llm_model"] = os.getenv("LLM_MODEL", "gpt-4o")
                docintel_endpoint = os.getenv("DOCINTEL_ENDPOINT")
                if docintel_endpoint:
                    kwargs["docintel_endpoint"] = docintel_endpoint

                result = md_converter.convert(source=file_path, **kwargs)
                final_markdown = result.markdown
                if final_markdown:
                    return {"status": "success", "markdown": final_markdown}
                else:
                    # MarkItDown might return empty if it can't handle it
                    if not error_message:  # Don't overwrite OCR error
                        error_message = (
                            "MarkItDown conversion resulted in empty content."
                        )
                    return {"status": "error", "message": error_message}

            except Exception as md_err:
                if not error_message:  # Don't overwrite OCR error
                    error_message = f"MarkItDown conversion failed: {str(md_err)}"
                return {"status": "error", "message": error_message}

        # Should not be reached if logic is correct
        return {
            "status": "error",
            "message": f"Conversion failed. Last error: {error_message or 'Unknown failure'}",
        }

    except Exception as e:
        return {
            "status": "error",
            "message": f"Unexpected error processing file: {str(e)}",
        }


# Wrap functions with FunctionTool
get_content_from_url_tool = FunctionTool(func=get_content_from_url)
get_content_from_path_tool = FunctionTool(func=get_content_from_path)
