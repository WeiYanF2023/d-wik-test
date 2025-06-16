"""
ADK Tools for Argo Agent.

This package contains FunctionTool instances converted from legacy implementations.
"""

from .general_search import general_search_tool
from .google_scholar import google_scholar_tool
from .markitdown import get_content_from_url_tool, get_content_from_path_tool
from .python_execution import python_execution_tool
from .unzip import unzip_tool
from .wikipedia import search_wikipedia_tool, get_wikipedia_page_tool, get_wikipedia_categories_tool
from .wolfram_alpha import wolfram_alpha_tool
from .youtube_understanding import youtube_understanding_tool
# Import the list of Google Maps tools (handles import errors internally)
from .google_maps import google_maps_tools
from .playwright_mcp import get_browser_tools

# Combine all tool instances into a single list for convenience, if desired
# Filter out None entries in case some tools failed to initialize (e.g., missing packages/keys)
_all_defined_tools = [
    general_search_tool,
    google_scholar_tool,
    get_content_from_url_tool,
    get_content_from_path_tool,
    python_execution_tool,
    unzip_tool,
    search_wikipedia_tool,
    get_wikipedia_page_tool,
    get_wikipedia_categories_tool,
    wolfram_alpha_tool,
    youtube_understanding_tool,
] + google_maps_tools # Add the list of Google Maps tools

# Filter out any None values that might occur if a tool failed to initialize
available_tools = [tool for tool in _all_defined_tools if tool is not None]

# Define __all__ based on the available tools for explicit export
__all__ = [tool.name for tool in available_tools]

# You can also export the list directly if needed elsewhere:
# exportable_tools_list = available_tools
