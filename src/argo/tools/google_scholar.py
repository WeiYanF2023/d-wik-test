#!/usr/bin/env python
"""
ADK Function Tool for Google Scholar Search.
"""

from typing import Dict, Any
from scholarly import scholarly
from google.adk.tools import FunctionTool

async def search_google_scholar(query: str, k: int = 5) -> Dict[str, Any]:
    """
    Search Google Scholar for academic papers based on the provided query using the 'scholarly' library.

    Args:
        query: The search query for Google Scholar.
        k: The maximum number of search results to return (default is 5).

    Returns:
        A dictionary containing:
        - 'status': 'success' or 'error'.
        - 'results': A list of dictionaries, each representing a paper with 'title', 'authors', 'abstract', and 'url' (if successful and results found).
        - 'message': An error message (if status is 'error') or "No results found..." if successful but no results.
    """
    try:
        # Perform the search asynchronously if possible, or handle potential blocking
        # Note: 'scholarly' might be blocking; consider running in a thread executor if performance is critical
        search_query = scholarly.search_pubs(query)

        results_data = []
        count = 0

        # Extract the requested number of results
        for paper in search_query:
            if count >= k:
                break

            bib = paper.get('bib', {})
            title = bib.get('title', 'No title available')
            authors = bib.get('author', ['No authors available'])
            # Ensure authors are strings if they are not already
            if isinstance(authors, list):
                authors_str = ', '.join(map(str, authors))
            else:
                authors_str = str(authors) # Handle single author case or unexpected format

            abstract = bib.get('abstract', 'No abstract available')
            eprint_url = paper.get('eprint_url', 'No URL available')

            # Format the result dictionary
            result_item = {
                "title": title,
                "authors": authors_str,
                "abstract": abstract,
                "url": eprint_url
            }

            results_data.append(result_item)
            count += 1

        # Return formatted results or a message if none found
        if results_data:
            return {"status": "success", "results": results_data}
        else:
            return {"status": "success", "results": [], "message": f"No results found for the query: '{query}'."}

    except Exception as e:
        return {"status": "error", "message": f"Error searching Google Scholar: {str(e)}"}

# Wrap the function with FunctionTool
google_scholar_tool = FunctionTool(func=search_google_scholar)
