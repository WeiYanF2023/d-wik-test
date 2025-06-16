#!/usr/bin/env python
"""
ADK Function Tool for Web Search using Google Custom Search API.
"""

import os
from typing import List, Dict, Any

from googleapiclient.discovery import build
from dotenv import load_dotenv
from google.adk.tools import FunctionTool

load_dotenv()

def _clean_up_snippets(items: List[dict]) -> None:
    """
    Remove non-breaking space and trailing whitespace from snippets.

    Args:
        items: The search results that contain snippets that have to be cleaned up.
    """
    for item in items:
        item.update({k: v.replace('\xa0', ' ').strip() if k == 'snippet' else v for k, v in item.items()})

async def web_search(search_term: str, num_result: int = 5) -> Dict[str, Any]:
    """
    Search the web using the Google Custom Search API for a given term.

    Regular query arguments can also be used, like appending site:reddit.com or after:2024-04-30.
    If available and/or requested, the links of the search results should be used in a follow-up
    request using a different tool (e.g., markitdown) to get the full content.

    Requires the following environment variables to be set:
    - CSE_API_KEY: Your Custom Search Engine API Key (might be the same as GOOGLE_API_KEY).
    - ENGINE_ID: Your Custom Search Engine ID.
    Optional environment variables:
    - SERVICE_NAME: Custom Search service name (default: 'customsearch').
    - COUNTRY_REGION: Country restriction (default: 'us').
    - GEOLOCATION: Geolocation bias (default: 'us').
    - RESULT_LANGUAGE: Result language restriction (default: 'lang_en').

    Args:
        search_term: The search term to search for.
        num_result: The number of search results to return (default is 5, max depends on API config).

    Returns:
        A dictionary containing:
        - 'status': 'success' or 'error'.
        - 'results': A list of dictionaries, each with 'title', 'link', 'snippet' (if successful and results found).
        - 'message': An error message (if status is 'error') or "No search results found." if successful but no results.
    """
    try:
        cse_api_key = os.getenv('CSE_API_KEY') # Use GOOGLE_API_KEY if CSE_API_KEY not set
        engine_id = os.getenv('ENGINE_ID')

        if not cse_api_key or not engine_id:
            return {
                "status": "error",
                "message": "Missing required environment variables: CSE_API_KEY or ENGINE_ID."
            }

        service = build(
            os.getenv('SERVICE_NAME', 'customsearch'),
            "v1",
            # developerKey=api_key
        )

        response = service.cse().list(
            q=search_term,
            key=cse_api_key,
            cx=engine_id,
            cr=os.getenv('COUNTRY_REGION', 'us'),
            gl=os.getenv('GEOLOCATION', 'us'),
            lr=os.getenv('RESULT_LANGUAGE', 'lang_en'),
            num=num_result,
            fields='items(title,link,snippet)'
        ).execute()

        results = response.get('items', [])
        _clean_up_snippets(results)

        if not results:
            return {"status": "success", "results": [], "message": "No search results found."}

        # Return structured data
        search_results = [
            {"title": r.get('title'), "link": r.get('link'), "snippet": r.get('snippet')}
            for r in results
        ]
        return {"status": "success", "results": search_results}

    except Exception as e:
        return {"status": "error", "message": f"Error during web search: {str(e)}"}

# Wrap the function with FunctionTool
general_search_tool = FunctionTool(func=web_search)
