#!/usr/bin/env python
"""
ADK Function Tools for interacting with Wikipedia using the wikipedia-api library.
"""

import wikipediaapi
from typing import Dict, Any, List
from google.adk.tools import FunctionTool


# Create a Wikipedia API instance with a user agent
def _get_wiki_api(language: str = "en") -> wikipediaapi.Wikipedia:
    """
    Get a Wikipedia API instance with the specified language.

    Args:
        language: The language code for Wikipedia (default is 'en' for English).

    Returns:
        A Wikipedia API instance configured for the specified language.
    """
    # Using a generic but descriptive user agent is recommended.
    return wikipediaapi.Wikipedia(
        user_agent="ADKAgent/1.0 (https://github.com/google/adk-python; example-agent)",
        language=language,
        extract_format=wikipediaapi.ExtractFormat.WIKI,  # Use WIKI format for Markdown-like text
    )


async def search_wikipedia(
    title: str, language: str = "en", max_results: int = 10
) -> Dict[str, Any]:
    """
    Retrieves a specific Wikipedia page by title and lists pages linked from it.

    Attempts to find an exact page match for the given title. If found, it returns
    the main page's details and details of pages linked from it, up to max_results.

    Args:
        title: The exact title of the page to look for on Wikipedia.
        language: The language code for Wikipedia (default is 'en').
        max_results: Maximum number of results (including the main page and linked pages) to return (default is 10).

    Returns:
        A dictionary containing:
        - 'status': 'success' or 'error'.
        - 'results': A list of dictionaries, each with 'title', 'summary', 'url' (if successful).
        - 'message': An error message (if status is 'error') or "No pages found..." if successful but no results.
    """
    try:
        wiki_wiki = _get_wiki_api(language)
        # Attempt to get the page by exact title
        page = wiki_wiki.page(title)

        results_data = []
        processed_titles = set()  # Keep track of added titles to avoid duplicates

        # 1. Add exact match if it exists
        if page.exists():
            results_data.append(
                {
                    "title": page.title,
                    "summary": page.summary[:250] + "..."
                    if len(page.summary) > 250
                    else page.summary,  # Slightly longer summary
                    "url": page.fullurl,
                }
            )
            processed_titles.add(page.title)

        # 2. Add linked pages (if exact match found) or similar pages (if not)
        pages_to_consider = []
        if page.exists():
            # Limit links to avoid excessive processing
            # Only consider links if the main page exists
            pages_to_consider = list(page.links.values())[:max_results] # Get links up to max_results
        # If page doesn't exist, pages_to_consider remains empty

        # 3. Process considered pages (only runs if page existed and had links)
        for link_page in pages_to_consider:
            if len(results_data) >= max_results:
                break
            # Check existence and avoid duplicates
            if link_page.exists() and link_page.title not in processed_titles:
                results_data.append(
                    {
                        "title": link_page.title,
                        "summary": link_page.summary[:250] + "..."
                        if len(link_page.summary) > 250
                        else link_page.summary,
                        "url": link_page.fullurl,
                    }
                )
                processed_titles.add(link_page.title)

        if results_data:
            return {"status": "success", "results": results_data}
        else:
            # If the initial page didn't exist, results_data is empty.
            # If it existed but had no links processed, results_data has only the main page.
            return {
                "status": "success",
                "results": [],
                "message": f"Wikipedia page '{title}' not found or no linked pages processed.",
            }

    except Exception as e:
        return {
            "status": "error",
            "results": [],
            "message": f"Error searching Wikipedia: {str(e)}",
        }


async def get_wikipedia_page(
    title: str, language: str = "en", sections_depth: int = 3
) -> Dict[str, Any]:
    """
    Get the full text content of a specific Wikipedia page by its exact title.

    Args:
        title: The exact title of the Wikipedia page (case-sensitive).
        language: The language code for Wikipedia (default is 'en').
        sections_depth: How many levels of subsections to include (default is 3). Use 0 for summary only.

    Returns:
        A dictionary containing:
        - 'status': 'success' or 'error'.
        - 'title': The title of the page retrieved (if successful).
        - 'url': The full URL of the page (if successful).
        - 'content': The full text content (summary + sections) of the page (if successful).
        - 'message': An error message (if status is 'error').
    """
    try:
        wiki_wiki = _get_wiki_api(language)
        page = wiki_wiki.page(title)

        if not page.exists():
            return {
                "status": "error",
                "content": None,
                "message": f"Wikipedia page '{title}' not found in language '{language}'.",
            }

        content_parts = [f"# {page.title}", f"\n{page.summary}\n"]

        def process_sections(
            sections: List[wikipediaapi.WikipediaPageSection], current_depth=1
        ):
            if sections_depth == 0 or current_depth > sections_depth:
                return

            for section in sections:
                # Add section heading with appropriate Markdown level
                content_parts.append(f"\n{'#' * (current_depth + 1)} {section.title}\n")

                # Add section text if it's not just the title again
                section_text = section.text.strip()
                if section_text and section_text != section.title:
                    content_parts.append(f"{section_text}\n")

                # Process subsections recursively
                process_sections(section.sections, current_depth + 1)

        # Process top-level sections if depth > 0
        if sections_depth > 0:
            process_sections(page.sections)

        # Add metadata
        content_parts.append(f"\n---\n*Source URL: {page.fullurl}*")

        full_content = "\n".join(content_parts)

        return {
            "status": "success",
            "title": page.title,
            "url": page.fullurl,
            "content": full_content,
        }

    except Exception as e:
        return {
            "status": "error",
            "content": None,
            "message": f"Error getting Wikipedia page '{title}': {str(e)}",
        }


async def get_wikipedia_categories(title: str, language: str = "en") -> Dict[str, Any]:
    """
    Get the categories associated with a specific Wikipedia page by its exact title.

    Args:
        title: The exact title of the Wikipedia page (case-sensitive).
        language: The language code for Wikipedia (default is 'en').

    Returns:
        A dictionary containing:
        - 'status': 'success' or 'error'.
        - 'title': The title of the page checked (if successful).
        - 'categories': A list of category names (cleaned, without 'Category:' prefix) (if successful).
        - 'message': An error message (if status is 'error') or "No categories found..." if successful but none exist.
    """
    try:
        wiki_wiki = _get_wiki_api(language)
        page = wiki_wiki.page(title)

        if not page.exists():
            return {
                "status": "error",
                "categories": [],
                "message": f"Wikipedia page '{title}' not found in language '{language}'.",
            }

        # Categories are returned as a dict { 'Category:Name': PageObject }
        raw_categories = list(page.categories.keys())
        cleaned_categories = [
            cat.replace("Category:", "").strip() for cat in raw_categories
        ]

        if not cleaned_categories:
            return {
                "status": "success",
                "title": page.title,
                "categories": [],
                "message": f"No categories found for Wikipedia page '{title}'.",
            }

        return {
            "status": "success",
            "title": page.title,
            "categories": cleaned_categories,
        }

    except Exception as e:
        return {
            "status": "error",
            "categories": [],
            "message": f"Error getting Wikipedia categories for '{title}': {str(e)}",
        }


# Wrap functions with FunctionTool
search_wikipedia_tool = FunctionTool(func=search_wikipedia)
get_wikipedia_page_tool = FunctionTool(func=get_wikipedia_page)
get_wikipedia_categories_tool = FunctionTool(func=get_wikipedia_categories)
