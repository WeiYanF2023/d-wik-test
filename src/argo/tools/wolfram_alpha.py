#!/usr/bin/env python
"""
ADK Function Tool to query the Wolfram Alpha LLM API.
"""

import os
import httpx
from typing import Dict, Any
from dotenv import load_dotenv
from google.adk.tools import FunctionTool

load_dotenv()

# Constants
WOLFRAM_API_URL = "https://www.wolframalpha.com/api/v1/llm-api"

async def query_wolfram_alpha(question: str) -> Dict[str, Any]:
    """
    Query Wolfram Alpha LLM API with a natural language question.

    Sends the question to Wolfram Alpha's specialized LLM API and returns the direct response.
    Best suited for questions requiring computation, math, science, data analysis,
    or factual lookups within Wolfram Alpha's knowledge base.

    Requires the WOLFRAM_APP_ID environment variable to be set. Get one from
    https://products.wolframalpha.com/llm-api/

    Args:
        question: The natural language question to ask Wolfram Alpha.

    Returns:
        A dictionary containing:
        - 'status': 'success' or 'error'.
        - 'result': The direct plain text response from the Wolfram Alpha LLM API (if successful).
        - 'message': An error message detailing the failure (if status is 'error').
    """
    app_id = os.environ.get("WOLFRAM_APP_ID")
    if not app_id:
        return {
            "status": "error",
            "result": None,
            "message": "Error: WOLFRAM_APP_ID environment variable is not set."
        }

    try:
        async with httpx.AsyncClient() as client:
            response = await client.post(
                WOLFRAM_API_URL,
                params={"appid": app_id, "input": question},
                timeout=60.0  # Allow ample time for potentially complex queries
            )
            response.raise_for_status() # Raise HTTPStatusError for 4xx/5xx responses

        # Decode the response content
        result_text = response.content.decode('utf-8')

        return {
            "status": "success",
            "result": result_text,
            "message": "Query successful."
        }

    except httpx.HTTPStatusError as e:
        # Specific handling for HTTP errors from the API
        error_detail = f"HTTP {e.response.status_code}: {e.response.text}"
        return {
            "status": "error",
            "result": None,
            "message": f"Error querying Wolfram Alpha API: {error_detail}"
        }
    except httpx.RequestError as e:
        # Handle client-side request errors (network, DNS, timeout before response)
         return {
            "status": "error",
            "result": None,
            "message": f"Network or request error querying Wolfram Alpha: {str(e)}"
        }
    except Exception as e:
        # Catch any other unexpected errors during the process
        return {
            "status": "error",
            "result": None,
            "message": f"An unexpected error occurred: {str(e)}"
        }

# Wrap the function with FunctionTool
wolfram_alpha_tool = FunctionTool(func=query_wolfram_alpha)
