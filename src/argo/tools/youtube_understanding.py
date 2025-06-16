#!/usr/bin/env python
"""
ADK Function Tool for analyzing YouTube videos using Vertex AI Gemini.
"""
import os
from typing import Dict, Any
from urllib.parse import urlparse

import vertexai
from vertexai.generative_models import GenerativeModel, Part, GenerationResponse
from google.api_core import exceptions as google_exceptions # Import google exceptions

from dotenv import load_dotenv
from google.adk.tools import FunctionTool

load_dotenv()

# Constants for Gemini model
GEMINI_MODEL_ID = os.getenv("YOUTUBE_GEMINI_MODEL", "gemini-1.5-pro-preview-0409") # Use env var or default

# Initialize Vertex AI client - needs project and location
# GOOGLE_API_KEY is often used for Google AI Studio, Vertex usually uses ADC.
# Let's rely on Application Default Credentials (ADC) for Vertex AI.
# Ensure GOOGLE_APPLICATION_CREDENTIALS env var is set or gcloud auth login/application-default login was run.
PROJECT_ID = os.environ.get("GOOGLE_PROJECT_ID")
LOCATION = os.environ.get("GOOGLE_LOCATION", "us-central1") # Default location if not set

gemini_model = None
initialization_error = None
try:
    if not PROJECT_ID:
        raise ValueError("GOOGLE_PROJECT_ID environment variable is required for Vertex AI.")
    vertexai.init(project=PROJECT_ID, location=LOCATION)
    gemini_model = GenerativeModel(GEMINI_MODEL_ID)
except ValueError as init_val_err:
    initialization_error = f"Vertex AI Initialization Error: {init_val_err}"
    print(initialization_error)
except Exception as init_err:
    initialization_error = f"Failed to initialize Vertex AI or Gemini Model: {init_err}"
    print(initialization_error)


# Helper function
def _is_valid_youtube_url(url: str) -> bool:
    """Validate if a URL is a proper YouTube video URL."""
    try:
        parsed_url = urlparse(url)
        # Check for standard youtube.com/watch or youtu.be formats
        is_youtube_domain = parsed_url.netloc in ("www.youtube.com", "youtube.com", "youtu.be")
        is_watch_path = "watch" in parsed_url.path and "v=" in parsed_url.query
        is_short_url = parsed_url.netloc == "youtu.be" and parsed_url.path # Path should exist for short URLs

        return is_youtube_domain and (is_watch_path or is_short_url)
    except Exception:
        return False


async def analyze_youtube_video(url: str, query: str) -> Dict[str, Any]:
    """
    Analyze a YouTube video using the configured Vertex AI Gemini model.

    Requires Vertex AI to be initialized (GOOGLE_PROJECT_ID, GOOGLE_LOCATION env vars, and ADC auth).
    The specific Gemini model used can be set via YOUTUBE_GEMINI_MODEL env var (defaults to gemini-1.5-pro-preview-0409).

    Args:
        url: URL of the YouTube video to analyze (must be a valid YouTube video URL).
        query: The question or task for Gemini to perform based on the video content.

    Returns:
        A dictionary containing:
        - 'status': 'success' or 'error'.
        - 'analysis_result': The text response from Gemini (if successful).
        - 'message': An error message (if status is 'error') or details about the success.
    """
    if initialization_error:
        return {"status": "error", "analysis_result": None, "message": initialization_error}
    if not gemini_model:
         # Should be caught by initialization_error, but as a safeguard
         return {"status": "error", "analysis_result": None, "message": "Gemini model not available."}

    if not _is_valid_youtube_url(url):
        return {
            "status": "error",
            "analysis_result": None,
            "message": "Invalid YouTube URL provided. Use format like 'https://www.youtube.com/watch?v=...' or 'https://youtu.be/...'"
        }

    try:
        # Create message parts for Gemini
        # Ensure the URI is correctly passed for Vertex AI processing
        video_part = Part.from_uri(
            uri=url,
            mime_type="video/youtube" # Use specific mime type for YouTube URLs if supported, else generic video/*
        )
        prompt_parts = [video_part, query] # Order might matter, often multimodal input first

        # Generate response from Gemini asynchronously
        # Use generate_content_async for async operation
        response: GenerationResponse = await gemini_model.generate_content_async(prompt_parts)

        # Extract text result
        # Add safety checks for response structure
        if response and response.candidates and response.candidates[0].content.parts:
             analysis_text = response.text # .text helper usually combines parts
        else:
             # Handle cases where response is empty or blocked
             finish_reason = response.candidates[0].finish_reason if response and response.candidates else "Unknown"
             safety_ratings = response.candidates[0].safety_ratings if response and response.candidates else "N/A"
             return {
                 "status": "error",
                 "analysis_result": None,
                 "message": f"Gemini did not return valid content. Finish Reason: {finish_reason}. Safety Ratings: {safety_ratings}"
             }


        return {
            "status": "success",
            "analysis_result": analysis_text,
            "message": "YouTube video analyzed successfully."
        }
    except google_exceptions.GoogleAPIError as api_err:
         # Catch specific Google API errors
         return {
            "status": "error",
            "analysis_result": None,
            "message": f"Vertex AI API Error analyzing YouTube video: {api_err}"
        }
    except Exception as e:
        # Catch other potential errors (network, invalid args to API, etc.)
        return {
            "status": "error",
            "analysis_result": None,
            "message": f"Unexpected error analyzing YouTube video: {str(e)}"
        }

# Wrap the function with FunctionTool
youtube_understanding_tool = FunctionTool(func=analyze_youtube_video)
