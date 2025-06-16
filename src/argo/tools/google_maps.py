#!/usr/bin/env python
"""
ADK Function Tools for interacting with various Google Maps APIs.

Requires the GOOGLE_MAPS_API_KEY environment variable to be set.
You can obtain the API key from the Google Cloud Console:
https://console.cloud.google.com/projectselector2/google/maps-apis/credentials

Ensure the necessary APIs (Places, Directions, Address Validation, Geocoding, etc.)
are enabled for your key in the Google Cloud Console.
"""

import os
from datetime import datetime
from typing import List, Optional, Dict, Any

from dotenv import load_dotenv
from google.adk.tools import FunctionTool

try:
    import googlemaps
    from google.maps import places_v1
    _GOOGLE_MAPS_AVAILABLE = True
except ImportError:
    print("Warning: googlemaps or google-maps-places package not found. Google Maps tools will be unavailable. Install using: pip install googlemaps google-maps-places")
    _GOOGLE_MAPS_AVAILABLE = False

load_dotenv()

# --- Client Initialization ---
googlemaps_client = None
places_client = None
initialization_error = None

if _GOOGLE_MAPS_AVAILABLE:
    try:
        api_key = os.getenv("GOOGLE_MAPS_API_KEY")
        if not api_key:
            raise ValueError("GOOGLE_MAPS_API_KEY environment variable is not set.")
        googlemaps_client = googlemaps.Client(key=api_key)
        places_client = places_v1.PlacesClient() # Uses ADC or API key from env implicitly if configured
    except Exception as e:
        initialization_error = f"Failed to initialize Google Maps clients: {e}"
        print(f"Warning: {initialization_error}")
        googlemaps_client = None
        places_client = None
else:
    initialization_error = "googlemaps/google-maps-places packages not installed."

# --- Helper to check client availability ---
def _check_clients_available() -> Optional[str]:
    if not _GOOGLE_MAPS_AVAILABLE:
        return initialization_error
    if initialization_error:
        return initialization_error
    if not googlemaps_client or not places_client:
        return "Google Maps clients are not initialized (check API key and installation)."
    return None

# --- ADK Function Tools ---

async def search_places(query: str) -> Dict[str, Any]:
    """
    Search for places using Google Maps Places API (Text Search).

    Requires the Places API to be enabled for the GOOGLE_MAPS_API_KEY.

    Args:
        query (str): The text query to search for (e.g., "restaurants in London", "Statue of Liberty").

    Returns:
        A dictionary containing:
        - 'status': 'success' or 'error'.
        - 'places': A list of dictionaries with detailed place information (name, address, rating, etc.) if successful.
        - 'message': An error message if status is 'error'.
    """
    client_error = _check_clients_available()
    if client_error or not places_client:
        return {"status": "error", "places": [], "message": client_error or "Places client not available."}

    try:
        request = places_v1.SearchTextRequest(text_query=query)
        # Requesting all fields using wildcard. Adjust field mask for efficiency if needed.
        response = places_client.search_text(request=request, metadata=[("x-goog-fieldmask", "*")])

        places_data = []
        for place in response.places:
            # Safely access potentially missing fields
            hours_desc = []
            if place.regular_opening_hours and place.regular_opening_hours.weekday_descriptions:
                 hours_desc = list(place.regular_opening_hours.weekday_descriptions)

            reviews_data = []
            if place.reviews:
                 reviews_data = [
                     {"text": review.text.text if review.text else None, "rating": review.rating}
                     for review in place.reviews
                 ]

            place_info = {
                "name": place.display_name.text if place.display_name else None,
                "address": place.formatted_address,
                "rating": place.rating,
                "reviews": reviews_data,
                "place_id": place.id,
                "phone": place.international_phone_number,
                "website": place.website_uri,
                "hours": hours_desc,
                # Add more fields as needed from the place object
                "location": {"latitude": place.location.latitude, "longitude": place.location.longitude} if place.location else None,
                "types": list(place.types) if place.types else [],
            }
            places_data.append(place_info)

        return {"status": "success", "places": places_data}

    except Exception as e:
        return {"status": "error", "places": [], "message": f"Error searching Google Maps Places: {str(e)}"}

async def get_directions(
    origin: str,
    destination: str,
    mode: str = "driving",
    departure_time_str: Optional[str] = None, # Use string for easier LLM input
    avoid: Optional[List[str]] = None,
) -> Dict[str, Any]:
    """
    Get directions between two locations using Google Maps Directions API.

    Requires the Directions API to be enabled for the GOOGLE_MAPS_API_KEY.

    Args:
        origin (str): Starting point address or coordinates (e.g., "New York, NY", "40.7128,-74.0060").
        destination (str): Destination address or coordinates (e.g., "Los Angeles, CA", "34.0522,-118.2437").
        mode (str, optional): Travel mode. Options: "driving", "walking", "bicycling", "transit". Defaults to "driving".
        departure_time_str (str, optional): Desired departure time (for transit) as ISO 8601 string (e.g., "2024-12-25T09:00:00Z"). If 'now', uses current time.
        avoid (List[str], optional): Features to avoid: "tolls", "highways", "ferries", "indoor".

    Returns:
        A dictionary containing:
        - 'status': 'success' or 'error'.
        - 'directions': The raw directions result from Google Maps API (usually a list of routes) if successful.
        - 'message': An error message if status is 'error'.
    """
    client_error = _check_clients_available()
    if client_error or not googlemaps_client:
        return {"status": "error", "directions": None, "message": client_error or "Google Maps client not available."}

    departure_dt = None
    if departure_time_str:
        if departure_time_str.lower() == 'now':
            departure_dt = datetime.now()
        else:
            try:
                departure_dt = datetime.fromisoformat(departure_time_str.replace('Z', '+00:00'))
            except ValueError:
                return {"status": "error", "directions": None, "message": "Invalid departure_time_str format. Use ISO 8601 (e.g., '2024-12-25T09:00:00Z') or 'now'."}

    try:
        result = googlemaps.directions(
            origin,
            destination,
            mode=mode,
            departure_time=departure_dt,
            avoid=avoid
        )
        # The result is typically a list of route dictionaries
        return {"status": "success", "directions": result}
    except Exception as e:
        return {"status": "error", "directions": None, "message": f"Error getting directions: {str(e)}"}

async def validate_address(
    address: str, region_code: str = "US", locality: Optional[str] = None, enable_usps_cass: bool = False
) -> Dict[str, Any]:
    """
    Validate an address using Google Maps Address Validation API.

    Requires the Address Validation API to be enabled for the GOOGLE_MAPS_API_KEY.

    Args:
        address (str): The address lines to validate (can be multi-line string).
        region_code (str): The CLDR region code (e.g., "US", "GB", "CA"). Defaults to "US".
        locality (str, optional): The locality (city/town) to help validation.
        enable_usps_cass (bool): Enable USPS CASS certification for US addresses (requires agreement). Defaults to False.

    Returns:
        A dictionary containing:
        - 'status': 'success' or 'error'.
        - 'validation_result': The raw validation result from the API if successful.
        - 'message': An error message if status is 'error'.
    """
    client_error = _check_clients_available()
    if client_error or not googlemaps_client:
        return {"status": "error", "validation_result": None, "message": client_error or "Google Maps client not available."}

    try:
        # The API expects a list of address lines, but often a single string works if formatted well.
        # Passing the input directly might be sufficient. If issues arise, consider splitting `address` by newline.
        address_lines = [address]
        result = googlemaps_client.addressvalidation(
            address_lines,
            regionCode=region_code,
            locality=locality,
            enableUspsCass=enable_usps_cass
        )
        return {"status": "success", "validation_result": result}
    except Exception as e:
        return {"status": "error", "validation_result": None, "message": f"Error validating address: {str(e)}"}

async def geocode_address(address: str, region: Optional[str] = None) -> Dict[str, Any]:
    """
    Convert an address into geographic coordinates (latitude/longitude) using Google Maps Geocoding API.

    Requires the Geocoding API to be enabled for the GOOGLE_MAPS_API_KEY.

    Args:
        address (str): The address to geocode (e.g., "1600 Amphitheatre Parkway, Mountain View, CA").
        region (str, optional): The region code (e.g., "us", "gb") to bias results.

    Returns:
        A dictionary containing:
        - 'status': 'success' or 'error'.
        - 'geocode_results': A list of geocoding results from the API (each with geometry, address components, etc.) if successful.
        - 'message': An error message if status is 'error'.
    """
    client_error = _check_clients_available()
    if client_error or not googlemaps_client:
        return {"status": "error", "geocode_results": [], "message": client_error or "Google Maps client not available."}

    try:
        result = googlemaps_client.geocode(address, region=region)
        return {"status": "success", "geocode_results": result}
    except Exception as e:
        return {"status": "error", "geocode_results": [], "message": f"Error geocoding address: {str(e)}"}

async def reverse_geocode(
    lat: float, lng: float, result_type: Optional[List[str]] = None, location_type: Optional[List[str]] = None
) -> Dict[str, Any]:
    """
    Convert geographic coordinates (latitude/longitude) into a human-readable address using Google Maps Reverse Geocoding API.

    Requires the Geocoding API to be enabled for the GOOGLE_MAPS_API_KEY.

    Args:
        lat (float): Latitude coordinate.
        lng (float): Longitude coordinate.
        result_type (List[str], optional): Filters results to specific address types (e.g., ["postal_code", "locality"]).
        location_type (List[str], optional): Filters results to specific location types (e.g., ["ROOFTOP", "APPROXIMATE"]).

    Returns:
        A dictionary containing:
        - 'status': 'success' or 'error'.
        - 'reverse_geocode_results': A list of address results from the API if successful.
        - 'message': An error message if status is 'error'.
    """
    client_error = _check_clients_available()
    if client_error or not googlemaps_client:
        return {"status": "error", "reverse_geocode_results": [], "message": client_error or "Google Maps client not available."}

    try:
        result = googlemaps_client.reverse_geocode((lat, lng), result_type=result_type, location_type=location_type)
        return {"status": "success", "reverse_geocode_results": result}
    except Exception as e:
        return {"status": "error", "reverse_geocode_results": [], "message": f"Error reverse geocoding: {str(e)}"}

async def get_distance_matrix(
    origins: List[str],
    destinations: List[str],
    mode: str = "driving",
    departure_time_str: Optional[str] = None, # Use string for easier LLM input
    avoid: Optional[List[str]] = None,
) -> Dict[str, Any]:
    """
    Calculate travel distance and time for a matrix of origins and destinations using Google Maps Distance Matrix API.

    Requires the Distance Matrix API to be enabled for the GOOGLE_MAPS_API_KEY.

    Args:
        origins (List[str]): List of starting addresses or coordinates.
        destinations (List[str]): List of destination addresses or coordinates.
        mode (str, optional): Travel mode ("driving", "walking", "bicycling", "transit"). Defaults to "driving".
        departure_time_str (str, optional): Desired departure time (ISO 8601 string or 'now'). Relevant for traffic/transit.
        avoid (List[str], optional): Features to avoid ("tolls", "highways", "ferries", "indoor").

    Returns:
        A dictionary containing:
        - 'status': 'success' or 'error'.
        - 'distance_matrix_result': The raw result dictionary from the API (containing rows, elements with distance/duration) if successful.
        - 'message': An error message if status is 'error'.
    """
    client_error = _check_clients_available()
    if client_error or not googlemaps_client:
        return {"status": "error", "distance_matrix_result": None, "message": client_error or "Google Maps client not available."}

    departure_dt = None
    if departure_time_str:
        if departure_time_str.lower() == 'now':
            departure_dt = datetime.now()
        else:
            try:
                departure_dt = datetime.fromisoformat(departure_time_str.replace('Z', '+00:00'))
            except ValueError:
                return {"status": "error", "distance_matrix_result": None, "message": "Invalid departure_time_str format. Use ISO 8601 or 'now'."}

    try:
        result = googlemaps_client.distance_matrix(
            origins,
            destinations,
            mode=mode,
            departure_time=departure_dt,
            avoid=avoid
        )
        return {"status": "success", "distance_matrix_result": result}
    except Exception as e:
        return {"status": "error", "distance_matrix_result": None, "message": f"Error getting distance matrix: {str(e)}"}

async def get_elevation(lat: float, lng: float) -> Dict[str, Any]:
    """
    Get the elevation (height above sea level) for specific geographic coordinates using Google Maps Elevation API.

    Requires the Elevation API to be enabled for the GOOGLE_MAPS_API_KEY.

    Args:
        lat (float): Latitude coordinate.
        lng (float): Longitude coordinate.

    Returns:
        A dictionary containing:
        - 'status': 'success' or 'error'.
        - 'elevation_results': A list containing elevation data (elevation, resolution, location) from the API if successful.
        - 'message': An error message if status is 'error'.
    """
    client_error = _check_clients_available()
    if client_error or not googlemaps_client:
        return {"status": "error", "elevation_results": [], "message": client_error or "Google Maps client not available."}

    try:
        # API expects list of locations, even for one point
        result = googlemaps_client.elevation([(lat, lng)])
        return {"status": "success", "elevation_results": result}
    except Exception as e:
        return {"status": "error", "elevation_results": [], "message": f"Error getting elevation: {str(e)}"}

async def get_timezone(lat: float, lng: float, timestamp_str: Optional[str] = None) -> Dict[str, Any]:
    """
    Get timezone information for a location using Google Maps Time Zone API.

    Requires the Time Zone API to be enabled for the GOOGLE_MAPS_API_KEY.

    Args:
        lat (float): Latitude coordinate.
        lng (float): Longitude coordinate.
        timestamp_str (str, optional): Timestamp as ISO 8601 string (e.g., "2024-12-25T15:00:00Z") or 'now'. Defaults to current time if None. Used for DST calculations.

    Returns:
        A dictionary containing:
        - 'status': 'success' or 'error'.
        - 'timezone_result': A dictionary with timezone details (dstOffset, rawOffset, timeZoneId, timeZoneName) from the API if successful.
        - 'message': An error message if status is 'error'.
    """
    client_error = _check_clients_available()
    if client_error or not googlemaps_client:
        return {"status": "error", "timezone_result": None, "message": client_error or "Google Maps client not available."}

    timestamp_dt = datetime.now() # Default to now
    if timestamp_str:
        if timestamp_str.lower() == 'now':
            timestamp_dt = datetime.now()
        else:
            try:
                timestamp_dt = datetime.fromisoformat(timestamp_str.replace('Z', '+00:00'))
            except ValueError:
                return {"status": "error", "timezone_result": None, "message": "Invalid timestamp_str format. Use ISO 8601 or 'now'."}

    try:
        result = googlemaps_client.timezone(location=(lat, lng), timestamp=timestamp_dt)
        return {"status": "success", "timezone_result": result}
    except Exception as e:
        return {"status": "error", "timezone_result": None, "message": f"Error getting timezone: {str(e)}"}


# --- Tool Instantiation ---
search_places_tool = FunctionTool(func=search_places) if _GOOGLE_MAPS_AVAILABLE else None
get_directions_tool = FunctionTool(func=get_directions) if _GOOGLE_MAPS_AVAILABLE else None
validate_address_tool = FunctionTool(func=validate_address) if _GOOGLE_MAPS_AVAILABLE else None
geocode_address_tool = FunctionTool(func=geocode_address) if _GOOGLE_MAPS_AVAILABLE else None
reverse_geocode_tool = FunctionTool(func=reverse_geocode) if _GOOGLE_MAPS_AVAILABLE else None
get_distance_matrix_tool = FunctionTool(func=get_distance_matrix) if _GOOGLE_MAPS_AVAILABLE else None
get_elevation_tool = FunctionTool(func=get_elevation) if _GOOGLE_MAPS_AVAILABLE else None
get_timezone_tool = FunctionTool(func=get_timezone) if _GOOGLE_MAPS_AVAILABLE else None

# Filter out None tools in case of import error
_all_tools = [
    search_places_tool,
    get_directions_tool,
    validate_address_tool,
    geocode_address_tool,
    reverse_geocode_tool,
    get_distance_matrix_tool,
    get_elevation_tool,
    get_timezone_tool,
]

google_maps_tools = [tool for tool in _all_tools if tool is not None]
