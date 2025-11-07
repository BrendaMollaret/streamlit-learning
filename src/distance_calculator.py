"""
Distance calculator using a local Nominatim (Docker) geocoding API.

Changes
- Queries are routed to a local Nominatim container at localhost:8080
- All geocoding is constrained to the Province (state) of Mendoza, Argentina
- Removes public OSM rate-limit policy delays; tuned for local service
"""
from __future__ import annotations

from ctypes import addressof
import math
import os
import time
from dataclasses import dataclass
from typing import Optional, Tuple

import requests

# Use local Nominatim container
NOMINATIM_SEARCH_URL = "http://localhost:8080/search"
NOMINATIM_REVERSE_URL = "https://nominatim.openstreetmap.org/reverse"
DEFAULT_CITY = "Mendoza"
DEFAULT_COUNTRY = "Argentina"
DEFAULT_STATE = "Mendoza"

# Default UA is optional for local; keep for consistency
DEFAULT_USER_AGENT = os.getenv(
    "NOMINATIM_USER_AGENT",
    "calcdist-distance-calculator/1.0 (local nominatim)",
)


class GeocodingError(Exception):
    """Base error for geocoding issues."""


class InvalidAddressError(GeocodingError):
    """Raised when no geocoding result is found for the given address."""


class RateLimitError(GeocodingError):
    """Raised when the geocoding API rate limits requests beyond allowed retries."""


class NetworkError(GeocodingError):
    """Raised when network/connectivity issues prevent contacting the API."""


@dataclass
class GeoPoint:
    lat: float
    lon: float


def _get_session(user_agent: Optional[str] = None, timeout: float = 10.0) -> requests.Session:
    """Create a configured requests.Session with proper headers and timeouts."""
    ua = user_agent or DEFAULT_USER_AGENT
    s = requests.Session()
    s.headers.update({
        "User-Agent": ua,
        "Accept": "application/json",
    })
    # Using a default timeout via a wrapper function for requests
    s.request = _wrap_request_with_timeout(s.request, timeout)
    return s


def _wrap_request_with_timeout(original_request, timeout: float):
    """Wrap session.request to enforce a default timeout for all calls."""
    def wrapped(method, url, **kwargs):
        if "timeout" not in kwargs:
            kwargs["timeout"] = timeout
        return original_request(method, url, **kwargs)
    return wrapped


def geocode_address(
    street: str,
    city: str = DEFAULT_CITY,  # Ignored; enforced to state-only
    country: str = DEFAULT_COUNTRY,  # Ignored; enforced via countrycodes
    *,
    user_agent: Optional[str] = None,
    per_request_delay: float = 0.0,
    max_attempts: int = 3,
    timeout: float = 10.0,
) -> GeoPoint:
    """Geocode a street address using local Nominatim, constrained to Mendoza Province.

    Parameters
    - street: Street address or free-text query (e.g., "San Martín 123", "Godoy Cruz")
    - user_agent: Optional UA header
    - per_request_delay: Optional delay between retries (defaults to 0 for local)
    - max_attempts: Maximum retry attempts on transient errors
    - timeout: Per-request timeout in seconds

    Returns
    - GeoPoint(lat, lon)
    """
    session = _get_session(user_agent=user_agent, timeout=timeout)

    # Strict province/country scope
    # viewbox bounding Mendoza province roughly: minlon, minlat, maxlon, maxlat
    VIEWBOX_MENDOZA = "-70,-36,-66,-32"

    # Use free-text query to improve hit rate on non-street inputs
    params = {
        "q": street,
        "state": DEFAULT_STATE,
        "countrycodes": "ar",
        "format": "jsonv2",
        "addressdetails": 0,
        "limit": 1,
        "bounded": 1,
        "viewbox": VIEWBOX_MENDOZA,
    }

    attempt = 0
    backoff = max(per_request_delay, 0.0)

    while attempt < max_attempts:
        attempt += 1
        try:
            if per_request_delay > 0:
                time.sleep(per_request_delay)
            resp = session.get(NOMINATIM_SEARCH_URL, params=params)
        except requests.exceptions.RequestException as e:
            if attempt >= max_attempts:
                raise NetworkError(f"Network error contacting local geocoding API: {e}") from e
            time.sleep(min(backoff, 2.0))
            backoff = min(backoff * 2, 2.0)
            continue

        if resp.status_code >= 500:
            if attempt >= max_attempts:
                raise GeocodingError(f"Local geocoding server error: HTTP {resp.status_code}")
            time.sleep(min(backoff, 2.0))
            backoff = min(backoff * 2, 2.0)
            continue
        elif resp.status_code != 200:
            raise GeocodingError(f"Local geocoding API error: HTTP {resp.status_code}")

        try:
            results = resp.json()
        except ValueError as e:
            raise GeocodingError("Invalid JSON response from local geocoding API") from e

        if not results:
            raise InvalidAddressError(f"Address not found within Mendoza province: {street}")

        r = results[0]
        try:
            lat = float(r["lat"])
            lon = float(r["lon"])
        except (KeyError, ValueError) as e:
            raise GeocodingError("Geocoding response missing coordinates") from e

        return GeoPoint(lat=lat, lon=lon)

    raise GeocodingError("Unexpected geocoding failure")


def haversine_km(p1: GeoPoint, p2: GeoPoint) -> float:
    """Compute the haversine distance between two points in kilometers."""
    R = 6371.0088  # IUGG mean Earth radius in kilometers
    lat1 = math.radians(p1.lat)
    lat2 = math.radians(p2.lat)
    dlat = lat2 - lat1
    dlon = math.radians(p2.lon - p1.lon)

    a = math.sin(dlat / 2) ** 2 + math.cos(lat1) * math.cos(lat2) * math.sin(dlon / 2) ** 2
    c = 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))
    return R * c


def calculate_distance_km(
    address1: str,
    address2: str,
    *,
    city: str = DEFAULT_CITY,
    country: str = DEFAULT_COUNTRY,
    user_agent: Optional[str] = None,
    per_request_delay: float = 1.0,
    timeout: float = 10.0,
) -> float:
    """Calculate straight-line distance in km between two street addresses.

    Parameters
    - address1/address2: Street addresses (e.g., "San Martín 123")
    - city: Defaults to "Mendoza"
    - country: Defaults to "Argentina"
    - user_agent: Optional custom User-Agent header
    - per_request_delay: Courtesy delay between requests (seconds)
    - timeout: Per-request timeout (seconds)

    Returns
    - Distance in kilometers, rounded to 2 decimal places.

    Raises
    - InvalidAddressError, AmbiguousAddressError, RateLimitError, NetworkError, GeocodingError
    """
    p1 = geocode_address(
        address1,
        city=city,
        country=country,
        user_agent=user_agent,
        per_request_delay=per_request_delay,
        timeout=timeout,
    )
    # Ensure we respect 1 req/sec between separate requests
    time.sleep(per_request_delay)
    p2 = geocode_address(
        address2,
        city=city,
        country=country,
        user_agent=user_agent,
        per_request_delay=per_request_delay,
        timeout=timeout,
    )

    dist = haversine_km(p1, p2)
    return round(dist, 2)


def reverse_geocode(
    lat: float,
    lon: float,
    *,
    user_agent: Optional[str] = None,
    per_request_delay: float = 1.0,
    timeout: float = 10.0,
) -> str:
    """Reverse-geocode coordinates to a human-readable address using public OSM Nominatim.

    Parameters
    - lat: Latitude
    - lon: Longitude
    - user_agent: Optional custom User-Agent header
    - per_request_delay: Courtesy delay between requests (seconds)
    - timeout: Per-request timeout in seconds

    Returns
    - Address string (Nominatim "display_name").

    Raises
    - InvalidAddressError, NetworkError, GeocodingError
    """
    session = _get_session(user_agent=user_agent, timeout=timeout)
    params = {
        "lat": lat,
        "lon": lon,
        "format": "jsonv2",
        "addressdetails": 1,
    }

    try:
        resp = session.get(NOMINATIM_REVERSE_URL, params=params)
    except requests.exceptions.RequestException as e:
        raise NetworkError(f"Network error contacting public reverse geocoding API: {e}") from e

    if resp.status_code >= 500:
        raise GeocodingError(f"Reverse geocoding server error: HTTP {resp.status_code}")
    elif resp.status_code != 200:
        raise GeocodingError(f"Reverse geocoding API error: HTTP {resp.status_code}")

    try:
        data = resp.json()
    except ValueError as e:
        raise GeocodingError("Invalid JSON response from reverse geocoding API") from e

    address = data.get("display_name")
    if not address:
        raise InvalidAddressError("No address found for given coordinates")

    return address


__all__ = [
    "calculate_distance_km",
    "geocode_address",
    "haversine_km",
    "GeoPoint",
    "GeocodingError",
    "InvalidAddressError",
    # "AmbiguousAddressError",
    "RateLimitError",
    "NetworkError",
    "reverse_geocode",
]

