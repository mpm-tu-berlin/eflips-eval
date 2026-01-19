"""
Routing functions for geographic trip plots using OpenRouteService API.

This module provides routing functions that calculate route geometries between
stations using the OpenRouteService API. Results are cached to minimize API calls.
"""

import asyncio
import hashlib
import logging
import os
import pickle
import warnings
from tempfile import gettempdir
from typing import List, Optional, Tuple

import httpx
import polyline

logger = logging.getLogger(__name__)


def _get_cache_directory() -> str:
    """
    Get the cache directory for routing results.

    The cache directory can be configured via the EFLIPS_ROUTING_CACHE environment
    variable. If not set, defaults to a temp directory specific to the current user.

    :return: Path to the cache directory
    """
    if "EFLIPS_ROUTING_CACHE" in os.environ:
        cache_dir = os.environ["EFLIPS_ROUTING_CACHE"]
    else:
        cache_dir = os.path.join(gettempdir(), f"eflips-routing-cache-{os.getuid()}")
    os.makedirs(cache_dir, exist_ok=True)
    return cache_dir


def _get_cache_key(coords: Tuple[Tuple[float, float], ...], profile: str) -> str:
    """
    Generate a cache key for a routing request.

    :param coords: Tuple of coordinates (lat, lon) along the route
    :param profile: Routing profile (e.g., "driving-car")
    :return: Cache filename
    """
    key_str = f"{coords}_{profile}"
    return hashlib.md5(key_str.encode()).hexdigest() + ".pkl"


def _load_cached_geometry(cache_key: str) -> Optional[List[Tuple[float, float]]]:
    """
    Load cached route geometry if available.

    :param cache_key: Cache key for the route
    :return: Cached geometry as list of (lat, lon) tuples, or None if not cached
    """
    cache_path = os.path.join(_get_cache_directory(), cache_key)
    if os.path.exists(cache_path):
        try:
            with open(cache_path, "rb") as f:
                ret_val = pickle.load(f)
                if not type(ret_val) == list:
                    raise ValueError(f"Cached geometry file {cache_path} is not a list")
                if not all(
                    type(coord) == tuple and len(coord) == 2 for coord in ret_val
                ):
                    raise ValueError(
                        f"Cached geometry file {cache_path} has invalid format"
                    )
                return ret_val
        except Exception as e:
            logger.warning(f"Failed to load cache file {cache_key}: {e}")
            # Delete corrupted cache file
            try:
                os.remove(cache_path)
            except Exception:
                pass
    return None


def _save_cached_geometry(cache_key: str, geometry: List[Tuple[float, float]]) -> None:
    """
    Save route geometry to cache.

    :param cache_key: Cache key for the route
    :param geometry: Route geometry as list of (lat, lon) tuples
    """
    cache_path = os.path.join(_get_cache_directory(), cache_key)
    try:
        with open(cache_path, "wb") as f:
            pickle.dump(geometry, f)
    except Exception as e:
        logger.warning(f"Failed to save cache file {cache_key}: {e}")


def get_openrouteservice_config() -> Tuple[str, Optional[str], str]:
    """
    Get OpenRouteService API configuration from environment variables.

    This function reads configuration from environment variables:
    - OPENROUTESERVICE_BASE_URL: Base URL for the API
    - OPENROUTESERVICE_API_KEY: API key for authentication (optional for custom instances)
    - OPENROUTESERVICE_PROFILE: Routing profile to use (default: "driving-car")

    :return: Tuple of (base_url, api_key, profile)
    :raises ValueError: If OPENROUTESERVICE_BASE_URL is not set
    """
    base_url = os.environ.get("OPENROUTESERVICE_BASE_URL")
    if not base_url:
        raise ValueError(
            "OPENROUTESERVICE_BASE_URL environment variable must be set for "
            "ROUTE_SHAPES_AND_GEO_LOOKUP mode"
        )

    if base_url.endswith("/"):
        base_url = base_url[:-1]

    api_key = os.environ.get("OPENROUTESERVICE_API_KEY")
    if not api_key:
        warnings.warn(
            "OPENROUTESERVICE_API_KEY is not set. This only works on custom instances without authentication."
        )

    profile = os.environ.get("OPENROUTESERVICE_PROFILE", "driving-car")

    return base_url, api_key, profile


async def _route_through_stations_async(
    station_coords: List[Tuple[float, float]],
    base_url: str,
    api_key: Optional[str],
    client: httpx.AsyncClient,
    profile: str = "driving-car",
) -> List[Tuple[float, float]]:
    """
    Internal async function to calculate route through multiple stations.

    :param station_coords: List of station coordinates (lat, lon)
    :param base_url: OpenRouteService base URL
    :param api_key: API key (optional for custom instances)
    :param client: Shared httpx AsyncClient
    :param profile: Routing profile (default: "driving-car")
    :return: Route geometry as list of (lat, lon) tuples
    """
    if len(station_coords) < 2:
        return station_coords

    # Generate cache key from all coordinates
    cache_key = _get_cache_key(tuple(station_coords), profile)

    # Try cache first
    cached = _load_cached_geometry(cache_key)
    if cached is not None:
        logger.debug(f"Cache hit for route with {len(station_coords)} stations")
        return cached

    logger.debug(
        f"Cache miss, calling API for route with {len(station_coords)} stations"
    )

    try:
        # Convert from (lat, lon) to [lon, lat] for OpenRouteService
        coords = [[coord[1], coord[0]] for coord in station_coords]

        # Remove duplicates while preserving order
        seen = set()
        unique_coords = []
        for c in coords:
            coord_tuple = (c[0], c[1])
            if coord_tuple not in seen:
                seen.add(coord_tuple)
                unique_coords.append(c)
        coords = unique_coords

        if len(coords) < 2:
            logger.warning(
                f"All station coordinates are identical for {len(station_coords)} stations, "
                "falling back to straight lines"
            )
            return station_coords

        # Prepare request
        url = f"{base_url}/v2/directions/{profile}"
        headers = {}
        if api_key:
            headers["Authorization"] = api_key

        body = {"coordinates": coords}

        # Make async HTTP POST request
        response = await client.post(url, json=body, headers=headers)
        response.raise_for_status()
        result = response.json()

        # Extract and decode geometry
        if "routes" not in result or not result["routes"]:
            logger.warning(
                f"No routes found for {len(station_coords)} stations, "
                "falling back to straight lines"
            )
            geometry = station_coords
        else:
            encoded_geometry = result["routes"][0]["geometry"]
            decoded = polyline.decode(encoded_geometry)

            # Convert back to (lat, lon)
            geometry = [(float(coord[0]), float(coord[1])) for coord in decoded]

        # Cache result
        _save_cached_geometry(cache_key, geometry)

        logger.info(f"Successfully routed through {len(station_coords)} stations")
        return geometry

    except (httpx.HTTPError, httpx.TimeoutException) as e:
        logger.error(f"Routing API error for {len(station_coords)} stations: {e}")
        logger.warning("Falling back to straight lines between stations")
        return station_coords
    except (KeyError, ValueError) as e:
        logger.error(f"Invalid API response for {len(station_coords)} stations: {e}")
        logger.warning("Falling back to straight lines between stations")
        return station_coords
    except Exception as e:
        logger.error(
            f"Unexpected routing error for {len(station_coords)} stations: {e}"
        )
        logger.warning("Falling back to straight lines between stations")
        return station_coords


async def _calculate_route_geometries_async(
    station_lists: List[List[Tuple[float, float]]],
    base_url: str,
    api_key: Optional[str],
) -> List[List[Tuple[float, float]]]:
    """
    Internal async function to calculate route geometries concurrently.

    :param station_lists: List of station coordinate lists, each for one trip
    :param base_url: OpenRouteService base URL
    :param api_key: API key (optional for custom instances)
    :return: List of route geometries, each as a list of (lat, lon) tuples
    """
    if not station_lists:
        return []

    logger.info(f"Calculating routes for {len(station_lists)} trips")

    # Use a single shared client for all requests
    async with httpx.AsyncClient(timeout=30.0) as client:
        # Create all tasks and run them concurrently
        tasks = [
            _route_through_stations_async(stations, base_url, api_key, client)
            for stations in station_lists
        ]
        geometries = await asyncio.gather(*tasks)

    logger.info(f"Completed routing for {len(station_lists)} trips")

    return list(geometries)


def calculate_route_geometries(
    station_lists: List[List[Tuple[float, float]]],
    base_url: str,
    api_key: Optional[str],
) -> List[List[Tuple[float, float]]]:
    """
    Calculate route geometries for multiple trips with IO concurrency.

    This function processes routing requests concurrently using asyncio for IO operations.
    Each trip routes through all its stations in a single API call. Results are cached
    to minimize API calls. The async complexity is hidden behind a synchronous interface.

    :param station_lists: List of station coordinate lists, each for one trip
    :param base_url: OpenRouteService base URL
    :param api_key: API key (optional for custom instances)
    :return: List of route geometries, each as a list of (lat, lon) tuples
    """
    return asyncio.run(
        _calculate_route_geometries_async(station_lists, base_url, api_key)
    )
