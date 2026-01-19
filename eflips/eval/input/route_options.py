"""
Route calculation options for geographic visualizations.

This module defines the RouteCalculationMode enum, which controls how route coordinates
are calculated in geographic trip plots.
"""

from enum import Enum


class RouteCalculationMode(Enum):
    """
    Enum controlling how route coordinates are calculated in geographic visualizations.

    This enum is used to specify the method for obtaining route coordinates when
    visualizing trips on a map. Different modes offer different trade-offs between
    accuracy, performance, and data requirements.
    """

    STATIONS_ONLY = "stations_only"
    """
    Always use station coordinates only.

    This mode constructs routes by connecting station points from:
    - departure_station
    - assoc_route_stations (intermediate stops)
    - arrival_station

    This is the simplest mode and requires no external data or API calls, but provides
    straight-line connections between stations rather than actual road routes.
    """

    ROUTE_SHAPES = "route_shapes"
    """
    Use Route.geom if available, fallback to STATIONS_ONLY if not present.

    This mode attempts to use the pre-calculated route geometry stored in Route.geom.
    If Route.geom is None (not available), it automatically falls back to using station
    coordinates only.

    This is the default mode and provides the best balance of accuracy and performance
    when route geometries are available in the database.
    """

    ROUTE_SHAPES_AND_GEO_LOOKUP = "route_shapes_and_geo_lookup"
    """
    Use Route.geom if available, otherwise perform routing lookups using OpenRouteService.

    This mode provides the most accurate routes by:
    1. Using Route.geom if it's available in the database
    2. Otherwise, making API calls to OpenRouteService to calculate routes between consecutive stations

    This mode requires:
    - OPENROUTESERVICE_BASE_URL environment variable
    - OPENROUTESERVICE_API_KEY environment variable

    The routing lookups are performed between all consecutive station pairs on each route,
    and results are cached to minimize API calls. On API failures, falls back to
    straight-line connections between stations.

    Note: This mode may incur API costs and requires an active internet connection.
    """
