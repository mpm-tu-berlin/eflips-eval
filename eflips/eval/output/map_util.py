"""
Utility functions for interactive map generation.

This module provides helper functions for creating interactive folium maps,
following the DRY (Don't Repeat Yourself) principle and clean code practices.
"""

from collections import defaultdict
from typing import Dict, List, Tuple, Optional, Set

import geoalchemy2.shape
import numpy as np
from eflips.model import Depot, Event, Rotation, Route, Station, Trip
from sqlalchemy.orm import Session, joinedload


def generate_depot_colors(depot_ids: List[int]) -> Dict[int, Dict[str, str]]:
    """
    Generate a consistent color palette for depots using folium-compatible colors.

    Returns both icon color names (for folium Icons) and hex colors (for PolyLines).
    Special case: If only one depot exists, returns black for all.

    :param depot_ids: List of depot IDs to generate colors for

    :return: Dictionary mapping depot_id to dict with 'icon_color' and 'hex_color'
    """
    # Folium Icon compatible colors with their hex equivalents
    # Excluding white, beige, lightgray as they're hard to see on maps
    FOLIUM_COLORS = [
        ("red", "#d9534f"),
        ("blue", "#5bc0de"),
        ("green", "#5cb85c"),
        ("purple", "#9b59b6"),
        ("orange", "#f0ad4e"),
        ("darkred", "#8b0000"),
        ("darkblue", "#00008b"),
        ("darkgreen", "#006400"),
        ("cadetblue", "#5f9ea0"),
        ("darkpurple", "#663399"),
        ("lightred", "#ff6b6b"),
        ("lightblue", "#add8e6"),
        ("lightgreen", "#90ee90"),
        ("pink", "#ff69b4"),
        ("gray", "#808080"),
        ("black", "#000000"),
    ]

    if len(depot_ids) <= 1:
        return {
            depot_id: {"icon_color": "black", "hex_color": "#000000"}
            for depot_id in depot_ids
        }

    # Use colors cyclically if more depots than colors
    color_map = {}
    for i, depot_id in enumerate(depot_ids):
        icon_color, hex_color = FOLIUM_COLORS[i % len(FOLIUM_COLORS)]
        color_map[depot_id] = {"icon_color": icon_color, "hex_color": hex_color}

    return color_map


def extract_coordinates_from_geometry(
    geom: Optional[geoalchemy2.elements.WKBElement],
    route: Optional[Route] = None,
) -> List[Tuple[float, float]]:
    """
    Extract (lat, lon) tuples from route geometry or station points.

    Based on the logic from eflips.eval.input.prepare.geographic_trip_plot().
    If geometry is provided, extracts coordinates from it.
    Otherwise, constructs coordinates from route's departure, intermediate, and arrival stations.

    :param geom: GeoAlchemy2 geometry element (route shape)
    :param route: Route object (needed if geom is None for fallback)
    :return: List of (latitude, longitude) tuples
    """
    if geom is not None:
        line_geom = geoalchemy2.shape.to_shape(geom)
        # Note: geometry coords are (lon, lat), we need (lat, lon) for folium
        return [(float(point[1]), float(point[0])) for point in line_geom.coords]

    if route is None:
        raise ValueError("Route object must be provided when geometry is None")

    # Fallback: construct from stations
    line_coords = []

    # Add departure station
    point_geom = geoalchemy2.shape.to_shape(route.departure_station.geom)  # type: ignore[arg-type]
    lon, lat = point_geom.x, point_geom.y
    line_coords.append((lat, lon))

    # Add intermediate stations
    for assoc in sorted(route.assoc_route_stations, key=lambda x: x.elapsed_distance):
        if assoc.location is not None:
            station_coordinates = geoalchemy2.shape.to_shape(assoc.location)  # type: ignore[arg-type]
        else:
            station_coordinates = geoalchemy2.shape.to_shape(assoc.station.geom)
        lon, lat = station_coordinates.x, station_coordinates.y
        line_coords.append((lat, lon))

    # Add arrival station
    point_geom = geoalchemy2.shape.to_shape(route.arrival_station.geom)  # type: ignore[arg-type]
    lon, lat = point_geom.x, point_geom.y
    line_coords.append((lat, lon))

    return line_coords


def calculate_map_center(stations: List[Station]) -> Tuple[float, float]:
    """
    Calculate the mean latitude and longitude for map centering.

    :param stations: List of Station objects with geometry

    :return: Tuple of (latitude, longitude)
    """
    if not stations:
        # Default center (roughly central Europe)
        return (50.0, 10.0)

    latitudes = []
    longitudes = []

    for station in stations:
        point = geoalchemy2.shape.to_shape(station.geom)  # type: ignore[arg-type]
        latitudes.append(point.y)
        longitudes.append(point.x)

    return (float(np.mean(latitudes)), float(np.mean(longitudes)))


def get_vehicle_counts_by_depot(
    scenario_id: int, session: Session
) -> Dict[int, Dict[str, int]]:
    """
    Get vehicle counts grouped by depot and vehicle type.

    Query pattern:
    1. Get all Rotations for scenario with vehicle and vehicle_type loaded
    2. For each rotation:

        - Get depot via first trip's departure station (matches Depot.station_id)
        - Track vehicle_id and vehicle_type_name

    3. Group by depot_id -> vehicle_type_name -> count unique vehicle_ids

    :param scenario_id: The scenario ID
    :param session: SQLAlchemy session

    :return: Dictionary {depot_id: {vehicle_type_name: vehicle_count}}
    """
    from eflips.model import Vehicle

    # Query rotations with necessary relationships loaded
    rotations = (
        session.query(Rotation)
        .filter(Rotation.scenario_id == scenario_id)
        .filter(
            Rotation.vehicle_id.isnot(None)
        )  # Only rotations with assigned vehicles
        .options(
            joinedload(Rotation.vehicle).joinedload(Vehicle.vehicle_type),
            joinedload(Rotation.trips)
            .joinedload(Trip.route)
            .joinedload(Route.departure_station),
        )
        .all()
    )

    # Get depot mapping (station_id -> depot_id)
    depots = session.query(Depot).filter(Depot.scenario_id == scenario_id).all()
    station_to_depot = {depot.station_id: depot.id for depot in depots}

    # Track vehicles by depot and type
    # Structure: {depot_id: {vehicle_type_name: set(vehicle_ids)}}
    depot_vehicles: Dict[int, Dict[str, Set[int]]] = defaultdict(
        lambda: defaultdict(set)
    )

    for rotation in rotations:
        if not rotation.trips:
            continue

        # Get depot from first trip's departure station
        first_trip = sorted(rotation.trips, key=lambda t: t.departure_time)[0]
        departure_station_id = first_trip.route.departure_station_id

        if departure_station_id not in station_to_depot:
            continue

        depot_id = station_to_depot[departure_station_id]
        vehicle_type_name = rotation.vehicle.vehicle_type.name
        vehicle_id = rotation.vehicle_id

        depot_vehicles[depot_id][vehicle_type_name].add(vehicle_id)

    # Convert sets to counts
    result: Dict[int, Dict[str, int]] = {}
    for depot_id, vehicle_types in depot_vehicles.items():
        result[depot_id] = {
            vtype_name: len(vehicle_ids)
            for vtype_name, vehicle_ids in vehicle_types.items()
        }

    return result


def get_routes_with_events(scenario_id: int, session: Session) -> List[Route]:
    """
    Get only routes that have corresponding Events.

    This is important: only routes with actual simulation events should be displayed.

    Query pattern:

    1. Query distinct Route IDs from Events via Trip
    2. Load routes with geometry and stations

    :param scenario_id: The scenario ID
    :param session: SQLAlchemy session
    :return: List of Route objects that have associated events
    """
    # Get distinct route IDs that have events
    route_ids_with_events = (
        session.query(Route.id)
        .join(Trip, Route.id == Trip.route_id)
        .join(Event, Trip.id == Event.trip_id)
        .filter(Route.scenario_id == scenario_id)
        .distinct()
        .all()
    )

    route_ids = [route_id[0] for route_id in route_ids_with_events]

    if not route_ids:
        return []

    # Load routes with necessary relationships
    routes = (
        session.query(Route)
        .filter(Route.id.in_(route_ids))
        .options(
            joinedload(Route.departure_station),
            joinedload(Route.arrival_station),
            joinedload(Route.line),
            joinedload(Route.assoc_route_stations).joinedload(
                Route.assoc_route_stations.property.mapper.class_.station
            ),
        )
        .all()
    )

    return routes
