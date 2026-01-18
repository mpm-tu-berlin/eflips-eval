import asyncio
from asyncio import Task
from collections import Counter
from datetime import datetime
from typing import Dict, List, Tuple, Any, Sequence

import httpx
import pandas as pd
import sqlalchemy
from eflips.model import Rotation, Trip, Route, Station, TripType
from geoalchemy2.shape import to_shape
from shapely import wkb  # type: ignore
from shapely.geometry.linestring import LineString  # type: ignore

from eflips.eval.input.route_options import RouteCalculationMode
from eflips.eval.input.routing import (
    get_openrouteservice_config,
    calculate_route_geometries,
    _route_through_stations_async,
)


def rotation_info(
    scenario_id: int,
    session: sqlalchemy.orm.session.Session,
    rotation_ids: None | int | List[int] = None,
) -> pd.DataFrame:
    """
    This function provides information about the rotations in a scenario. This information can be provided even before
    the simulation has been run. It creates a dataframe with the following columns:

    - rotation_id: the id of the rotation
    - rotation_name: the name of the rotation
    - vehicle_type_id: the id of the vehicle type
    - vehicle_type_name: the name of the vehicle type
    - total_distance: the total distance of the rotation
    - time_start: the departure of the first trip
    - time_end: the arrival of the last trip
    - line_name: the name of the line, which is the first part of the rotation name. Used for sorting
    - line_is_unified: True if the rotation only contains one line
    - start_station: the name of the departure station
    - end_station: the name of the arrival station

    :param scenario_id: The scenario id for which to create the dataframe
    :param session: An sqlalchemy session to an eflips-model database
    :param rotation_ids: A list of rotation ids to filter for. If None, all rotations are included
    :return: a pandas DataFrame
    """

    result: List[Dict[str, int | float | str | datetime]] = []

    rotations = (
        session.query(Rotation)
        .filter(Rotation.scenario_id == scenario_id)
        .options(
            sqlalchemy.orm.joinedload(Rotation.trips)
            .joinedload(Trip.route)
            .joinedload(Route.line),
            sqlalchemy.orm.joinedload(Rotation.vehicle_type),
            sqlalchemy.orm.joinedload(Rotation.trips)
            .joinedload(Trip.route)
            .joinedload(Route.departure_station),
            sqlalchemy.orm.joinedload(Rotation.trips)
            .joinedload(Trip.route)
            .joinedload(Route.arrival_station),
        )
    )

    if rotation_ids is not None:
        if isinstance(rotation_ids, int):
            rotation_ids = [rotation_ids]
        rotations = rotations.filter(Rotation.id.in_(rotation_ids))

    for rotation in rotations:
        # The rotation distance comes form the routes of the trips
        distance = 0.0
        for trip in rotation.trips:
            distance += trip.route.distance / 1000

        # We want to be able to sort and/or group by line. Therefore we need to identify the predominant line name for each
        # rotation.
        line_names: Dict[str, int] = {}
        for trip in rotation.trips:
            line_name = trip.route.line.name if trip.route.line is not None else "N/A"
            if line_name not in line_names:
                line_names[line_name] = 0
            line_names[line_name] += 1
        line_name = Counter(line_names).most_common(1)[0][0]

        result.append(
            {
                "rotation_id": rotation.id,
                "rotation_name": (
                    rotation.name
                    if rotation.name is not None
                    else f"Unnamed Rotation ({rotation.id})"
                ),
                "vehicle_type_id": rotation.vehicle_type_id,
                "vehicle_type_name": rotation.vehicle_type.name,
                "total_distance": distance,
                "line_name": line_name,
                "line_is_unified": len(line_names)
                == 1,  # True if there is only one line in the rotation
                "time_start": rotation.trips[0].departure_time,
                "time_end": rotation.trips[-1].arrival_time,
                "start_station": rotation.trips[0].route.departure_station.name,
                "end_station": rotation.trips[-1].route.arrival_station.name,
            }
        )

    df = pd.DataFrame(result)

    df.sort_values(by=["line_name", "time_start"], inplace=True)

    return df


def _station_to_coord(station: Station) -> Tuple[float, float]:
    """
    Convert a station's geometry to (lat, lon) coordinates.

    :param station: The station object
    :return: Tuple of (latitude, longitude)
    """
    point = to_shape(station.geom)  # type: ignore[arg-type]
    return (point.y, point.x)


def _extract_geom_coords(trip: Trip) -> List[Tuple[float, float]]:
    """
    Extract coordinates from trip.route.geom.

    :param trip: The trip object
    :return: List of (lat, lon) tuples representing the route geometry
    """
    line_geom: LineString = to_shape(trip.route.geom)  # type: ignore
    return [(float(point[1]), float(point[0])) for point in line_geom.coords]


def _extract_station_coords(trip: Trip) -> List[Tuple[float, float]]:
    """
    Extract coordinates from departure, intermediate, and arrival stations.

    :param trip: The trip object
    :return: List of (lat, lon) tuples connecting all stations
    """
    line_coords = []

    # Departure station
    point_geom = to_shape(trip.route.departure_station.geom)  # type: ignore[arg-type]
    lon, lat = point_geom.x, point_geom.y
    line_coords.append((lat, lon))

    # Intermediate stations
    for assoc in trip.route.assoc_route_stations:
        if assoc.location is not None:
            station_coordinates = to_shape(assoc.location)  # type: ignore[arg-type]
        else:
            station_coordinates = to_shape(assoc.station.geom)
        lon, lat = station_coordinates.x, station_coordinates.y
        line_coords.append((lat, lon))

    # Arrival station
    point_geom = to_shape(trip.route.arrival_station.geom)  # type: ignore[arg-type]
    lon, lat = point_geom.x, point_geom.y
    line_coords.append((lat, lon))

    return line_coords


def _get_all_station_coords(trip: Trip) -> List[Tuple[float, float]]:
    """
    Get all station coordinates for a trip in order.

    :param trip: The trip object
    :return: List of (lat, lon) tuples for all stations on the route
    """
    coords = []

    # Departure station
    coords.append(_station_to_coord(trip.route.departure_station))

    # Intermediate stations
    for assoc in sorted(
        trip.route.assoc_route_stations, key=lambda a: a.elapsed_distance
    ):
        coords.append(_station_to_coord(assoc.station))

    # Arrival station
    coords.append(_station_to_coord(trip.route.arrival_station))

    return coords


def _split_stations_into_chunks(
    stations: List[Tuple[float, float]], max_chunk_size: int = 50
) -> List[List[Tuple[float, float]]]:
    """
    Split a list of station coordinates into overlapping chunks.

    OpenRouteService has a limit on the number of waypoints per request (typically 50).
    This function splits long routes into chunks with 1-point overlap to ensure continuity.

    :param stations: List of station coordinates
    :param max_chunk_size: Maximum waypoints per chunk (default: 50)
    :return: List of coordinate chunks with overlap
    """
    if len(stations) <= max_chunk_size:
        return [stations]

    chunks = []
    start = 0

    while start < len(stations):
        end = min(start + max_chunk_size, len(stations))
        chunks.append(stations[start:end])

        # Next chunk starts at the last point of current chunk (overlap)
        # unless we've reached the end
        if end < len(stations):
            start = end - 1
        else:
            break

    return chunks


def _combine_route_geometries(
    geometries: List[List[Tuple[float, float]]]
) -> List[Tuple[float, float]]:
    """
    Combine multiple route geometries into a single continuous route.

    Removes duplicate points at chunk boundaries that were created by overlap.

    :param geometries: List of route geometries (each a list of coordinates)
    :return: Combined route geometry
    """
    if not geometries:
        return []

    if len(geometries) == 1:
        return geometries[0]

    combined = list(geometries[0])

    for geometry in geometries[1:]:
        # Skip the first point of subsequent geometries (overlap from previous chunk)
        combined.extend(geometry[1:])

    return combined


async def _process_rotations_with_routing(
    rotations: sqlalchemy.orm.query.Query[Rotation],
    base_url: str,
    api_key: str | None,
    profile: str,
    passenger_trips_only: bool = False,
) -> List[Dict[str, int | float | str | datetime | List[Tuple[float, float]]]]:
    """
    Async helper to process rotations with immediate route lookup.

    Starts routing tasks immediately when encountering trips without geom,
    allowing concurrent execution during iteration. Handles routes with >50
    waypoints by splitting into chunks and reassembling.

    :param rotations: Query result of rotations to process
    :param base_url: OpenRouteService base URL
    :param api_key: API key (optional for custom instances)
    :param profile: Routing profile (e.g., "driving-car", "driving-hgv")
    :param passenger_trips_only: Whether to filter for passenger trips only
    :return: List of result dictionaries
    """
    result: List[
        Dict[str, int | float | str | datetime | List[Tuple[float, float]]]
    ] = []
    tasks_to_resolve: List[
        Tuple[int, List[asyncio.Task[List[tuple[float, float]]]]]
    ] = []
    line_coords: (
        Any  # make mypy happy (and making it more narrow doesn't seem to work :( )
    )

    # Create shared HTTP client for all routing requests
    async with httpx.AsyncClient(timeout=30.0) as client:
        # Iterate through rotations
        for rotation in rotations:
            # Extract rotation metadata
            origin_depot_id = rotation.trips[0].route.departure_station_id
            origin_depot_name = rotation.trips[0].route.departure_station.name
            line_name = rotation.trips[0].route.line.name
            vehicle_type_id = rotation.vehicle_type_id
            vehicle_type_name = rotation.vehicle_type.name

            for trip in rotation.trips:
                if passenger_trips_only and trip.trip_type != TripType.PASSENGER:
                    continue
                # If route has geom, extract coordinates directly
                if trip.route.geom is not None:
                    line_coords = _extract_geom_coords(trip)
                else:
                    # Get station coordinates
                    station_coords = _get_all_station_coords(trip)

                    # Split into chunks if more than 50 waypoints
                    chunks = _split_stations_into_chunks(
                        station_coords, max_chunk_size=50
                    )

                    # Create async task for each chunk and START IMMEDIATELY
                    chunk_tasks = []
                    for chunk in chunks:
                        task = asyncio.create_task(
                            _route_through_stations_async(
                                chunk, base_url, api_key, client, profile
                            )
                        )
                        chunk_tasks.append(task)
                        # Yield control to event loop so task can start executing
                        await asyncio.sleep(0)

                    # Store all chunk tasks for later resolution
                    tasks_to_resolve.append((len(result), chunk_tasks))
                    # Temporarily store tasks as placeholder; will be replaced with actual coordinates
                    # after asyncio.gather() completes in the resolution phase
                    line_coords = chunk_tasks

                # Append result with either coordinates or tasks
                result.append(
                    {
                        "rotation_id": rotation.id,
                        "rotation_name": rotation.name,
                        "vehicle_type_id": vehicle_type_id,
                        "vehicle_type_name": vehicle_type_name,
                        "originating_depot_id": origin_depot_id,
                        "originating_depot_name": origin_depot_name,
                        "distance": trip.route.distance,
                        "coordinates": line_coords,
                        "line_name": line_name,
                    }
                )

        # Resolve all routing tasks and combine chunks
        for result_idx, chunk_tasks in tasks_to_resolve:
            # Await all chunk tasks
            chunk_geometries = await asyncio.gather(*chunk_tasks)
            # Combine geometries from all chunks
            combined_geometry = _combine_route_geometries(chunk_geometries)
            result[result_idx]["coordinates"] = combined_geometry

    return result


def geographic_trip_plot(
    scenario_id: int,
    session: sqlalchemy.orm.session.Session,
    rotation_ids: None | int | List[int] = None,
    route_calculation_mode: RouteCalculationMode = RouteCalculationMode.ROUTE_SHAPES,
    passenger_trips_only: bool = True,
) -> pd.DataFrame:
    """
    This function creates a dataframe that can be used to visualize the geographic distribution of rotations. It creates
    a dataframe with one row for each trip and the following columns:

    - rotation_id: the id of the rotation
    - rotation_name: the name of the rotation
    - vehicle_type_id: the id of the vehicle type
    - vehicle_type_name: the name of the vehicle type
    - originating_depot_id: the id of the originating depot
    - originating_depot_name: the name of the originating depot
    - distance: the distance of the route
    - coordinates: An array of (lat, lon) tuples with the coordinates of the route - the shape if set, otherwise the stops
    - line_name: the name of the line, which is the first part of the rotation name. Used for sorting

    :param scenario_id: The scenario id for which to create the dataframe
    :param session: An sqlalchemy session to an eflips-model database
    :param rotation_ids: A list of rotation ids to filter for. If None, all rotations are included
    :param route_calculation_mode: RouteCalculationMode enum controlling how coordinates are obtained.
        - STATIONS_ONLY: Use station points only
        - ROUTE_SHAPES: Use Route.geom if available, fallback to stations (default)
        - ROUTE_SHAPES_AND_GEO_LOOKUP: Use Route.geom if available, else lookup via OpenRouteService API
    :param passenger_trips_only: If True, only passenger trips are included
    :return: a pandas DataFrame
    """
    rotations_q = session.query(Rotation).filter(Rotation.scenario_id == scenario_id)
    if rotation_ids is not None:
        if isinstance(rotation_ids, int):
            rotation_ids = [rotation_ids]
        rotations_q = rotations_q.filter(Rotation.id.in_(rotation_ids))
    rotations_q = rotations_q.options(
        sqlalchemy.orm.joinedload(Rotation.trips)
        .joinedload(Trip.route)
        .joinedload(Route.line)
    )
    rotations_q = rotations_q.options(
        sqlalchemy.orm.joinedload(Rotation.vehicle_type),
    )
    rotations_q = rotations_q.options(
        sqlalchemy.orm.joinedload(Rotation.trips)
        .joinedload(Trip.route)
        .joinedload(Route.departure_station),
    )

    # Handle ROUTE_SHAPES_AND_GEO_LOOKUP mode separately with async processing
    if route_calculation_mode == RouteCalculationMode.ROUTE_SHAPES_AND_GEO_LOOKUP:
        base_url, api_key, profile = get_openrouteservice_config()
        return pd.DataFrame(
            asyncio.run(
                _process_rotations_with_routing(
                    rotations_q,
                    base_url,
                    api_key,
                    profile,
                    passenger_trips_only=passenger_trips_only,
                )
            )
        )

    # Handle STATIONS_ONLY and ROUTE_SHAPES modes (synchronous processing)
    result: List[
        Dict[str, int | float | str | datetime | List[Tuple[float, float]]]
    ] = []

    for rotation in rotations_q:
        origin_depot_id = rotation.trips[0].route.departure_station_id
        origin_depot_name = rotation.trips[0].route.departure_station.name
        line_name = rotation.trips[0].route.line.name
        vehicle_type_id = rotation.vehicle_type_id
        vehicle_type_name = rotation.vehicle_type.name

        for trip in rotation.trips:
            if passenger_trips_only and trip.trip_type != TripType.PASSENGER:
                continue
            # Obtain the coordinates based on the selected mode
            if route_calculation_mode == RouteCalculationMode.STATIONS_ONLY:
                # Always use station coordinates
                line_coords = _extract_station_coords(trip)

            elif route_calculation_mode == RouteCalculationMode.ROUTE_SHAPES:
                # Use Route.geom if available, fallback to stations
                if trip.route.geom is not None:
                    line_coords = _extract_geom_coords(trip)
                else:
                    line_coords = _extract_station_coords(trip)

            else:
                raise ValueError(
                    f"Unknown route_calculation_mode: {route_calculation_mode}"
                )

            result.append(
                {
                    "rotation_id": rotation.id,
                    "rotation_name": rotation.name,
                    "vehicle_type_id": vehicle_type_id,
                    "vehicle_type_name": vehicle_type_name,
                    "originating_depot_id": origin_depot_id,
                    "originating_depot_name": origin_depot_name,
                    "distance": trip.route.distance,
                    "coordinates": line_coords,
                    "line_name": line_name,
                }
            )

    return pd.DataFrame(result)


def single_rotation_info(
    rotation_id: int,
    session: sqlalchemy.orm.session.Session,
) -> pd.DataFrame:
    """
    This methods provides information over the trips in a single rotation and returns a pandas DataFrame with the
    following columns:

    - trip_id: the id of the trip
    - trip_type: the type of the trip
    - line_name: the name of the line
    - route_name: the name of the route
    - distance: the distance of the route
    - departure_time: the departure time of the trip
    - arrival_time: the arrival time of the trip
    - departure_station_name: the name of the departure station
    - departure_station_id: the id of the departure station
    - arrival_station_name: the name of the arrival station
    - arrival_station_id: the id of the arrival station

    :param rotation_id: The id of the rotation to get the information for
    :param session: An sqlalchemy session to an eflips-model database
    :return: A pandas DataFrame
    """

    rotation = (
        session.query(Rotation)
        .filter(Rotation.id == rotation_id)
        .options(sqlalchemy.orm.joinedload(Rotation.trips).joinedload(Trip.route))
        .one()
    )

    result: List[Dict[str, int | float | str | datetime]] = []

    for trip in rotation.trips:
        result.append(
            {
                "trip_id": trip.id,
                "trip_type": trip.trip_type,
                "line_name": rotation.name,
                "route_name": trip.route.name,
                "distance": trip.route.distance,
                "departure_time": trip.departure_time,
                "arrival_time": trip.arrival_time,
                "departure_station_name": trip.route.departure_station.name,
                "departure_station_id": trip.route.departure_station.id,
                "arrival_station_name": trip.route.arrival_station.name,
                "arrival_station_id": trip.route.arrival_station.id,
            }
        )

    return pd.DataFrame(result)
