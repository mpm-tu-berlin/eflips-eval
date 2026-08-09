import asyncio
from asyncio import Task
from collections import Counter
from datetime import datetime, timedelta
from typing import Dict, List, Set, Tuple, Any, Sequence

import httpx
import pandas as pd
import sqlalchemy
from eflips.model import (
    AssocRouteStation,
    Depot,
    Line,
    Rotation,
    Route,
    Station,
    StopTime,
    Trip,
    TripType,
)
from geoalchemy2.shape import to_shape
from shapely import wkb  # type: ignore
from shapely.geometry.linestring import LineString  # type: ignore

from eflips.eval.input.line_axis import (
    LineAxis,
    build_line_axis,
    depot_slots,
)
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


#: The columns of :func:`time_distance_diagram`, in order. Named so that a line with no
#: trips at all can still return an empty frame that has them, rather than one that has no
#: columns and makes the visualize step raise ``KeyError``.
TIME_DISTANCE_COLUMNS: List[str] = [
    "line_id",
    "line_name",
    "trip_id",
    "trip_kind",
    "rotation_id",
    "rotation_name",
    "rotation_is_single_line",
    "route_id",
    "stop_index",
    "station_id",
    "station_name",
    "station_is_depot",
    "position",
    "is_margin",
    "is_axis_reference",
    "arrival_time",
    "departure_time",
    "trip_departure_time",
]

#: One stop of one trip: station, arrival, and departure once the dwell is added.
_TripStop = Tuple[Station, datetime, datetime]


def _stops_of_trip(trip: Trip) -> List[_TripStop]:
    """
    The stops of a trip, falling back to its route's endpoints when it has no stop times.

    eflips-model permits a trip with no :class:`eflips.model.StopTime` at all, and the
    deadhead trips that eflips-depot generates routinely have none. Such a trip is still a
    movement from one place to another and belongs on the diagram, so its route's two
    endpoints stand in for the missing stops.

    :param trip: the trip to read
    :return: ``(station, arrival, departure)`` per stop, in travel order
    """
    if trip.stop_times:
        return [
            (
                stop_time.station,
                stop_time.arrival_time,
                stop_time.arrival_time
                + (stop_time.dwell_duration or timedelta(seconds=0)),
            )
            for stop_time in trip.stop_times
        ]
    return [
        (trip.route.departure_station, trip.departure_time, trip.departure_time),
        (trip.route.arrival_station, trip.arrival_time, trip.arrival_time),
    ]


def _route_stop_indices(
    stop_station_ids: Sequence[int], route_station_ids: Sequence[int]
) -> List[int | None]:
    """
    Match a trip's stops onto its route's stops, as an ordered subsequence.

    eflips-model only guarantees that a trip's stop times are a *subsequence* of its route's
    stations -- its own validator explicitly allows "associated route stations without stop
    times". Pairing them off by position would therefore mis-place every stop after the first
    skipped one. Walking the two sequences in step gives the true route stop index, which is
    what tells the two visits of a ring line's terminus apart.

    :param stop_station_ids: the station of each stop of the trip, in travel order
    :param route_station_ids: the station of each stop of the route, in travel order
    :return: the route stop index for each trip stop, ``None`` where it could not be matched
    """
    indices: List[int | None] = []
    cursor = 0
    for station_id in stop_station_ids:
        match: int | None = None
        for candidate in range(cursor, len(route_station_ids)):
            if route_station_ids[candidate] == station_id:
                match = candidate
                cursor = candidate + 1
                break
        indices.append(match)
    return indices


def _depot_station_ids(
    scenario_id: int, session: sqlalchemy.orm.session.Session, derived: Set[int]
) -> Set[int]:
    """
    Which stations are depots, from the ``Depot`` table and from the schedule graph.

    The two sources are unioned rather than preferred one over the other. A scenario built by
    eflips-depot has ``Depot`` rows; one imported from BVG-XML has none and can only be read
    off the schedule, where a depot is what a rotation starts and ends at. Taking both means
    neither kind of scenario needs a mode flag, and a garage that the ``Depot`` table happens
    to have forgotten is still recognised.

    :param scenario_id: the scenario the line belongs to
    :param session: An sqlalchemy session to an eflips-model database
    :param derived: stations found at the outer end of this line's depot legs
    :return: the station ids that count as depots
    """
    declared = {
        depot.station_id
        for depot in session.query(Depot).filter(Depot.scenario_id == scenario_id)
    }
    return declared | derived


def time_distance_diagram(
    line_id: int,
    session: sqlalchemy.orm.session.Session,
    include_depot_trips: bool = True,
) -> pd.DataFrame:
    """
    Prepare one line's trips for a time-distance diagram (Bildfahrplan, Marey diagram).

    Every route variant of the line -- both directions, short turns, diversions, ring routes
    -- is projected onto a single distance axis, so that one number, ``position``, says where
    a stop belongs on the horizontal axis of the diagram. Pull-out and pull-in trips are
    placed in margin bands beyond the ends of the line, since a depot lies off the line's
    course and would otherwise stretch the axis over the whole city.

    The dataframe has one row per stop of one trip, sorted by trip and stop, with the
    following columns:

    - line_id: the id of the line, the same for every row
    - line_name: the name of the line, the same for every row
    - trip_id: the id of the trip; the rows sharing one are one line on the diagram
    - trip_kind: "passenger", "pull_out" or "pull_in"
    - rotation_id: the id of the rotation the trip belongs to
    - rotation_name: the name of that rotation
    - rotation_is_single_line: True if that rotation only ever serves this line
    - route_id: the id of the route the trip runs over
    - stop_index: the position of this stop within the trip, counting from zero
    - station_id: the id of the station
    - station_name: the name of the station
    - station_is_depot: True if the station houses a depot
    - position: how far along the line this stop is, in metres. Outside the line's own
      extent for the depot end of a pull-out or pull-in trip
    - is_margin: True if this stop is drawn in a depot margin beyond the end of the line
    - is_axis_reference: True if this stop lies on the route variant that defines the axis
    - arrival_time: when the vehicle arrives at the station
    - departure_time: when it leaves again, later than the arrival if the vehicle dwells
    - trip_departure_time: the departure of the whole trip, repeated on each of its rows

    All three times are absolute and timezone-aware. The service day they belong to is
    deliberately *not* decided here: it depends on local time, which
    :func:`eflips.eval.input.visualize.time_distance_diagram` applies together with the
    timezone. ``trip_departure_time`` is carried on every row so that the visualization can
    put a trip running through the small hours on a single service day rather than splitting
    it across two.

    The plot has no title, per the conventions of this package. The figures a caller may want
    for one are all in the dataframe::

        line = df["line_name"].iloc[0]
        trips = df.loc[df["trip_kind"] == "passenger", "trip_id"].nunique()
        rotations = df["rotation_id"].nunique()

    :param line_id: The id of the line to create the dataframe for
    :param session: An sqlalchemy session to an eflips-model database
    :param include_depot_trips: Whether to include the pull-out and pull-in trips that feed
        and follow this line. If False, only passenger trips are returned
    :return: A pandas DataFrame
    """
    line = session.query(Line).filter(Line.id == line_id).one()

    # Only the routes that carry passengers may shape the axis. An importer may well attach
    # a depot leg to the line it serves, and a depot sits kilometres off the course in an
    # arbitrary direction, so letting one in squeezes the line's own stops into a corner.
    passenger_route_ids = {
        row[0]
        for row in session.query(Trip.route_id)
        .join(Route)
        .filter(Route.line_id == line_id, Trip.trip_type == TripType.PASSENGER)
        .distinct()
    }

    routes = (
        session.query(Route)
        .filter(Route.id.in_(passenger_route_ids))
        .options(
            sqlalchemy.orm.joinedload(Route.assoc_route_stations).joinedload(
                AssocRouteStation.station
            )
        )
        .all()
        if passenger_route_ids
        else []
    )

    route_stops: Dict[int, List[Tuple[int, float]]] = {}
    for route in routes:
        if route.assoc_route_stations:
            route_stops[route.id] = [
                (assoc.station_id, assoc.elapsed_distance)
                for assoc in route.assoc_route_stations
            ]
        else:
            # A route with no stop associations still has two ends and a length.
            route_stops[route.id] = [
                (route.departure_station_id, 0.0),
                (route.arrival_station_id, route.distance),
            ]

    axis = build_line_axis(route_stops)
    if axis.reference_route_id is None:
        return pd.DataFrame(columns=TIME_DISTANCE_COLUMNS)

    trips = (
        session.query(Trip)
        .filter(Trip.route_id.in_(passenger_route_ids))
        .filter(Trip.trip_type == TripType.PASSENGER)
        .options(
            sqlalchemy.orm.joinedload(Trip.stop_times).joinedload(StopTime.station),
            sqlalchemy.orm.joinedload(Trip.rotation),
            sqlalchemy.orm.joinedload(Trip.route),
        )
        .all()
    )

    # A rotation is "single line" when every passenger trip it runs is on this line.
    rotation_ids = {trip.rotation_id for trip in trips}
    lines_per_rotation: Dict[int, Set[int]] = {}
    if rotation_ids:
        for rotation_id, other_line_id in (
            session.query(Trip.rotation_id, Route.line_id)
            .join(Route)
            .filter(Trip.rotation_id.in_(rotation_ids))
            .filter(Trip.trip_type == TripType.PASSENGER)
            .distinct()
        ):
            lines_per_rotation.setdefault(rotation_id, set()).add(other_line_id)

    depot_legs, attachments = _collect_depot_legs(
        line_id, rotation_ids, axis, session, include_depot_trips
    )
    depot_ids = _depot_station_ids(
        line.scenario_id, session, {station_id for station_id, _ in attachments}
    )
    slots = depot_slots(
        axis, sorted({station_id for station_id, _ in attachments}), attachments
    )

    result: List[Dict[str, Any]] = []
    for trip in trips:
        kind = "passenger"
        rotation_lines = lines_per_rotation.get(trip.rotation_id, set())
        result.extend(
            _rows_for_trip(
                trip=trip,
                line=line,
                kind=kind,
                is_single_line=len(rotation_lines) <= 1,
                axis=axis,
                slots=slots,
                depot_ids=depot_ids,
                anchor=None,
            )
        )
    for trip, kind, anchor in depot_legs:
        rotation_lines = lines_per_rotation.get(trip.rotation_id, set())
        result.extend(
            _rows_for_trip(
                trip=trip,
                line=line,
                kind=kind,
                is_single_line=len(rotation_lines) <= 1,
                axis=axis,
                slots=slots,
                depot_ids=depot_ids,
                anchor=anchor,
            )
        )

    if not result:
        return pd.DataFrame(columns=TIME_DISTANCE_COLUMNS)

    df = pd.DataFrame(result, columns=TIME_DISTANCE_COLUMNS)
    for column in ("arrival_time", "departure_time", "trip_departure_time"):
        df[column] = pd.to_datetime(df[column], utc=True)
    return df.sort_values(["trip_id", "stop_index"]).reset_index(drop=True)


def _collect_depot_legs(
    line_id: int,
    rotation_ids: Set[int],
    axis: LineAxis,
    session: sqlalchemy.orm.session.Session,
    include_depot_trips: bool,
) -> Tuple[List[Tuple[Trip, str, float | None]], List[Tuple[int, float | None]]]:
    """
    The pull-out trips that feed this line and the pull-in trips that follow it.

    A rotation's first trip belongs to the line of its *first passenger trip*, and its last
    trip to the line of its last. That is the only reading under which a rotation shared
    between two lines does not have its depot legs drawn twice, once on each line's diagram.

    Each leg is anchored to where the adjoining passenger trip starts or ends, rather than to
    the depot station's average position: on a ring line the terminus has two positions and
    their average is the middle of the diagram, which is precisely where the leg does not go.

    :param line_id: the line being drawn
    :param rotation_ids: the rotations that serve it
    :param axis: the line's distance axis
    :param session: An sqlalchemy session to an eflips-model database
    :param include_depot_trips: when False, returns nothing
    :return: a ``(trip, kind, anchor position)`` list and the ``(station id, anchor
        position)`` attachments the margins are laid out from
    """
    if not include_depot_trips or not rotation_ids:
        return [], []

    rotations = (
        session.query(Rotation)
        .filter(Rotation.id.in_(rotation_ids))
        .options(
            sqlalchemy.orm.joinedload(Rotation.trips).joinedload(Trip.route),
            sqlalchemy.orm.joinedload(Rotation.trips).joinedload(Trip.stop_times),
        )
        .all()
    )

    legs: List[Tuple[Trip, str, float | None]] = []
    attachments: List[Tuple[int, float | None]] = []
    for rotation in rotations:
        passenger = [
            trip for trip in rotation.trips if trip.trip_type == TripType.PASSENGER
        ]
        if not passenger:
            continue
        first, last = rotation.trips[0], rotation.trips[-1]

        if (
            first.trip_type != TripType.PASSENGER
            and passenger[0].route.line_id == line_id
        ):
            anchor = axis.position_for(passenger[0].route_id, 0)
            legs.append((first, "pull_out", anchor))
            attachments.append((_stops_of_trip(first)[0][0].id, anchor))

        if (
            last.trip_type != TripType.PASSENGER
            and passenger[-1].route.line_id == line_id
        ):
            stops = _stops_of_trip(passenger[-1])
            anchor = axis.position_for(passenger[-1].route_id, len(stops) - 1)
            if anchor is None:
                anchor = axis.position_of_station(stops[-1][0].id)
            legs.append((last, "pull_in", anchor))
            attachments.append((_stops_of_trip(last)[-1][0].id, anchor))

    return legs, attachments


def _rows_for_trip(
    trip: Trip,
    line: Line,
    kind: str,
    is_single_line: bool,
    axis: LineAxis,
    slots: Dict[int, float],
    depot_ids: Set[int],
    anchor: float | None,
) -> List[Dict[str, Any]]:
    """
    One trip as a list of stop rows, or an empty list if it cannot be placed on the axis.

    :param trip: the trip to lay out
    :param line: the line being drawn
    :param kind: "passenger", "pull_out" or "pull_in"
    :param is_single_line: whether the trip's rotation only serves this line
    :param axis: the line's distance axis
    :param slots: depot station id -> its position in a margin
    :param depot_ids: the station ids that count as depots
    :param anchor: for a depot leg, where on the axis it attaches
    :return: one dictionary per stop, ready for the dataframe
    """
    stops = _stops_of_trip(trip)
    if len(stops) < 2:
        return []

    if kind == "passenger":
        route_station_ids = [
            assoc.station_id for assoc in trip.route.assoc_route_stations
        ] or [trip.route.departure_station_id, trip.route.arrival_station_id]
        route_indices = _route_stop_indices(
            [station.id for station, _, _ in stops], route_station_ids
        )
    else:
        route_indices = [None] * len(stops)

    rows: List[Dict[str, Any]] = []
    for index, ((station, arrival, departure), route_index) in enumerate(
        zip(stops, route_indices)
    ):
        position: float | None = None
        is_margin = False
        if route_index is not None:
            position = axis.position_for(trip.route_id, route_index)
        if position is None and station.id in slots:
            # The depot end of a pull-out or pull-in trip: it lives in a margin band.
            position = slots[station.id]
            is_margin = True
        if position is None and anchor is not None:
            position = anchor
        if position is None:
            position = axis.position_of_station(station.id)
        if position is None:
            continue

        rows.append(
            {
                "line_id": line.id,
                "line_name": line.name,
                "trip_id": trip.id,
                "trip_kind": kind,
                "rotation_id": trip.rotation_id,
                "rotation_name": trip.rotation.name
                or f"Unnamed Rotation ({trip.rotation_id})",
                "rotation_is_single_line": is_single_line,
                "route_id": trip.route_id,
                "stop_index": index,
                "station_id": station.id,
                "station_name": station.name,
                "station_is_depot": station.id in depot_ids,
                # Whole metres: a week of a busy line is a few hundred thousand of these
                # written verbatim into the HTML, and the decimals buy nothing.
                "position": float(round(position)),
                "is_margin": is_margin,
                "is_axis_reference": trip.route_id == axis.reference_route_id,
                "arrival_time": arrival,
                "departure_time": departure,
                "trip_departure_time": trip.departure_time,
            }
        )

    return rows if len(rows) >= 2 else []
