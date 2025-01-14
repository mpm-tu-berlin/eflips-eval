from collections import Counter
from datetime import datetime
from typing import Dict, List

import pandas as pd
import sqlalchemy
from eflips.model import Rotation, Trip, Route
from shapely import wkb  # type: ignore


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
                "rotation_name": rotation.name,
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


def geographic_trip_plot(
    scenario_id: int,
    session: sqlalchemy.orm.session.Session,
    rotation_ids: None | int | List[int] = None,
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
    - coordinates: An array of *(lon, lat)* tuples with the coordinates of the route - the shape if set, otherwise the stops
    - line_name: the name of the line, which is the first part of the rotation name. Used for sorting

    :param scenario_id:
    :param session:
    :param rotation_ids:
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

    result: List[Dict[str, int | float | str | datetime]] = []
    for rotation in rotations_q:
        origin_depot_id = rotation.trips[0].route.departure_station_id
        origin_depot_name = rotation.trips[0].route.departure_station.name
        line_name = rotation.trips[0].route.line.name
        vehicle_type_id = rotation.vehicle_type_id
        vehicle_type_name = rotation.vehicle_type.name

        for trip in rotation.trips:
            # Obtain the coordinates of the route
            if trip.route.geom is not None:
                raise NotImplementedError(
                    "Geometries are not yet supported. Check if the code below 'just works'."
                    "If not, you need to implement the conversion to coordinates."
                )
                line_geom = wkb.loads(bytes(trip.route.geom.data))
                line_coords = [(point.y, point.x) for point in line_geom.coords]
            else:
                line_coords = []
                point_geom = wkb.loads(bytes(trip.route.departure_station.geom.data))
                lon, lat = point_geom.x, point_geom.y
                line_coords.append((lat, lon))
                for assoc in trip.route.assoc_route_stations:
                    if assoc.location is not None:
                        station_coordinates = wkb.loads(bytes(assoc.location.data))
                    else:
                        station_coordinates = wkb.loads(bytes(assoc.station.geom.data))
                    lon, lat = station_coordinates.x, station_coordinates.y
                    line_coords.append((lat, lon))
                point_geom = wkb.loads(bytes(trip.route.arrival_station.geom.data))
                lon, lat = point_geom.x, point_geom.y
                line_coords.append((lat, lon))

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
