from datetime import datetime
from typing import Dict, List

import sqlalchemy
import pandas as pd
from eflips.model import Event, Rotation, Vehicle


def rotation_name_for_sorting(rotation_name: str) -> int:
    """
    Takes a rotation name, which is a sequence of a string (containing X[0-9]+ or N[0-9]+ or M[0-9]+ or [0-9+])
    followed by a '/' followed by an integer. The function first makes sure that it is only a single rotation descriptor
    by discarding everything after the first ' ' character, if it exists. Then it splits the string into two parts and
    replaces X by 9, N, by 10 in the first part. It then turns both numbers into a string and add leading zeros to the
    second part, so that the two parts have the same length. It then concatenates the two parts and returns the result as
    an integer.

    :param rotation_name: The rotation name
    :return: a sortable integer
    """
    if " " in rotation_name:
        rotation_name = rotation_name.split(" ")[0]
    rotation_name_parts = rotation_name.split("/")
    first_part = int(
        rotation_name_parts[0].replace("X", "9").replace("N", "10").replace("M", "11")
    )
    second_part = int(rotation_name_parts[1])
    return first_part * 1000 + second_part


def departure_arrival_soc(
    scenario_id: int, session: sqlalchemy.orm.session.Session
) -> pd.DataFrame:
    """
    This function creates a dataframe with the SoC at departure and arrival for each trip.
    The columns are
    - event_id: the associated event id
    - rotation_id: the associated rotation id (for the trip the vehicle is going on or returning from)
    - rotation_name: the name of the rotation
    - vehicle_type_id: the vehicle type id
    - vehicle_type_name: the name of the vehicle type
    - vehicle_id: the vehicle id
    - vehicle_name: the name of the vehicle
    - time: the time at which this SoC was recorded (for departure, this is the departure time from the depot, for
      arrival, this is the arrival time at the depot)
    - soc: the state of charge at the given time
    - event_type: the type of event, either "Departure" or "Arrival"

    :param scenario_id: The scenario id for which to create the dataframe
    :param session: An sqlalchemy session to an eflips-model database

    :return: A pandas DataFrame
    """

    vehicles = session.query(Vehicle).filter(Vehicle.scenario_id == scenario_id)
    result: List[Dict[str, str | int | float | datetime]] = []

    for vehicle in vehicles:
        events = (
            session.query(Event)
            .filter(Event.vehicle_id == vehicle.id)
            .order_by(Event.time_start)
            .all()
        )
        for i in range(len(events)):
            # The interesting events for us are the ones immediately before or after an event at a depot.
            # That meand the previous or nect event has an area_id, the event does not

            if events[i].area_id is None:
                if i > 0 and events[i - 1].area_id is not None:
                    # This is a departure event
                    result.append(
                        {
                            "rotation_id": events[i].trip.rotation_id,
                            "rotation_name": events[i].trip.rotation.name,
                            "vehicle_type_id": vehicle.vehicle_type_id,
                            "vehicle_type_name": vehicle.vehicle_type.name,
                            "vehicle_id": vehicle.id,
                            "vehicle_name": vehicle.name,
                            "time": events[i].time_start,
                            "soc": events[i].soc_start * 100,
                            "event_type": "Departure",
                        }
                    )
                elif i < len(events) - 1 and events[i + 1].area_id is not None:
                    # This is an arrival event
                    result.append(
                        {
                            "rotation_id": events[i].trip.rotation_id,
                            "rotation_name": events[i].trip.rotation.name,
                            "vehicle_type_id": vehicle.vehicle_type_id,
                            "vehicle_type_name": vehicle.vehicle_type.name,
                            "vehicle_id": vehicle.id,
                            "vehicle_name": vehicle.name,
                            "time": events[i].time_end,
                            "soc": events[i].soc_end * 100,
                            "event_type": "Arrival",
                        }
                    )

    return pd.DataFrame(result)


def rotation_info(
    scenario_id: int, session: sqlalchemy.orm.session.Session
) -> pd.DataFrame:
    """
    This function provides information about the rotations in a scenario. This information can be provided even before
    the simulation has been run. It creates a dataframe withe the following columns:

    - rotation_id: the id of the rotation
    - rotation_name: the name of the rotation
    - vehicle_type_id: the id of the vehicle type
    - vehicle_type_name: the name of the vehicle type
    - total_distance: the total distance of the rotation
    - time_start: the departure of the first trip
    - time_end: the arrival of the last trip
    - line_name: the name of the line, which is the first part of the rotation name. Used for sorting

    :param scenario_id: The scenario id for which to create the dataframe
    :param session: An sqlalchemy session to an eflips-model database
    :return: a pandas DataFrame
    """

    result: List[Dict[str, int | float | str | datetime]] = []

    rotations = session.query(Rotation).filter(Rotation.scenario_id == scenario_id)
    for rotation in rotations:
        # The rotation distance comes form the routes of the trips
        distance = 0.0
        for trip in rotation.trips:
            distance += trip.route.distance / 1000

        result.append(
            {
                "rotation_id": rotation.id,
                "rotation_name": rotation.name,
                "vehicle_type_id": rotation.vehicle_type_id,
                "vehicle_type_name": rotation.vehicle_type.name,
                "total_distance": distance,
                "time_start": rotation.trips[0].departure_time,
                "time_end": rotation.trips[-1].arrival_time,
            }
        )

    # We want to properly sort by roation name, which is a bit intricate, as it's a string of two numbers divided by a
    # '/' character. We can't just sort by the string, as "10/11" would come after "10/1". We need to split the string
    # into its components and sort by them.
    df = pd.DataFrame(result)
    df["line_name"] = [r.split("/")[0] for r in df["rotation_name"]]

    df.sort_values(by=["line_name", "time_start"], inplace=True)

    return df
