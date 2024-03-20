from datetime import datetime
from typing import Dict, List

import sqlalchemy
import pandas as pd

from eflips.model import Rotation


def rotation_name_for_sorting(rotation_name: str) -> str:
    """
    Takes a rotation name, which in the BVG is a string of two numbers separated by a '/' character, and returns a string
    that can be used for sorting. The first part of the string is returned for BVG rotations.

    Other rotation names are not supported and will return the rotation name itself.

    :param rotation_name: The rotation name
    :return: a sortable string
    """
    if "/" in rotation_name:
        return rotation_name.split("/")[0]
    else:
        return rotation_name


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

    :param scenario_id: The scenario id for which to create the dataframe
    :param session: An sqlalchemy session to an eflips-model database
    :param rotation_ids: A list of rotation ids to filter for. If None, all rotations are included
    :return: a pandas DataFrame
    """

    result: List[Dict[str, int | float | str | datetime]] = []

    rotations = session.query(Rotation).filter(Rotation.scenario_id == scenario_id)

    if rotation_ids is not None:
        if isinstance(rotation_ids, int):
            rotation_ids = [rotation_ids]
        rotations = rotations.filter(Rotation.id.in_(rotation_ids))

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

    # We want to properly sort by rotation name, which is a bit intricate, as it's a string of two numbers divided by a
    # '/' character. We can't just sort by the string, as "10/11" would come after "10/1". We need to split the string
    # into its components and sort by them.
    df = pd.DataFrame(result)
    df["line_name"] = df["rotation_name"].apply(rotation_name_for_sorting)

    df.sort_values(by=["line_name", "time_start"], inplace=True)

    return df
