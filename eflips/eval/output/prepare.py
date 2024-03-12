import os

from datetime import datetime, timedelta
from typing import Dict, List

import sqlalchemy
from sqlalchemy import func
from sqlalchemy.orm import Session
import pandas as pd
from eflips.model import Event, Vehicle, Scenario


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


def depot_event(scenario_id: int, session: Session) -> pd.DataFrame:
    """
    This function creates a dataframe with all the events at the depot for a given scenario.
    The columns are
    - time_start: the start time of the event in datetime format
    - time_end: the end time of the event in datetime format
    - vehicle_id: the unique vehicle identifier which could be used for querying the vehicle in the database
    - event_type: the type of event specified in the eflips model. See :class:`eflips.model.EventType` for more information
    - area_id: the unique area identifier which could be used for querying the area in the database
    - trip_id: the unique trip identifier which could be used for querying the trip in the database
    - station_id: the unique station identifier which could be used for querying the station in the database
    - location: the location of the event. This could be "depot", "trip" or "station"


    :param scenario_id: The unique identifier of the scenario
    :param session: A :class:`sqlalchemy.orm.session.Session` object to an eflips-model database
    :return: A pandas DataFrame
    """

    event_list_for_plot: List[Dict[str, int | float | str | datetime]] = []
    events_from_db = (
        session.query(Event)
        .filter(Event.scenario_id == scenario_id)
        .order_by(Event.vehicle_id, Event.time_start)
        .all()
    )

    for event in events_from_db:

        location = None
        if event.area_id is not None:
            location = "depot"
        elif event.trip_id is not None:
            location = "trip"
        elif event.station_id is not None:
            location = "station"
        event_list_for_plot.append(
            {
                "time_start": event.time_start,
                "time_end": event.time_end,
                "vehicle_id": str(event.vehicle_id),
                "event_type": event.event_type.name,
                "area_id": event.area_id,
                "trip_id": event.trip_id,
                "station_id": event.station_id,
                "location": location

            }
        )

    return pd.DataFrame(event_list_for_plot)


def vehicle_soc(scenario_id: int, session: Session) -> pd.DataFrame:
    pass
