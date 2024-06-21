import zoneinfo
from datetime import datetime, timedelta
from typing import Dict, List, Iterable, Tuple, Optional

import numpy as np
import numpy.typing as npt
import pandas as pd
import sqlalchemy
from eflips.model import Event, Rotation, Vehicle, Trip, EventType
from sqlalchemy.orm import Session


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
            # That means the previous or next event has an area_id, the event does not

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


def depot_event(
    scenario_id: int, session: Session, vehicle_ids: None | int | List[int] = None
) -> pd.DataFrame:
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
    :param vehicle_ids: A list of vehicle ids to filter for. If None, all vehicles are included
    :return: A pandas DataFrame
    """

    event_list_for_plot: List[Dict[str, int | float | str | datetime | None]] = []
    events_from_db_q = (
        session.query(Event)
        .filter(Event.scenario_id == scenario_id)
        .order_by(Event.vehicle_id, Event.time_start)
    )

    if vehicle_ids is not None:
        if isinstance(vehicle_ids, int):
            vehicle_ids = [vehicle_ids]
        events_from_db_q = events_from_db_q.filter(Event.vehicle_id.in_(vehicle_ids))

    events_from_db = events_from_db_q.all()

    for event in events_from_db:
        location = None
        if event.area_id is not None:
            location = "Depot"
        elif event.trip_id is not None:
            location = "Trip"
        elif event.station_id is not None:
            location = "Station"

        event_list_for_plot.append(
            {
                "time_start": event.time_start,
                "time_end": event.time_end,
                "soc_start": event.soc_start,
                "soc_end": event.soc_end,
                "vehicle_id": str(event.vehicle_id),
                "vehicle_type_id": event.vehicle.vehicle_type_id,
                "vehicle_type_name": event.vehicle.vehicle_type.name,
                "event_type": event.event_type.name.replace("_", " ").title(),
                "area_id": event.area_id,
                "trip_id": event.trip_id,
                "station_id": event.station_id,
                "location": location,
            }
        )

    df = pd.DataFrame(event_list_for_plot)
    df.sort_values(by=["vehicle_type_name", "time_start"], inplace=True)

    return df


def power_and_occupancy(
    aread_id: int | Iterable[int],
    session: sqlalchemy.orm.session.Session,
    temporal_resolution: int = 60,
) -> pd.DataFrame:
    """
    This function creates a dataframe containing a timeseries of the power and occupancy of the given area(s).
    The columns are:
    - time: the time at which the data was recorded
    - power: the summed power consumption of the area(s) at the given time
    - occupancy: the summed occupancy of the area(s) at the given time

    :param aread_id: The id of the area for which to create the dataframe
    :param session: An sqlalchemy session to an eflips-model database
    :param temporal_resolution: The temporal resolution of the timeseries in seconds. Default is 60 seconds.
    :return: A pandas DataFrame
    """
    if isinstance(aread_id, int):
        aread_id = [aread_id]
    events = session.query(Event).filter(Event.area_id.in_(aread_id))

    start_time_row = (
        session.query(Event.time_start)
        .filter(Event.area_id.in_(aread_id))
        .order_by(Event.time_start)
        .first()
    )
    if start_time_row is None:
        raise ValueError("No events found for the given area_id")
    start_time = start_time_row[0]  # Oh, if we had nullability operators in Python…
    end_time_row = (
        session.query(Event.time_end)
        .filter(Event.area_id.in_(aread_id))
        .order_by(Event.time_end.desc())
        .first()
    )
    if end_time_row is None:
        raise ValueError("No events found for the given area_id")
    end_time = end_time_row[0]  # Oh, if we had nullability operators in Python…

    # Round the start and end times to the nearest temporal_resolution
    start_time = start_time - timedelta(seconds=start_time.second % temporal_resolution)
    end_time = end_time + timedelta(
        seconds=temporal_resolution - end_time.second % temporal_resolution
    )

    # Create a 1-second interval time series
    time: npt.NDArray[np.datetime64] = np.arange(
        start_time, end_time, timedelta(seconds=temporal_resolution)
    )
    time_as_unix = np.arange(
        start_time.timestamp(), end_time.timestamp(), temporal_resolution
    )
    energy = np.zeros(time.shape[0])
    occupancy = np.zeros(time.shape[0])

    # For each event:
    # Convert SoC to energy
    # Resample to 1-second intervals
    # Add the energy to the energy series
    for event in events:
        if event.timeseries is not None:
            this_event_times: List[datetime] = [datetime.fromisoformat(t) for t in event.timeseries["time"]]  # type: ignore
            this_event_socs: List[float] = event.timeseries["soc"]  # type: ignore
        else:
            this_event_times = []
            this_event_socs = []
        # Attach the event's time_start and soc to the timeseries at the beginning
        this_event_times.insert(0, event.time_start)
        this_event_socs.insert(0, event.soc_start)
        # Attach the event's time_end and soc to the timeseries at the end
        this_event_times.append(event.time_end - timedelta(seconds=1))
        this_event_socs.append(event.soc_end)
        this_event_unix_times = np.array([t.timestamp() for t in this_event_times])

        # Validation: the timeseries should be sorted and the socs should be in the range [0, 1] and monotonically increasing
        assert all(
            this_event_times[i] <= this_event_times[i + 1]
            for i in range(len(this_event_times) - 1)
        )
        assert all(
            this_event_socs[i] <= this_event_socs[i + 1]
            for i in range(len(this_event_socs) - 1)
        )

        # Convert from SoC to enerhgy using the vehicle types battery capacity
        this_event_energy = (
            np.array(this_event_socs) * event.vehicle.vehicle_type.battery_capacity
        )  # kWh

        # Resample the energy to 1-second intervals
        this_event_energy = np.interp(
            time_as_unix, this_event_unix_times, this_event_energy
        )
        energy += this_event_energy

        # For occupancy, we create an entry at the beginning and end of the event, then resample to 1-second
        # intervals with left=0 and right=0
        this_event_occupancy = np.interp(
            time_as_unix,
            [event.time_start.timestamp(), event.time_end.timestamp() - 1],
            [1, 1],
            left=0,
            right=0,
        )
        occupancy += this_event_occupancy

    # Calculate the power from the energy
    power = (np.diff(energy) / np.diff(time_as_unix).astype(float)) * 3600  # kW

    # Create the dataframe
    # First, change the time to the local timezone

    tz = datetime.now().astimezone().tzinfo
    time_as_local_tz = pd.to_datetime(time).tz_localize("UTC").tz_convert(tz)

    result = pd.DataFrame(
        {
            "time": time_as_local_tz[:-1],
            "power": power,
            "occupancy": occupancy[:-1],
        }
    )
    return result


def specific_energy_consumption(scenario_id: int, session: Session) -> pd.DataFrame:
    """
    Creates a dataframe of all the trip energy consumptions and distances for the given scenario.
    The dataframe contains the following columns:
    - trip_id: the unique identifier of the trip
    - route_id: the unique identifier of the route
    - route_name: the name of the route
    - distance: the distance of the route in km
    - energy_consumption: the energy consumption of the trip in kWh
    - vehicle_type_id: the unique identifier of the vehicle type
    - vehicle_type_name: the name of the vehicle type

    :param scenario_id: The unique identifier of the scenario
    :param session: An sqlalchemy session to an eflips-model database
    :return: A pandas DataFrame
    """
    events = (
        session.query(Event)
        .filter(Event.scenario_id == scenario_id)
        .filter(Event.event_type == EventType.DRIVING)
        .options(sqlalchemy.orm.joinedload(Event.trip).joinedload(Trip.route))
    )
    result: List[Dict[str, int | float | str]] = []
    for event in events:
        trip = event.trip
        delta_soc = event.soc_end - event.soc_start
        delta_energy = delta_soc * trip.rotation.vehicle_type.battery_capacity * -1
        result.append(
            {
                "trip_id": trip.id,
                "route_id": trip.route_id,
                "route_name": trip.route.name,
                "distance": trip.route.distance / 1000,
                "energy_consumption": delta_energy,
                "vehicle_type_id": trip.rotation.vehicle_type_id,
                "vehicle_type_name": trip.rotation.vehicle_type.name,
            }
        )
    return pd.DataFrame(result)


def vehicle_soc(
    vehicle_id: int,
    session: Session,
    timezone: Optional[zoneinfo.ZoneInfo] = zoneinfo.ZoneInfo("Europe/Berlin"),
) -> Tuple[pd.DataFrame, Dict[str, List[Tuple[str, datetime, datetime]]]]:
    """
    This function takes in a vehicle id and returns a description what happened to the vehicle over time.
    The dataframe contains the following columns:
    - time: the time at which the SoC was recorded
    - soc: the state of charge at the given time

    Additionally, a dictionary for the different kinds of events is returned. For each kind of event, a list of Tuples
    with a description of the event, the start time and the end time is returned.

    The kinds of events are:
    - "rotation": A list of rotation names and the time the rotation started and ended
    - "charging": A list of the location of the charging and the time the charging started and ended

    :param timezone: Explicit timezone information to use for the visualization. Default is Europe/Berlin
    :param vehicle_id: the unique identifier of the vehicle
    :param session: A :class:`sqlalchemy.orm.session.Session` object to an eflips-model database
    :return: A pandas DataFrame
    """
    events_from_db = (
        session.query(Event)
        .filter(Event.vehicle_id == vehicle_id)
        .order_by(Event.time_start)
        .all()
    )

    descriptions: Dict[str, List[Tuple[str, datetime, datetime]]] = {
        "rotation": [],
        "charging": [],
    }

    # Go through all events and connect the soc_start and soc_end and time_start and time_end
    all_times = []
    all_soc = []

    for event in events_from_db:
        all_times.append(event.time_start.astimezone(timezone))
        all_soc.append(event.soc_start)

        if event.timeseries is not None:
            this_event_times: List[datetime] = [
                datetime.fromisoformat(t).astimezone(timezone) for t in event.timeseries["time"]  # type: ignore
            ]
            this_event_socs: List[float] = event.timeseries["soc"]  # type: ignore
            all_times.extend(this_event_times)
            all_soc.extend(this_event_socs)

        all_times.append(event.time_end.astimezone(timezone))
        all_soc.append(event.soc_end)

        if (
            event.event_type == EventType.CHARGING_DEPOT
            or event.event_type == EventType.CHARGING_OPPORTUNITY
        ):
            if event.area is not None:
                name = event.area.name + " in " + event.area.depot.name
            else:
                name = event.station.name

            descriptions["charging"].append(
                (
                    name,
                    event.time_start.astimezone(timezone),
                    event.time_end.astimezone(timezone),
                )
            )

    for rotation in (
        session.query(Rotation).filter(Rotation.vehicle_id == vehicle_id).all()
    ):
        descriptions["rotation"].append(
            (
                rotation.name,
                rotation.trips[0].departure_time.astimezone(timezone),
                rotation.trips[-1].arrival_time.astimezone(timezone),
            )
        )

    return pd.DataFrame({"time": all_times, "soc": all_soc}), descriptions
