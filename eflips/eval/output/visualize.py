from typing import Dict

import pandas as pd
import plotly.express as px  # type: ignore
import plotly.graph_objs as go  # type: ignore
from plotly.subplots import make_subplots  # type: ignore


def get_color_scheme(
    color_scheme: str,
) -> Dict[str, str | Dict[str, str] | str | px.colors.sequential]:
    """
    This function returns a dictionary with the color scheme to be used in the gantt chart
    :param color_scheme: A string representing the color scheme to be used in the gantt chart. It can be one of the following:
    - "Event type"
    - "SOC"
    - "Location"
    :return: A dictionary with the color scheme
    """
    color = None
    color_discrete_map = None
    color_continuous_scale = None

    match color_scheme:
        case "Event type":
            color = "event_type"
            color_discrete_map = {
                "CHARGING_DEPOT": "forestgreen",
                "DRIVING": "skyblue",
                "SERVICE": "salmon",
                "STANDBY_DEPARTURE": "orange",
            }
            color_continuous_scale = ""
        case "SOC":
            color = "soc_end"
            color_discrete_map = None
            color_continuous_scale = px.colors.sequential.Viridis
        case "Location":
            color = "location"
            color_discrete_map = {
                "depot": "salmon",
                "trip": "forestgreen",
                "station": "steelblue",
            }
            color_continuous_scale = ""

    color_scheme_dict = {
        "color": color,
        "color_discrete_map": color_discrete_map,
        "color_continuous_scale": color_continuous_scale,
    }

    return color_scheme_dict


def departure_arrival_soc(prepared_data: pd.DataFrame) -> go.Figure:
    """
    This function visualizes the departure and arrival SoC using plotly
    :param prepared_data: The result of the departure_arrival_soc function, a dataframe with the following columns:

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


    :return: A plotly figure object
    """

    fig = px.scatter(
        prepared_data,
        x="time",
        y="soc",
        symbol="event_type",
        color="vehicle_type_name",
        hover_data=["rotation_name"],
        labels={
            "soc": "State of Charge (%)",
            "time": "Time",
            "rotation_name": "Rotation Name",
            "vehicle_type_name": "Vehicle Type",
            "event_type": "Event Type",
        },
    )
    fig.update(layout_yaxis_range=[0, 100])
    return fig


def depot_event(
    prepared_data: pd.DataFrame, color_scheme: str = "Event type"
) -> go.Figure:
    """
    This function visualizes all events as a gantt chart using plotly
    :param prepared_data: The result of the depot_event function, a dataframe with the following columns:
    :param color_scheme: A string representing the color scheme to be used in the gantt chart. It can be one of the following:
    - "Event type"
    - "SOC"
    - "Location"

    - time_start: the start time of the event in datetime format
    - time_end: the end time of the event in datetime format
    - vehicle_id: the unique vehicle identifier which could be used for querying the vehicle in the database
    - event_type: the type of event specified in the eflips model. See :class:`eflips.model.EventType` for more information
    - area_id: the unique area identifier which could be used for querying the area in the database
    - trip_id: the unique trip identifier which could be used for querying the trip in the database
    - station_id: the unique station identifier which could be used for querying the station in the database
    - location: the location of the event. This could be "depot", "trip" or "station"

    :return: A plotly figure object

    """
    color_scheme_dict = get_color_scheme(color_scheme)
    color = color_scheme_dict["color"]
    color_discrete_map = color_scheme_dict["color_discrete_map"]
    color_continuous_scale = color_scheme_dict["color_continuous_scale"]

    fig = px.timeline(
        prepared_data,
        x_start="time_start",
        x_end="time_end",
        y="vehicle_id",
        color=color,
        color_discrete_map=color_discrete_map,
        color_continuous_scale=color_continuous_scale,
        labels={
            "time_start": "Start Time",
            "time_end": "End Time",
            "soc_start": "Start State of Charge (%)",
            "soc_end": "End State of Charge (%)",
            "area_id": "Area ID",
        },
    )
    fig.update_layout(legend={"orientation": "h", "y": 1.02, "title": "Event Type"})

    return fig


def power_and_occupancy(prepared_data: pd.DataFrame) -> go.Figure:
    """
    This function visualizes the power and occupancy using plotly
    :param prepared_data: The result of the power_and_occupancy function, a dataframe with the following columns:

    - time: the time at which the power and occupancy was recorded
    - power: the power at the given time
    - occupancy: the occupancy at the given time

    :return: A plotly figure object
    """
    fig = make_subplots(specs=[[{"secondary_y": True}]])
    fig.add_trace(
        go.Scatter(x=prepared_data["time"], y=prepared_data["power"], name="Power"),
        secondary_y=False,
    )
    fig.add_trace(
        go.Scatter(
            x=prepared_data["time"], y=prepared_data["occupancy"], name="Occupancy"
        ),
        secondary_y=True,
    )
    # Set x-axis title
    fig.update_xaxes(title_text="Time")

    # Set y-axes titles
    fig.update_yaxes(title_text="Power (kW)", secondary_y=False)
    fig.update_yaxes(title_text="Number of Vehicles in Depot", secondary_y=True)
    return fig


def vehicle_soc(prepared_data: pd.DataFrame) -> go.Figure:
    pass
