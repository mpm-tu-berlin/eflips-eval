from datetime import datetime
from typing import Dict, List, Tuple

import pandas as pd
import plotly.express as px  # type: ignore
import plotly.graph_objs as go  # type: ignore
from plotly.subplots import make_subplots  # type: ignore


def get_color_scheme(
    color_scheme: str,
) -> Dict[str, str | dict[str, str] | None]:
    """
    This function returns a dictionary with the color scheme to be used in the gantt chart
    :param color_scheme: A string representing the color scheme to be used in the gantt chart. It can be one of the following:
    - "event_type"
    - "soc"
    - "location"
    :return: A dictionary with the color scheme
    """
    color = None
    color_discrete_map = None
    color_continuous_scale = None

    match color_scheme:
        case "event_type":
            color = "event_type"
            color_discrete_map = {
                "Charging Depot": "forestgreen",
                "Charging Opportunity": "lightgreen",
                "Driving": "skyblue",
                "Service": "salmon",
                "Standby Departure": "orange",
                "Standby": "yellow",
                "Precondition": "lightgrey",
            }
            color_continuous_scale = ""
        case "soc":
            color = "soc_end"
            color_discrete_map = None
            color_continuous_scale = px.colors.sequential.Viridis
        case "location":
            color = "location"
            color_discrete_map = {
                "Depot": "salmon",
                "Trip": "forestgreen",
                "Station": "steelblue",
            }
            color_continuous_scale = ""
        case _:
            raise ValueError("Invalid color scheme")

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

    lower_range = min(prepared_data["soc"])
    fig.update(layout_yaxis_range=[min(lower_range - 1, 0), 100])
    return fig


def depot_event(
    prepared_data: pd.DataFrame, color_scheme: str = "event_type"
) -> go.Figure:
    """
    This function visualizes all events as a gantt chart using plotly
    :param prepared_data: The result of the depot_event function, a dataframe with the following columns:
    :param color_scheme: A string representing the color scheme to be used in the gantt chart. It can be one of the following:
    - "event_type"
    - "soc"
    - "location"
    see :func:`get_color_scheme` for more information.

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
        pattern_shape="vehicle_type_name",
        labels={
            "time_start": "Start Time",
            "time_end": "End Time",
            "soc_start": "Start State of Charge (%)",
            "soc_end": "End State of Charge (%)",
            "area_id": "Area ID",
            "vehicle_id": "Vehicle ID",
        },
    )
    if color_scheme == "soc":
        legend_title = "State of Charge"
        fig.update_layout(coloraxis=dict(colorbar=dict(orientation="h", y=-0.15)))
    else:
        legend_title = color_scheme.replace("_", " ").title()

    vehicle_types = prepared_data["vehicle_type_name"].unique()

    for bar in fig.data:
        for vehicle_type in vehicle_types:
            if vehicle_type in bar.name:
                # Substituting legendgroup with vehicle type
                bar.legendgroup = str(vehicle_type)
                bar.legendgrouptitle = {"text": vehicle_type}
                break

    fig.update_xaxes(title_text="Time")
    fig.update_yaxes(title_text="Vehicles", secondary_y=False)
    fig.update_layout(
        legend={"title": legend_title + " , " + "Vehicle Type"},
        yaxis={"showticklabels": False},
    )
    fig.update_layout(legend=dict(groupclick="togglegroup"))

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


def specific_energy_consumption(prepared_data: pd.DataFrame) -> go.Figure:
    """
    Creates a histogram of the specific energy consumption of the vehicles, colored by vehicle type
    :param prepared_data: A DataFrame with the following columns:
        - trip_id: the unique identifier of the trip
        - route_id: the unique identifier of the route
        - route_name: the name of the route
        - distance: the distance of the route in km
        - energy_consumption: the energy consumption of the trip in kWh
        - vehicle_type_id: the unique identifier of the vehicle type
        - vehicle_type_name: the name of the vehicle type
    :return: a plotly figure object
    """
    prepared_data["specific_energy_consumption"] = (
        prepared_data["energy_consumption"] / prepared_data["distance"]
    )
    bin_count = prepared_data.shape[0] // 10
    fig = go.Figure()
    for vehicle_type_name, data in prepared_data.groupby("vehicle_type_name"):
        fig.add_trace(
            go.Histogram(x=data["specific_energy_consumption"], name=vehicle_type_name)
        )

    # Overlay both histograms
    fig.update_layout(
        barmode="overlay",
        xaxis_title="Specific Energy Consumption (kWh/km)",
        yaxis_title="Count",
    )
    # Reduce opacity to see both histograms
    fig.update_traces(opacity=0.75)
    return fig


def vehicle_soc(
    prepared_data: pd.DataFrame,
    descriptions: Dict[str, List[Tuple[str, datetime, datetime]]],
) -> go.Figure:
    """
    This function visualizes the state of charge of a vehicle over time using plotly. Optionally, it can also visualize
    event types or event locations by vertical rectangles.
    :param prepared_data: A dataframe with the following columns:
    - time: the time at which the SoC was recorded
    - soc: the state of charge at the given time

    :param descriptions: A dictionary with the following keys
    - "trip": A list of route names and the time the trip started and ended
    - "rotation": A list of rotation names and the time the rotation started and ended
    - "charging": A list of the location of the charging and the time the charging started and ended
    for each key, the value is a list of tuples with the following structure:
    - (name, start_time, end_time)

    :return: A plotly figure object
    """

    fig = px.line(
        prepared_data,
        x="time",
        y="soc",
        labels={"time": "Time", "soc": "State of Charge (%)"},
    )

    colors = ["red", "green", "blue"]

    for event_type, event_list in descriptions.items():
        color = colors.pop(0)
        for event in event_list:
            fig.add_vrect(
                x0=event[1],
                x1=event[2],
                line_width=0,
                fillcolor=color,
                opacity=0.25,
                annotation_text=event[0],
                annotation_textangle=270,
            )
    return fig
