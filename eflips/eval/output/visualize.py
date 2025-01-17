from datetime import datetime, timedelta
from typing import Dict, List, Tuple
from zoneinfo import ZoneInfo

import matplotlib.animation as animation
import matplotlib.patches as patches
import matplotlib.pyplot as plt
import pandas as pd
import plotly.express as px  # type: ignore
import plotly.graph_objs as go  # type: ignore
from eflips.model import AreaType, Area
from matplotlib.figure import Figure
from plotly.subplots import make_subplots  # type: ignore

from eflips.eval.output.util import _draw_area, _is_occupied


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
        case "area_type":
            color = "area_type"
            color_discrete_map = {
                "Direct": "salmon",
                "Line": "forestgreen",
                "Other": "lightgrey",
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
    prepared_data: pd.DataFrame,
    color_scheme: str = "event_type",
    timezone: ZoneInfo = ZoneInfo("Europe/Berlin"),
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

    # Go through the dataframe and fix the timezones
    for col in ["time_start", "time_end"]:
        prepared_data[col] = prepared_data[col].dt.tz_convert(timezone)

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


def power_and_occupancy(
    prepared_data: pd.DataFrame, timezone: ZoneInfo = ZoneInfo("Europe/Berlin")
) -> go.Figure:
    """
    This function visualizes the power and occupancy using plotly
    :param prepared_data: The result of the power_and_occupancy function, a dataframe with the following columns:

    - time: the time at which the power and occupancy was recorded
    - power: the power at the given time
    - occupancy_charging: the summed occupancy (actively charing vehicles) of the area(s) at the given time
    - occupancy_total: the summed occupancy of the area(s) at the given time, including all events

    :return: A plotly figure object
    """

    # Go through the dataframe and fix the timezones
    prepared_data["time"] = prepared_data["time"].dt.tz_convert(timezone)

    fig = make_subplots(specs=[[{"secondary_y": True}]])
    fig.add_trace(
        go.Scatter(x=prepared_data["time"], y=prepared_data["power"], name="Power"),
        secondary_y=False,
    )
    fig.add_trace(
        go.Scatter(
            x=prepared_data["time"],
            y=prepared_data["occupancy_charging"],
            name="Occupancy (Charging)",
        ),
        secondary_y=True,
    )
    fig.add_trace(
        go.Scatter(
            x=prepared_data["time"],
            y=prepared_data["occupancy_total"],
            name="Occupancy (Total)",
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
    timezone: ZoneInfo = ZoneInfo("Europe/Berlin"),
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

    # Go through the prepared_data and fix the timezones
    prepared_data["time"] = prepared_data["time"].dt.tz_convert(timezone)
    # For the event descriptions, fix the timezones
    for event_type, event_list in descriptions.items():
        descriptions[event_type] = [
            (name, start_time.astimezone(timezone), end_time.astimezone(timezone))
            for (name, start_time, end_time) in event_list
        ]

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


def depot_layout(
    area_blocks: List[List[Area]],
) -> Tuple[Dict[int, List[patches.Rectangle]], Figure]:
    """
    This function visualizes the depot layout using matplotlib

    :param area_blocks: a list of lists of :class:`eflips.model.Area` objects.

    :return: a dictionary with the area id as the key and a list of :class:`matplotlib.patches.Rectangle` objects as the value and a :class:`matplotlib.figure.Figure` object
    """
    plt.rcParams["figure.dpi"] = 600
    fig, ax = plt.subplots()

    buffer = 0.9
    block_distance = 10

    area_dict = {}

    for j in range(len(area_blocks)):
        areas = area_blocks[j]
        for i in range(len(areas)):
            area = areas[i]
            vehicle_width = 2.55
            vehicle_length = 12
            if area.area_type == AreaType.LINE:
                center = (
                    i * (vehicle_width * 2 + buffer) + j * block_distance,
                    0,
                )
            elif area.area_type == AreaType.DIRECT_ONESIDE:
                center = (
                    i * (vehicle_length + buffer) + j * block_distance,
                    0,
                )
            else:
                raise NotImplementedError

            slots_in_area = _draw_area(ax, area, center)

            area_dict[area.id] = slots_in_area

    # Set the x and y axis limits
    plt.axis("equal")
    return area_dict, fig


def animate(
    frame: int,
    area_dict: Dict[int, List[patches.Rectangle]],
    area_occupancy: Dict[Tuple[int, int], List[Tuple[int, int]]],
    animation_start: datetime,
    time_resolution: int = 120,
) -> None:
    """
    This function animates the depot activity using matplotlib
    :param frame: the current frame to be rendered
    :param area_dict: a dictionary with the area id as the key and a list of :class:`matplotlib.patches.Rectangle` objects as the value
    :param area_occupancy: a dictionary containing the occupancy of each slot in the depot. See :func:`depot_activity_animation` for more information
    :param animation_start: a datetime object representing the start time of the animation
    :param time_resolution: an integer representing the time interval between 2 frames in seconds
    :return: None
    """
    for area_id, slots in area_dict.items():
        for slot_id, slot in enumerate(slots):
            slot_occupancy = area_occupancy[(area_id, slot_id)]
            slot_occupancy = [
                (int(s[0] / time_resolution), int(s[1] / time_resolution))
                for s in slot_occupancy
            ]

            slot.set_facecolor(
                "green" if _is_occupied(frame, slot_occupancy) else "lightgrey"
            )

    if hasattr(animate, "frame_text"):
        animate.frame_text.remove()

    # Add the current frame number

    current_time = animation_start + timedelta(seconds=frame * time_resolution)

    xlim = plt.gca().get_xlim()
    ylim = plt.gca().get_ylim()

    # Set the text position to the top right corner
    animate.frame_text = plt.text(  # type: ignore
        xlim[1],
        ylim[1] + 20,
        f"{current_time}",
        fontsize=12,
        horizontalalignment="right",
        verticalalignment="top",
    )


def depot_activity_animation(
    area_blocks: List[List[Area]],
    area_occupancy: Dict[Tuple[int, int], List[Tuple[int, int]]],
    animation_range: Tuple[datetime, datetime],
    time_resolution: int = 120,
) -> animation.FuncAnimation:
    """
    This function visualizes the depot activity as an animation using matplotlib
    :param area_blocks: A list of lists of :class:`eflips.model.Area` objects. Each list represents a Direct area of
    several line areas with the same length and vehicle type
    :param area_occupancy: A dictionary with the following key:
    - (area_id, slot_id): A tuple representing the area and slot id
    The value is a list of 2-tuples representing the start and end time of the occupancy in seconds since the animation start
    :param animation_range: A tuple representing the start and end time of the to-be-animated events.
    :param time_resolution: Time interval between 2 frames in seconds

    :return: a :class:`matplotlib.animation.FuncAnimation` object
    """

    frames_end = int(
        (animation_range[1] - animation_range[0]).total_seconds() / time_resolution
    )

    area_dict, fig = depot_layout(area_blocks)
    ani = animation.FuncAnimation(
        fig,
        animate,  # type: ignore
        frames=frames_end,
        interval=1,
        fargs=(area_dict, area_occupancy, animation_range[0], time_resolution),
    )

    return ani
