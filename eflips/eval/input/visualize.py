import dash_cytoscape as cyto  # type: ignore
import pandas as pd
import plotly.express as px  # type: ignore
import plotly.graph_objs as go  # type: ignore
from eflips.model import TripType


def rotation_info(prepared_data: pd.DataFrame) -> go.Figure:
    """
    This function visualizes the rotation information using plotly

    :param prepared_data: The result of the rotation_info function, a dataframe with the following columns:

    - rotation_id: the id of the rotation
    - rotation_name: the name of the rotation
    - vehicle_type_id: the id of the vehicle type
    - vehicle_type_name: the name of the vehicle type
    - total_distance: the total distance of the rotation
    - time_start: the departure of the first trip
    - time_end: the arrival of the last trip
    - line_name: the name of the line, which is the first part of the rotation name. Used for sorting

    :return: A plotly figure object
    """

    fig = px.timeline(
        prepared_data,
        x_start="time_start",
        x_end="time_end",
        y="rotation_name",
        color="total_distance",
        labels={
            "rotation_name": "Rotation Name",
            "total_distance": "Total Distance (km)",
            "line_name": "Line Name",
        },
        hover_data=["vehicle_type_name"],
        pattern_shape="line_name",
    )
    fig.update_layout(legend_orientation="h")
    return fig


def single_rotation_info(prepared_data: pd.DataFrame) -> cyto.Cytoscape:
    """
    This plots a single rotation as a network graph. The nodes are the stops and the edges are the trips between the
    stops.

    :param prepared_data: The result of the rotation_info function, a dataframe with the following columns:
        - trip_id: the id of the trip
        - trip_type: the type of the trip
        - line_name: the name of the line
        - route_name: the name of the route
        - distance: the distance of the route
        - time_start: the departure time of the trip
        - time_end: the arrival time of the trip
        - departure_station_name: the name of the departure station
        - departure_station_id: the id of the departure station
        - arrival_station_name: the name of the arrival station
        - arrival_station_id: the id of the arrival station

    :return: A Dash Cytoscape object. It can be added to a Dash layout.
    """

    # Create a list of elements, the stations are nodes and the trips are edges
    stations_already_added = set()
    elements = []
    for i, row in prepared_data.iterrows():
        # Handle the station (node) information
        if row.departure_station_id not in stations_already_added:
            elements.append(
                {
                    "data": {
                        "id": str(row.departure_station_id),
                        "label": row.departure_station_name,
                    }
                }
            )
            stations_already_added.add(row.departure_station_id)
        if row.arrival_station_id not in stations_already_added:
            elements.append(
                {
                    "data": {
                        "id": str(row.arrival_station_id),
                        "label": str(row.arrival_station_name),
                    }
                }
            )
            stations_already_added.add(row.arrival_station_id)

        # Handle the trip (edge) information
        type_str = "Passenger" if row.trip_type == TripType.PASSENGER else "Deadhead"
        color = "#9dbaea" if row.trip_type == TripType.PASSENGER else "#f4a261"
        trip_str = (
            f"{type_str} trip {row.departure_station_name} ({row.departure_time.strftime('%H:%M')})"
            f" -> {row.arrival_station_name} ({row.arrival_time.strftime('%H:%M')})"
        )
        elements.append(
            {
                "data": {
                    "source": str(row.departure_station_id),
                    "target": str(row.arrival_station_id),
                    "label": trip_str,
                    "color": color,
                }
            }
        )

    cytograph = cyto.Cytoscape(
        id="cytoscape",
        elements=elements,
        layout={"name": "cose"},
        style={"width": "1000px", "height": "1000px"},
        stylesheet=[
            {
                "selector": "node",
                "style": {"label": "data(label)", "background-color": "#11479e"},
            },
            {
                "selector": "edge",
                "style": {
                    "label": "data(label)",
                    "curve-style": "bezier",
                    "target-arrow-shape": "triangle",
                    "line-color": "data(color)",
                    "target-arrow-color": "data(color)",
                },
            },
        ],
    )

    return cytograph
