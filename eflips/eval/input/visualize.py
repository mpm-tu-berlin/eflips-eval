from zoneinfo import ZoneInfo

import dash_cytoscape as cyto  # type: ignore
import folium  # type: ignore
import pandas as pd
import plotly.express as px  # type: ignore
import plotly.graph_objs as go  # type: ignore
import seaborn as sns  # type: ignore
from eflips.model import TripType


def rotation_info(
    prepared_data: pd.DataFrame, timezone: ZoneInfo = ZoneInfo("Europe/Berlin")
) -> go.Figure:
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
    - line_is_unified: True if the rotation only contains one line
    - start_station: the name of the departure station
    - end_station: the name of the arrival station

    :return: A plotly figure object
    """

    # Go through the dataframe and fix the timezones
    for col in ["time_start", "time_end"]:
        prepared_data[col] = prepared_data[col].dt.tz_convert(timezone)

    fig = px.timeline(
        prepared_data,
        x_start="time_start",
        x_end="time_end",
        y="rotation_name",
        color="total_distance",
        labels={
            "time_start": "Departure Time",
            "time_end": "Arrival Time",
            "rotation_name": "Rotation Name",
            "total_distance": "Total Distance (km)",
            "line_name": "Line Name",
            "vehicle_type_name": "Vehicle Type Name",
            "start_station": "Start Station",
            "end_station": "End Station",
        },
        hover_name="rotation_name",
        hover_data=[
            "vehicle_type_name",
            "total_distance",
            "start_station",
            "end_station",
        ],
        pattern_shape="line_name",
    )
    fig.update_layout(legend_orientation="h")
    return fig


def geographic_trip_plot(prepared_data: pd.DataFrame) -> folium.Map:
    """
    This function visualizes the trips on a map using folium. The trips are lines between the departure and arrival
    stations.



    :param prepared_data: A Pandas dataframe with the following columns:
            - rotation_id: the id of the rotation
            - rotation_name: the name of the rotation
            - vehicle_type_id: the id of the vehicle type
            - vehicle_type_name: the name of the vehicle type
            - originating_depot_id: the id of the originating depot
            - originating_depot_name: the name of the originating depot
            - distance: the distance of the route
            - coordinates: An array of *(lon, lat)* tuples with the coordinates of the route - the shape if set, otherwise the stops
            - line_name: the name of the line, which is the first part of the rotation name. Used for sorting
    :return: A folium map object
    """
    # Use seaborn to create a color palette for each originating depot
    palette = sns.color_palette(
        "husl", n_colors=len(prepared_data["originating_depot_id"].unique())
    )
    colors = {}
    for depot_name in prepared_data["originating_depot_name"].unique():
        color = palette.pop(0)
        # Turn the color into a hex string
        color = "#{:02x}{:02x}{:02x}".format(*[int(c * 255) for c in color])
        colors[depot_name] = color

    # Obtain the mean latitude and longitude for the map center
    lat_center = (
        prepared_data["coordinates"]
        .apply(lambda x: sum([c[0] for c in x]) / len(x))
        .mean()
    )
    lon_center = (
        prepared_data["coordinates"]
        .apply(lambda x: sum([c[1] for c in x]) / len(x))
        .mean()
    )

    map = folium.Map(
        location=[lat_center, lon_center], zoom_start=11, tiles="Cartodb dark_matter"
    )
    for i, row in prepared_data.iterrows():
        color = colors[row["originating_depot_name"]]
        pl = folium.PolyLine(row["coordinates"], color=color, weight=2.5, opacity=1)
        map.add_child(pl)
    return map


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
