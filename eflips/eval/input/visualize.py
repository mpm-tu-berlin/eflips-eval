import pandas as pd
import plotly.express as px  # type: ignore
import plotly.graph_objs as go  # type: ignore


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
