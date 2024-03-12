import pandas as pd
import plotly.express as px  # type: ignore
from sqlalchemy import create_engine
from sqlalchemy.orm import Session
import eflips.eval.data_preparation
import plotly.graph_objs as go  # type: ignore


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


if __name__ == "__main__":
    engine = create_engine("postgresql://ludger:moosemoose@localhost/bvg_schedule_all")
    session = Session(engine)
    SCENARIO_ID = 8

    prepared_data = eflips.eval.data_preparation.departure_arrival_soc(
        SCENARIO_ID, session
    )
    fig = departure_arrival_soc(prepared_data)
    fig.show()

    prepared_data = eflips.eval.data_preparation.rotation_info(SCENARIO_ID, session)
    fig = rotation_info(prepared_data)
    fig.show()
