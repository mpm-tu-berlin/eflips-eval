import os
from datetime import datetime

import pytz
from eflips.model import Area, Line, Vehicle, Depot

import eflips.eval.input.prepare as input_prepare
import eflips.eval.output.prepare as output_prepare
import eflips.eval.input.visualize as input_visualize
import eflips.eval.output.visualize as output_visualize

import plotly.express as px  # type: ignore
from sqlalchemy import create_engine
from sqlalchemy.orm import Session

if __name__ == "__main__":
    if (
        "DATABASE_URL" not in os.environ
        or os.environ["DATABASE_URL"] is None
        or os.environ["DATABASE_URL"] == ""
    ):
        raise ValueError(
            "The database url must be specified either as an argument or as the environment variable DATABASE_URL."
        )
    engine = create_engine(os.environ["DATABASE_URL"])
    session = Session(engine)
    SCENARIO_ID = 8

    # Example of the load and occupancy visualization
    all_areas = session.query(Area).filter(Area.scenario_id == SCENARIO_ID).all()
    all_area_ids = [area.id for area in all_areas]
    prepared_data = output_prepare.power_and_occupancy(all_area_ids, session)
    fig = output_visualize.power_and_occupancy(prepared_data)
    fig.show()

    # Example of the specific energy consumption visualization
    prepared_data = output_prepare.specific_energy_consumption(SCENARIO_ID, session)
    fig = output_visualize.specific_energy_consumption(prepared_data)
    fig.show()

    # Example of using the arrival and departure SoC visualization
    prepared_data = output_prepare.departure_arrival_soc(SCENARIO_ID, session)
    fig = output_visualize.departure_arrival_soc(prepared_data)
    fig.show()

    # Example of using the rotation info visualization
    prepared_data = input_prepare.rotation_info(SCENARIO_ID, session)
    fig = input_visualize.rotation_info(prepared_data)
    fig.show()

    # Example of using the time-distance diagram (Bildfahrplan), one line at a time. The
    # height is worth setting: a whole service day at the default height squeezes a
    # half-hour journey into a nearly horizontal line.
    for line in session.query(Line).filter(Line.scenario_id == SCENARIO_ID):
        prepared_data = input_prepare.time_distance_diagram(line.id, session)
        if prepared_data.empty:
            continue
        fig = input_visualize.time_distance_diagram(prepared_data, height=1400)
        # The package's visualize functions set no title; add one from the data if wanted.
        passenger_trips = prepared_data.loc[
            prepared_data["trip_kind"] == "passenger", "trip_id"
        ].nunique()
        fig.update_layout(
            title=f"Line {line.name}: {passenger_trips} passenger trips, "
            f"{prepared_data['rotation_id'].nunique()} rotations"
        )
        fig.show()

    # Example of using the depot event visualization
    prepared_data = output_prepare.depot_event(SCENARIO_ID, session)
    fig = output_visualize.depot_event(prepared_data, color_scheme="event_type")
    fig.show()
    #
    # Example of using the vehicle soc visualization
    example_vehicle_id = (
        session.query(Vehicle.id)
        .filter(Vehicle.scenario_id == SCENARIO_ID)
        .limit(1)
        .one()[0]
    )
    prepared_data, descriptions = output_prepare.vehicle_soc(
        example_vehicle_id, session
    )
    fig = output_visualize.vehicle_soc(prepared_data, descriptions)
    fig.show()

    # Example of using the depot activity visualization
    depot_id = (
        session.query(Depot.id)
        .filter(Depot.scenario_id == SCENARIO_ID)
        .limit(1)
        .one()[0]
    )

    area_blocks = output_prepare.depot_layout(depot_id, session)
    _, fig = output_visualize.depot_layout(area_blocks)
    fig.show()

    tz = pytz.timezone("Europe/Berlin")

    animation_range = (
        tz.localize(datetime(2023, 7, 1, 21, 0)),
        tz.localize(datetime(2023, 7, 2, 2, 0)),
    )
    depot_activity = output_prepare.depot_activity(depot_id, session, animation_range)

    animation = output_visualize.depot_activity_animation(
        area_blocks,
        depot_activity,
        animation_range=animation_range,
        time_resolution=120,
    )
    animation.save("depot_activity.mp4", writer="ffmpeg", fps=5)
