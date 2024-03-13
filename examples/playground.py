import os

from eflips.model import Area

import eflips.eval.input.prepare as input_prepare
import eflips.eval.output.prepare as output_prepare
import eflips.eval.input.visualize as input_visualize
import eflips.eval.output.visualize as output_visualize

import plotly.express as px  # type: ignore
from sqlalchemy import create_engine
from sqlalchemy.orm import Session

if __name__ == "__main__":
    engine = create_engine(os.environ.get("DATABASE_URL"))
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

    # Example of using the depot event visualization
    prepared_data = output_prepare.depot_event(SCENARIO_ID, session)
    fig = output_visualize.depot_event(prepared_data)
    fig.show()
