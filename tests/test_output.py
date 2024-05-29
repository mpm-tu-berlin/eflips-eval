"""
This file contains tests for the "output" visualizations, which are visualizations that can be done after the simulation
has been run.
"""

import pytest
from eflips.depot.api import (
    simulate_scenario,
    simple_consumption_simulation,
)
from eflips.model import Area, Vehicle

import eflips.eval.output.prepare
import eflips.eval.output.visualize
from tests.base import BaseTest


class TestOutput(BaseTest):
    @pytest.fixture(scope="module")
    def post_simulation_scenario(self, session, scenario):
        # Run the simulation
        simple_consumption_simulation(scenario, initialize_vehicles=True)
        simulate_scenario(scenario)
        session.commit()
        simple_consumption_simulation(scenario, initialize_vehicles=False)

        return scenario

    def test_fixtures(self, session, scenario, post_simulation_scenario):
        assert scenario == post_simulation_scenario
        assert len(scenario.events) > 0

    def test_departure_arrival_soc(self, session, post_simulation_scenario):
        prepared_data = eflips.eval.output.prepare.departure_arrival_soc(
            post_simulation_scenario.id, session
        )
        fig = eflips.eval.output.visualize.departure_arrival_soc(prepared_data)
        assert fig is not None

    def test_depot_event(self, session, post_simulation_scenario):
        prepared_data = eflips.eval.output.prepare.depot_event(
            post_simulation_scenario.id, session
        )
        fig = eflips.eval.output.visualize.depot_event(prepared_data)
        assert fig is not None

    def test_power_and_occupancy(self, session, post_simulation_scenario):
        all_areas = (
            session.query(Area)
            .filter(Area.scenario_id == post_simulation_scenario.id)
            .all()
        )
        all_area_ids = [area.id for area in all_areas]
        prepared_data = eflips.eval.output.prepare.power_and_occupancy(
            all_area_ids, session
        )
        fig = eflips.eval.output.visualize.power_and_occupancy(prepared_data)
        assert fig is not None

    def test_specific_energy_consumption(self, session, post_simulation_scenario):
        prepared_data = eflips.eval.output.prepare.specific_energy_consumption(
            post_simulation_scenario.id, session
        )
        fig = eflips.eval.output.visualize.specific_energy_consumption(prepared_data)
        assert fig is not None

    def test_vehicle_soc(self, session, post_simulation_scenario):
        vehicle_id = (
            session.query(Vehicle.id)
            .filter(Vehicle.scenario_id == post_simulation_scenario.id)
            .limit(1)
            .one()[0]
        )
        prepared_data, descriptions = eflips.eval.output.prepare.vehicle_soc(
            vehicle_id, session
        )
        fig = eflips.eval.output.visualize.vehicle_soc(prepared_data, descriptions)
        assert fig is not None
