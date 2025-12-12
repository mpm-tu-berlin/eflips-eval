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

    def test_interactive_map(self, session, post_simulation_scenario):
        """
        Test the interactive map generation with depots, routes, stations, and termini.
        Saves the map to a temporary directory.
        """
        import os
        import tempfile

        # Prepare data for the interactive map
        prepared_data = eflips.eval.output.prepare.interactive_map_data(
            post_simulation_scenario.id, session
        )

        # Verify the structure of prepared data
        assert "scenarios" in prepared_data
        assert "map_center" in prepared_data
        assert "global_color_map" in prepared_data

        scenario_data = prepared_data["scenarios"][post_simulation_scenario.id]
        assert "name" in scenario_data
        assert "depots" in scenario_data
        assert "routes" in scenario_data
        assert "termini_electrified" in scenario_data
        assert "termini_unelectrified" in scenario_data

        # Verify we have data
        assert len(scenario_data["depots"]) > 0, "Should have at least one depot"
        assert len(scenario_data["routes"]) > 0, "Should have routes with events"

        # Check depot data structure
        depot = scenario_data["depots"][0]
        assert "id" in depot
        assert "name" in depot
        assert "latitude" in depot
        assert "longitude" in depot
        assert "vehicle_counts" in depot
        assert "icon_color" in depot
        assert "hex_color" in depot

        # Verify icon_color is a valid folium color name
        valid_folium_colors = {
            "red",
            "blue",
            "green",
            "purple",
            "orange",
            "darkred",
            "darkblue",
            "darkgreen",
            "cadetblue",
            "darkpurple",
            "lightred",
            "lightblue",
            "lightgreen",
            "pink",
            "gray",
            "black",
        }
        assert (
            depot["icon_color"] in valid_folium_colors
        ), f"Invalid folium color: {depot['icon_color']}"

        # Verify hex_color is a valid hex color
        assert depot["hex_color"].startswith("#"), "hex_color should start with #"
        assert (
            len(depot["hex_color"]) == 7
        ), "hex_color should be 7 characters (#RRGGBB)"

        # Check route data structure
        route = scenario_data["routes"][0]
        assert "coordinates" in route
        assert "depot_id" in route
        assert "route_name" in route
        assert len(route["coordinates"]) > 0, "Route should have coordinates"

        # Create the interactive map
        with tempfile.TemporaryDirectory() as temp_dir:
            os.makedirs(temp_dir, exist_ok=True)
            # Create the map without plot directories first
            folium_map = eflips.eval.output.visualize.interactive_map(prepared_data)
            assert folium_map is not None

            # Save to temp directory
            map_path = os.path.join(temp_dir, "test_interactive_map.html")
            folium_map.save(map_path)

            # Verify file was created
            assert os.path.exists(map_path), "Map HTML file should be created"
            assert os.path.getsize(map_path) > 0, "Map HTML file should not be empty"

            # Test with plot directories (even though they don't exist, should not error)
            station_plot_dir = os.path.join(temp_dir, "stations")
            depot_plot_dir = os.path.join(temp_dir, "depots")
            os.makedirs(station_plot_dir, exist_ok=True)
            os.makedirs(depot_plot_dir, exist_ok=True)

            folium_map_with_dirs = eflips.eval.output.visualize.interactive_map(
                prepared_data,
                station_plot_dir=station_plot_dir,
                depot_plot_dir=depot_plot_dir,
            )
            assert folium_map_with_dirs is not None

            # Save map with plot directories
            map_with_dirs_path = os.path.join(
                temp_dir, "test_interactive_map_with_dirs.html"
            )
            folium_map_with_dirs.save(map_with_dirs_path)
            assert os.path.exists(map_with_dirs_path)

        # Test with multiple scenarios
        prepared_data_multi = eflips.eval.output.prepare.interactive_map_data(
            [post_simulation_scenario.id], session  # List with single scenario
        )
        folium_map_multi = eflips.eval.output.visualize.interactive_map(
            prepared_data_multi
        )
        assert folium_map_multi is not None
