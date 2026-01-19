"""
This file contains tests for the "input" visualizations, which are visualizations that can be done before the simulation
has been run.
"""

import dash_cytoscape
import eflips.depot.api
import folium
import pandas as pd
import plotly.graph_objs as go  # type: ignore
import pytest
import httpx
from unittest.mock import patch, AsyncMock
import polyline
from eflips.model import (
    Rotation,
    Route,
)

import eflips.eval.input.prepare
import eflips.eval.input.visualize
from eflips.eval.input.route_options import RouteCalculationMode
from tests.base import BaseTest


class TestInput(BaseTest):

    def test_rotation_info(self, session, scenario):

        df = eflips.eval.input.prepare.rotation_info(scenario.id, session)

        # The following columns should be present
        #  - rotation_id: the id of the rotation
        # - rotation_name: the name of the rotation
        # - vehicle_type_id: the id of the vehicle type
        # - vehicle_type_name: the name of the vehicle type
        # - total_distance: the total distance of the rotation
        # - time_start: the departure of the first trip
        # - time_end: the arrival of the last trip
        # - line_name: the name of the line, which is the first part of the rotation name. Used for sorting
        assert "rotation_id" in df.columns
        assert "rotation_name" in df.columns
        assert "vehicle_type_id" in df.columns
        assert "vehicle_type_name" in df.columns
        assert "total_distance" in df.columns
        assert "time_start" in df.columns
        assert "time_end" in df.columns
        assert "line_name" in df.columns

        fig = eflips.eval.input.visualize.rotation_info(df)
        assert fig is not None
        assert isinstance(fig, go.Figure)

    def test_rotation_info_single_rotation(self, scenario, session):
        rotation_id = (
            session.query(Rotation)
            .filter(Rotation.scenario_id == scenario.id)
            .first()
            .id
        )
        df_1 = eflips.eval.input.prepare.rotation_info(
            1, session, rotation_ids=rotation_id
        )
        df_2 = eflips.eval.input.prepare.rotation_info(
            1, session, rotation_ids=[rotation_id]
        )
        assert len(df_1) == 1
        assert len(df_2) == 1
        assert df_1.equals(df_2)

    def test_geographic_trip_plot(self, scenario, session):
        rotation_id = (
            session.query(Rotation)
            .filter(Rotation.scenario_id == scenario.id)
            .first()
            .id
        )
        df_1 = eflips.eval.input.prepare.geographic_trip_plot(rotation_id, session)
        assert df_1 is not None
        assert isinstance(df_1, pd.DataFrame)

        # The following columns should be present
        # - rotation_id: the id of the rotation
        # - rotation_name: the name of the rotation
        # - vehicle_type_id: the id of the vehicle type
        # - vehicle_type_name: the name of the vehicle type
        # - originating_depot_id: the id of the originating depot
        # - originating_depot_name: the name of the originating depot
        # - distance: the distance of the route
        # - coordinates: An array of *(lon, lat)* tuples with the coordinates of the route - the shape if set, otherwise the stops
        # - line_name: the name of the line, which is the first part of the rotation name. Used for sorting
        assert "rotation_id" in df_1.columns
        assert "rotation_name" in df_1.columns
        assert "vehicle_type_id" in df_1.columns
        assert "vehicle_type_name" in df_1.columns
        assert "originating_depot_id" in df_1.columns
        assert "originating_depot_name" in df_1.columns
        assert "distance" in df_1.columns
        assert "coordinates" in df_1.columns
        assert "line_name" in df_1.columns

        my_map = eflips.eval.input.visualize.geographic_trip_plot(df_1)
        assert my_map is not None
        assert isinstance(my_map, folium.Map)

    def test_rotation_info_single_roatation_2(self, scenario, session):
        rotation_id = (
            session.query(Rotation)
            .filter(Rotation.scenario_id == scenario.id)
            .first()
            .id
        )
        df_1 = eflips.eval.input.prepare.single_rotation_info(rotation_id, session)
        assert df_1 is not None

        # The following columns should be present
        # - trip_id: the id of the trip
        # - trip_type: the type of the trip
        # - line_name: the name of the line
        # - route_name: the name of the route
        # - distance: the distance of the route
        # - departure_time: the departure time of the trip
        # - arrival_time: the arrival time of the trip
        # - departure_station_name: the name of the departure station
        # - departure_station_id: the id of the departure station
        # - arrival_station_name: the name of the arrival station
        # - arrival_station_id: the id of the arrival station
        assert "trip_id" in df_1.columns
        assert "trip_type" in df_1.columns
        assert "line_name" in df_1.columns
        assert "route_name" in df_1.columns
        assert "distance" in df_1.columns
        assert "departure_time" in df_1.columns
        assert "arrival_time" in df_1.columns
        assert "departure_station_name" in df_1.columns
        assert "departure_station_id" in df_1.columns
        assert "arrival_station_name" in df_1.columns
        assert "arrival_station_id" in df_1.columns

        my_cyto = eflips.eval.input.visualize.single_rotation_info(df_1)
        assert my_cyto is not None
        assert isinstance(my_cyto, dash_cytoscape.Cytoscape)

    def test_geographic_trip_plot_stations_only(self, scenario, session):
        """Test that STATIONS_ONLY mode always uses station coordinates."""
        df = eflips.eval.input.prepare.geographic_trip_plot(
            scenario.id,
            session,
            route_calculation_mode=RouteCalculationMode.STATIONS_ONLY,
        )

        assert len(df) > 0
        assert "coordinates" in df.columns

        # Verify all trips have coordinates
        for coords in df["coordinates"]:
            assert isinstance(coords, list)
            assert len(coords) >= 2  # At least departure and arrival

    def test_geographic_trip_plot_route_shapes(self, scenario, session):
        """Test ROUTE_SHAPES mode uses geom when available."""
        df = eflips.eval.input.prepare.geographic_trip_plot(
            scenario.id,
            session,
            route_calculation_mode=RouteCalculationMode.ROUTE_SHAPES,
        )

        assert len(df) > 0
        assert "coordinates" in df.columns

        # Verify all trips have coordinates
        for coords in df["coordinates"]:
            assert isinstance(coords, list)
            assert len(coords) >= 2

    def test_geographic_trip_plot_route_shapes_fallback(self, scenario, session):
        """Test ROUTE_SHAPES falls back to stations when geom unavailable."""
        # Clear geom from all routes in the scenario
        session.query(Route).filter(Route.scenario_id == scenario.id).update(
            {"geom": None}, synchronize_session=False
        )
        session.commit()

        df = eflips.eval.input.prepare.geographic_trip_plot(
            scenario.id,
            session,
            route_calculation_mode=RouteCalculationMode.ROUTE_SHAPES,
        )

        # Should still work using station coords
        assert len(df) > 0
        assert "coordinates" in df.columns

        # Verify coordinates from stations (should have at least 2 points)
        for coords in df["coordinates"]:
            assert isinstance(coords, list)
            assert len(coords) >= 2

    def test_geographic_trip_plot_geo_lookup_missing_env(
        self, scenario, session, monkeypatch
    ):
        """Test ROUTE_SHAPES_AND_GEO_LOOKUP raises error with missing env vars."""
        # Remove environment variables
        monkeypatch.delenv("OPENROUTESERVICE_BASE_URL", raising=False)
        monkeypatch.delenv("OPENROUTESERVICE_API_KEY", raising=False)

        # Clear geom to force routing lookup
        session.query(Route).filter(Route.scenario_id == scenario.id).update(
            {"geom": None}, synchronize_session=False
        )
        session.commit()

        with pytest.raises(ValueError, match="OPENROUTESERVICE_BASE_URL"):
            eflips.eval.input.prepare.geographic_trip_plot(
                scenario.id,
                session,
                route_calculation_mode=RouteCalculationMode.ROUTE_SHAPES_AND_GEO_LOOKUP,
            )

    def test_geographic_trip_plot_geo_lookup_with_geom(
        self, scenario, session, monkeypatch
    ):
        """Test ROUTE_SHAPES_AND_GEO_LOOKUP uses geom when available (no API calls)."""
        # Set env vars even though they won't be used
        monkeypatch.setenv("OPENROUTESERVICE_BASE_URL", "http://mock")
        monkeypatch.setenv("OPENROUTESERVICE_API_KEY", "mock-key")

        # If routes have geom, it should use them without calling API
        df = eflips.eval.input.prepare.geographic_trip_plot(
            scenario.id,
            session,
            route_calculation_mode=RouteCalculationMode.ROUTE_SHAPES_AND_GEO_LOOKUP,
        )

        assert len(df) > 0
        assert "coordinates" in df.columns

        for coords in df["coordinates"]:
            assert isinstance(coords, list)
            if len(coords) > 0:  # Some routes may have geom
                assert len(coords) >= 2

    @pytest.fixture
    def mock_ors_response(self):
        """Mock OpenRouteService API response."""
        return {
            "routes": [
                {
                    "geometry": "u`rgH_afjA???",  # Encoded polyline
                    "summary": {"distance": 1200.0, "duration": 120.0},
                }
            ]
        }

    def test_routing_cache_hit(self, monkeypatch, tmpdir):
        """Test that cached routes are reused without API calls."""
        from eflips.eval.input.routing import (
            _get_cache_key,
            _save_cached_geometry,
            _load_cached_geometry,
        )

        # Use temporary cache directory
        monkeypatch.setenv("EFLIPS_ROUTING_CACHE", str(tmpdir))

        # Create test data
        coords = ((52.5, 13.4), (52.6, 13.5))
        profile = "driving-car"
        geometry = [(52.5, 13.4), (52.55, 13.45), (52.6, 13.5)]

        # Save to cache
        cache_key = _get_cache_key(coords, profile)
        _save_cached_geometry(cache_key, geometry)

        # Load from cache
        loaded = _load_cached_geometry(cache_key)
        assert loaded == geometry

    def test_chunking_large_routes(self):
        """Test that routes with >50 waypoints are properly chunked."""
        from eflips.eval.input.prepare import (
            _split_stations_into_chunks,
            _combine_route_geometries,
        )

        # Create route with 120 stations (should create 3 chunks with overlap)
        stations = [(float(i), float(i)) for i in range(120)]

        chunks = _split_stations_into_chunks(stations, max_chunk_size=50)

        # Should have 3 chunks
        assert len(chunks) == 3

        # First chunk: 0-49 (50 stations)
        assert len(chunks[0]) == 50
        assert chunks[0][0] == (0.0, 0.0)
        assert chunks[0][-1] == (49.0, 49.0)

        # Second chunk: 49-98 (50 stations, overlaps at 49)
        assert len(chunks[1]) == 50
        assert chunks[1][0] == (49.0, 49.0)
        assert chunks[1][-1] == (98.0, 98.0)

        # Third chunk: 98-119 (22 stations, overlaps at 98)
        assert len(chunks[2]) == 22
        assert chunks[2][0] == (98.0, 98.0)
        assert chunks[2][-1] == (119.0, 119.0)

        # Test recombination
        mock_geometries = [
            [(0.0, 0.0), (1.0, 1.0), (2.0, 2.0)],
            [(2.0, 2.0), (3.0, 3.0), (4.0, 4.0)],  # Overlaps at (2.0, 2.0)
            [(4.0, 4.0), (5.0, 5.0)],  # Overlaps at (4.0, 4.0)
        ]

        combined = _combine_route_geometries(mock_geometries)

        # Should have 6 unique points (overlaps removed)
        assert len(combined) == 6
        assert combined == [
            (0.0, 0.0),
            (1.0, 1.0),
            (2.0, 2.0),
            (3.0, 3.0),
            (4.0, 4.0),
            (5.0, 5.0),
        ]

    @pytest.mark.asyncio
    async def test_routing_api_timeout(self, monkeypatch, tmpdir):
        """Test fallback behavior when API times out."""
        from eflips.eval.input.routing import _route_through_stations_async

        monkeypatch.setenv("EFLIPS_ROUTING_CACHE", str(tmpdir))

        stations = [(52.5, 13.4), (52.6, 13.5)]

        # Mock client that raises timeout
        async def mock_post(*args, **kwargs):
            raise httpx.TimeoutException("Request timed out")

        mock_client = AsyncMock()
        mock_client.post = mock_post

        # Should fall back to station coords
        result = await _route_through_stations_async(
            stations, "http://mock", "key", mock_client, "driving-car"
        )

        assert result == stations  # Fallback to straight line

    @pytest.mark.asyncio
    async def test_routing_api_invalid_response(self, monkeypatch, tmpdir):
        """Test fallback behavior when API returns invalid data."""
        from eflips.eval.input.routing import _route_through_stations_async

        monkeypatch.setenv("EFLIPS_ROUTING_CACHE", str(tmpdir))

        stations = [(52.5, 13.4), (52.6, 13.5)]

        # Mock client that returns invalid JSON
        async def mock_post(*args, **kwargs):
            response = AsyncMock()
            response.raise_for_status = AsyncMock()
            response.json = AsyncMock(return_value={"invalid": "data"})
            return response

        mock_client = AsyncMock()
        mock_client.post = mock_post

        # Should fall back to station coords
        result = await _route_through_stations_async(
            stations, "http://mock", "key", mock_client, "driving-car"
        )

        assert result == stations  # Fallback to straight line

    def test_geographic_trip_plot_passenger_trips_only(self, scenario, session):
        """Test that passenger_trips_only parameter filters correctly."""
        from eflips.model import TripType, Trip

        # Get all trips count
        df_all = eflips.eval.input.prepare.geographic_trip_plot(
            scenario.id,
            session,
            route_calculation_mode=RouteCalculationMode.STATIONS_ONLY,
            passenger_trips_only=False,
        )

        # Get passenger trips only
        df_passenger = eflips.eval.input.prepare.geographic_trip_plot(
            scenario.id,
            session,
            route_calculation_mode=RouteCalculationMode.STATIONS_ONLY,
            passenger_trips_only=True,
        )

        # Count passenger trips in scenario
        passenger_count = (
            session.query(Trip)
            .filter(
                Trip.route.has(Route.scenario_id == scenario.id),
                Trip.trip_type == TripType.PASSENGER,
            )
            .count()
        )

        # Verify filtering works
        assert len(df_passenger) == passenger_count
        assert len(df_passenger) <= len(df_all)

    def test_geographic_trip_plot_with_mocked_ors(
        self, scenario, session, monkeypatch, tmpdir
    ):
        """Test ROUTE_SHAPES_AND_GEO_LOOKUP with mocked OpenRouteService."""
        from eflips.eval.input.routing import _route_through_stations_async

        monkeypatch.setenv("OPENROUTESERVICE_BASE_URL", "http://mock-ors")
        monkeypatch.setenv("OPENROUTESERVICE_API_KEY", "mock-key")
        monkeypatch.setenv("EFLIPS_ROUTING_CACHE", str(tmpdir))

        # Clear all route geometries to force API calls
        session.query(Route).filter(Route.scenario_id == scenario.id).update(
            {"geom": None}, synchronize_session=False
        )
        session.commit()

        # Create mock polyline geometry
        test_coords = [(52.5, 13.4), (52.55, 13.45), (52.6, 13.5)]
        encoded = polyline.encode(test_coords)

        # Mock the async HTTP client
        async def mock_post(*args, **kwargs):
            response = AsyncMock()
            response.raise_for_status = AsyncMock()
            response.json = AsyncMock(
                return_value={
                    "routes": [
                        {
                            "geometry": encoded,
                            "summary": {"distance": 1200.0, "duration": 120.0},
                        }
                    ]
                }
            )
            return response

        with patch("httpx.AsyncClient") as mock_client_class:
            mock_client = AsyncMock()
            mock_client.post = mock_post
            mock_client.__aenter__ = AsyncMock(return_value=mock_client)
            mock_client.__aexit__ = AsyncMock()
            mock_client_class.return_value = mock_client

            # Run the function
            df = eflips.eval.input.prepare.geographic_trip_plot(
                scenario.id,
                session,
                route_calculation_mode=RouteCalculationMode.ROUTE_SHAPES_AND_GEO_LOOKUP,
            )

        # Verify results
        assert len(df) > 0
        assert "coordinates" in df.columns

        # All routes should have coordinates
        for coords in df["coordinates"]:
            assert isinstance(coords, list)
            assert len(coords) >= 2
