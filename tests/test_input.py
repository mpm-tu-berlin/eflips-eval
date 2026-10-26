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
    Line,
    Rotation,
    Route,
)

import eflips.eval.input.prepare
import eflips.eval.input.visualize
from eflips.eval.input.line_axis import build_line_axis, depot_slots
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

    def test_time_distance_diagram(self, scenario, session):
        line = (
            session.query(Line)
            .filter(Line.scenario_id == scenario.id, Line.name == "Oberstadt")
            .one()
        )

        df = eflips.eval.input.prepare.time_distance_diagram(line.id, session)

        # Every column the docstring promises must be there
        for column in eflips.eval.input.prepare.TIME_DISTANCE_COLUMNS:
            assert column in df.columns

        assert len(df) > 0
        assert (df["line_id"] == line.id).all()
        assert (df["line_name"] == "Oberstadt").all()

        # The three time columns must be timezone-aware: visualize converts them, which
        # would raise on naive timestamps.
        for column in ("arrival_time", "departure_time", "trip_departure_time"):
            assert df[column].dt.tz is not None

        # Exactly one route variant defines the axis.
        assert df.loc[df["is_axis_reference"], "route_id"].nunique() == 1

        # The two directions of the line must land on the same distance axis, running
        # opposite ways along it. This is the whole point of the axis construction: without
        # it the return trips would be drawn back-to-front.
        positions_by_route = {
            route_id: rows.sort_values("stop_index")["position"].tolist()
            for route_id, rows in df[df["trip_kind"] == "passenger"]
            .groupby("route_id")
            .__iter__()
        }
        forward = [p for p in positions_by_route.values() if p[0] < p[-1]]
        backward = [p for p in positions_by_route.values() if p[0] > p[-1]]
        assert forward and backward, "expected both directions of the line"
        assert set(forward[0]) == set(backward[0])

        fig = eflips.eval.input.visualize.time_distance_diagram(df)
        assert fig is not None
        assert isinstance(fig, go.Figure)

        # The house rules: a visualize function sets neither a title nor a viewport size.
        assert fig.layout.title.text is None
        assert fig.layout.height is None
        assert fig.layout.width is None

    def test_time_distance_diagram_service_day(self, scenario, session):
        """The hours after midnight belong to the service day that started the evening before."""
        line = (
            session.query(Line)
            .filter(Line.scenario_id == scenario.id, Line.name == "Oberstadt")
            .one()
        )
        df = eflips.eval.input.prepare.time_distance_diagram(line.id, session)
        fig = eflips.eval.input.visualize.time_distance_diagram(df)

        ticks = list(fig.layout.yaxis.ticktext)
        assert "24:00" in ticks
        assert "26:00" in ticks
        # An extended clock never wraps back to 00:00 -- that is the whole point of it.
        assert "00:00" not in ticks
        assert "02:00" not in ticks

        # Every passenger trip is filed under exactly one service day, so the button counts
        # must add up to the number of passenger trips. This is what catches a trip being
        # split across two days or dropped at the boundary.
        buttons = fig.layout.updatemenus[0].buttons
        assert buttons[-1].label == "All days"
        counted = sum(int(b.label.rsplit("(", 1)[1].rstrip(")")) for b in buttons[:-1])
        assert counted == df.loc[df["trip_kind"] == "passenger", "trip_id"].nunique()

        # Four traces per service day, and only one day is shown to start with.
        assert len(fig.data) == (len(buttons) - 1) * 4
        assert sum(1 for trace in fig.data if trace.visible) == 4

    def test_time_distance_diagram_height(self, scenario, session):
        line = (
            session.query(Line)
            .filter(Line.scenario_id == scenario.id, Line.name == "Oberstadt")
            .one()
        )
        df = eflips.eval.input.prepare.time_distance_diagram(line.id, session)

        assert (
            eflips.eval.input.visualize.time_distance_diagram(df).layout.height is None
        )
        fig = eflips.eval.input.visualize.time_distance_diagram(df, height=1400)
        assert fig.layout.height == 1400

    def test_time_distance_diagram_no_trips(self, scenario, session):
        """A line with routes but no trips gives an empty frame that still has the columns."""
        line = (
            session.query(Line)
            .filter(
                Line.scenario_id == scenario.id, Line.name == "Holländisches Viertel"
            )
            .one()
        )

        df = eflips.eval.input.prepare.time_distance_diagram(line.id, session)
        assert df.empty
        for column in eflips.eval.input.prepare.TIME_DISTANCE_COLUMNS:
            assert column in df.columns

        fig = eflips.eval.input.visualize.time_distance_diagram(df)
        assert isinstance(fig, go.Figure)
        assert len(fig.data) == 0


class TestLineAxis:
    """
    The distance axis on its own, without a database.

    The shared scenario fixture has no ring line, no short-turn variant and no depot legs,
    so the cases the axis construction actually exists for can only be reached by handing it
    route data directly.
    """

    def test_opposite_directions_share_the_axis(self):
        out = [(1, 0.0), (2, 2500.0), (3, 5000.0)]
        back = [(3, 0.0), (2, 2500.0), (1, 5000.0)]

        axis = build_line_axis({10: out, 11: back})

        assert axis.route_positions[10] == [0.0, 2500.0, 5000.0]
        # The return route is projected onto the same axis, so its metres run backwards.
        assert axis.route_positions[11] == [5000.0, 2500.0, 0.0]
        assert axis.extent == (0.0, 5000.0)

    def test_short_turn_sits_inside_the_reference(self):
        full = [(1, 0.0), (2, 2500.0), (3, 5000.0)]
        short = [(1, 0.0), (2, 2500.0)]

        axis = build_line_axis({10: full, 12: short})

        assert axis.route_positions[12] == [0.0, 2500.0]

    def test_variant_past_the_terminus_extrapolates_in_metres(self):
        """A stop the reference does not know keeps its own metres, so the axis stays metric."""
        full = [(1, 0.0), (2, 2500.0), (3, 5000.0)]
        longer = [(1, 0.0), (2, 2500.0), (3, 5000.0), (4, 6200.0)]

        axis = build_line_axis({10: full, 13: longer})

        assert axis.route_positions[13] == [0.0, 2500.0, 5000.0, 6200.0]
        assert axis.extent == (0.0, 6200.0)

    def test_ring_line_gives_one_station_two_positions(self):
        """On a loop a station is two places on the axis, and both must survive."""
        loop = [(1, 0.0), (2, 1000.0), (3, 2000.0), (2, 3000.0), (1, 4000.0)]

        axis = build_line_axis({20: loop})
        positions = axis.route_positions[20]

        assert positions[0] != positions[4]  # the terminus, at both ends
        assert positions[1] != positions[3]  # the intermediate stop, served twice
        assert axis.position_for(20, 0) == 0.0
        assert axis.position_for(20, 4) == 4000.0

    def test_empty_and_degenerate_input(self):
        assert build_line_axis({}).reference_route_id is None
        # A route of a single stop cannot span anything and is ignored.
        assert build_line_axis({1: [(1, 0.0)]}).reference_route_id is None

    def test_depot_slots_land_outside_the_line(self):
        axis = build_line_axis({10: [(1, 0.0), (2, 2500.0), (3, 5000.0)]})
        low, high = axis.extent

        # One depot feeding the low end, one feeding the high end.
        slots = depot_slots(axis, [91, 92], [(91, 0.0), (92, 5000.0)])

        assert slots[91] < low
        assert slots[92] > high

    def test_depot_slots_fan_out_on_the_same_side(self):
        axis = build_line_axis({10: [(1, 0.0), (2, 2500.0), (3, 5000.0)]})
        low, _ = axis.extent

        slots = depot_slots(axis, [91, 92], [(91, 0.0), (92, 0.0)])

        assert slots[91] < low and slots[92] < low
        assert slots[91] != slots[92]
