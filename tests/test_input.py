"""
This file contains tests for the "input" visualizations, which are visualizations that can be done before the simulation
has been run.
"""

import eflips.depot.api
import plotly.graph_objs as go  # type: ignore
from eflips.model import (
    Rotation,
)

import eflips.eval.input.prepare
import eflips.eval.input.visualize
from eflips.eval.input.prepare import rotation_name_for_sorting
from tests.base import BaseTest


class TestInput(BaseTest):

    def test_rotation_name_for_for_sorting(self):
        assert rotation_name_for_sorting("A/B") == "A"
        assert rotation_name_for_sorting("A/B honig") == "A"
        assert rotation_name_for_sorting("A") == "A"
        assert rotation_name_for_sorting(None) is None

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
