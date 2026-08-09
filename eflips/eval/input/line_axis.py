"""
The distance axis of a time-distance diagram.

A line is not a single route. The database gives it a dozen route variants -- both
directions, short turns, diversions, loops -- and a time-distance diagram needs all of them
projected onto one distance axis. This module builds that projection.

The construction anchors every variant onto the longest one. Because ``elapsed_distance`` is
in metres on both, a station the reference route does not know can still be placed by
carrying the variant's own metres outward from the nearest shared station, so the axis stays
a distance axis rather than degenerating into an ordinal list of stops.

The result is deliberately keyed by *route stop index*, not by station. A ring line visits
the same station twice, and on such a route a station has no single position: it is both
metre 0 and the last metre of the axis. Handing out one position per station draws every
ring line trip as a full-width jump back to the start.

Nothing here touches the database or pandas, so the projection can be exercised directly
with hand-written route data -- which matters, because loops and short turns are awkward to
express in a database fixture.
"""

from dataclasses import dataclass, field
from statistics import mean
from typing import Dict, List, Mapping, Optional, Sequence, Tuple

#: Two candidate positions for one station further apart than this mean the station cannot
#: be used as an anchor: it is a loop terminus, or a variant genuinely serves it twice.
AMBIGUOUS_M: float = 300.0

#: One route's stop sequence: ``(station id, elapsed distance in metres)`` in travel order.
RouteStops = Sequence[Tuple[int, float]]


@dataclass
class LineAxis:
    """
    The distance axis of one line, and where each route variant sits on it.

    :param route_positions: route id -> position in metres per route stop index, in the
        route's own travel order
    :param station_positions: station id -> a representative position, averaged over the
        station's occurrences. Only for callers that need a rough answer, such as "which end
        of the line is this depot near?" -- on a loop it is the midpoint of two real
        positions and therefore a place the station never is.
    :param reference_route_id: the variant whose own metres define the axis, or ``None``
        when the line has no usable route
    """

    route_positions: Dict[int, List[float]] = field(default_factory=dict)
    station_positions: Dict[int, float] = field(default_factory=dict)
    reference_route_id: Optional[int] = None

    @property
    def extent(self) -> Tuple[float, float]:
        """
        The lowest and highest position on the axis proper, in metres.

        :return: a ``(low, high)`` tuple, ``(0.0, 1.0)`` when the axis is empty
        """
        if self.reference_route_id is None:
            return (0.0, 1.0)
        positions = self.route_positions[self.reference_route_id]
        return (min(positions), max(positions))

    def position_for(self, route_id: int, stop_index: int) -> Optional[float]:
        """
        Where one stop of one route sits on the axis.

        :param route_id: the route the stop belongs to
        :param stop_index: the stop's index in the route's travel order
        :return: the position in metres, or ``None`` if the route is not on this axis
        """
        positions = self.route_positions.get(route_id)
        if positions is None or not 0 <= stop_index < len(positions):
            return None
        return positions[stop_index]

    def position_of_station(self, station_id: int) -> Optional[float]:
        """
        A representative position for a station, averaged over its occurrences.

        :param station_id: the station to place
        :return: the position in metres, or ``None`` if the station is not on this axis
        """
        return self.station_positions.get(station_id)


def _anchor_map(candidates: Mapping[int, List[float]]) -> Dict[int, float]:
    """
    The stations that have one unambiguous position, which are the ones worth anchoring on.

    :param candidates: station id -> every position proposed for it so far
    :return: station id -> position, for the stations whose proposals agree
    """
    return {
        station_id: mean(values)
        for station_id, values in candidates.items()
        if max(values) - min(values) <= AMBIGUOUS_M
    }


def _oriented(sequence: RouteStops, reverse: bool) -> List[Tuple[int, float]]:
    """
    Flip a route's stop sequence so that its metres run the same way as the reference.

    :param sequence: the route's stops in travel order
    :param reverse: whether to turn the sequence around
    :return: the stops, possibly reversed, with ``elapsed_distance`` re-based
    """
    if not reverse:
        return list(sequence)
    total = sequence[-1][1]
    return [(station_id, total - elapsed) for station_id, elapsed in reversed(sequence)]


def _project(
    sequence: RouteStops, reference: Mapping[int, float]
) -> Optional[List[float]]:
    """
    Map one route's ``elapsed_distance`` scale onto the reference scale.

    Anchors are the stations the route shares with the reference. Between two anchors the
    mapping is affine; outside the outermost pair it carries the route's own metres forward,
    which is the right extrapolation because both scales are metres.

    :param sequence: the route's stops, already oriented to run with the reference
    :param reference: station id -> position, the stations available to anchor on
    :return: a position per element of ``sequence``, or ``None`` if nothing anchors
    """
    anchors = sorted(
        (elapsed, reference[station_id])
        for station_id, elapsed in sequence
        if station_id in reference
    )
    if not anchors:
        return None

    result: List[float] = []
    for _, elapsed in sequence:
        if len(anchors) == 1 or elapsed <= anchors[0][0]:
            own, ref = anchors[0]
            result.append(ref + (elapsed - own))
        elif elapsed >= anchors[-1][0]:
            own, ref = anchors[-1]
            result.append(ref + (elapsed - own))
        else:
            for (own_a, ref_a), (own_b, ref_b) in zip(anchors, anchors[1:]):
                if own_a <= elapsed <= own_b:
                    span = own_b - own_a
                    fraction = (elapsed - own_a) / span if span else 0.0
                    result.append(ref_a + fraction * (ref_b - ref_a))
                    break
            else:  # pragma: no cover - the bracketing above is exhaustive
                result.append(anchors[-1][1])
    return result


def _agrees_with_reference(
    sequence: RouteStops, reference: Mapping[int, float]
) -> bool:
    """
    Whether a route runs in the same direction as the reference, judged by anchor slope.

    :param sequence: the route's stops in travel order
    :param reference: station id -> position, the stations available to anchor on
    :return: ``True`` when the route's metres grow with the reference's
    """
    anchors = [
        (elapsed, reference[station_id])
        for station_id, elapsed in sequence
        if station_id in reference
    ]
    if len(anchors) < 2:
        return True
    forward = sum(
        1
        for (own_a, ref_a), (own_b, ref_b) in zip(anchors, anchors[1:])
        if (own_b - own_a) * (ref_b - ref_a) >= 0
    )
    return forward * 2 >= len(anchors) - 1


def build_line_axis(routes: Mapping[int, RouteStops]) -> LineAxis:
    """
    Build the distance axis of one line from all of its route variants.

    Pass only the variants that carry passengers. Depot legs are often attached to the line
    they serve, and a depot sits kilometres off the course in an arbitrary direction, so
    letting one into the axis squeezes the line's own stops into a corner of the diagram.

    :param routes: route id -> that route's stops as ``(station id, elapsed distance)`` in
        travel order, as ``AssocRouteStation`` already provides them
    :return: the line's axis
    """
    usable = {route_id: stops for route_id, stops in routes.items() if len(stops) >= 2}
    if not usable:
        return LineAxis()

    # The longest variant defines the axis outright: its own metres are the scale.
    reference_id = max(
        usable, key=lambda route_id: (len(usable[route_id]), usable[route_id][-1][1])
    )
    reference_stops = usable[reference_id]
    route_positions: Dict[int, List[float]] = {
        reference_id: [elapsed for _, elapsed in reference_stops]
    }
    candidates: Dict[int, List[float]] = {}
    for station_id, elapsed in reference_stops:
        candidates.setdefault(station_id, []).append(elapsed)

    # Two passes: a variant that shares nothing with the reference may still share something
    # with a variant already merged in, so retry it against the grown axis.
    pending = [route_id for route_id in usable if route_id != reference_id]
    for _ in range(2):
        merged: List[int] = []
        anchors = _anchor_map(candidates)
        for route_id in pending:
            stops = usable[route_id]
            forward = _agrees_with_reference(stops, anchors)
            sequence = _oriented(stops, reverse=not forward)
            projected = _project(sequence, anchors)
            if projected is None:
                continue
            # _oriented reversed the travel order; undo that so the array stays aligned with
            # the route's own stop order, which is what a trip's stops are ordered by.
            route_positions[route_id] = (
                projected if forward else list(reversed(projected))
            )
            for (station_id, _), position in zip(sequence, projected):
                candidates.setdefault(station_id, []).append(position)
            merged.append(route_id)
        pending = [route_id for route_id in pending if route_id not in merged]
        if not pending:
            break

    return LineAxis(
        route_positions=route_positions,
        station_positions={
            station_id: mean(values) for station_id, values in candidates.items()
        },
        reference_route_id=reference_id,
    )


def depot_slots(
    axis: LineAxis,
    depot_station_ids: Sequence[int],
    attachments: Sequence[Tuple[int, Optional[float]]],
) -> Dict[int, float]:
    """
    Place each depot in the left or right margin beyond the ends of the line.

    A depot goes into the margin nearer the line stations it actually connects to, so a
    pull-out to the western terminus is drawn entering from the west rather than crossing the
    whole diagram. Within a margin the depots are fanned out so their legs stay apart.

    :param axis: the line's axis, which supplies the extent the margins sit outside of
    :param depot_station_ids: the depot stations to place
    :param attachments: ``(depot station id, position on the axis)`` for every depot leg of
        this line, which is what decides which end of the line each depot serves. The
        position may be ``None`` when the leg's other end is not on the axis.
    :return: depot station id -> position in metres, outside the axis extent
    """
    low, high = axis.extent
    span = max(high - low, 1.0)
    margin = 0.10 * span

    touches: Dict[int, List[float]] = {
        station_id: [] for station_id in depot_station_ids
    }
    for station_id, position in attachments:
        if station_id in touches and position is not None:
            touches[station_id].append(position)

    midpoint = (low + high) / 2
    left: List[int] = []
    right: List[int] = []
    for station_id in depot_station_ids:
        seen = touches.get(station_id) or []
        near = mean(seen) if seen else midpoint
        (left if near < midpoint else right).append(station_id)

    slots: Dict[int, float] = {}
    for index, station_id in enumerate(left):
        slots[station_id] = low - margin * (1 + index * 0.45)
    for index, station_id in enumerate(right):
        slots[station_id] = high + margin * (1 + index * 0.45)
    return slots
