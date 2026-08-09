from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional, Tuple
from zoneinfo import ZoneInfo

import dash_cytoscape as cyto  # type: ignore
import folium  # type: ignore
import pandas as pd
import plotly.express as px  # type: ignore
import plotly.graph_objs as go  # type: ignore
import seaborn as sns  # type: ignore
from eflips.model import TripType


def rotation_info(
    prepared_data: pd.DataFrame, timezone: ZoneInfo = ZoneInfo("Europe/Berlin")
) -> go.Figure:
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
    - line_is_unified: True if the rotation only contains one line
    - start_station: the name of the departure station
    - end_station: the name of the arrival station

    :return: A plotly figure object
    """

    # Go through the dataframe and fix the timezones
    for col in ["time_start", "time_end"]:
        prepared_data[col] = prepared_data[col].dt.tz_convert(timezone)

    fig = px.timeline(
        prepared_data,
        x_start="time_start",
        x_end="time_end",
        y="rotation_name",
        color="total_distance",
        labels={
            "time_start": "Departure Time",
            "time_end": "Arrival Time",
            "rotation_name": "Rotation Name",
            "total_distance": "Total Distance (km)",
            "line_name": "Line Name",
            "vehicle_type_name": "Vehicle Type Name",
            "start_station": "Start Station",
            "end_station": "End Station",
        },
        hover_name="rotation_name",
        hover_data=[
            "vehicle_type_name",
            "total_distance",
            "start_station",
            "end_station",
        ],
        pattern_shape="line_name",
    )
    fig.update_layout(legend_orientation="h")
    return fig


def geographic_trip_plot(prepared_data: pd.DataFrame) -> folium.Map:
    """
    This function visualizes the trips on a map using folium. The trips are lines between the departure and arrival
    stations.



    :param prepared_data: A Pandas dataframe with the following columns:
            - rotation_id: the id of the rotation
            - rotation_name: the name of the rotation
            - vehicle_type_id: the id of the vehicle type
            - vehicle_type_name: the name of the vehicle type
            - originating_depot_id: the id of the originating depot
            - originating_depot_name: the name of the originating depot
            - distance: the distance of the route
            - coordinates: An array of *(lon, lat)* tuples with the coordinates of the route - the shape if set, otherwise the stops
            - line_name: the name of the line, which is the first part of the rotation name. Used for sorting
    :return: A folium map object
    """
    # Use seaborn to create a color palette for each originating depot
    palette = sns.color_palette(
        "husl", n_colors=len(prepared_data["originating_depot_id"].unique())
    )
    colors = {}
    for depot_name in prepared_data["originating_depot_name"].unique():
        color = palette.pop(0)
        # Turn the color into a hex string
        color = "#{:02x}{:02x}{:02x}".format(*[int(c * 255) for c in color])
        colors[depot_name] = color

    # Obtain the mean latitude and longitude for the map center
    lat_center = (
        prepared_data["coordinates"]
        .apply(lambda x: sum([c[0] for c in x]) / len(x))
        .mean()
    )
    lon_center = (
        prepared_data["coordinates"]
        .apply(lambda x: sum([c[1] for c in x]) / len(x))
        .mean()
    )

    map = folium.Map(
        location=[lat_center, lon_center], zoom_start=11, tiles="Cartodb dark_matter"
    )
    for i, row in prepared_data.iterrows():
        color = colors[row["originating_depot_name"]]
        pl = folium.PolyLine(row["coordinates"], color=color, weight=2.5, opacity=1)
        map.add_child(pl)
    return map


def single_rotation_info(prepared_data: pd.DataFrame) -> cyto.Cytoscape:
    """
    This plots a single rotation as a network graph. The nodes are the stops and the edges are the trips between the
    stops.

    :param prepared_data: The result of the rotation_info function, a dataframe with the following columns:
        - trip_id: the id of the trip
        - trip_type: the type of the trip
        - line_name: the name of the line
        - route_name: the name of the route
        - distance: the distance of the route
        - time_start: the departure time of the trip
        - time_end: the arrival time of the trip
        - departure_station_name: the name of the departure station
        - departure_station_id: the id of the departure station
        - arrival_station_name: the name of the arrival station
        - arrival_station_id: the id of the arrival station

    :return: A Dash Cytoscape object. It can be added to a Dash layout.
    """

    # Create a list of elements, the stations are nodes and the trips are edges
    stations_already_added = set()
    elements = []
    for i, row in prepared_data.iterrows():
        # Handle the station (node) information
        if row.departure_station_id not in stations_already_added:
            elements.append(
                {
                    "data": {
                        "id": str(row.departure_station_id),
                        "label": row.departure_station_name,
                    }
                }
            )
            stations_already_added.add(row.departure_station_id)
        if row.arrival_station_id not in stations_already_added:
            elements.append(
                {
                    "data": {
                        "id": str(row.arrival_station_id),
                        "label": str(row.arrival_station_name),
                    }
                }
            )
            stations_already_added.add(row.arrival_station_id)

        # Handle the trip (edge) information
        type_str = "Passenger" if row.trip_type == TripType.PASSENGER else "Deadhead"
        color = "#9dbaea" if row.trip_type == TripType.PASSENGER else "#f4a261"
        trip_str = (
            f"{type_str} trip {row.departure_station_name} ({row.departure_time.strftime('%H:%M')})"
            f" -> {row.arrival_station_name} ({row.arrival_time.strftime('%H:%M')})"
        )
        elements.append(
            {
                "data": {
                    "source": str(row.departure_station_id),
                    "target": str(row.arrival_station_id),
                    "label": trip_str,
                    "color": color,
                }
            }
        )

    cytograph = cyto.Cytoscape(
        id="cytoscape",
        elements=elements,
        layout={"name": "cose"},
        style={"width": "1000px", "height": "1000px"},
        stylesheet=[
            {
                "selector": "node",
                "style": {"label": "data(label)", "background-color": "#11479e"},
            },
            {
                "selector": "edge",
                "style": {
                    "label": "data(label)",
                    "curve-style": "bezier",
                    "target-arrow-shape": "triangle",
                    "line-color": "data(color)",
                    "target-arrow-color": "data(color)",
                },
            },
        ],
    )

    return cytograph


#: All service days are drawn against one date, so that the y axis is a time of day and the
#: days of a week can be compared by flipping between them.
_SERVICE_DAY_REFERENCE = datetime(2000, 1, 3)

#: The service day hours the y axis is ticked at. It starts at the service day boundary and
#: runs past 24:00, because those hours are what the extended clock exists for. It reaches
#: 29:00 rather than stopping at the 27:00 boundary because that boundary binds *departures*:
#: a trip leaving at 26:52 arrives well after 27:00 and its last stop needs a gridline too.
#: Ticks beyond the data cost nothing, since the axis autoranges to what is drawn.
_SERVICE_DAY_HOURS = range(3, 30)

_WEEKDAYS = ("Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun")

#: Category -> (legend label, colour, dash, width). Four categories, so hue is doing honest
#: categorical work; the dash repeats the distinction as a second channel, so it survives a
#: greyscale print and colour-vision deficiency.
_TRIP_STYLES: Dict[str, Tuple[str, str, str, float]] = {
    "single": ("Passenger trip, single-line rotation", "#2a78d6", "solid", 1.4),
    "multi": ("Passenger trip, multi-line rotation", "#eb6834", "solid", 1.4),
    "pull_out": ("Pull-out trip", "#1baf7a", "dash", 1.2),
    "pull_in": ("Pull-in trip", "#4a3aa7", "dot", 1.2),
}

#: How many station labels the distance axis may carry before they are thinned.
_MAX_X_TICKS = 28


def _extended_clock(seconds: int) -> str:
    """
    Seconds since the start of a service day's calendar date as a service-day clock.

    02:18 on the following morning becomes ``26:18:00``. This is the convention timetables
    use, and it exists for exactly the case that makes it necessary here: written as
    ``02:18`` a late night run cannot be told from an early morning one, and on a
    time-distance diagram -- where it is drawn at the *bottom* of the day, below 23:00 -- the
    plain clock reads as an error rather than as the small hours.

    :param seconds: seconds since midnight of the service day's own date
    :return: the time as ``HH:MM:SS``, with hours running past 24
    """
    hours, remainder = divmod(seconds, 3600)
    minutes, secs = divmod(remainder, 60)
    return f"{hours:02d}:{minutes:02d}:{secs:02d}"


def _distance_axis_ticks(
    prepared_data: pd.DataFrame, max_ticks: int
) -> Tuple[List[float], List[str]]:
    """
    Tick positions and station names for the distance axis.

    The ticks come from the route variant that defines the axis, taken over every trip that
    runs it so that a stop no single trip serves is still labelled. Thinning goes by
    *distance* rather than by every n-th station, because the axis is metric: three stops
    within 600 m of each other would otherwise print three labels on top of one another while
    a 2 km gap stayed bare. Both ends of the line are kept whatever the spacing, and the
    depots in the margins are always labelled.

    A ring line needs no special handling. Its terminus holds two different positions, so it
    contributes two rows and gets a tick at each end of the axis.

    :param prepared_data: the frame from
        :func:`eflips.eval.input.prepare.time_distance_diagram`
    :param max_ticks: how many station labels the axis may carry
    :return: the tick positions and their labels
    """
    reference = prepared_data[prepared_data["is_axis_reference"]]
    entries: List[Tuple[float, str]] = []
    if not reference.empty:
        unique = (
            reference[["position", "station_name", "station_is_depot"]]
            .drop_duplicates(subset=["position", "station_name"])
            .sort_values("position")
        )
        entries = [
            (
                float(position),
                f"{name} (Depot)" if is_depot else str(name),
            )
            for position, name, is_depot in zip(
                unique["position"].tolist(),
                unique["station_name"].tolist(),
                unique["station_is_depot"].tolist(),
            )
        ]

    if len(entries) > 2:
        low, high = entries[0][0], entries[-1][0]
        # The gap has to hold whatever the station count: two stops 10 m apart collide even
        # on a line that has only a dozen of them.
        minimum_gap = (high - low) / max_ticks
        kept = [entries[0]]
        for entry in entries[1:-1]:
            if entry[0] - kept[-1][0] >= minimum_gap:
                kept.append(entry)
        if entries[-1][0] - kept[-1][0] < minimum_gap and len(kept) > 1:
            kept.pop()
        kept.append(entries[-1])
        entries = kept

    margins = prepared_data[prepared_data["is_margin"]]
    if not margins.empty:
        unique_margins = margins[["position", "station_name"]].drop_duplicates()
        for position, name in zip(
            unique_margins["position"].tolist(),
            unique_margins["station_name"].tolist(),
        ):
            entries.append((float(position), f"{name} (Depot)"))

    entries.sort()
    return [position for position, _ in entries], [label for _, label in entries]


def _service_day_buttons(
    days: List[pd.Timestamp],
    passenger_counts: Dict[pd.Timestamp, int],
    categories: List[str],
) -> List[Dict[str, Any]]:
    """
    A button per service day, plus one that overlays the whole period.

    :param days: the service days present, in order
    :param passenger_counts: service day -> how many passenger trips it carries
    :param categories: the trip categories, in the order the traces were added
    :param days: the service days present, in order
    :return: the ``updatemenus`` list for the layout
    """
    stride = len(categories)
    buttons: List[Dict[str, Any]] = []
    for index, day in enumerate(days):
        visible = [False] * (len(days) * stride)
        for offset in range(stride):
            visible[index * stride + offset] = True
        label = f"{_WEEKDAYS[day.weekday()]} {day.strftime('%d.%m.')}"
        buttons.append(
            {
                "label": f"{label} ({passenger_counts.get(day, 0)})",
                "method": "restyle",
                "args": [{"visible": visible}],
            }
        )
    buttons.append(
        {
            "label": "All days",
            "method": "restyle",
            "args": [{"visible": [True] * (len(days) * stride)}],
        }
    )
    return [
        {
            "type": "buttons",
            "direction": "right",
            "showactive": True,
            "x": 0,
            "xanchor": "left",
            "y": 1.02,
            "yanchor": "bottom",
            "pad": {"t": 2, "b": 2, "l": 2, "r": 2},
            "font": {"size": 11},
            "buttons": buttons,
        }
    ]


def time_distance_diagram(
    prepared_data: pd.DataFrame,
    timezone: ZoneInfo = ZoneInfo("Europe/Berlin"),
    service_day_start: timedelta = timedelta(hours=3),
    height: Optional[int] = None,
) -> go.Figure:
    """
    Plot one line's trips as a time-distance diagram (Bildfahrplan, Marey diagram).

    Time runs downward and distance along the line runs left to right, so one trip is one
    polyline: sloping where the vehicle moves, vertical where it stands. Pull-out and pull-in
    trips are drawn in margin bands beyond the two ends of the line, and trips whose rotation
    only ever serves this line are drawn in a different colour and dash from those whose
    rotation is shared with another line.

    The y axis is a *service day*, not a calendar day. It starts at ``service_day_start`` and
    the hours after midnight are labelled 24:00, 25:00, 26:00 rather than 00:00, 01:00,
    02:00, so that a run leaving at 26:18 sits at the bottom of the diagram where it belongs
    -- it is the tail of that day's night service, not an early run of the next day. Each
    trip is placed on the service day it *departs* in, so a trip running through the boundary
    is never split in two.

    Every service day in the data gets its own set of traces and a button to select it; the
    day with the most passenger trips is shown first, and a final button overlays them all.

    This function sets no height, per the conventions of this package. A whole service day at
    the default height makes a 30-minute journey nearly horizontal, so pass ``height=1400``
    or thereabouts when rendering to HTML.

    :param prepared_data: The result of the
        :func:`eflips.eval.input.prepare.time_distance_diagram` function, a dataframe with
        the following columns:

        - trip_id: the id of the trip; the rows sharing one become one polyline
        - trip_kind: "passenger", "pull_out" or "pull_in"
        - rotation_name: the name of the rotation, shown on hover
        - rotation_is_single_line: True if the rotation only ever serves this line
        - stop_index: the position of the stop within the trip
        - station_name: the name of the station, shown on hover and on the axis
        - station_is_depot: True if the station houses a depot
        - position: how far along the line the stop is, in metres
        - is_margin: True if the stop is drawn in a depot margin
        - is_axis_reference: True if the stop lies on the route defining the axis
        - arrival_time: when the vehicle arrives, timezone-aware
        - departure_time: when it leaves again, timezone-aware
        - trip_departure_time: the departure of the whole trip, timezone-aware

    :param timezone: The timezone the service day is cut in. Service days are a local-time
        idea, so this decides which day a trip belongs to as well as how times are shown
    :param service_day_start: How far into the calendar day a service day begins. Three
        hours is the usual convention in public transit
    :param height: The height of the figure in pixels, or None to leave it to the caller
    :return: A plotly figure object
    """
    figure = go.Figure()
    figure.update_layout(
        xaxis_title="Distance along the Line",
        yaxis_title="Time of Day (Service Day)",
    )

    if prepared_data.empty:
        if height is not None:
            figure.update_layout(height=height)
        return figure

    data = prepared_data.copy()

    # Local time first: a service day is a local-time idea, so the boundary cannot be cut
    # until the timestamps are in the timezone the operator works in.
    local_arrival = data["arrival_time"].dt.tz_convert(timezone).dt.tz_localize(None)
    local_departure = (
        data["departure_time"].dt.tz_convert(timezone).dt.tz_localize(None)
    )
    local_trip = (
        data["trip_departure_time"].dt.tz_convert(timezone).dt.tz_localize(None)
    )

    # The whole trip's service day, not each stop's: deriving it per stop would put a trip
    # that runs through the boundary on two different days and draw it as a 24-hour vertical
    # line straight through the diagram.
    service_day = (local_trip - service_day_start).dt.normalize()
    data["service_day"] = service_day
    data["y_arrival"] = _SERVICE_DAY_REFERENCE + (local_arrival - service_day)
    data["y_departure"] = _SERVICE_DAY_REFERENCE + (local_departure - service_day)

    data["category"] = data["trip_kind"].where(
        data["trip_kind"] != "passenger",
        data["rotation_is_single_line"].map({True: "single", False: "multi"}),
    )

    arrival_clock = (local_arrival - service_day).dt.total_seconds().astype(int)
    departure_clock = (local_departure - service_day).dt.total_seconds().astype(int)
    data["hover_arrival"] = [
        f"{name} · {_extended_clock(seconds)} · {rotation}"
        for name, seconds, rotation in zip(
            data["station_name"], arrival_clock, data["rotation_name"]
        )
    ]
    data["hover_departure"] = [
        f"{name} · {_extended_clock(seconds)} · {rotation}"
        for name, seconds, rotation in zip(
            data["station_name"], departure_clock, data["rotation_name"]
        )
    ]

    days = sorted(data["service_day"].unique())
    categories = list(_TRIP_STYLES)
    passenger_counts = {
        day: int(
            data.loc[
                (data["service_day"] == day) & (data["trip_kind"] == "passenger"),
                "trip_id",
            ].nunique()
        )
        for day in days
    }

    data = data.sort_values(["service_day", "category", "trip_id", "stop_index"])

    # Traces are added day-major, category-minor, and every category is present for every
    # day even when it is empty: the button arithmetic depends on that stride.
    for day in days:
        for category in categories:
            label, colour, dash, width = _TRIP_STYLES[category]
            subset = data[(data["service_day"] == day) & (data["category"] == category)]
            xs: List[Optional[float]] = []
            ys: List[Optional[datetime]] = []
            texts: List[str] = []
            for _, trip_rows in subset.groupby("trip_id", sort=False):
                positions = trip_rows["position"].tolist()
                arrivals = trip_rows["y_arrival"].tolist()
                departures = trip_rows["y_departure"].tolist()
                hover_arrivals = trip_rows["hover_arrival"].tolist()
                hover_departures = trip_rows["hover_departure"].tolist()
                for index, position in enumerate(positions):
                    xs.append(float(position))
                    ys.append(arrivals[index])
                    texts.append(str(hover_arrivals[index]))
                    if departures[index] != arrivals[index]:
                        # The vehicle dwells: a vertical segment, time passing in one place.
                        xs.append(float(position))
                        ys.append(departures[index])
                        texts.append(str(hover_departures[index]))
                # A gap, so the next trip starts a new polyline rather than continuing this
                # one. One trace per trip would be cleaner but unopenable: a busy line has
                # several hundred trips a day of forty stops each.
                xs.append(None)
                ys.append(None)
                texts.append("")

            figure.add_trace(
                go.Scattergl(
                    x=xs,
                    y=ys,
                    mode="lines",
                    name=label,
                    legendgroup=category,
                    showlegend=bool(xs),
                    line={"color": colour, "width": width, "dash": dash},
                    text=texts,
                    hovertemplate="<b>%{text}</b><br>" + label + "<extra></extra>",
                    connectgaps=False,
                    visible=False,
                )
            )

    default_day = max(days, key=lambda day: passenger_counts.get(day, 0))
    default_index = days.index(default_day)
    for offset in range(len(categories)):
        figure.data[default_index * len(categories) + offset].visible = True

    figure.update_layout(
        updatemenus=_service_day_buttons(days, passenger_counts, categories)
    )

    tick_positions, tick_labels = _distance_axis_ticks(data, _MAX_X_TICKS)
    hour_ticks = [
        _SERVICE_DAY_REFERENCE + timedelta(hours=hour) for hour in _SERVICE_DAY_HOURS
    ]
    hour_labels = [f"{hour:02d}:00" for hour in _SERVICE_DAY_HOURS]

    reference = data[data["is_axis_reference"]]
    figure.update_layout(
        xaxis={
            "title": "Distance along the Line",
            "tickmode": "array",
            "tickvals": tick_positions,
            "ticktext": tick_labels,
            "tickangle": -60,
            "automargin": True,
        },
        yaxis={
            "title": "Time of Day (Service Day)",
            "autorange": "reversed",
            # Explicit ticks rather than a date format, so that the hours after midnight
            # read 24:00/25:00/26:00 and stay visibly part of the same service day.
            "tickmode": "array",
            "tickvals": hour_ticks,
            "ticktext": hour_labels,
            "automargin": True,
        },
        hovermode="closest",
        legend={
            "orientation": "h",
            "yanchor": "bottom",
            "y": 1.07,
            "xanchor": "right",
            "x": 1.0,
        },
    )

    if not reference.empty:
        # The two rules that separate the line proper from the depot margins.
        for position in (reference["position"].min(), reference["position"].max()):
            figure.add_vline(x=float(position), line_width=1, line_color="#b6b5ad")

    if height is not None:
        figure.update_layout(height=height)

    return figure
