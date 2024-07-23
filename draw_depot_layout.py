import os
from math import ceil
from typing import List, Tuple
from datetime import datetime, timedelta
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import matplotlib.transforms as transforms
import matplotlib.animation as animation
import numpy as np

import pytz

from eflips.model import VehicleType, Vehicle, Area, Process, AreaType, Depot, Event

from sqlalchemy import create_engine
from sqlalchemy.orm import Session


def draw_area(ax, area, center, angle, buffer=0.9):
    slots_in_area = []
    if area.area_type == AreaType.LINE:
        vehicle_type = area.vehicle_type
        assert vehicle_type is not None, "Vehicle type must be specified for line area"
        for i in range(area.capacity):
            slot_center = (center[0], center[1] + (vehicle_type.length + buffer) * i)
            slot = draw_slot(
                ax, area.vehicle_type, slot_center, angle, "grey", text=str(i)
            )
            slots_in_area.append(slot)

    elif area.area_type == AreaType.DIRECT_ONESIDE:
        raise NotImplementedError
    elif area.area_type == AreaType.DIRECT_TWOSIDE:
        raise NotImplementedError
    else:
        raise ValueError("Invalid area type")

    return slots_in_area


def draw_slot(ax, vehicle_type, center, angle, color, buffer=0.9, text=None):
    # Create a Rectangle patch

    assert (
        vehicle_type.width is not None and vehicle_type.length is not None
    ), "Vehicle must have valid width and length. "
    size = (vehicle_type.width + buffer, vehicle_type.length)
    rect = patches.Rectangle(center, size[0], size[1], linewidth=1, facecolor=color)

    # Create an Affine2D transform object and apply the rotation and translation
    t = transforms.Affine2D().rotate_deg(angle)

    # Set the transform for the rectangle
    rect.set_transform(t + ax.transData)

    # Add the patch to the Axes
    ax.add_patch(rect)
    rect_center = (center[0] + size[0] / 2, center[1] + size[1] / 2)

    ax.text(
        rect_center[0],
        rect_center[1],
        text,
        ha="center",
        va="center",
        transform=t + ax.transData,
    )

    return rect


def get_slot_occupancy(area_id, slot_id, simulation_start) -> List[Tuple]:
    events = (
        session.query(Event)
        .filter(Event.area_id == area_id, Event.subloc_no == slot_id)
        .order_by(Event.time_start)
        .all()
    )

    return [
        (
            int(ceil((event.time_start - simulation_start).total_seconds() / 300)),
            int(ceil((event.time_end - simulation_start).total_seconds() / 300)),
        )
        for event in events
    ]


def is_available(time, available_times):
    for time_tuple in available_times:
        if time_tuple[0] < time < time_tuple[1]:
            return True
    return False


def animate(frame, area_dict, area_occupancy):

    for area_id, slots in area_dict.items():
        for slot_id, slot in enumerate(slots):
            slot_occupancy = area_occupancy[(area_id, slot_id)]
            slot.set_facecolor(
                "green" if is_available(frame, slot_occupancy) else "grey"
            )


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
    # Call the function with center, size and angle
    fig, ax = plt.subplots()

    areas = (
        session.query(Area)
        .filter(Area.scenario_id == SCENARIO_ID, Area.area_type == AreaType.LINE)
        .all()
    )

    buffer = 0.9

    area_dict = {}
    area_occupancy = {}

    dummy_simulation_start = datetime(
        2023, 6, 30, 20, 0, tzinfo=pytz.timezone("Europe/Berlin")
    )

    for i in range(len(areas)):
        area = areas[i]
        vehicle_type = area.vehicle_type

        assert vehicle_type is not None, "Vehicle type must be specified for line area"

        slots_in_area = draw_area(
            ax, area, (i * (vehicle_type.width * 2 + buffer), 0), 0
        )

        area_dict[area.id] = slots_in_area
        for s in range(len(slots_in_area)):
            area_occupancy[area.id, s] = get_slot_occupancy(
                area.id, s, dummy_simulation_start
            )

    # Set the x and y axis limits
    plt.xlim(-10, 100)
    plt.ylim(-10, 175)
    plt.axis("equal")

    # frames: how many frames to generate. The function will be called for each frame with i as the frame number
    # interval: delay between frames in milliseconds
    # frames should be the total stored time in the depot

    ani = animation.FuncAnimation(
        fig,
        animate,
        frames=3000,
        interval=1,
        fargs=(area_dict, area_occupancy),
        save_count=3000,
    )
    #
    # # fps: frames per second when displayed. Basically means how fast I will see the animation
    ani.save("rectangle_animation.gif", writer="imagemagick", fps=10)

    plt.show()
