from datetime import datetime
from math import ceil
from typing import List, Tuple, Optional

from eflips.model import AreaType, VehicleType, Event, Area
from matplotlib.axes import Axes
from matplotlib import patches, transforms
from sqlalchemy.orm import Session


def _draw_slot(
    ax: Axes,
    vehicle_type: VehicleType,
    center: Tuple[float, float],
    angle: int,
    color: str,
    buffer: float = 0.9,
    text: Optional[str] = None,
) -> patches.Rectangle:
    """
    This function takes all the parameters needed to draw a slot and returns a :class:`matplotlib.patches.Rectangle` object.

    :param ax: a :class:`matplotlib.axes.Axes` object for the plot
    :param vehicle_type: a :class:`eflips.model.VehicleType` object representing the vehicle type
    :param center: a tuple representing the center of the slot
    :param angle: an integer representing the angle of the slot in degrees
    :param color: a string representing the color of the slot
    :param buffer: a float representing the buffer distance around the slot
    :param text: a string representing the text to be displayed in the slot

    :return: a :class:`matplotlib.patches.Rectangle` object
    """
    if vehicle_type.length is None or vehicle_type.width is None:
        # Use default 12m bus size
        size = (2.55 + buffer, 12)
    else:
        size = (vehicle_type.width + buffer, vehicle_type.length)
    rect = patches.Rectangle(center, size[0], size[1], linewidth=1, facecolor=color)

    # Create an Affine2D transform object and apply the rotation and translation
    # Rotate around its own center
    t = transforms.Affine2D().rotate_deg_around(center[0], center[1], angle)

    # Set the transform for the rectangle
    # rect.set_transform(t)
    rect.set_transform(t + ax.transData)

    # Add the patch to the Axes
    ax.add_patch(rect)
    rect_center = (center[0] + size[0] / 2, center[1] + size[1] / 2)

    if text is not None:
        ax.text(
            rect_center[0],
            rect_center[1],
            text,
            ha="center",
            va="center",
            transform=t + ax.transData,
            fontsize=3,
        )

    return rect


def _draw_area(
    ax: Axes, area: Area, center: Tuple[float, float], buffer: float = 0.9
) -> List[patches.Rectangle]:
    """
    This function returns a list of :class:`matplotlib.patches.Rectangle` objects representing the slots in the area

    :param ax: a :class:`matplotlib.axes.Axes` object for the plot
    :param area: a :class:`eflips.model.Area` object representing the area
    :param center: a tuple representing the center of the area
    :param buffer: a float representing the buffer distance between the areas

    :return: a list of :class:`matplotlib.patches.Rectangle` objects
    """
    slots_in_area = []

    if area.vehicle_type is None:
        vehicle_type = VehicleType(length=12, width=2.55)

    else:
        vehicle_type = area.vehicle_type

    vehicle_length = vehicle_type.length if vehicle_type.length is not None else 12

    for i in range(area.capacity):
        slot_center = (center[0], center[1] + (vehicle_length + buffer) * i)
        slot = _draw_slot(
            ax,
            vehicle_type,
            slot_center,
            0 if area.area_type == AreaType.LINE else 45,
            "lightgrey",
            text=str(i),
        )
        slots_in_area.append(slot)

    return slots_in_area


def _is_occupied(time: int, occupied_times: List[Tuple[int, int]]) -> bool:
    """
    This function checks if a given time is within the available times

    :param time: the time to be checked
    :param occupied_times: a list of 2-tuples representing the start and end time of the occupied slots

    :return: a boolean value. True for this time is occupied, False otherwise
    """
    for time_tuple in occupied_times:
        if time_tuple[0] <= time <= time_tuple[1]:
            return True
    return False


def _get_slot_occupancy(
    area_id: int,
    slot_id: int,
    session: Session,
    animation_range: Tuple[datetime, datetime],
) -> List[Tuple[int, int]]:
    """
    This function returns the occupancy of a slot in a given area

    :param area_id: the id of the area :param slot_id: the id of the slot
    :param session: a :class:`sqlalchemy.orm.Session` object
    :param animation_range: a tuple of two :class:`datetime.datetime` objects representing the start and end time that
           the animation should render

    :return: a list of 2-tuples representing the start and end time in seconds relative to
             animation_range[0] of the occupied slots

    """

    events = (
        session.query(Event)
        .filter(Event.area_id == area_id, Event.subloc_no == slot_id)
        .order_by(Event.time_start)
        .all()
    )

    occupancy = []
    for event in events:
        if event.time_end < animation_range[0] or event.time_start > animation_range[1]:
            continue
        else:
            start = max((event.time_start - animation_range[0]).total_seconds(), 0)
            end = (event.time_end - animation_range[0]).total_seconds()
            occupancy.append((int(start), int(end)))

    return occupancy
