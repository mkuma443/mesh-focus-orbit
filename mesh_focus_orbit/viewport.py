"""Viewport event helpers shared by MFO operators.

This module has no registration side effects and no add-on runtime state.
"""
from mathutils import Vector


def sculpt_cursor_region_coordinate(context, event):
    """Return an explicit cursor coordinate in the owning WINDOW region."""
    try:
        region = context.region
        if region is None or str(region.type) != "WINDOW":
            return None
        width = float(region.width)
        height = float(region.height)
        if width <= 0.0 or height <= 0.0:
            return None
        mouse_x = getattr(event, "mouse_x", None)
        mouse_y = getattr(event, "mouse_y", None)
        if mouse_x is not None and mouse_y is not None:
            x = float(mouse_x) - float(region.x)
            y = float(mouse_y) - float(region.y)
            if 0.0 <= x < width and 0.0 <= y < height:
                return Vector((x, y))
            return None
        region_x = getattr(event, "mouse_region_x", None)
        region_y = getattr(event, "mouse_region_y", None)
        if region_x is None or region_y is None:
            return None
        x = float(region_x)
        y = float(region_y)
        if 0.0 <= x < width and 0.0 <= y < height:
            return Vector((x, y))
    except (AttributeError, ReferenceError, RuntimeError, TypeError, ValueError):
        return None
    return None
