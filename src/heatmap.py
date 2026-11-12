"""Seating heat map: where faces were seen during a session, drawn in pseudo-colour.

Every detected face adds one sighting to a coarse grid; at the end of the session the grid is
mapped to colours (blue = seen a few times, red = seen in most frames) and blended over a frame.
"""
import cv2
import numpy as np


def new_heat(frame_shape, cell=8):
    """An empty heat grid for a frame, `cell` frame pixels per grid cell."""
    return np.zeros((frame_shape[0] // cell + 1, frame_shape[1] // cell + 1), np.float32)


def add_boxes(heat, boxes, cell=8):
    """Add one sighting per box (x1, y1, x2, y2 in frame pixels) as an ellipse of the box's own
    size, so a large near face and a small far face each count once over the area they cover."""
    for x1, y1, x2, y2 in boxes:
        mark = np.zeros_like(heat)
        centre = (int(round((x1 + x2) / 2 / cell)), int(round((y1 + y2) / 2 / cell)))
        axes = (max(1, int(round((x2 - x1) / 2 / cell))), max(1, int(round((y2 - y1) / 2 / cell))))
        cv2.ellipse(mark, centre, axes, 0, 0, 360, 1.0, -1)
        heat += mark
    return heat


def heat_overlay(frame, heat, alpha=0.55):
    """Blend a non-negative heat array over a frame in pseudo-colour (hot = high)."""
    heat = cv2.resize(heat.astype(np.float32), (frame.shape[1], frame.shape[0]), interpolation=cv2.INTER_CUBIC)
    norm = np.clip(heat / max(float(heat.max()), 1e-9), 0, 1)
    colour = cv2.applyColorMap((norm * 255).astype(np.uint8), cv2.COLORMAP_JET)
    weight = (alpha * np.sqrt(norm))[..., None]            # transparent where nothing was seen
    return (frame * (1 - weight) + colour * weight).astype(np.uint8)
