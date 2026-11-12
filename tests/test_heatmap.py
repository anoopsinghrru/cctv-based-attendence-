import numpy as np

from src.heatmap import add_boxes, heat_overlay, new_heat


def test_each_face_counts_once_over_the_area_it_covers():
    # a large near face and a small far face
    grid = add_boxes(new_heat((240, 320)), [(40, 40, 120, 140), (250, 30, 266, 50)])
    assert grid.shape == (31, 41) and grid.max() == 1
    assert grid[11, 10] == 1 and grid[5, 32] == 1 and grid[25, 30] == 0
    assert grid[:, :20].sum() > 10 * grid[:, 25:].sum()
    assert add_boxes(grid, [(40, 40, 120, 140)]).max() == 2          # seen again: counts add up


def test_overlay_colours_only_where_faces_were_seen():
    frame = np.full((60, 80, 3), 100, np.uint8)
    heat = np.zeros((6, 8), np.float32)
    heat[3, 4] = 5
    over = heat_overlay(frame, heat)
    assert over.shape == frame.shape and over.dtype == np.uint8
    assert np.array_equal(over[0, 0], frame[0, 0]) and not np.array_equal(over[35, 45], frame[35, 45])
    assert np.array_equal(heat_overlay(frame, np.zeros((6, 8), np.float32)), frame)   # nothing seen: frame unchanged
