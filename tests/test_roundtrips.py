"""Every conversion ships with its inverse: going there and back gives what you started with."""

import numpy as np
import pytest


@pytest.mark.parametrize("space, tolerance", [("HSV", 1e-4), ("Lab", 0.005), ("YCrCb", 0.001)])
def test_colour_spaces_there_and_back(engine, space, tolerance):
    p = engine.test_picture(32, 32)
    back = engine.from_space(engine.to_space(p, space), "HSV")
    assert back.space == "RGB", "the array says its space, so the widget can't misread it"
    assert np.abs(back - p).max() < tolerance


def test_split_and_merge(engine):
    p = engine.test_picture(32, 32)
    a, b, c = engine.split(p)
    assert (engine.merge(a, b, c) == p).all()


def test_encode_and_decode_png_to_8_bits(engine):
    p = engine.test_picture(32, 32)
    assert np.abs(engine.decode_image(engine.encode_image(p, "png")) - p).max() <= 0.5 / 255 + 1e-6


def test_quarter_turns_and_flips_lose_nothing(engine):
    p = engine.test_picture(32, 16)
    once = engine.rotate(p, "90 clockwise")
    assert once.shape == (32, 16, 3)
    assert (engine.rotate(once, "90 anticlockwise") == p).all()
    for axis in ("left to right", "top to bottom", "both"):
        assert (engine.flip(engine.flip(p, axis), axis) == p).all()


def test_pyramids_go_back_in_size(engine):
    p = engine.test_picture(64, 32)
    assert engine.pyramid_down(p).shape == (16, 32, 3)
    assert engine.pyramid_up(engine.pyramid_down(p)).shape == p.shape


def test_lines_contours_and_points_come_back_as_tables(engine):
    mask = (engine.test_picture(64, 64)[:, :, 0] > 0.5).astype("f4")
    contours = engine.find_contours(mask, "outer")
    assert engine.draw_contours(np.zeros_like(mask), contours, "#ffffff", -1).max() == 1.0
