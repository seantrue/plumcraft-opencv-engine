"""Each node called as a plain method: no host, no PlumCraft, quick to run and to debug."""

import cv2
import numpy as np
import pytest
from plumcraft_engine import ConfigError, OpError, image


def square(side: int = 64) -> np.ndarray:
    """A white square on black, one channel."""
    a = np.zeros((side, side), dtype="f4")
    a[side // 4 : 3 * side // 4, side // 4 : 3 * side // 4] = 1.0
    return a


RED = np.zeros((8, 8, 3), dtype="f4")
RED[:, :, 0] = 1.0


def test_red_stays_red_through_every_way_in_and_out(engine, tmp_path):
    png = engine.encode_image(RED)
    pixels = cv2.imdecode(np.frombuffer(png.data, dtype="u1"), cv2.IMREAD_COLOR)
    assert pixels[0, 0].tolist() == [0, 0, 255], "the file is RGB in OpenCV's BGR order"
    assert engine.decode_image(png)[0, 0].tolist() == [1.0, 0.0, 0.0]
    path = tmp_path / "red.png"
    path.write_bytes(png.data)
    loaded = engine.load_image(str(path))
    assert loaded[0, 0].tolist() == [1.0, 0.0, 0.0] and loaded.space == "RGB" and loaded.names == ("red", "green", "blue")


@pytest.mark.parametrize("format, media", [("png", "image/png"), ("jpeg", "image/jpeg"), ("webp", "image/webp")])
def test_encoded_pictures_carry_a_thumbnail(engine, format, media):
    encoded = engine.encode_image(engine.test_picture(64, 48), format)
    assert encoded.meta["media"] == media and encoded.thumb.shape == (48, 64, 4)


def test_a_picture_file_gets_a_thumbnail(engine):
    t = engine.picture_thumb(engine.encode_image(RED))
    assert t.shape == (8, 8, 4) and t[0, 0].tolist() == [255, 0, 0, 255], "red, in RGB order"
    assert engine.picture_thumb(image(b"\x89PNG\r\n\x1a\n" + b"junk" * 10)) is None


def test_load_image_says_what_went_wrong(engine, tmp_path):
    with pytest.raises(ConfigError):
        engine.load_image("")
    with pytest.raises(OpError):
        engine.load_image(str(tmp_path / "gone.png"))
    with pytest.raises(OpError):
        engine.decode_image(image(b"\x89PNG\r\n\x1a\n" + b"junk" * 10))


def test_test_picture(engine):
    p = engine.test_picture(64, 48)
    assert p.shape == (48, 64, 3) and p.dtype == np.float32 and p.space == "RGB"
    assert 0.0 <= p.min() and p.max() <= 1.0


def test_to_grey_is_luminance(engine):
    green = np.zeros((4, 4, 3), dtype="f4")
    green[:, :, 1] = 1.0
    assert engine.to_grey(green)[0, 0] == pytest.approx(0.587, abs=0.001)


def test_to_space_says_what_it_is_in_opencvs_units(engine):
    hsv = engine.to_space(RED, "HSV")
    assert hsv.space == "HSV" and hsv.names == ("hue", "saturation", "value")
    assert hsv[0, 0].tolist() == pytest.approx([0.0, 1.0, 1.0])
    assert engine.to_space(RED, "Lab")[0, 0, 0] == pytest.approx(53.2, abs=0.1), "L is 0-100"


def test_equalize_spreads_tones_and_keeps_shape(engine):
    ramp = np.tile(np.linspace(0.4, 0.6, 64, dtype="f4"), (64, 1))
    out = engine.equalize(ramp)
    assert out.shape == ramp.shape and out.max() - out.min() > 0.9
    assert engine.equalize(engine.test_picture(32, 32)).shape == (32, 32, 3)


def test_blurs_smooth_and_even_kernels_are_made_odd(engine):
    a = square()
    assert 0.0 < engine.gaussian_blur(a, 3.0)[32 - 16, 32] < 1.0
    speck = np.zeros((16, 16), dtype="f4")
    speck[8, 8] = 1.0
    assert engine.median_blur(speck, 4).max() == 0.0, "a median removes a speck, and 4 became 5"
    assert engine.bilateral(engine.test_picture(32, 32), 5, 0.1, 5.0).shape == (32, 32, 3)


def test_derivatives_are_signed(engine):
    a = square()
    assert engine.sobel(a, "x", 3).min() < 0.0 < engine.sobel(a, "x", 3).max()
    assert engine.sobel(a, "y", 3)[32, 32] == 0.0, "nothing changes inside the square"
    assert engine.laplacian(a, 3).min() < 0.0 < engine.laplacian(a, 3).max()


def test_canny_finds_the_edges_of_a_square(engine):
    edges = engine.canny(square(), 0.2, 0.5)
    assert set(np.unique(edges)) == {0.0, 1.0}
    assert edges[32, 32] == 0.0, "not inside"
    assert edges[:, 16:18].any(), "on the left edge"
    assert (engine.canny(square(), 0.5, 0.2) == edges).all(), "a low above the high swaps"


def test_thresholds(engine):
    ramp = np.tile(np.linspace(0, 1, 16, dtype="f4"), (16, 1))
    assert engine.threshold(ramp, 0.5, "fixed", False)[0].tolist() == [0.0] * 8 + [1.0] * 8
    assert engine.threshold(ramp, 0.5, "fixed", True)[0, 0] == 1.0
    dark_and_light = np.where(square() > 0, 0.8, 0.2).astype("f4")
    assert (engine.threshold(dark_and_light, 0.01, "otsu", False) == square()).all(), "Otsu ignores level"
    with pytest.raises(ConfigError):
        engine.threshold(RED, 0.5, "fixed", False)
    assert engine.adaptive_threshold(ramp, 15, 0.02, False).shape == ramp.shape


def test_morphology(engine):
    a = square()
    a[2, 2] = 1.0
    opened = engine.morphology(a, "open", "rectangle", 3, 1)
    assert opened[2, 2] == 0.0 and opened[32, 32] == 1.0, "open removes the speck, keeps the square"
    assert engine.morphology(a, "dilate", "rectangle", 3, 1).sum() > a.sum() > engine.morphology(a, "erode", "rectangle", 3, 1).sum()


def test_resize_keeps_the_shape_unless_told(engine):
    p = engine.test_picture(64, 32)
    assert engine.resize(p, 16, 0, "area").shape == (8, 16, 3)
    assert engine.resize(p, 16, 16, "cubic").shape == (16, 16, 3)


def test_a_bad_picture_is_a_config_error_and_nan_is_not_a_crash(engine):
    for bad in (np.zeros(4), np.zeros((2, 2, 2, 2)), np.zeros((2, 2, 7)), np.zeros((0, 3))):
        with pytest.raises(ConfigError):
            engine.gaussian_blur(bad, 1.0)
    nan = np.full((8, 8), np.nan, dtype="f4")
    assert engine.gaussian_blur(nan, 1.0).max() == 0.0
    with pytest.raises(ConfigError):
        engine.merge(np.zeros((2, 2)), np.zeros((2, 2)), np.zeros((3, 3)))


# ---------- colour, filters, and tables of features ----------


def test_normalize_convert_scale_clahe(engine):
    assert (engine.normalize(square() * 0.5 + 0.25)[[0, 32], [0, 32]] == [0.0, 1.0]).all()
    assert engine.normalize(np.full((4, 4), 0.3, "f4")).max() == 0.0, "flat"
    assert engine.convert_scale(np.full((2, 2), 0.5, "f4"), 3.0, 0.0).max() == 1.0, "clipped"
    assert engine.clahe(engine.test_picture(64, 64), 2.0, 4).shape == (64, 64, 3)


def test_in_range_picks_a_hue(engine):
    hsv = engine.to_space(RED, "HSV")
    assert engine.in_range(hsv, 0, 10, 0.5, 1, 0.5, 1).min() == 1.0
    assert engine.in_range(hsv, 100, 140, 0.5, 1, 0.5, 1).max() == 0.0
    assert engine.in_range(square(), 0.5, 2).sum() == square().sum()


def test_filter_2d_takes_a_kernel_and_refuses_junk(engine):
    a = square()
    assert (engine.filter_2d(a, "0 0 0; 0 1 0; 0 0 0", False) == a).all(), "identity"
    assert engine.box_filter(a, 3)[16, 16] < 1.0
    assert engine.scharr(a, "x")[32, 32] == 0.0
    for bad in ("", "1 2; 3", "a b", "1 " * 16):
        with pytest.raises(ConfigError):
            engine.filter_2d(a, bad, False)


def table_column(table, name):
    return next(c["values"] for c in table["columns"] if c["name"] == name)


def test_hough_finds_a_square_and_draws_it_back(engine):
    edges = engine.canny(square(128), 0.2, 0.5)
    lines = engine.hough_lines(edges, 20, 30, 5)
    assert len(table_column(lines, "x1")) >= 4 and max(table_column(lines, "length")) > 50
    drawn = engine.draw_segments(np.zeros((128, 128), "f4"), lines, "#ff0000", 1)
    assert drawn.shape == (128, 128, 3) and drawn[:, :, 0].max() > 0.5 and drawn[:, :, 1].max() == 0.0, "red, in RGB order"
    empty = engine.hough_lines(np.zeros((16, 16), "f4"), 20, 30, 5)
    assert table_column(empty, "x1") == []


def test_contours_measure_and_draw_back(engine):
    table = engine.find_contours(square(64), "outer")
    assert table_column(table, "width") == [32] and table_column(table, "area")[0] == pytest.approx(31 * 31)
    filled = engine.draw_contours(np.zeros((64, 64), "f4"), table, "#00ff00", -1)
    assert filled[32, 32].tolist() == [0.0, 1.0, 0.0], "filled, in green"
    assert engine.draw_contours(np.zeros((8, 8), "f4"), {"columns": [{"name": "points", "type": "STRING", "values": []}]}, "#00ff00", 1).max() == 0.0
    with pytest.raises(ConfigError):
        engine.draw_contours(np.zeros((8, 8), "f4"), {"columns": [{"name": "points", "values": ["a,b"]}]}, "#00ff00", 1)
    with pytest.raises(ConfigError):
        engine.draw_contours(np.zeros((8, 8), "f4"), {"columns": []}, "#00ff00", 1)


def test_points_find_corners_and_draw_back(engine):
    corners = engine.good_features(square(64), 10, 0.05, 8.0)
    assert len(table_column(corners, "x")) == 4
    p = engine.test_picture(128, 96)
    assert "response" in [c["name"] for c in engine.orb_keypoints(p, 50)["columns"]]
    assert engine.draw_points(np.zeros((64, 64), "f4"), corners, "#ffff00", 3).max() > 0.5
    with pytest.raises(ConfigError):
        engine.draw_points(np.zeros((8, 8), "f4"), corners, "plum", 3)


def test_shapes_are_numbered_measured_and_grown(engine):
    a = square(64)
    a[:4, :4] = 1.0
    labels, count = engine.connected_components(a)
    assert count == 2 and set(np.unique(labels)) == {0, 1, 2} and labels.dtype.kind == "i"
    distance = engine.distance_transform(square(64))
    assert distance[32, 32] == pytest.approx(16, abs=1) and distance[0, 0] == 0.0
    grown = engine.watershed(engine.test_picture(64, 64), labels)
    assert grown.dtype.kind == "i" and grown.shape == (64, 64)
    with pytest.raises(ConfigError):
        engine.watershed(engine.test_picture(64, 64), np.zeros((4, 4), "i4"))
    assert engine.structuring_element("cross", 3).tolist() == [[0, 1, 0], [1, 1, 1], [0, 1, 0]]


def test_crop_and_warp(engine):
    p = engine.test_picture(64, 48)
    assert engine.crop(p, 60, 40, 100, 100).shape == (8, 4, 3), "cut at the edge"
    with pytest.raises(ConfigError):
        engine.crop(p, 500, 0, 10, 10)
    assert (engine.warp_affine(p, 0.0, 1.0, 0.0, 0.0) == p).all(), "nothing asked, nothing changed"
    assert engine.warp_affine(p, 30.0, 1.0, 5.0, 0.0).shape == p.shape


@pytest.mark.parametrize("name", ["stylization", "detail_enhance", "edge_preserving", "denoise", "cartoon"])
def test_photo_effects_keep_the_size_and_change_the_picture(engine, name):
    p = engine.test_picture(64, 48)
    out = getattr(engine, name)(p)
    assert out.shape == p.shape and out.space == "RGB"


def test_pencil_sketch_gives_two_pictures(engine):
    grey, colour = engine.pencil_sketch(engine.test_picture(64, 48))
    assert grey.shape == (48, 64) and colour.shape == (48, 64, 3)


def test_inpaint_fills_a_hole(engine):
    p = np.full((32, 32, 3), 0.5, "f4")
    hole = np.zeros((32, 32), "f4")
    hole[12:20, 12:20] = 1.0
    p[12:20, 12:20] = 0.0
    assert engine.inpaint(p, hole, 3.0, "telea")[16, 16, 0] == pytest.approx(0.5, abs=0.05)
    with pytest.raises(ConfigError):
        engine.inpaint(p, np.zeros((4, 4), "f4"), 3.0, "telea")


def test_kmeans_is_repeatable_and_counts_every_pixel(engine):
    p = engine.test_picture(32, 32)
    few, palette = engine.kmeans_colours(p, 4, 1, 7)
    again, _ = engine.kmeans_colours(p, 4, 1, 7)
    assert (few == again).all(), "the same seed, the same result"
    assert len(np.unique(few.reshape(-1, 3), axis=0)) <= 4 and sum(table_column(palette, "pixels")) == 32 * 32


def test_grab_cut_finds_a_subject(engine):
    p = np.zeros((64, 64, 3), "f4")
    p[20:44, 20:44] = (0.9, 0.2, 0.1)
    mask = engine.grab_cut(p, 0.2, 0.2, 0.8, 0.8, 3, 1)
    assert mask[32, 32] == 1.0 and mask[2, 2] == 0.0
    with pytest.raises(ConfigError):
        engine.grab_cut(p, 0.5, 0.5, 0.5, 0.5, 3, 1)


def test_combining_pictures(engine):
    a, b = np.full((4, 4), 0.25, "f4"), np.full((4, 4), 0.75, "f4")
    assert engine.add_weighted(a, b, 0.5, 0.5, 0.0)[0, 0] == pytest.approx(0.5)
    assert engine.abs_diff(a, b)[0, 0] == pytest.approx(0.5)
    with pytest.raises(ConfigError):
        engine.abs_diff(a, np.zeros((2, 2), "f4"))
    assert engine.bitwise(np.ones((2, 2), "f4"), None, "not", None).max() == 0.0
    with pytest.raises(ConfigError):
        engine.bitwise(a, None, "and", None)
    p = engine.test_picture(16, 16)
    mask = np.zeros((16, 16), "f4")
    mask[:8] = 1.0
    kept = engine.bitwise(p, p, "and", mask)
    assert kept[12].max() == 0.0 and kept[2].max() > 0.0


def test_video_frame(engine, tmp_path):
    path = tmp_path / "v.avi"
    writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"MJPG"), 10.0, (32, 24))
    assert writer.isOpened()
    for i in range(10):
        writer.write(np.full((24, 32, 3), (i * 25, 0, 0), "u1"))
    writer.release()
    assert engine.video_frame(str(path), 0.0).shape == (24, 32, 3)
    assert engine.video_frame(str(path), 0.5)[12, 16, 2] > engine.video_frame(str(path), 0.0)[12, 16, 2], "later, and BGR's blue is RGB's third"
    with pytest.raises(OpError):
        engine.video_frame(str(path), 500.0)
    with pytest.raises(OpError):
        engine.video_frame(str(tmp_path / "gone.avi"), 0.0)
    with pytest.raises(OpError):
        engine.video_frame("x\x00y", 0.0)
    with pytest.raises(ConfigError):
        engine.video_frame("", 0.0)
