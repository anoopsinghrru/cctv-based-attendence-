"""Units 1, 3, 4 and 5 of the syllabus: each operation is checked against a property it must have."""
import cv2
import numpy as np
import pytest

from src import dip_ops as ops
from src import preprocess as pp
from tests.test_preprocess import scene


# ------------------------------------------------------------------ Unit 1
def test_sampling_and_quantization():
    img = scene()
    sub = ops.subsample(img, 4)
    assert sub.shape == img.shape and np.array_equal(sub[::4, ::4], img[::4, ::4])
    assert np.array_equal(sub[0:4, 0:4], np.broadcast_to(img[0, 0], (4, 4, 3)))      # blocks of one sample
    for bits in (1, 2, 4):
        q = ops.quantize(img, bits)
        assert len(np.unique(q)) <= 2 ** bits
    assert pp.psnr(img, ops.quantize(img, 6)) > pp.psnr(img, ops.quantize(img, 3)) > pp.psnr(img, ops.quantize(img, 1))
    assert pp.psnr(img, ops.subsample(img, 2)) > pp.psnr(img, ops.subsample(img, 8))


# ------------------------------------------------------------------ Unit 2: homomorphic
def test_homomorphic_evens_out_lighting_and_keeps_overall_brightness():
    img = scene(130)
    ramp = np.linspace(0.35, 1.0, img.shape[1], dtype=np.float32)[None, :, None]     # light falling off to one side
    shaded = (img * ramp).astype(np.uint8)
    out, H = pp.homomorphic_parts(shaded, 0.5, 1.5, 30)
    assert pp.illumination_spread(out) < pp.illumination_spread(shaded) * 0.8
    assert abs(pp.mean_brightness(out) - pp.mean_brightness(shaded)) < 12
    h, w = H.shape
    assert abs(H[h // 2, w // 2] - 0.5) < 1e-3 and abs(H[0, 0] - 1.5) < 1e-2          # low gain at DC, high far out
    assert out.shape == shaded.shape and out.dtype == np.uint8


def test_homomorphic_is_an_optional_pipeline_stage(cfg):
    img = scene()
    assert "homomorphic" not in pp.run_stages(img, cfg["preprocess"])[0]
    on = dict(cfg["preprocess"], homomorphic=True)
    stages, _ = pp.run_stages(img, on)
    assert "homomorphic" in stages and len(stages) == 6


# ------------------------------------------------------------------ Unit 3: restoration
def test_psfs_are_normalised():
    for psf in (ops.gaussian_psf(2.0), ops.motion_psf(15, 0), ops.motion_psf(15, 45)):
        assert abs(float(psf.sum()) - 1) < 1e-5 and psf.min() >= 0
    m = ops.motion_psf(15, 0)
    assert m[7].sum() > 0.99                                  # horizontal motion: all weight on the middle row


@pytest.mark.parametrize("psf", [ops.gaussian_psf(2.0), ops.motion_psf(13, 20)])
def test_inverse_filter_recovers_a_noise_free_blur(psf):
    img = scene()
    blurred = ops.degrade(img, psf)
    assert pp.psnr(img, blurred) < 30
    assert pp.psnr(img, ops.inverse_filter(blurred, psf, cutoff=0.02)) > pp.psnr(img, blurred) + 3


def test_wiener_beats_inverse_filtering_when_there_is_noise():
    img, psf = scene(), ops.gaussian_psf(2.0)
    noisy = ops.degrade(img, psf, noise_sigma=6)
    inverse = pp.psnr(img, ops.inverse_filter(noisy, psf, cutoff=0.02))      # divides noise by tiny |H|
    wiener = pp.psnr(img, ops.wiener_filter(noisy, psf, k=0.01))
    assert wiener > inverse + 3 and wiener > pp.psnr(img, noisy)
    # a larger K trusts the data less: smoother, less noise amplification
    assert pp.estimate_noise(ops.wiener_filter(noisy, psf, k=0.1)) < pp.estimate_noise(ops.wiener_filter(noisy, psf, k=0.001))


def test_pseudo_colour_and_slicing():
    g = pp.luminance(scene())
    for name in ops.COLOURMAPS:
        assert ops.pseudo_colour(g, name).shape == g.shape + (3,)
    sliced = ops.intensity_slice(g, 4)
    assert len(np.unique(sliced.reshape(-1, 3), axis=0)) <= 4
    assert len(ops.hsv_channels(scene())) == 3 and len(ops.ycrcb_channels(scene())) == 3
    frame = np.full((60, 80, 3), 100, np.uint8)
    heat = np.zeros((6, 8), np.float32)
    heat[3, 4] = 5
    over = ops.heat_overlay(frame, heat)
    assert np.array_equal(over[0, 0], frame[0, 0]) and not np.array_equal(over[35, 45], frame[35, 45])
    # a large near face and a small far face each count once, over the area they cover
    grid = ops.add_boxes(ops.new_heat((240, 320)), [(40, 40, 120, 140), (250, 30, 266, 50)])
    assert grid.shape == (31, 41) and grid.max() == 1
    assert grid[11, 10] == 1 and grid[5, 32] == 1 and grid[25, 30] == 0
    assert grid[:, :20].sum() > 10 * grid[:, 25:].sum()
    assert ops.add_boxes(grid, [(40, 40, 120, 140)]).max() == 2          # seen again: counts add up


# ------------------------------------------------------------------ Unit 4: compression
def test_huffman_is_a_prefix_code_within_one_bit_of_entropy():
    counts = {0: 45, 1: 13, 2: 12, 3: 16, 4: 9, 5: 5}
    codes = ops.huffman_codes(counts)
    assert {len(codes[0])} == {1} and len(codes[5]) == 4                       # textbook example
    words = sorted(codes.values())
    assert not any(b.startswith(a) for a, b in zip(words, words[1:]))           # no code is a prefix of another
    total = sum(counts.values())
    avg = sum(counts[s] * len(codes[s]) for s in counts) / total
    ent = ops.entropy_bits(list(counts.values()))
    assert ent <= avg < ent + 1 and abs(avg - 2.24) < 1e-9
    msg = [0, 1, 2, 3, 4, 5, 0, 0, 3]
    assert ops.huffman_decode(ops.huffman_encode(msg, codes), codes) == msg
    assert ops.huffman_codes({7: 10}) == {7: "0"}


def test_huffman_stats_on_an_image():
    s = ops.huffman_stats(scene())
    assert 0 < s["entropy"] <= s["average_length"] < s["entropy"] + 1 <= 9
    assert 0.9 < s["efficiency"] <= 1 and s["ratio"] > 1


def test_arithmetic_coding_is_lossless_and_close_to_entropy():
    rng = np.random.default_rng(0)
    msg = rng.choice([0, 1, 2, 3], size=4000, p=[0.7, 0.15, 0.1, 0.05]).tolist()
    counts = {s: msg.count(s) for s in set(msg)}
    bits = ops.arithmetic_encode(msg, counts)
    assert ops.arithmetic_decode(bits, counts, len(msg)) == msg
    ent = ops.entropy_bits(list(counts.values()))
    per_symbol = len(bits) / len(msg)
    assert ent <= per_symbol < ent + 0.02                                        # fractions of a bit per symbol
    huff = ops.huffman_codes(counts)
    assert per_symbol < sum(counts[s] * len(huff[s]) for s in counts) / len(msg)   # beats Huffman here


def test_lzw_is_lossless_and_compresses_repetition():
    data = b"TOBEORNOTTOBEORTOBEORNOT" * 20
    codes = ops.lzw_encode(data)
    assert ops.lzw_decode(codes) == data and len(codes) * 12 < len(data) * 8 / 2
    noise = np.random.default_rng(1).integers(0, 256, 3000, dtype=np.uint8).tobytes()
    assert ops.lzw_decode(ops.lzw_encode(noise)) == noise                        # still lossless on noise
    assert ops.lzw_encode(b"A") == [65]


def test_lossless_comparison_orders_methods_sensibly():
    small = cv2.resize(scene(), (160, 120))
    rows, _ = ops.lossless_comparison(small)
    bpp = {r["method"].split(",")[0].split(" (")[0]: r["bits_per_pixel"] for r in rows}
    assert bpp["Entropy"] <= bpp["Arithmetic coding"] <= bpp["Huffman coding"] < 8


def test_jpeg_quantization_tables_and_steps():
    assert np.array_equal(ops.jpeg_table(50), ops.JPEG_LUMA)
    assert ops.jpeg_table(10).mean() > ops.jpeg_table(50).mean() > ops.jpeg_table(90).mean()
    assert ops.jpeg_table(100).max() == 1
    g = pp.luminance(scene())[:117, :203]                                        # not a multiple of 8
    low, nz_low, _ = ops.jpeg_steps(g, 10)
    high, nz_high, _ = ops.jpeg_steps(g, 90)
    assert low.shape == g.shape and nz_low < nz_high < 1
    assert pp.psnr(g, high) > pp.psnr(g, low) + 5
    d10, s10 = ops.jpeg_roundtrip(scene(), 10)
    d90, s90 = ops.jpeg_roundtrip(scene(), 90)
    assert s10 < s90 and pp.psnr(scene(), d90) > pp.psnr(scene(), d10)


# ------------------------------------------------------------------ Unit 4: morphology
def blob_mask():
    m = np.zeros((120, 160), np.uint8)
    cv2.rectangle(m, (20, 20), (70, 80), 255, -1)       # a block ...
    cv2.rectangle(m, (38, 40), (50, 55), 0, -1)         # ... with a hole
    cv2.circle(m, (120, 60), 18, 255, -1)               # a disc
    m[100, 10] = m[5, 150] = 255                        # two specks
    return m


def test_erosion_dilation_opening_closing():
    m = blob_mask()
    assert cv2.countNonZero(ops.erode(m, 5)) < cv2.countNonZero(m) < cv2.countNonZero(ops.dilate(m, 5))
    opened = ops.opening(m, 5)
    assert opened[100, 10] == 0 and opened[5, 150] == 0 and opened[60, 120] == 255      # specks gone, disc kept
    assert np.all(opened <= m)                                                           # opening never adds pixels
    closed = ops.closing(m, 21)
    assert closed[47, 44] == 255 and np.all(closed >= m)                                 # hole closed, never removes
    assert np.array_equal(ops.opening(opened, 5), opened)                                # idempotent


def test_boundary_and_hole_filling():
    m = ops.opening(blob_mask(), 5)
    b = ops.boundary(m)
    assert 0 < cv2.countNonZero(b) < cv2.countNonZero(m) and b[60, 120] == 0 and b[20, 45] == 255
    filled = ops.fill_holes(m)
    assert m[47, 44] == 0 and filled[47, 44] == 255
    assert filled[0, 0] == 0 and cv2.countNonZero(filled) - cv2.countNonZero(m) == 13 * 16


def test_connected_components_and_labels():
    labels, items = ops.components(blob_mask())
    assert len(items) == 4 and items[0]["area"] > items[1]["area"] > items[2]["area"] == 1
    labels, items = ops.components(blob_mask(), min_area=50)
    assert len(items) == 2 and labels[100, 10] == 0
    disc = items[1]
    assert abs(disc["cx"] - 120) < 1 and abs(disc["cy"] - 60) < 1
    colours = ops.colour_labels(labels)
    assert colours.shape == (120, 160, 3) and not colours[0, 0].any() and colours[60, 120].any()


def test_skeleton_and_thinning_reduce_a_bar_to_a_line():
    bar = np.zeros((60, 140), np.uint8)
    cv2.rectangle(bar, (10, 22), (130, 36), 255, -1)
    for fn in (ops.skeleton, ops.thin):
        out = fn(bar)
        assert 0 < cv2.countNonZero(out) < cv2.countNonZero(bar) * 0.2
        assert np.all(out <= bar)                                                 # lies inside the shape
        ys = np.nonzero(out[:, 40:100])[0]
        assert ys.size and ys.min() >= 27 and ys.max() <= 31                      # runs along the middle
    assert cv2.countNonZero(ops.thin(bar)[:, 70]) == 1                            # exactly one pixel wide


def test_head_count_on_synthetic_faces():
    img = np.full((540, 960, 3), (190, 190, 190), np.uint8)                       # grey wall
    skin = (120, 150, 215)                                                        # BGR skin tone
    centres = [(150, 150), (400, 140), (650, 160), (300, 380), (780, 390)]
    for x, y in centres:
        cv2.ellipse(img, (x, y), (30, 38), 0, 0, 360, skin, -1)
        cv2.circle(img, (x - 10, y - 8), 4, (40, 40, 40), -1)                     # eyes: holes in the skin mask
        cv2.circle(img, (x + 10, y - 8), 4, (40, 40, 40), -1)
    cv2.circle(img, (880, 80), 3, skin, -1)                                       # a skin-coloured speck
    out = ops.head_count(img)
    assert out["count"] == 5
    assert out["skin"][150 - 8, 150 - 10] == 0 and out["filled"][150 - 8, 150 - 10] == 255
    got = sorted((round(c["cx"]), round(c["cy"])) for c in out["regions"])
    assert all(abs(gx - x) <= 3 and abs(gy - y) <= 3 for (gx, gy), (x, y) in zip(got, sorted(centres)))
    assert ops.draw_regions(img, out["regions"]).shape == img.shape


# ------------------------------------------------------------------ Unit 5: segmentation
def test_point_and_line_detection():
    g = np.full((60, 60), 100, np.uint8)
    g[30, 30] = 255
    pts = ops.point_detect(g)
    assert pts[30, 30] == 255 and cv2.countNonZero(pts) == 1
    g = np.full((60, 60), 60, np.uint8)
    g[20, 5:55] = 220                       # horizontal line
    g[5:55, 40] = 220                       # vertical line
    hz, vt = ops.line_detect(g, "horizontal"), ops.line_detect(g, "vertical")
    assert hz[20, 10] == 255 and hz[10, 40] == 0 and vt[10, 40] == 255 and vt[20, 10] == 0


def test_edge_operators_find_a_step_edge():
    g = np.full((80, 80), 50, np.uint8)
    g[:, 40:] = 200
    for method in ("sobel", "prewitt", "roberts", "log", "canny"):
        e = ops.edges(g, method)
        assert e.shape == g.shape and e[:, 37:43].max() > 100, method
        assert e[:, :30].max() == 0 and e[:, 50:].max() == 0, method
    assert cv2.countNonZero(ops.edges(g, "canny")[10:70]) == 60                  # Canny: one pixel thin


def test_thresholding_separates_two_populations():
    rng = np.random.default_rng(3)
    g = np.concatenate([rng.normal(60, 10, 5000), rng.normal(180, 12, 5000)]).clip(0, 255).astype(np.uint8).reshape(100, 100)
    t_otsu, mask = ops.threshold_otsu(g)
    t_iter, _ = ops.threshold_global(g)
    assert 85 < t_otsu < 155 and 85 < t_iter < 155          # anywhere in the gap between the two modes
    assert abs(t_otsu - cv2.threshold(g, 0, 255, cv2.THRESH_OTSU)[0]) <= 2      # matches OpenCV's Otsu
    assert abs(int((mask > 0).sum()) - 5000) < 30
    curve = ops.otsu_curve(g)
    assert curve.shape == (256,) and int(np.argmax(curve)) == t_otsu
    shaded = np.tile(np.linspace(40, 220, 200), (60, 1)).astype(np.uint8)
    shaded[20:40, ::20] = np.clip(shaded[20:40, ::20].astype(int) - 35, 0, 255)  # dark marks on a gradient
    assert (ops.threshold_adaptive(shaded)[30, ::20] == 0).mean() > 0.8         # found on both the dark and bright side
    assert (ops.threshold_otsu(shaded)[1][30, ::20] == 0).mean() <= 0.7         # a global threshold misses the bright side


def test_region_growing_fills_one_region_only():
    g = np.full((60, 80), 40, np.uint8)
    cv2.rectangle(g, (10, 10), (35, 45), 200, -1)
    cv2.rectangle(g, (50, 10), (70, 45), 205, -1)         # a similar region that is not connected
    mask = ops.region_grow(g, (20, 20), tolerance=10)
    assert cv2.countNonZero(mask) == 26 * 36 and mask[20, 60] == 0 and mask[5, 5] == 0
    assert cv2.countNonZero(ops.region_grow(g, (20, 20), tolerance=255)) == g.size


def test_hough_transform_finds_lines():
    edge = np.zeros((200, 200), np.uint8)
    cv2.line(edge, (10, 50), (190, 50), 255, 1)           # horizontal: theta = 90 deg, rho = 50
    cv2.line(edge, (120, 10), (120, 190), 255, 1)         # vertical: theta = 0, rho = 120
    acc, rhos, thetas = ops.hough_accumulator(edge)
    peaks = set()
    work = acc.copy()
    for _ in range(2):
        r, t = np.unravel_index(np.argmax(work), work.shape)
        peaks.add((int(rhos[r]), int(round(np.rad2deg(thetas[t])))))
        work[max(0, r - 5):r + 6, max(0, t - 5):t + 6] = 0
    assert peaks == {(-50, -90), (120, 0)} or peaks == {(50, 90), (120, 0)}
    lines = ops.hough_lines(edge, threshold=60, min_length=100)
    assert len(lines) >= 2 and ops.draw_lines(edge, lines).shape == (200, 200, 3)
    assert ops.hough_lines(np.zeros((50, 50), np.uint8)) == []


def test_contours_follow_region_boundaries():
    m = np.zeros((100, 100), np.uint8)
    cv2.circle(m, (50, 50), 20, 255, -1)
    found = ops.contours(m)
    assert len(found) == 1 and 100 < len(found[0]) < 180                         # about the circumference
    assert ops.contours(np.zeros((40, 40), np.uint8)) == []
