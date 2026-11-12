import cv2
import numpy as np

from src import preprocess as pp


def scene(mean=130, seed=0):
    """A textured synthetic frame (gradients + shapes) with roughly the given brightness."""
    rng = np.random.default_rng(seed)
    img = np.tile(np.linspace(mean - 60, mean + 60, 320, dtype=np.float32), (240, 1))
    img = cv2.merge([img, img * 0.9, img * 1.05])
    for _ in range(25):
        x, y = int(rng.integers(0, 300)), int(rng.integers(0, 220))
        cv2.rectangle(img, (x, y), (x + 18, y + 18), float(rng.integers(20, 235)), -1)
    return img.clip(0, 255).astype(np.uint8)


def test_histogram_is_normalised_and_gives_mean():
    img = scene()
    h = pp.histogram(img)
    assert h.shape == (256,) and abs(h.sum() - 1) < 1e-6
    assert abs(pp.mean_brightness(img) - pp.luminance(img).mean()) < 0.5


def test_gamma_only_applied_to_dark_frames():
    bright = scene(140)
    out, g = pp.auto_gamma(bright, 95, 125)
    assert g == 1.0 and out is bright

    dark = pp.gamma_correct(scene(140), 2.5)
    assert pp.mean_brightness(dark) < 95
    out, g = pp.auto_gamma(dark, 95, 125)
    assert 0.35 <= g < 1.0
    assert pp.mean_brightness(out) > pp.mean_brightness(dark) + 20


def test_gamma_is_clamped_on_black_frames():
    black = np.zeros((50, 50, 3), np.uint8)
    out, g = pp.auto_gamma(black)
    assert g == 1.0 and out.max() == 0
    almost = np.full((50, 50, 3), 2, np.uint8)
    _, g = pp.auto_gamma(almost)
    assert g == 0.35


def test_noise_estimate_tracks_added_noise():
    clean = scene()
    rng = np.random.default_rng(1)
    levels = []
    for sigma in (0, 5, 15):
        # the same noise on all three channels, so the luminance noise is exactly sigma
        noise = rng.normal(0, sigma, clean.shape[:2])[:, :, None]
        noisy = (clean.astype(np.float32) + noise).clip(0, 255).astype(np.uint8)
        levels.append(pp.estimate_noise(noisy))
    assert levels[0] < levels[1] < levels[2]
    assert 3.5 < levels[1] < 7 and 12 < levels[2] < 18    # close to the true sigma


def test_denoise_reduces_noise_and_none_is_identity():
    rng = np.random.default_rng(2)
    noisy = (scene().astype(np.float32) + rng.normal(0, 12, (240, 320, 3))).clip(0, 255).astype(np.uint8)
    for method in ("bilateral", "nlm"):
        assert pp.estimate_noise(pp.denoise(noisy, method, 5, 60, 60, 10)) < pp.estimate_noise(noisy)
    assert pp.denoise(noisy, "none") is noisy


def test_clahe_raises_contrast_but_less_harshly_than_global_he():
    flat = (scene().astype(np.float32) * 0.3 + 90).astype(np.uint8)        # low-contrast frame
    std = lambda im: float(pp.luminance(im).std())
    clahe, he = pp.clahe_lab(flat, 2.0, 8), pp.global_hist_eq(flat)
    assert std(flat) < std(clahe) < std(he)
    assert clahe.shape == flat.shape and clahe.dtype == np.uint8


def test_unsharp_sharpens_and_zero_amount_is_identity():
    soft = cv2.GaussianBlur(scene(), (0, 0), 2)
    assert pp.blur_score(pp.unsharp_mask(soft, 0.8, 1.5)) > pp.blur_score(soft)
    assert pp.unsharp_mask(soft, 0) is soft


def test_blur_score_orders_sharp_over_blurred():
    sharp = scene()
    assert pp.blur_score(sharp) > pp.blur_score(cv2.GaussianBlur(sharp, (0, 0), 1.5)) \
        > pp.blur_score(cv2.GaussianBlur(sharp, (0, 0), 4))


def test_lanczos_upscale_size():
    out = pp.lanczos_upscale(np.zeros((20, 30, 3), np.uint8), 2.5)
    assert out.shape == (50, 75, 3)


def test_tiles_cover_frame_and_overlap():
    w, h = 1920, 1080
    tiles = pp.make_tiles(w, h, (3, 2), 0.25)
    assert len(tiles) == 6
    cover = np.zeros((h, w), np.uint8)
    for x0, y0, x1, y1 in tiles:
        assert 0 <= x0 < x1 <= w and 0 <= y0 < y1 <= h
        cover[y0:y1, x0:x1] += 1
    assert cover.min() >= 1                         # every pixel is inside some tile
    assert cover.max() >= 2                         # neighbours share a band
    assert (tiles[0][2] - tiles[1][0]) > 100        # horizontal overlap wide enough for a face


# ------------------------------------------------------------------ measurement helpers
def test_added_noise_and_psnr():
    img = scene()
    assert pp.psnr(img, img) == float("inf")
    light, heavy = pp.add_noise(img, 5), pp.add_noise(img, 25)
    assert pp.psnr(img, light) > pp.psnr(img, heavy) > 15
    assert pp.estimate_noise(heavy) > pp.estimate_noise(light) > pp.estimate_noise(img)


def test_laplacian_map_and_lab_planes():
    flat = np.full((60, 60, 3), 120, np.uint8)
    assert pp.laplacian_map(flat).max() == 0
    edge = flat.copy()
    edge[:, 30:] = 200
    assert pp.laplacian_map(edge)[:, 28:32].max() > 200 and pp.laplacian_map(edge)[:, :20].max() == 0
    l, a, b = pp.lab_channels(scene())
    assert l.shape == a.shape == b.shape == (240, 320)


def test_clahe_changes_lightness_only():
    colourful = scene().copy()
    colourful[..., 2] = np.clip(colourful[..., 2].astype(int) + 60, 0, 255)
    before, after = pp.lab_channels(colourful), pp.lab_channels(pp.clahe_lab(colourful))
    shift = float(np.mean(np.hypot(after[1].astype(np.float32) - before[1], after[2].astype(np.float32) - before[2])))
    assert shift < 3                                              # a* and b* stay where they were
    assert np.abs(after[0].astype(int) - before[0]).mean() > shift


def test_pipeline_stages_and_enable_switch(cfg):
    dark = pp.gamma_correct(scene(140), 2.5)
    stages, gamma = pp.run_stages(dark, cfg["preprocess"])
    assert len(stages) == 5 and gamma < 1
    out, info = pp.enhance_frame(dark, cfg["preprocess"])
    assert info["mean_after"] > info["mean_before"] and out.shape == dark.shape
    off = dict(cfg["preprocess"], enabled=False)
    same, info = pp.enhance_frame(dark, off)
    assert same is dark and info["gamma"] == 1.0
