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


# ------------------------------------------------------------------ DIP fundamentals
def test_intensity_transforms():
    img = scene()
    assert np.array_equal(pp.negative(pp.negative(img)), img)
    assert pp.mean_brightness(pp.log_transform(img)) > pp.mean_brightness(img)      # log brightens
    flat = (img.astype(np.float32) * 0.3 + 90).astype(np.uint8)
    stretched = pp.contrast_stretch(flat)
    assert pp.luminance(stretched).std() > 2 * pp.luminance(flat).std()
    assert pp.luminance(stretched).min() < 15 and pp.luminance(stretched).max() > 240   # full range used
    assert np.allclose(pp.transfer_curve("identity"), np.arange(256))
    g = pp.transfer_curve("gamma", 0.5)
    assert g[0] == 0 and abs(g[255] - 255) < 1e-6 and g[64] > 64                     # gamma < 1 lifts shadows
    assert pp.transfer_curve("negative")[0] == 255


def test_cdf_and_global_equalisation_flattens_it():
    flat = (scene().astype(np.float32) * 0.3 + 90).astype(np.uint8)
    c = pp.cdf(flat)
    assert abs(c[-1] - 1) < 1e-6 and np.all(np.diff(c) >= -1e-9)
    ideal = np.arange(1, 257) / 256.0
    assert np.abs(pp.cdf(pp.global_hist_eq(flat)) - ideal).mean() < np.abs(c - ideal).mean() * 0.5


def test_noise_models_and_psnr():
    img = scene()
    assert pp.psnr(img, img) == float("inf")
    light, heavy = pp.add_noise(img, "gaussian", 5), pp.add_noise(img, "gaussian", 25)
    assert pp.psnr(img, light) > pp.psnr(img, heavy) > 15
    sp = pp.add_noise(img, "sp", 10)
    changed = np.any(sp != img, axis=2).mean()
    assert 0.06 < changed < 0.12                                  # about 10 % of pixels
    assert set(np.unique(sp[np.any(sp != img, axis=2)])) <= {0, 255}


def test_median_wins_on_salt_and_pepper_and_bilateral_keeps_edges():
    img = scene()
    sp = pp.add_noise(img, "sp", 8)
    score = {m: pp.psnr(img, pp.smooth(sp, m, 5)) for m in ("mean", "gaussian", "median")}
    assert score["median"] > score["mean"] + 3 and score["median"] > score["gaussian"] + 3
    noisy = pp.add_noise(img, "gaussian", 10)
    for m in ("mean", "gaussian", "median", "bilateral", "nlm"):
        assert pp.smooth(noisy, m, 5).shape == img.shape
    assert pp.blur_score(pp.smooth(img, "bilateral", 5)) > pp.blur_score(pp.smooth(img, "mean", 5))
    assert pp.smooth(img, "none") is img


def test_sharpening_and_edge_maps():
    soft = cv2.GaussianBlur(scene(), (0, 0), 2)
    assert pp.blur_score(pp.laplacian_sharpen(soft, 0.6)) > pp.blur_score(soft)
    flat = np.full((60, 60, 3), 120, np.uint8)
    assert pp.laplacian_map(flat).max() == 0 and pp.sobel_magnitude(flat).max() == 0
    edge = flat.copy()
    edge[:, 30:] = 200
    assert pp.sobel_magnitude(edge)[:, 28:32].max() > 200 and pp.sobel_magnitude(edge)[:, :20].max() == 0


def test_frequency_domain():
    img = scene()
    spec = pp.fft_spectrum(img)
    h, w = spec.shape
    assert spec.shape == img.shape[:2] and spec[h // 2, w // 2] == 255          # DC term is the peak, centred
    low = pp.frequency_mask((64, 64), "low", 10, "ideal")
    assert low[32, 32] == 1 and low[0, 0] == 0 and np.allclose(low + pp.frequency_mask((64, 64), "high", 10, "ideal"), 1)
    for profile in ("gaussian", "butterworth"):
        m = pp.frequency_mask((64, 64), "low", 10, profile)
        assert m[32, 32] > 0.99 and m[32, 42] < m[32, 36] < 1 and m[0, 0] < 0.05   # smooth roll-off
    smooth, _ = pp.frequency_filter(img, "low", 15, "gaussian")
    assert pp.blur_score(smooth) < pp.blur_score(img) * 0.5                      # low-pass removes detail
    wide, _ = pp.frequency_filter(img, "low", 500, "gaussian")
    assert np.abs(wide.astype(int) - pp.luminance(img).astype(int)).mean() < 2  # huge cut-off = identity
    # blur moves energy out of the high frequencies, which is what the blur score detects
    assert pp.high_frequency_energy(cv2.GaussianBlur(img, (0, 0), 2)) < pp.high_frequency_energy(img) * 0.5


def test_interpolation_and_colour_planes():
    small = scene()[:40, :60]
    for name in pp.INTERPOLATIONS:
        assert pp.resize_with(small, 3, name).shape == (120, 180, 3)
    assert pp.blur_score(pp.resize_with(small, 4, "nearest")) > pp.blur_score(pp.resize_with(small, 4, "bilinear"))
    l, a, b = pp.lab_channels(scene())
    assert l.shape == a.shape == b.shape == (240, 320)
    colourful = scene().copy()
    colourful[..., 2] = np.clip(colourful[..., 2].astype(int) + 60, 0, 255)

    def shift(out):
        return float(np.mean(np.hypot(*(np.asarray(x, np.float32) - y for x, y in
                                         zip(pp.lab_channels(out)[1:], pp.lab_channels(colourful)[1:])))))
    assert shift(pp.clahe_lab(colourful)) < shift(pp.equalize_rgb(colourful))   # LAB keeps the colours


def test_pipeline_stages_and_enable_switch(cfg):
    dark = pp.gamma_correct(scene(140), 2.5)
    stages, gamma = pp.run_stages(dark, cfg["preprocess"])
    assert len(stages) == 5 and gamma < 1
    out, info = pp.enhance_frame(dark, cfg["preprocess"])
    assert info["mean_after"] > info["mean_before"] and out.shape == dark.shape
    off = dict(cfg["preprocess"], enabled=False)
    same, info = pp.enhance_frame(dark, off)
    assert same is dark and info["gamma"] == 1.0
