"""DIP stage: every image-processing step applied before detection / recognition.

Pipeline (in order):
  1. brightness histogram  -> automatic gamma correction when the frame is dark
  2. bilateral / Non-Local-Means denoising
  3. CLAHE on the L channel of LAB (global histogram equalisation kept for comparison)
  4. unsharp masking
  5. Lanczos upscaling of small face crops + Laplacian-variance blur score
  6. overlapping tiles so distant faces are magnified for the detector
"""
from collections import OrderedDict

import cv2
import numpy as np


# ---------------------------------------------------------------- 1. brightness / gamma
def luminance(img):
    return cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) if img.ndim == 3 else img


def histogram(img):
    """256-bin luminance histogram, normalised to sum 1."""
    h = cv2.calcHist([luminance(img)], [0], None, [256], [0, 256]).ravel()
    return h / max(h.sum(), 1)


def mean_brightness(img):
    """Mean luminance computed from the histogram (0..255)."""
    return float(np.dot(histogram(img), np.arange(256)))


def gamma_correct(img, gamma):
    lut = (np.power(np.arange(256) / 255.0, gamma) * 255.0).clip(0, 255).astype(np.uint8)
    return cv2.LUT(img, lut)


def auto_gamma(img, dark_mean=95, target_mean=125):
    """If the frame is dark, pick gamma so that mean -> target_mean. Returns (img, gamma)."""
    mean = mean_brightness(img)
    if mean >= dark_mean or mean < 1:
        return img, 1.0
    gamma = np.log(target_mean / 255.0) / np.log(mean / 255.0)
    gamma = float(np.clip(gamma, 0.35, 1.0))
    return gamma_correct(img, gamma), gamma


# ---------------------------------------------------------------- 2. denoising
_NOISE_KERNEL = np.array([[1, -2, 1], [-2, 4, -2], [1, -2, 1]], np.float32)


def estimate_noise(img):
    """Fast noise standard deviation estimate (Immerkaer 1996): the kernel cancels
    image structure, so what is left is mostly sensor noise."""
    gray = luminance(img).astype(np.float32)
    h, w = gray.shape
    resp = np.abs(cv2.filter2D(gray, -1, _NOISE_KERNEL)[1:-1, 1:-1]).sum()
    return float(np.sqrt(np.pi / 2) * resp / (6.0 * max((w - 2) * (h - 2), 1)))


def denoise(img, method="bilateral", d=5, sigma_color=35, sigma_space=35, nlm_h=5):
    if method == "bilateral":
        return cv2.bilateralFilter(img, d, sigma_color, sigma_space)
    if method == "nlm":
        return cv2.fastNlMeansDenoisingColored(img, None, nlm_h, nlm_h, 7, 21)
    return img


# ---------------------------------------------------------------- 3. contrast
def clahe_lab(img, clip=2.0, grid=8):
    """Contrast-limited adaptive histogram equalisation on L of LAB (colour is untouched)."""
    lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    l = cv2.createCLAHE(clipLimit=clip, tileGridSize=(grid, grid)).apply(l)
    return cv2.cvtColor(cv2.merge([l, a, b]), cv2.COLOR_LAB2BGR)


def global_hist_eq(img):
    """Plain global histogram equalisation on L of LAB - only used as a comparison."""
    lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    return cv2.cvtColor(cv2.merge([cv2.equalizeHist(l), a, b]), cv2.COLOR_LAB2BGR)


# ---------------------------------------------------------------- 4. sharpening
def unsharp_mask(img, amount=0.6, sigma=1.5):
    """sharp = img + amount * (img - gaussian_blur(img))"""
    if amount <= 0:
        return img
    blur = cv2.GaussianBlur(img, (0, 0), sigma)
    return cv2.addWeighted(img, 1.0 + amount, blur, -amount, 0)


# ---------------------------------------------------------------- 5. crops
def lanczos_upscale(img, scale):
    h, w = img.shape[:2]
    return cv2.resize(img, (max(1, round(w * scale)), max(1, round(h * scale))),
                      interpolation=cv2.INTER_LANCZOS4)


def blur_score(img):
    """Variance of the Laplacian: high = sharp, low = blurry."""
    return float(cv2.Laplacian(luminance(img), cv2.CV_64F).var())


# ---------------------------------------------------------------- 6. tiles
def make_tiles(width, height, grid=(3, 2), overlap=0.25):
    """Overlapping tile rectangles (x0, y0, x1, y1) covering the whole frame."""
    cols, rows = grid
    tw = width / (cols - (cols - 1) * overlap)
    th = height / (rows - (rows - 1) * overlap)
    tiles = []
    for r in range(rows):
        for c in range(cols):
            x0 = int(round(c * tw * (1 - overlap)))
            y0 = int(round(r * th * (1 - overlap)))
            x1 = width if c == cols - 1 else int(round(x0 + tw))
            y1 = height if r == rows - 1 else int(round(y0 + th))
            tiles.append((x0, y0, min(x1, width), min(y1, height)))
    return tiles


# ---------------------------------------------------------------- full pipeline
def run_stages(img, pcfg):
    """Apply stages 1-4 and return every intermediate image (used by dip_demo.py)."""
    stages = OrderedDict()
    stages["original"] = img
    out, gamma = auto_gamma(img, pcfg["dark_mean"], pcfg["target_mean"])
    stages[f"gamma ({gamma:.2f})"] = out
    # Brightening a dark frame also amplifies its noise, so the filter strength follows the
    # measured noise level; contrast and sharpening are eased off to avoid boosting it again.
    noise = estimate_noise(out)
    calm = float(np.clip(pcfg.get("noise_ok", 4.0) / max(noise, 1e-6), 0.25, 1.0))
    strength = 1.0 / calm
    out = denoise(out, pcfg["denoise"], pcfg["bilateral_d"] + (4 if strength > 1.5 else 0),
                  min(120, pcfg["bilateral_sigma_color"] * strength),
                  pcfg["bilateral_sigma_space"], min(20, pcfg["nlm_h"] * strength))
    stages[f"denoise ({pcfg['denoise']}, noise {noise:.1f})"] = out
    if pcfg.get("clahe", True):
        out = clahe_lab(out, max(1.0, pcfg["clahe_clip"] * calm), pcfg["clahe_grid"])
    stages["CLAHE (LAB-L)"] = out
    out = unsharp_mask(out, pcfg["unsharp_amount"] * calm, pcfg["unsharp_sigma"])
    stages["unsharp mask"] = out
    return stages, gamma


def enhance_frame(img, pcfg):
    """Stages 1-4 on a frame. Returns (enhanced, info)."""
    info = {"mean_before": mean_brightness(img), "gamma": 1.0}
    if not pcfg.get("enabled", True):
        info["mean_after"] = info["mean_before"]
        return img, info
    stages, gamma = run_stages(img, pcfg)
    out = next(reversed(stages.values()))
    info.update(gamma=gamma, mean_after=mean_brightness(out))
    return out, info
