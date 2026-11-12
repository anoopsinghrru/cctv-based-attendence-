"""Syllabus operations beyond the enhancement pipeline, organised by course unit.

Unit 1  sampling and quantization
Unit 3  image restoration (inverse and Wiener filtering), pseudo-colour, colour models
Unit 4  compression (Huffman, arithmetic, LZW, JPEG) and morphological processing
Unit 5  segmentation (point / line / edge detection, thresholding, region growing, Hough)

Everything is written out with NumPy / basic OpenCV calls so that each step of the textbook
algorithm is visible. Used by the DIP Lab page, scripts/evaluate_syllabus.py and, for the
head count and heat map, by the attendance session itself.
"""
import heapq
from collections import deque

import cv2
import numpy as np

from . import preprocess as pp


# ================================================================ Unit 1: sampling, quantization
def subsample(img, factor):
    """Keep every `factor`-th pixel (spatial sampling), shown at the original size."""
    h, w = img.shape[:2]
    return cv2.resize(img[::factor, ::factor], (w, h), interpolation=cv2.INTER_NEAREST)


def quantize(img, bits):
    """Reduce the number of grey levels per channel to 2**bits (intensity quantization)."""
    step = 256 // (2 ** int(bits))
    return ((img // step) * step + step // 2).astype(np.uint8)


# ================================================================ Unit 3: restoration
def gaussian_psf(sigma, size=None):
    size = size or int(6 * sigma + 1) | 1
    ax = np.arange(size) - size // 2
    k = np.exp(-(ax[:, None] ** 2 + ax[None, :] ** 2) / (2.0 * sigma ** 2))
    return (k / k.sum()).astype(np.float32)


def motion_psf(length, angle_deg=0.0):
    """Linear motion blur of `length` pixels at the given angle."""
    size = int(length) | 1
    k = np.zeros((size, size), np.float32)
    c = size // 2
    dx, dy = np.cos(np.deg2rad(angle_deg)), -np.sin(np.deg2rad(angle_deg))
    for t in np.linspace(-c, c, 4 * size):
        k[int(round(c + t * dy)), int(round(c + t * dx))] = 1
    return k / k.sum()


def otf(psf, shape):
    """Optical transfer function H(u, v): the PSF padded to the image size, centred at the origin."""
    h, w = shape[:2]
    pad = np.zeros((h, w), np.float32)
    kh, kw = psf.shape
    pad[:kh, :kw] = psf
    return np.fft.fft2(np.roll(pad, (-(kh // 2), -(kw // 2)), axis=(0, 1)))


def _per_channel(img, fn):
    if img.ndim == 2:
        return fn(img.astype(np.float32))
    return np.dstack([fn(img[..., c].astype(np.float32)) for c in range(img.shape[2])])


def degrade(img, psf, noise_sigma=0.0, seed=0):
    """g = h * f + n : blur with the PSF (circular convolution) and add Gaussian noise."""
    H = otf(psf, img.shape)
    rng = np.random.default_rng(seed)
    out = _per_channel(img, lambda f: np.real(np.fft.ifft2(np.fft.fft2(f) * H)))
    if noise_sigma:
        out = out + rng.normal(0, noise_sigma, out.shape)
    return np.clip(out, 0, 255).astype(np.uint8)


def inverse_filter(img, psf, cutoff=0.1):
    """F = G / H, with frequencies where |H| < cutoff set to zero (plain division explodes there)."""
    H = otf(psf, img.shape)
    mag = np.abs(H)
    inv = np.where(mag >= cutoff, 1.0 / np.where(mag >= cutoff, H, 1), 0)
    return np.clip(_per_channel(img, lambda g: np.real(np.fft.ifft2(np.fft.fft2(g) * inv))), 0, 255).astype(np.uint8)


def wiener_filter(img, psf, k=0.01):
    """Minimum mean-square-error filter: F = conj(H) / (|H|^2 + K) * G."""
    H = otf(psf, img.shape)
    w = np.conj(H) / (np.abs(H) ** 2 + k)
    return np.clip(_per_channel(img, lambda g: np.real(np.fft.ifft2(np.fft.fft2(g) * w))), 0, 255).astype(np.uint8)


# ---------------------------------------------------------------- colour and pseudo-colour
COLOURMAPS = {"jet": cv2.COLORMAP_JET, "hot": cv2.COLORMAP_HOT, "viridis": cv2.COLORMAP_VIRIDIS,
              "turbo": cv2.COLORMAP_TURBO, "bone": cv2.COLORMAP_BONE}


def pseudo_colour(gray, cmap="jet"):
    """Map grey levels to colours (intensity-to-colour transformation)."""
    return cv2.applyColorMap(pp.luminance(gray), COLOURMAPS[cmap])


def intensity_slice(gray, slices=4, cmap="jet"):
    """Intensity slicing: the grey scale is cut into bands and each band gets one colour."""
    g = pp.luminance(gray)
    band = (g.astype(np.int32) * slices // 256).clip(0, slices - 1)
    return cv2.applyColorMap((band * 255 // max(slices - 1, 1)).astype(np.uint8), COLOURMAPS[cmap])


def hsv_channels(img):
    return cv2.split(cv2.cvtColor(img, cv2.COLOR_BGR2HSV))


def ycrcb_channels(img):
    return cv2.split(cv2.cvtColor(img, cv2.COLOR_BGR2YCrCb))


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


def heat_overlay(frame, heat, alpha=0.55, cmap="jet"):
    """Blend a non-negative heat array over a frame in pseudo-colour (hot = high)."""
    heat = cv2.resize(heat.astype(np.float32), (frame.shape[1], frame.shape[0]), interpolation=cv2.INTER_CUBIC)
    norm = np.clip(heat / max(float(heat.max()), 1e-9), 0, 1)
    colour = cv2.applyColorMap((norm * 255).astype(np.uint8), COLOURMAPS[cmap])
    weight = (alpha * np.sqrt(norm))[..., None]            # transparent where nothing was seen
    return (frame * (1 - weight) + colour * weight).astype(np.uint8)


# ================================================================ Unit 4: compression
def entropy_bits(counts):
    """First-order entropy in bits per symbol: the lower bound for symbol-by-symbol coding."""
    p = np.asarray(counts, np.float64)
    p = p[p > 0] / p.sum()
    return float(-(p * np.log2(p)).sum())


def huffman_codes(counts):
    """Huffman code table {symbol: bit string} for the symbols with a non-zero count."""
    heap = [(int(c), i, (s,)) for i, (s, c) in enumerate(sorted(counts.items())) if c > 0]
    if len(heap) == 1:
        return {heap[0][2][0]: "0"}
    codes = {s[0]: "" for _, _, s in heap}
    heapq.heapify(heap)
    tie = len(heap)
    while len(heap) > 1:                          # merge the two least probable groups
        c0, _, g0 = heapq.heappop(heap)
        c1, _, g1 = heapq.heappop(heap)
        for s in g0:
            codes[s] = "0" + codes[s]
        for s in g1:
            codes[s] = "1" + codes[s]
        heapq.heappush(heap, (c0 + c1, tie, g0 + g1))
        tie += 1
    return codes


def huffman_encode(symbols, codes):
    return "".join(codes[int(s)] for s in symbols)


def huffman_decode(bits, codes):
    inverse, out, cur = {v: k for k, v in codes.items()}, [], ""
    for b in bits:
        cur += b
        if cur in inverse:
            out.append(inverse[cur])
            cur = ""
    return out


def symbol_counts(gray):
    counts = np.bincount(pp.luminance(gray).ravel(), minlength=256)
    return {i: int(c) for i, c in enumerate(counts) if c}


def huffman_stats(gray):
    """Average code length, entropy, efficiency and compression ratio of Huffman-coding an image."""
    counts = symbol_counts(gray)
    codes = huffman_codes(counts)
    total = sum(counts.values())
    avg = sum(counts[s] * len(codes[s]) for s in counts) / total
    ent = entropy_bits(list(counts.values()))
    return {"entropy": ent, "average_length": avg, "efficiency": ent / avg, "ratio": 8.0 / avg,
            "symbols": len(counts), "codes": codes, "counts": counts}


def arithmetic_encode(symbols, counts):
    """Integer arithmetic coder with a fixed frequency model. Returns the list of output bits.

    The whole message becomes one number in [0, 1): each symbol narrows the interval in
    proportion to its probability, so frequent symbols cost a fraction of a bit.
    """
    order = sorted(counts)
    index = {s: i for i, s in enumerate(order)}
    cum = np.concatenate([[0], np.cumsum([counts[s] for s in order])]).tolist()
    total = cum[-1]
    full, half, quarter = (1 << 32) - 1, 1 << 31, 1 << 30
    low, high, pending, bits = 0, full, 0, []
    for s in symbols:
        i = index[int(s)]
        span = high - low + 1
        high = low + span * cum[i + 1] // total - 1
        low = low + span * cum[i] // total
        while True:
            if high < half:
                bits.append(0); bits.extend([1] * pending); pending = 0
            elif low >= half:
                bits.append(1); bits.extend([0] * pending); pending = 0
                low -= half; high -= half
            elif low >= quarter and high < 3 * quarter:
                pending += 1
                low -= quarter; high -= quarter
            else:
                break
            low, high = low * 2, high * 2 + 1
    pending += 1
    bits.append(0 if low < quarter else 1)
    bits.extend([1 if low < quarter else 0] * pending)
    return bits


def arithmetic_decode(bits, counts, n):
    order = sorted(counts)
    cum = np.concatenate([[0], np.cumsum([counts[s] for s in order])]).tolist()
    total = cum[-1]
    full, half, quarter = (1 << 32) - 1, 1 << 31, 1 << 30
    stream = iter(bits)
    value = 0
    for _ in range(32):
        value = value * 2 + next(stream, 0)
    low, high, out = 0, full, []
    for _ in range(n):
        span = high - low + 1
        target = ((value - low + 1) * total - 1) // span
        i = int(np.searchsorted(cum, target, side="right")) - 1
        out.append(order[i])
        high = low + span * cum[i + 1] // total - 1
        low = low + span * cum[i] // total
        while True:
            if high < half:
                pass
            elif low >= half:
                low -= half; high -= half; value -= half
            elif low >= quarter and high < 3 * quarter:
                low -= quarter; high -= quarter; value -= quarter
            else:
                break
            low, high, value = low * 2, high * 2 + 1, value * 2 + next(stream, 0)
    return out


def lzw_encode(data, max_bits=12):
    """LZW: the dictionary of repeated strings is built while coding, so nothing is sent with it."""
    table = {bytes([i]): i for i in range(256)}
    limit, out, cur = 1 << max_bits, [], b""
    for byte in bytes(data):
        nxt = cur + bytes([byte])
        if nxt in table:
            cur = nxt
        else:
            out.append(table[cur])
            if len(table) < limit:
                table[nxt] = len(table)
            cur = bytes([byte])
    if cur:
        out.append(table[cur])
    return out


def lzw_decode(codes, max_bits=12):
    table = {i: bytes([i]) for i in range(256)}
    limit = 1 << max_bits
    prev = table[codes[0]]
    out = bytearray(prev)
    for code in codes[1:]:
        entry = table[code] if code in table else prev + prev[:1]
        out += entry
        if len(table) < limit:
            table[len(table)] = prev + entry[:1]
        prev = entry
    return bytes(out)


def lossless_comparison(gray, max_bits=12):
    """Bits per pixel achieved by each lossless method on one grey image."""
    g = pp.luminance(gray)
    n = g.size
    stats = huffman_stats(g)
    arith = len(arithmetic_encode(g.ravel().tolist(), stats["counts"]))
    lzw = len(lzw_encode(g.tobytes(), max_bits)) * max_bits
    png = len(cv2.imencode(".png", g)[1]) * 8
    rows = [("Uncompressed", 8.0), ("Entropy (lower bound for these methods)", stats["entropy"]),
            ("Huffman coding", stats["average_length"]), ("Arithmetic coding", arith / n),
            (f"LZW, {max_bits}-bit codes", lzw / n), ("PNG (LZ77 + Huffman, with prediction)", png / n)]
    return [{"method": m, "bits_per_pixel": round(b, 3), "ratio": round(8.0 / b, 2)} for m, b in rows], stats


# ---------------------------------------------------------------- JPEG
JPEG_LUMA = np.array([[16, 11, 10, 16, 24, 40, 51, 61], [12, 12, 14, 19, 26, 58, 60, 55],
                      [14, 13, 16, 24, 40, 57, 69, 56], [14, 17, 22, 29, 51, 87, 80, 62],
                      [18, 22, 37, 56, 68, 109, 103, 77], [24, 35, 55, 64, 81, 104, 113, 92],
                      [49, 64, 78, 87, 103, 121, 120, 101], [72, 92, 95, 98, 112, 100, 103, 99]], np.float32)


def jpeg_table(quality):
    """The standard luminance quantization table scaled for a quality of 1..100."""
    quality = int(np.clip(quality, 1, 100))
    scale = 5000 / quality if quality < 50 else 200 - 2 * quality
    return np.clip(np.floor((JPEG_LUMA * scale + 50) / 100), 1, 255).astype(np.float32)


def jpeg_steps(gray, quality=50):
    """The core of baseline JPEG on a grey image: 8x8 DCT, quantize, dequantize, inverse DCT.

    Returns (reconstruction, share of coefficients that are non-zero, quantized coefficients).
    """
    g = pp.luminance(gray).astype(np.float32) - 128.0
    h, w = g.shape
    g = np.pad(g, ((0, (-h) % 8), (0, (-w) % 8)), mode="edge")
    table = jpeg_table(quality)
    out = np.empty_like(g)
    coeffs = np.empty_like(g)
    for y in range(0, g.shape[0], 8):
        for x in range(0, g.shape[1], 8):
            q = np.round(cv2.dct(g[y:y + 8, x:x + 8]) / table)
            coeffs[y:y + 8, x:x + 8] = q
            out[y:y + 8, x:x + 8] = cv2.idct(q * table)
    recon = np.clip(out[:h, :w] + 128, 0, 255).astype(np.uint8)
    return recon, float((coeffs != 0).mean()), coeffs


def jpeg_roundtrip(img, quality=50):
    """Encode with the real JPEG codec. Returns (decoded image, size in bytes)."""
    ok, buf = cv2.imencode(".jpg", img, [cv2.IMWRITE_JPEG_QUALITY, int(quality)])
    return cv2.imdecode(buf, cv2.IMREAD_UNCHANGED), int(buf.size)


# ================================================================ Unit 4: morphology
def kernel(k=3, shape="ellipse"):
    kind = {"ellipse": cv2.MORPH_ELLIPSE, "rect": cv2.MORPH_RECT, "cross": cv2.MORPH_CROSS}[shape]
    return cv2.getStructuringElement(kind, (int(k), int(k)))


def erode(mask, k=3, shape="ellipse"):
    return cv2.erode(mask, kernel(k, shape))


def dilate(mask, k=3, shape="ellipse"):
    return cv2.dilate(mask, kernel(k, shape))


def opening(mask, k=3, shape="ellipse"):
    """Erosion then dilation: removes specks smaller than the structuring element."""
    return dilate(erode(mask, k, shape), k, shape)


def closing(mask, k=3, shape="ellipse"):
    """Dilation then erosion: closes gaps and small holes."""
    return erode(dilate(mask, k, shape), k, shape)


def boundary(mask, k=3):
    """Boundary extraction: A minus (A eroded by B)."""
    return cv2.subtract(mask, erode(mask, k))


def fill_holes(mask):
    """Hole filling: flood the background from the border; what the flood cannot reach is a hole."""
    h, w = mask.shape
    padded = cv2.copyMakeBorder(mask, 1, 1, 1, 1, cv2.BORDER_CONSTANT, value=0)
    flood = padded.copy()
    cv2.floodFill(flood, np.zeros((h + 4, w + 4), np.uint8), (0, 0), 255)
    return cv2.bitwise_or(mask, cv2.bitwise_not(flood)[1:-1, 1:-1])


def components(mask, min_area=0):
    """Connected components (8-connectivity). Returns (label image, list of dicts), largest first."""
    n, labels, stats, cent = cv2.connectedComponentsWithStats(mask, connectivity=8)
    items = [{"label": i, "x": int(stats[i, 0]), "y": int(stats[i, 1]), "w": int(stats[i, 2]), "h": int(stats[i, 3]),
              "area": int(stats[i, 4]), "cx": float(cent[i, 0]), "cy": float(cent[i, 1])}
             for i in range(1, n) if stats[i, 4] >= min_area]
    keep = np.isin(labels, [c["label"] for c in items])
    return np.where(keep, labels, 0), sorted(items, key=lambda c: -c["area"])


def colour_labels(labels):
    """A distinct colour per connected component, black background."""
    hue = ((labels * 47) % 180).astype(np.uint8)
    hsv = np.dstack([hue, np.full_like(hue, 200), np.where(labels > 0, 255, 0).astype(np.uint8)])
    return cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)


def skeleton(mask):
    """Morphological skeleton (Lantuejoul): union over k of (A eroded k times) minus its opening."""
    cross = kernel(3, "cross")
    work, skel = (mask > 0).astype(np.uint8) * 255, np.zeros_like(mask)
    while cv2.countNonZero(work):
        eroded = cv2.erode(work, cross)
        skel = cv2.bitwise_or(skel, cv2.subtract(work, cv2.dilate(eroded, cross)))
        work = eroded
    return skel


def thin(mask, max_iter=200):
    """Zhang-Suen thinning: peel boundary pixels in two sub-iterations until one pixel wide."""
    img = (mask > 0).astype(np.uint8)
    for _ in range(max_iter):
        changed = False
        for step in (0, 1):
            p = np.pad(img, 1)
            n = [p[:-2, 1:-1], p[:-2, 2:], p[1:-1, 2:], p[2:, 2:], p[2:, 1:-1], p[2:, :-2], p[1:-1, :-2], p[:-2, :-2]]
            count = sum(n)
            trans = sum((n[i] == 0) & (n[(i + 1) % 8] == 1) for i in range(8))
            if step == 0:
                cond = (n[0] * n[2] * n[4] == 0) & (n[2] * n[4] * n[6] == 0)
            else:
                cond = (n[0] * n[2] * n[6] == 0) & (n[0] * n[4] * n[6] == 0)
            remove = (img == 1) & (count >= 2) & (count <= 6) & (trans == 1) & cond
            if remove.any():
                img[remove] = 0
                changed = True
        if not changed:
            break
    return img * 255


# ---------------------------------------------------------------- skin segmentation head count
def skin_mask(img):
    """Skin-coloured pixels by thresholding the chrominance planes of YCrCb (and avoiding dark pixels)."""
    y, cr, cb = ycrcb_channels(img)
    return ((cr >= 138) & (cr <= 175) & (cb >= 85) & (cb <= 128) & (y >= 60)).astype(np.uint8) * 255


def head_count(img, open_k=None, close_k=None, min_area=None, max_area=None):
    """Classical people count: skin segmentation, opening, closing, hole filling, connected
    components, then a size and shape filter. Returns every intermediate image.

    This is a segmentation cross-check, not recognition: it counts face-sized skin regions.
    """
    h, w = img.shape[:2]
    unit = max(w, h) / 1920.0
    open_k = open_k or max(3, int(round(7 * unit)) | 1)
    close_k = close_k or max(3, int(round(15 * unit)) | 1)
    min_area = min_area or int(900 * unit * unit)
    max_area = max_area or int(40000 * unit * unit)
    raw = skin_mask(cv2.GaussianBlur(img, (0, 0), 1.2 * max(unit, 0.4)))
    opened = opening(raw, open_k)
    closed = closing(opened, close_k)
    filled = fill_holes(closed)
    labels, items = components(filled, min_area)
    heads = [c for c in items if c["area"] <= max_area and 0.5 <= c["w"] / max(c["h"], 1) <= 1.8
             and c["area"] / (c["w"] * c["h"]) >= 0.45]
    keep = np.isin(labels, [c["label"] for c in heads])
    return {"skin": raw, "opened": opened, "closed": closed, "filled": filled,
            "labels": np.where(keep, labels, 0), "regions": heads, "count": len(heads)}


def draw_regions(img, regions, colour=(255, 160, 0)):
    out = img.copy()
    t = max(2, img.shape[1] // 600)
    for c in regions:
        cv2.rectangle(out, (c["x"], c["y"]), (c["x"] + c["w"], c["y"] + c["h"]), colour, t)
    return out


# ================================================================ Unit 5: segmentation
LINE_KERNELS = {"horizontal": [[-1, -1, -1], [2, 2, 2], [-1, -1, -1]], "+45 degrees": [[-1, -1, 2], [-1, 2, -1], [2, -1, -1]],
                "vertical": [[-1, 2, -1], [-1, 2, -1], [-1, 2, -1]], "-45 degrees": [[2, -1, -1], [-1, 2, -1], [-1, -1, 2]]}


def point_detect(gray, threshold=0.6):
    """Isolated points: a strong response to the 8-neighbour Laplacian mask."""
    r = np.abs(cv2.filter2D(pp.luminance(gray).astype(np.float32), -1,
                            np.array([[-1, -1, -1], [-1, 8, -1], [-1, -1, -1]], np.float32)))
    return (r >= threshold * r.max()).astype(np.uint8) * 255


def line_detect(gray, direction="horizontal", threshold=0.35):
    """One-pixel-wide lines in one direction, with the classic 3x3 line masks."""
    r = cv2.filter2D(pp.luminance(gray).astype(np.float32), -1, np.array(LINE_KERNELS[direction], np.float32))
    r = np.clip(r, 0, None)
    return (r >= threshold * max(float(r.max()), 1e-6)).astype(np.uint8) * 255


def edges(gray, method="canny", low=60, high=150):
    """Edge maps from first- and second-derivative operators."""
    g = pp.luminance(gray)
    f = g.astype(np.float32)
    if method == "canny":
        return cv2.Canny(g, low, high)
    if method == "log":                              # Laplacian of Gaussian (Marr-Hildreth)
        r = np.abs(cv2.Laplacian(cv2.GaussianBlur(f, (0, 0), 1.4), cv2.CV_32F, ksize=3))
        return np.clip(r / max(float(r.max()), 1e-6) * 255 * 2.5, 0, 255).astype(np.uint8)
    kernels = {"sobel": ([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], [[-1, -2, -1], [0, 0, 0], [1, 2, 1]]),
               "prewitt": ([[-1, 0, 1], [-1, 0, 1], [-1, 0, 1]], [[-1, -1, -1], [0, 0, 0], [1, 1, 1]]),
               "roberts": ([[1, 0], [0, -1]], [[0, 1], [-1, 0]])}[method]
    gx = cv2.filter2D(f, -1, np.array(kernels[0], np.float32))
    gy = cv2.filter2D(f, -1, np.array(kernels[1], np.float32))
    mag = cv2.magnitude(gx, gy)
    return np.clip(mag / max(float(mag.max()), 1e-6) * 255 * 2, 0, 255).astype(np.uint8)


def threshold_global(gray, tol=0.5):
    """Basic global thresholding: T = mean of the two class means, repeated until it settles."""
    g = pp.luminance(gray)
    t = float(g.mean())
    for _ in range(100):
        lo, hi = g[g <= t], g[g > t]
        new = 0.5 * ((lo.mean() if lo.size else 0) + (hi.mean() if hi.size else 255))
        if abs(new - t) < tol:
            break
        t = new
    return t, (g > t).astype(np.uint8) * 255


def otsu_curve(gray):
    """Between-class variance for every threshold 0..255 (Otsu maximises it)."""
    p = pp.histogram(gray).astype(np.float64)
    levels = np.arange(256)
    w0 = np.cumsum(p)
    mu = np.cumsum(p * levels)
    with np.errstate(divide="ignore", invalid="ignore"):
        var = (mu[-1] * w0 - mu) ** 2 / (w0 * (1 - w0))
    return np.nan_to_num(var)


def threshold_otsu(gray):
    t = int(np.argmax(otsu_curve(gray)))
    return t, (pp.luminance(gray) > t).astype(np.uint8) * 255


def threshold_adaptive(gray, block=31, offset=5):
    """Local thresholding: each pixel is compared with the mean of its neighbourhood."""
    return cv2.adaptiveThreshold(pp.luminance(gray), 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, int(block) | 1, offset)


def region_grow(img, seed, tolerance=12):
    """Region growing from a seed (x, y): 4-connected neighbours join while they stay within
    `tolerance` grey levels of the seed's value."""
    g = pp.luminance(img).astype(np.int32)
    h, w = g.shape
    x0, y0 = int(np.clip(seed[0], 0, w - 1)), int(np.clip(seed[1], 0, h - 1))
    ref = g[y0, x0]
    mask = np.zeros((h, w), np.uint8)
    mask[y0, x0] = 255
    queue = deque([(x0, y0)])
    while queue:
        x, y = queue.popleft()
        for nx, ny in ((x + 1, y), (x - 1, y), (x, y + 1), (x, y - 1)):
            if 0 <= nx < w and 0 <= ny < h and not mask[ny, nx] and abs(g[ny, nx] - ref) <= tolerance:
                mask[ny, nx] = 255
                queue.append((nx, ny))
    return mask


def hough_accumulator(edge_map, theta_steps=180):
    """The (rho, theta) accumulator of the Hough transform, built by voting. Returns (acc, rhos, thetas)."""
    ys, xs = np.nonzero(edge_map)
    h, w = edge_map.shape
    diag = int(np.ceil(np.hypot(h, w)))
    thetas = np.deg2rad(np.arange(-90, 90, 180.0 / theta_steps))
    acc = np.zeros((2 * diag + 1, len(thetas)), np.int32)
    for j, t in enumerate(thetas):
        rho = np.round(xs * np.cos(t) + ys * np.sin(t)).astype(int) + diag
        acc[:, j] = np.bincount(rho, minlength=2 * diag + 1)
    return acc, np.arange(-diag, diag + 1), thetas


def hough_lines(edge_map, threshold=80, min_length=60, max_gap=10):
    """Line segments from the probabilistic Hough transform: list of (x1, y1, x2, y2)."""
    lines = cv2.HoughLinesP(edge_map, 1, np.pi / 180, int(threshold), minLineLength=int(min_length), maxLineGap=int(max_gap))
    return [] if lines is None else [tuple(int(v) for v in ln) for ln in np.asarray(lines).reshape(-1, 4)]


def draw_lines(img, lines, colour=(0, 200, 255)):
    out = img.copy() if img.ndim == 3 else cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
    t = max(2, out.shape[1] // 500)
    for x1, y1, x2, y2 in lines:
        cv2.line(out, (x1, y1), (x2, y2), colour, t, cv2.LINE_AA)
    return out


def contours(mask, min_points=30):
    """Boundary following: the ordered outline of every region (edge linking into closed boundaries)."""
    found, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    return [c for c in found if len(c) >= min_points]
