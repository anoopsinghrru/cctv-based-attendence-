"""DIP Lab topics for the rest of the syllabus: sampling and quantization, homomorphic
filtering, restoration, pseudo-colour, compression, morphology, segmentation and recognition."""
import cv2
import numpy as np
import pandas as pd
import streamlit as st

from src import dip_ops as ops
from src import preprocess as pp
from src.config import ROOT
from src.face_engine import draw_faces
from ui import common as c
from ui import theme
from ui.lab_helpers import fit, gallery, note, rgb

ROLE = {
    "Sampling and quantization": ("Analysis", "A digital image is sampled in space and quantized in intensity. Both limits decide how small or how coarse a face can be and still be recognised."),
    "Homomorphic filtering": ("Optional pipeline stage", "Separates illumination from reflectance to even out one-sided lighting. Implemented as a switchable stage; measured, it did not improve detection, so it is off by default."),
    "Inverse and Wiener filtering": ("Analysis", "Restores a blurred, noisy frame when the blur is known. Shown for comparison: Wiener filtering gave a small recognition gain in the test, but a real camera's blur is not known."),
    "Pseudo-colour processing": ("Used in the session result", "Grey values are mapped to colours. Each attendance session saves a pseudo-colour heat map of where faces were seen."),
    "Lossless compression": ("Analysis", "Coding redundancy: Huffman, arithmetic and LZW coding of a frame, compared with its entropy."),
    "JPEG compression": ("Analysis", "CCTV video is always compressed. The DCT and quantization steps of JPEG, and what the quality setting does to recognition."),
    "Morphological processing": ("Analysis", "Erosion, dilation, opening, closing, hole filling, connected components, thinning and skeletons, applied to a skin-colour mask to count people."),
    "Point, line and edge detection": ("Analysis", "Derivative masks for isolated points, lines and edges."),
    "Thresholding and region growing": ("Analysis", "Global, Otsu and adaptive thresholds, and region growing from a seed."),
    "Hough transform and boundaries": ("Analysis", "Edge points vote for lines in the (rho, theta) plane; contours link boundary pixels into closed outlines."),
    "Recognition as classification": ("Pipeline: the matching step", "Each face is a pattern vector; it is assigned to the nearest enrolled student if it passes a decision threshold and a margin."),
}


def _table(name):
    """A measured result table written by scripts/evaluate_syllabus.py (None if not run yet)."""
    for folder in (ROOT / "docs", ROOT / "outputs" / "evaluation"):
        if (folder / name).exists():
            return pd.read_csv(folder / name)
    return None


def _measured(name, caption):
    df = _table(name)
    if df is not None:
        st.markdown(f"**Measured on the datasets:** {caption}")
        st.dataframe(df, width="stretch", hide_index=True)


# ------------------------------------------------------------------ Unit 1
def sampling(frame, full):
    st.latex(r"f(x, y) \;\rightarrow\; f(i\,\Delta x,\; j\,\Delta y) \qquad L = 2^{k}\ \text{grey levels} \qquad b = M \times N \times k\ \text{bits}")
    a, b = st.columns(2)
    factor = a.slider("Keep one pixel in every", 1, 16, 4)
    bits = b.slider("Bits per channel", 1, 8, 3)
    images = {"Original (8 bits)": frame, f"Sampled: 1 pixel in {factor}": ops.subsample(frame, factor),
              f"Quantized: {bits} bits, {2 ** bits} levels": ops.quantize(frame, bits),
              "Both": ops.quantize(ops.subsample(frame, factor), bits)}
    gallery(images, per_row=2)
    h, w = frame.shape[:2]
    theme.kv([("Samples kept", f"{(h // factor) * (w // factor):,} of {h * w:,} pixels"),
              ("Storage, 3 channels", f"{h * w * 3 * 8 / 8e6:.2f} MB → {(h // factor) * (w // factor) * 3 * bits / 8e6:.3f} MB"),
              ("PSNR after sampling", f"{pp.psnr(frame, images[f'Sampled: 1 pixel in {factor}']):.1f} dB"),
              ("PSNR after quantization", f"{pp.psnr(frame, images[f'Quantized: {bits} bits, {2 ** bits} levels']):.1f} dB")])
    note("Too few samples gives blocky faces (the checkerboard effect); too few grey levels gives false contours in "
         "smooth areas such as walls and skin. The face-size experiment in the report is the sampling limit of this "
         "system: recognition holds down to faces about 20 pixels wide.")
    _measured("syllabus_quantization.csv", "recognition accuracy against bits per channel (LFW pairs).")


# ------------------------------------------------------------------ Unit 2
def homomorphic(frame, full):
    st.latex(r"f = i \cdot r \;\Rightarrow\; \ln f = \ln i + \ln r \qquad H(u,v) = (\gamma_H - \gamma_L)\left[1 - e^{-c\,D^2(u,v)/D_0^2}\right] + \gamma_L")
    a, b, d, e = st.columns(4)
    shade = a.toggle("Simulate light from one side", value=True)
    gl = b.slider("γL (illumination gain)", 0.1, 1.0, 0.5, 0.05)
    gh = d.slider("γH (detail gain)", 1.0, 2.5, 1.5, 0.1)
    d0 = e.slider("Cut-off D₀", 5, 150, 30, 5)
    src = (frame * np.linspace(0.3, 1.0, frame.shape[1], dtype=np.float32)[None, :, None]).astype(np.uint8) if shade else frame
    out, H = pp.homomorphic_parts(src, gl, gh, d0)
    he = pp.global_hist_eq(src)
    gallery({"Input": src, "Homomorphic filter": out, "Global histogram equalisation": he,
             "H(u, v)": (H / H.max() * 255).astype(np.uint8), "Illumination estimate (heavy blur)": cv2.GaussianBlur(pp.luminance(src), (0, 0), 40),
             "After filtering": cv2.GaussianBlur(pp.luminance(out), (0, 0), 40)})
    cols = np.arange(src.shape[1])
    profile = pd.concat([pd.DataFrame({"column": cols, "brightness": pp.luminance(im).mean(axis=0), "image": n})
                         for n, im in (("input", src), ("homomorphic", out), ("global equalisation", he))], ignore_index=True)
    left, right = st.columns([1.3, 1])
    with left:
        st.markdown("**Mean brightness across the frame, left to right**")
        theme.lines(profile, "column", "brightness", "image", "column", "mean brightness")
    with right:
        theme.kv([("Lighting unevenness, input", f"{pp.illumination_spread(src):.1f}"),
                  ("Lighting unevenness, homomorphic", f"{pp.illumination_spread(out):.1f}"),
                  ("Lighting unevenness, equalisation", f"{pp.illumination_spread(he):.1f}"),
                  ("Mean brightness", f"{pp.mean_brightness(src):.0f} → {pp.mean_brightness(out):.0f}")])
    note("Illumination changes slowly across the frame (low frequencies); reflectance carries the detail (high "
         "frequencies). After the logarithm they add, so one filter can lower the first and raise the second. "
         "Unevenness is the spread of the mean brightness of an 8×8 grid of blocks.")
    _measured("syllabus_homomorphic.csv", "face detection against the classroom labels, with and without this stage. "
              "It evens the lighting but does not find more faces, so it stays off by default.")


# ------------------------------------------------------------------ Unit 3
def restoration(frame, full):
    st.latex(r"G = H F + N \qquad \hat F_{inverse} = \frac{G}{H} \qquad \hat F_{Wiener} = \frac{H^{*}}{|H|^{2} + K}\, G")
    a, b, d, e = st.columns(4)
    kind = a.selectbox("Degradation", ["Out of focus (Gaussian)", "Motion blur"])
    amount = b.slider("Blur strength", 1.0, 6.0, 2.0, 0.5)
    noise = d.slider("Noise σ", 0.0, 15.0, 4.0, 1.0)
    k = e.select_slider("Wiener K", [0.0005, 0.001, 0.003, 0.01, 0.03, 0.1, 0.3], value=0.01)
    src = fit(frame, 640)
    psf = ops.gaussian_psf(amount) if kind.startswith("Out") else ops.motion_psf(int(amount * 6) | 1, 15)
    blurred = ops.degrade(src, psf, noise)
    inverse, wiener = ops.inverse_filter(blurred, psf, 0.05), ops.wiener_filter(blurred, psf, k)
    score = {n: pp.psnr(src, im) for n, im in (("Degraded", blurred), ("Inverse filter", inverse), ("Wiener filter", wiener))}
    psf_view = cv2.resize((psf / psf.max() * 255).astype(np.uint8), (240, 240), interpolation=cv2.INTER_NEAREST)
    gallery({"Original": src, f"Degraded · {score['Degraded']:.1f} dB": blurred, "Point spread function h(x, y)": psf_view,
             f"Inverse filter · {score['Inverse filter']:.1f} dB": inverse, f"Wiener filter · {score['Wiener filter']:.1f} dB": wiener,
             "|H(u, v)|": (np.fft.fftshift(np.abs(ops.otf(psf, src.shape))) * 255).astype(np.uint8)})
    left, right = st.columns([1.2, 1])
    with left:
        theme.bars(pd.DataFrame({"image": list(score), "psnr": list(score.values())}), "image", "psnr", "PSNR against the original (dB)")
    with right:
        theme.kv([("Sharpness (Lap. var), degraded", f"{pp.blur_score(blurred):.0f}"),
                  ("Sharpness, Wiener result", f"{pp.blur_score(wiener):.0f}"),
                  ("Noise σ estimate, inverse result", f"{pp.estimate_noise(inverse):.1f}"),
                  ("Noise σ estimate, Wiener result", f"{pp.estimate_noise(wiener):.1f}")])
    note("Dividing by H restores the blur, but where |H| is tiny the noise is divided by a tiny number and takes "
         "over; here those frequencies are cut off. The Wiener filter adds K to the denominator, so it backs off "
         "exactly where the signal is weaker than the noise. Raise the noise to see inverse filtering fail first. "
         "Both need the blur to be known, which a real camera does not tell us.")
    _measured("syllabus_restoration.csv", "recognition on blurred, noisy LFW faces with and without restoration.")


def pseudo(frame, full):
    st.latex(r"g(x, y) = \big(\,r(f),\; g(f),\; b(f)\,\big) \qquad \text{intensity slicing: } f \in [l_{k}, l_{k+1}) \mapsto c_k")
    a, b = st.columns(2)
    cmap = a.selectbox("Colour map", list(ops.COLOURMAPS))
    slices = b.slider("Intensity slices", 2, 12, 5)
    gray = pp.luminance(frame)
    faces = c.engine().analyze(full, reject_blurry=False)
    heat = ops.add_boxes(ops.new_heat(full.shape), [f.bbox for f in faces])
    gallery({"Grey image": gray, f"Pseudo-colour ({cmap})": ops.pseudo_colour(gray, cmap),
             f"Intensity slicing, {slices} bands": ops.intensity_slice(gray, slices, cmap),
             "Edge strength in pseudo-colour": ops.pseudo_colour(pp.sobel_magnitude(frame), cmap),
             "Hue channel of HSV": ops.hsv_channels(frame)[0],
             f"Face positions in this frame ({len(faces)})": fit(ops.heat_overlay(full, cv2.GaussianBlur(heat, (0, 0), 2.0)))})
    note("The eye separates thousands of colours but only a few dozen grey levels, so mapping a measurement to colour "
         "makes small differences visible. The system uses this in every session result: face positions are "
         "accumulated over the whole class and shown as a heat map, which shows where students sat and which seats "
         "the camera sees poorly.")


# ------------------------------------------------------------------ Unit 4
@st.cache_data(show_spinner="Coding the frame ...")
def _lossless(data, shape):
    gray = np.frombuffer(data, np.uint8).reshape(shape)
    rows, stats = ops.lossless_comparison(gray)
    table = sorted(stats["counts"].items(), key=lambda kv: -kv[1])
    total = sum(stats["counts"].values())
    codes = [{"Grey level": s, "Probability": round(n / total, 4), "Huffman code": stats["codes"][s],
              "Code length": len(stats["codes"][s])} for s, n in table]
    return rows, codes, {k: stats[k] for k in ("entropy", "average_length", "efficiency", "symbols")}


def lossless(frame, full):
    st.latex(r"H = -\sum_{k} p_k \log_2 p_k \qquad L_{avg} = \sum_k p_k\, l_k \qquad C = \frac{8}{L_{avg}} \qquad R = 1 - \frac{1}{C}")
    gray = cv2.resize(pp.luminance(frame), (320, 180), interpolation=cv2.INTER_AREA)
    rows, codes, stats = _lossless(gray.tobytes(), gray.shape)
    df = pd.DataFrame(rows)
    left, right = st.columns([1, 1.25])
    with left:
        st.image(rgb(gray), caption="Frame coded here: 320 × 180 pixels, 8-bit grey", width="stretch")
        theme.kv([("Entropy", f"{stats['entropy']:.3f} bits / pixel"), ("Huffman average code length", f"{stats['average_length']:.3f} bits"),
                  ("Huffman efficiency", f"{100 * stats['efficiency']:.1f}%"), ("Grey levels in use", stats["symbols"]),
                  ("Coding redundancy at 8 bits", f"{100 * (1 - stats['entropy'] / 8):.1f}%")])
    with right:
        theme.bars(df.rename(columns={"bits_per_pixel": "bpp"}), "method", "bpp", "bits per pixel (lower is better)", fmt=".3f")
        st.dataframe(df.rename(columns={"method": "Method", "bits_per_pixel": "Bits per pixel", "ratio": "Compression ratio"}),
                     width="stretch", hide_index=True)
    st.markdown("**Huffman code table** (most and least probable grey levels)")
    a, b = st.columns(2)
    a.dataframe(pd.DataFrame(codes[:8]), width="stretch", hide_index=True)
    b.dataframe(pd.DataFrame(codes[-8:]), width="stretch", hide_index=True)
    note("Huffman gives short codes to frequent grey levels but needs a whole number of bits per symbol; arithmetic "
         "coding spends fractions of a bit and lands on the entropy. Both remove only coding redundancy, and a camera "
         "frame uses its grey levels fairly evenly, so the gain is small. LZW looks for repeated strings: on a noisy "
         "natural image it finds few and its 12-bit codes make the file larger. PNG does much better because it first "
         "predicts each pixel from its neighbours, which removes the spatial redundancy.")


def jpeg(frame, full):
    st.latex(r"F(u,v) = \tfrac14 C(u)C(v)\sum_{x=0}^{7}\sum_{y=0}^{7} f(x,y)\cos\tfrac{(2x+1)u\pi}{16}\cos\tfrac{(2y+1)v\pi}{16} \qquad F_q = \mathrm{round}\!\left(\frac{F}{Q}\right)")
    quality = st.slider("JPEG quality", 1, 100, 30)
    src = fit(frame, 640)
    gray = pp.luminance(src)
    recon, nonzero, coeffs = ops.jpeg_steps(gray, quality)
    decoded, size = ops.jpeg_roundtrip(src, quality)
    diff = cv2.convertScaleAbs(cv2.absdiff(gray, recon), alpha=8)
    y0, x0 = (gray.shape[0] // 2) // 8 * 8, (gray.shape[1] // 2) // 8 * 8
    gallery({"Original (luminance)": gray, f"Our DCT + quantization, quality {quality}": recon, "Error × 8": diff,
             f"Real JPEG codec · {size / 1024:.1f} kB": decoded, "Zoom: original": cv2.resize(gray[y0 - 24:y0 + 40, x0 - 24:x0 + 40], (256, 256), interpolation=cv2.INTER_NEAREST),
             "Zoom: after JPEG": cv2.resize(recon[y0 - 24:y0 + 40, x0 - 24:x0 + 40], (256, 256), interpolation=cv2.INTER_NEAREST)})
    left, right = st.columns(2)
    with left:
        theme.kv([("Non-zero DCT coefficients", f"{100 * nonzero:.1f}%"), ("PSNR, our reconstruction", f"{pp.psnr(gray, recon):.1f} dB"),
                  ("File size, real codec", f"{size / 1024:.1f} kB"), ("Compression ratio against raw", f"{src.size / size:.1f} : 1")])
        st.markdown("**Quantization table at this quality**")
        st.dataframe(pd.DataFrame(ops.jpeg_table(quality).astype(int)), width="stretch", hide_index=True)
    with right:
        st.markdown("**One 8×8 block: quantized DCT coefficients**")
        st.dataframe(pd.DataFrame(coeffs[y0:y0 + 8, x0:x0 + 8].astype(int)), width="stretch", hide_index=True)
        note("The top-left value is the block's average (DC). Most of the others become zero after division by the "
             "table, and runs of zeros are what the entropy coder then stores cheaply.")
    note("JPEG removes psychovisual redundancy: fine detail the eye barely notices is quantized coarsely. A face "
         "recogniser does depend on that detail, so heavy compression that still looks acceptable can already cost "
         "matches. CCTV recorders at low bitrates work in this range.")
    _measured("syllabus_jpeg.csv", "recognition accuracy and file size against JPEG quality (LFW faces 40 pixels wide).")


def morphology(frame, full):
    st.latex(r"A \ominus B,\quad A \oplus B,\quad A \circ B = (A \ominus B) \oplus B,\quad A \bullet B = (A \oplus B) \ominus B,\quad \beta(A) = A - (A \ominus B)")
    unit = max(full.shape[:2]) / 1920.0
    a, b, d = st.columns(3)
    open_k = a.slider("Opening size", 3, 25, max(3, int(round(7 * unit)) | 1), 2)
    close_k = b.slider("Closing size", 3, 41, max(3, int(round(15 * unit)) | 1), 2)
    shape = d.selectbox("Structuring element", ["ellipse", "rect", "cross"])
    hc = ops.head_count(full, open_k, close_k)
    faces = c.engine().analyze(full, reject_blurry=False)
    st.markdown("**Counting people with segmentation and morphology**")
    gallery({"1 · Skin mask (YCrCb threshold)": fit(hc["skin"]), "2 · Opening removes specks": fit(hc["opened"]),
             "3 · Closing joins broken regions": fit(hc["closed"]), "4 · Hole filling": fit(hc["filled"]),
             "5 · Connected components": fit(ops.colour_labels(hc["labels"])),
             f"6 · Face-sized regions kept: {hc['count']}": fit(ops.draw_regions(full, hc["regions"]))})
    st.markdown("**The basic operations on the filled mask**")
    mask = fit(hc["filled"], 640)
    mask = (mask > 127).astype(np.uint8) * 255
    gallery({f"Erosion ({shape}, 7)": ops.erode(mask, 7, shape), f"Dilation ({shape}, 7)": ops.dilate(mask, 7, shape),
             "Boundary extraction": ops.boundary(mask), "Morphological skeleton": ops.dilate(ops.skeleton(mask), 2),
             "Thinning (Zhang-Suen)": ops.dilate(ops.thin(mask), 2), "Opening − closing difference": cv2.absdiff(ops.opening(mask, 7, shape), ops.closing(mask, 7, shape))})
    theme.kv([("Skin pixels", f"{100 * (hc['skin'] > 0).mean():.1f}% of the frame"),
              ("Regions after hole filling", len(ops.components(hc["filled"])[1])),
              ("Face-sized regions kept", hc["count"]), ("Faces found by the detector in this frame", len(faces))])
    note("Skin colour is separated in the Cr and Cb planes, where it is compact and largely independent of brightness. "
         "Opening removes isolated pixels, closing joins the pieces of one face, hole filling closes eyes and mouths, "
         "and connected components turns the mask into countable regions. The count is unreliable in a classroom: "
         "wooden desks, hands and arms are skin-coloured too, and neighbouring faces merge.")
    _measured("syllabus_headcount.csv", "this count against the labelled number of students per frame, beside the face "
              "detector. The classical count is shown as a technique, not used for attendance.")


# ------------------------------------------------------------------ Unit 5
def edge_detection(frame, full):
    st.latex(r"\nabla f = \left[\frac{\partial f}{\partial x},\, \frac{\partial f}{\partial y}\right] \qquad |\nabla f| \approx |g_x| + |g_y| \qquad \nabla^2 f = \frac{\partial^2 f}{\partial x^2} + \frac{\partial^2 f}{\partial y^2}")
    a, b, d = st.columns(3)
    low = a.slider("Canny low threshold", 10, 200, 60, 10)
    high = b.slider("Canny high threshold", 50, 400, 150, 10)
    direction = d.selectbox("Line mask direction", list(ops.LINE_KERNELS))
    src = fit(frame, 640)
    maps = {"Roberts (2×2)": ops.edges(src, "roberts"), "Prewitt (3×3)": ops.edges(src, "prewitt"), "Sobel (3×3)": ops.edges(src, "sobel"),
            "Laplacian of Gaussian": ops.edges(src, "log"), f"Canny ({low}, {high})": ops.edges(src, "canny", low, high),
            f"Line mask: {direction}": ops.line_detect(src, direction)}
    gallery(maps)
    canny = maps[f"Canny ({low}, {high})"]
    theme.kv([("Edge pixels, Canny", f"{100 * (canny > 0).mean():.1f}% of the frame"),
              ("Isolated points found (Laplacian mask)", int((ops.point_detect(src) > 0).sum())),
              ("Mean gradient magnitude (Sobel)", f"{maps['Sobel (3×3)'].mean():.1f}")])
    st.markdown("**Masks used**")
    k = st.columns(4)
    for col, (name, mask) in zip(k, (("Sobel gₓ", [[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]]), ("Prewitt gₓ", [[-1, 0, 1], [-1, 0, 1], [-1, 0, 1]]),
                                    ("Laplacian (point)", [[-1, -1, -1], [-1, 8, -1], [-1, -1, -1]]), (f"Line: {direction}", ops.LINE_KERNELS[direction]))):
        col.caption(name)
        col.dataframe(pd.DataFrame(mask), width="stretch", hide_index=True)
    note("First-derivative masks (Roberts, Prewitt, Sobel) give thick edges whose strength is the gradient magnitude. "
         "The Laplacian of Gaussian is a second derivative and marks edges at its zero crossings. Canny smooths, "
         "takes the gradient, thins it to one pixel by non-maximum suppression, and keeps weak edges only when they "
         "connect to strong ones (the two thresholds).")


def thresholding(frame, full):
    st.latex(r"g(x,y) = \begin{cases}1 & f(x,y) > T\\ 0 & \text{otherwise}\end{cases} \qquad \sigma_B^2(T) = \frac{[m_G P_1(T) - m(T)]^2}{P_1(T)\,[1 - P_1(T)]}")
    a, b, d, e = st.columns(4)
    block = a.slider("Adaptive block size", 11, 101, 31, 10)
    sx = b.slider("Seed x (%)", 0, 100, 75)
    sy = d.slider("Seed y (%)", 0, 100, 30)
    tol = e.slider("Region tolerance", 2, 60, 14)
    src = fit(frame, 640)
    gray = pp.luminance(src)
    t_iter, m_iter = ops.threshold_global(gray)
    t_otsu, m_otsu = ops.threshold_otsu(gray)
    seed = (int(sx / 100 * (gray.shape[1] - 1)), int(sy / 100 * (gray.shape[0] - 1)))
    region = ops.region_grow(cv2.GaussianBlur(gray, (0, 0), 1.5), seed, tol)
    shown = src.copy()
    shown[region > 0] = (0.45 * shown[region > 0] + 0.55 * np.array([255, 160, 0])).astype(np.uint8)
    cv2.circle(shown, seed, 7, (0, 0, 255), -1)
    gallery({"Grey image": gray, f"Basic global threshold, T = {t_iter:.0f}": m_iter, f"Otsu, T = {t_otsu}": m_otsu,
             f"Adaptive (local mean, {block}×{block})": ops.threshold_adaptive(gray, block),
             f"Region grown from the seed ({100 * (region > 0).mean():.1f}% of frame)": shown, "Region mask": region})
    curve = ops.otsu_curve(gray)
    df = pd.concat([pd.DataFrame({"level": np.arange(256), "value": pp.histogram(gray) / pp.histogram(gray).max(), "curve": "histogram"}),
                    pd.DataFrame({"level": np.arange(256), "value": curve / max(curve.max(), 1e-9), "curve": "between-class variance"})])
    left, right = st.columns([1.3, 1])
    with left:
        st.markdown("**Histogram and Otsu's between-class variance** (both scaled to 1)")
        theme.lines(df, "level", "value", "curve", "threshold / grey level", "relative value")
    with right:
        theme.kv([("Basic global threshold", f"{t_iter:.0f}"), ("Otsu threshold", t_otsu),
                  ("Pixels above Otsu's threshold", f"{100 * (m_otsu > 0).mean():.1f}%"),
                  ("Separability (σ²B / σ²G)", f"{curve.max() / max(float(gray.var()), 1e-9):.2f}")])
    note("Otsu picks the threshold that maximises the variance between the two classes, which is the peak of the "
         "curve. A single global threshold fails when lighting changes across the frame; the adaptive threshold "
         "compares each pixel with its own neighbourhood instead. Region growing starts at the red seed and adds "
         "connected neighbours whose grey level stays within the tolerance: move the seed onto a wall or a desk.")


def hough(frame, full):
    st.latex(r"\rho = x\cos\theta + y\sin\theta \qquad \text{each edge point votes for every } (\rho, \theta) \text{ it could lie on}")
    a, b, d = st.columns(3)
    votes = a.slider("Votes needed", 30, 250, 90, 10)
    length = b.slider("Minimum line length", 20, 300, 100, 10)
    gap = d.slider("Largest gap to bridge", 0, 40, 8, 2)
    src = fit(frame, 640)
    edge_map = ops.edges(src, "canny")
    acc, rhos, thetas = ops.hough_accumulator(edge_map)
    lines = ops.hough_lines(edge_map, votes, length, gap)
    acc_view = cv2.resize((np.sqrt(acc / max(acc.max(), 1)) * 255).astype(np.uint8), (640, 360), interpolation=cv2.INTER_AREA)
    _, mask = ops.threshold_otsu(cv2.GaussianBlur(pp.luminance(src), (0, 0), 2))
    outlines = ops.contours(ops.opening(mask, 5), 80)
    drawn = src.copy()
    cv2.drawContours(drawn, outlines, -1, (255, 160, 0), 2)
    gallery({"Canny edge map": edge_map, "Accumulator: ρ (vertical) against θ (horizontal)": ops.pseudo_colour(acc_view, "hot"),
             f"Lines found: {len(lines)}": ops.draw_lines(src, lines), f"Region boundaries (contours): {len(outlines)}": drawn})
    angles = [np.degrees(np.arctan2(y2 - y1, x2 - x1)) for x1, y1, x2, y2 in lines]
    horizontal = sum(abs(a) < 20 for a in angles)
    r, t = np.unravel_index(int(np.argmax(acc)), acc.shape)
    theme.kv([("Edge points voting", f"{int((edge_map > 0).sum()):,}"), ("Strongest line", f"ρ = {rhos[r]} px, θ = {np.degrees(thetas[t]):.0f}°, {acc.max()} votes"),
              ("Lines found", len(lines)), ("Near-horizontal lines (desk and bench edges)", horizontal),
              ("Median tilt of those lines", f"{np.median([a for a in angles if abs(a) < 20]):+.1f}°" if horizontal else "–")])
    note("A line in the image is one point in the (ρ, θ) plane, and every edge point on it votes for that point, so "
         "lines show up as bright peaks in the accumulator even when the edge is broken. In a classroom the long "
         "peaks are desk and bench edges; their tilt tells how the camera is rotated. Contours follow each region's "
         "boundary pixel by pixel and link them into a closed outline.")


# ------------------------------------------------------------------ Unit 6
def recognition(frame, full):
    st.latex(r"d_j(\mathbf{x}) = \frac{\mathbf{x}\cdot\mathbf{m}_j}{\|\mathbf{x}\|\,\|\mathbf{m}_j\|} \qquad \text{assign } \mathbf{x} \rightarrow \omega_j \ \text{ if } d_j = \max_k d_k,\; d_j \ge T \ \text{ and } d_j - d_{2nd} \ge m")
    d, conf = c.db(), c.cfg()["recognition"]
    students = d.students()
    batches = sorted(students[students["photos"] > 0]["batch"].unique()) if len(students) else []
    if not batches:
        st.info("No students are enrolled yet, so there are no pattern classes to compare with.", icon=":material/group:")
        return
    faces = sorted(c.engine().analyze(full, reject_blurry=False), key=lambda f: -f.size)
    if not faces:
        st.info("No face was detected in this frame. Choose another frame.", icon=":material/face:")
        return
    a, b = st.columns(2)
    default = next((i for i, x in enumerate(batches) if x == "BSCS-A"), 0)
    batch = a.selectbox("Pattern classes (enrolled batch)", batches, index=default)
    i = b.selectbox("Unknown pattern (face in the frame)", range(len(faces)), format_func=lambda k: f"face {k + 1} · {faces[k].size:.0f} px wide")
    f = faces[i]
    rolls, mat = d.gallery(batch)
    uniq = sorted(set(rolls))
    per = np.full(len(uniq), -1.0)
    np.maximum.at(per, [uniq.index(r) for r in rolls], f.embedding @ mat.T)
    order = np.argsort(-per)
    names = students.set_index("roll_no")["name"]
    thr = conf["blurry_threshold"] if f.blur < conf["blurry_below"] else conf["match_threshold"]
    best, second = float(per[order[0]]), float(per[order[1]]) if len(uniq) > 1 else -1.0
    accepted = best >= thr and best - second >= conf["margin"]
    left, right = st.columns([1, 2.2])
    with left:
        st.image(rgb(cv2.resize(f.crop, (224, 224), interpolation=cv2.INTER_CUBIC)), caption="Aligned face: the pattern x (512 numbers)", width="stretch")
        if accepted:
            st.success(f"Assigned to {uniq[order[0]]} · {names.get(uniq[order[0]], '')}", icon=":material/check_circle:")
        else:
            st.warning("Rejected: " + ("below the threshold" if best < thr else "too close to the second-best class"), icon=":material/block:")
    with right:
        top = pd.DataFrame({"student": [f"{uniq[j]} · {names.get(uniq[j], '')}" for j in order[:10]], "similarity": [float(per[j]) for j in order[:10]]})
        theme.bars(top, "student", "similarity", "decision function d(x): cosine similarity", fmt=".3f")
        theme.kv([("Best class", f"{best:.3f}"), ("Second-best class", f"{second:.3f}"), ("Margin", f"{best - second:.3f} (needs {conf['margin']})"),
                  ("Threshold for this crop", f"{thr} ({'blurry' if f.blur < conf['blurry_below'] else 'sharp'}, blur score {f.blur:.0f})")])
    labels, colours = [""] * len(faces), [(160, 160, 160)] * len(faces)
    labels[i], colours[i] = "x", (80, 220, 80) if accepted else (60, 60, 235)
    st.image(rgb(fit(draw_faces(full, faces, labels, colours))), caption="The selected face in the frame", width="stretch")
    open_set = _table("open_set.csv")
    if open_set is not None:
        sub = open_set[(open_set["class_size"] == 40) & (open_set["margin"] == 0.06) & (open_set["probe"].isin(["clean, 110 px", "blurred, 40 px"]))]
        curves = pd.concat([sub.assign(value=sub["TAR_%"], curve=sub["probe"] + " · enrolled accepted"),
                            sub.assign(value=sub["FAR_%"] * 100, curve=sub["probe"] + " · strangers accepted ×100")])
        st.markdown("**Choosing the decision threshold T** (measured, class of 40)")
        theme.lines(curves, "threshold", "value", "curve", "threshold T", "% (strangers ×100)")
    note("This is a decision-theoretic classifier. Each enrolled student is a pattern class represented by prototype "
         "vectors (their photos); the decision function is the cosine similarity to the nearest prototype, which for "
         "unit-length vectors is the minimum-distance classifier. The threshold and the margin add a reject option, "
         "because most faces in a room may belong to no class at all.")


RENDER = {"Sampling and quantization": sampling, "Homomorphic filtering": homomorphic, "Inverse and Wiener filtering": restoration,
          "Pseudo-colour processing": pseudo, "Lossless compression": lossless, "JPEG compression": jpeg,
          "Morphological processing": morphology, "Point, line and edge detection": edge_detection,
          "Thresholding and region growing": thresholding, "Hough transform and boundaries": hough,
          "Recognition as classification": recognition}
