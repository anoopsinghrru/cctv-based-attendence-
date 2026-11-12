"""DIP Lab: every image-processing fundamental the system is built on, demonstrated live on a
frame of the footage, with the formula, adjustable parameters, measurements, and a note on
where the attendance pipeline uses it."""
import cv2
import numpy as np
import pandas as pd
import streamlit as st

from src import preprocess as pp
from src.face_engine import ARCFACE_DST, draw_faces, similarity_transform
from ui import common as c
from ui import theme

TOPICS = ["Intensity transformations", "Histogram processing", "Noise and spatial filtering",
          "Sharpening and blur measure", "Frequency domain", "Colour models",
          "Geometric transforms and interpolation", "Multi-scale tiling", "Complete pipeline"]
ROLE = {
    "Intensity transformations": ("Pipeline stage 1", "Gamma correction is applied automatically when the luminance histogram shows a dark frame."),
    "Histogram processing": ("Pipeline stages 1 and 3", "The histogram drives the brightness decision; CLAHE on the L channel is the contrast stage."),
    "Noise and spatial filtering": ("Pipeline stage 2", "A bilateral filter (Non-Local Means optional) removes noise; its strength follows the measured noise level."),
    "Sharpening and blur measure": ("Pipeline stages 4 and 5", "Unsharp masking is the sharpening stage; the variance of the Laplacian is the blur score that rejects unusable face crops."),
    "Frequency domain": ("Analysis", "Not a pipeline stage: it explains why blur lowers the blur score, and what the smoothing and sharpening stages do to the spectrum."),
    "Colour models": ("Pipeline stage 3", "Contrast is enhanced on the L channel of CIELAB so that colours are not shifted."),
    "Geometric transforms and interpolation": ("Pipeline stage 5", "Each face is warped to a 112×112 template by a similarity transform; small faces are Lanczos-upscaled first."),
    "Multi-scale tiling": ("Pipeline stage 6", "The frame is split into overlapping tiles so that distant faces are magnified for the detector."),
    "Complete pipeline": ("All stages", "The stages in the order the system applies them, with the measured effect of each."),
}


# ------------------------------------------------------------------ helpers
@st.cache_data(show_spinner=False)
def read_frame(path, index, mtime):
    cap = cv2.VideoCapture(path)
    cap.set(cv2.CAP_PROP_POS_FRAMES, index)
    ok, frame = cap.read()
    cap.release()
    return frame if ok else None


@st.cache_data(show_spinner=False)
def frame_count(path, mtime):
    cap = cv2.VideoCapture(path)
    n = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 1)
    cap.release()
    return max(n, 1)


def fit(img, width=880):
    return cv2.resize(img, (width, round(img.shape[0] * width / img.shape[1])), interpolation=cv2.INTER_AREA) \
        if img.shape[1] > width else img


def rgb(img):
    return cv2.cvtColor(img, cv2.COLOR_GRAY2RGB if img.ndim == 2 else cv2.COLOR_BGR2RGB)


def gallery(images, per_row=3):
    """Captioned images in a grid. `images` is a dict caption -> image."""
    items = list(images.items())
    for i in range(0, len(items), per_row):
        cols = st.columns(per_row)
        for col, (caption, img) in zip(cols, items[i:i + per_row]):
            col.image(rgb(img), caption=caption, width="stretch")


def hist_frame(images, fn=pp.histogram, scale=100.0):
    return pd.concat([pd.DataFrame({"level": np.arange(256), "value": fn(img) * scale, "image": name})
                      for name, img in images.items()], ignore_index=True)


def entropy(img):
    h = pp.histogram(img)
    h = h[h > 0]
    return float(-(h * np.log2(h)).sum())


def contrast(img):
    return float(pp.luminance(img).std())


def measure(images, extra=None):
    rows = []
    for name, img in images.items():
        row = {"Image": name, "Mean": round(pp.mean_brightness(img), 1), "Contrast (σ)": round(contrast(img), 1),
               "Entropy (bits)": round(entropy(img), 2), "Sharpness (Lap. var)": round(pp.blur_score(img), 1)}
        if extra:
            row.update(extra(name, img))
        rows.append(row)
    st.dataframe(pd.DataFrame(rows), width="stretch", hide_index=True)


# ------------------------------------------------------------------ topics
def intensity(frame, full):
    st.latex(r"s = 255\left(\frac{r}{255}\right)^{\gamma} \qquad \gamma_{auto} = \frac{\ln(\text{target}/255)}{\ln(\text{mean}/255)}")
    conf = c.cfg()["preprocess"]
    a, b = st.columns(2)
    dark = a.toggle("Simulate an under-exposed frame", value=True)
    src = pp.gamma_correct(frame, 2.4) if dark else frame
    auto, g_auto = pp.auto_gamma(src, conf["dark_mean"], conf["target_mean"])
    gamma = b.slider("Gamma", 0.2, 3.0, float(round(g_auto, 2)), 0.05)
    mean = pp.mean_brightness(src)
    st.markdown(f"<span class='av-chip'>mean brightness {mean:.0f}</span>"
                f"<span class='av-chip'>dark below {conf['dark_mean']}</span>"
                f"<span class='av-chip'>automatic gamma {g_auto:.2f}</span>"
                f"<span class='av-chip'>{'correction applied' if g_auto < 1 else 'no correction needed'}</span>",
                unsafe_allow_html=True)
    images = {"Input": src, f"Gamma {gamma:.2f}": pp.gamma_correct(src, gamma), "Log transform": pp.log_transform(src),
              "Contrast stretch (2–98%)": pp.contrast_stretch(src), "Negative": pp.negative(src)}
    gallery(images)
    left, right = st.columns(2)
    with left:
        st.markdown("**Transfer functions**")
        r = np.arange(256)
        curves = pd.concat([pd.DataFrame({"input": r, "output": pp.transfer_curve(k, gamma), "curve": n})
                            for k, n in (("identity", "identity"), ("gamma", f"gamma {gamma:.2f}"),
                                         ("log", "log"), ("negative", "negative"))], ignore_index=True)
        theme.lines(curves, "input", "output", "curve", "input level r", "output level s")
    with right:
        st.markdown("**Luminance histograms**")
        theme.lines(hist_frame({k: images[k] for k in list(images)[:2]}), "level", "value", "image",
                    "grey level", "% of pixels", area=True)
    measure(images)


def histogram(frame, full):
    st.latex(r"s_k = (L-1)\sum_{j=0}^{k} p_r(r_j) \qquad \text{CLAHE: equalise per tile, clip the histogram, interpolate}")
    a, b, d = st.columns(3)
    flat = a.toggle("Simulate a low-contrast frame", value=True)
    clip = b.slider("CLAHE clip limit", 1.0, 6.0, float(c.cfg()["preprocess"]["clahe_clip"]), 0.5)
    grid = d.slider("CLAHE tile grid", 2, 16, int(c.cfg()["preprocess"]["clahe_grid"]), 2)
    src = (frame.astype(np.float32) * 0.4 + 80).astype(np.uint8) if flat else frame
    images = {"Input": src, "Global histogram equalisation": pp.global_hist_eq(src),
              f"CLAHE (clip {clip}, {grid}×{grid})": pp.clahe_lab(src, clip, grid)}
    gallery(images)
    left, right = st.columns(2)
    with left:
        st.markdown("**Histograms**")
        theme.lines(hist_frame(images), "level", "value", "image", "grey level", "% of pixels", area=True)
    with right:
        st.markdown("**Cumulative distributions**")
        theme.lines(hist_frame(images, pp.cdf, 1.0), "level", "value", "image", "grey level", "cumulative share")
    measure(images)
    st.markdown("<div class='av-note'>Global equalisation makes the cumulative distribution a straight line, which "
                "over-brightens walls and windows. CLAHE equalises small tiles and clips each histogram, so contrast "
                "rises where the detail is, without that over-stretching.</div>", unsafe_allow_html=True)


def filtering(frame, full):
    st.latex(r"g(x,y) = \sum_{s,t} w(s,t)\, f(x+s,\,y+t) \qquad PSNR = 10\log_{10}\frac{255^2}{MSE}")
    a, b, d = st.columns(3)
    kind = a.selectbox("Noise model", ["gaussian", "salt and pepper"])
    amount = b.slider("Noise level (σ for Gaussian, % of pixels for salt and pepper)", 2.0, 40.0, 15.0, 1.0)
    k = d.slider("Kernel size", 3, 11, 5, 2)
    clean = fit(frame, 640)
    noisy = pp.add_noise(clean, "gaussian" if kind == "gaussian" else "sp", amount)
    images = {"Noisy input": noisy}
    for label, method in (("Mean filter", "mean"), ("Gaussian filter", "gaussian"), ("Median filter", "median"),
                          ("Bilateral filter", "bilateral"), ("Non-Local Means", "nlm")):
        images[label] = pp.smooth(noisy, method, k)
    scores = {n: pp.psnr(clean, im) for n, im in images.items()}
    gallery({f"{n} · PSNR {scores[n]:.1f} dB": im for n, im in images.items()})
    left, right = st.columns([1.1, 1])
    with left:
        theme.bars(pd.DataFrame({"filter": list(scores), "psnr": list(scores.values())}), "filter", "psnr",
                   "PSNR against the clean frame (dB)")
    with right:
        best = max((n for n in scores if n != "Noisy input"), key=scores.get)
        theme.kv([("Estimated noise σ of the input", f"{pp.estimate_noise(noisy):.1f}"),
                  ("Estimated noise σ, clean frame", f"{pp.estimate_noise(clean):.1f}"),
                  ("Best filter here", f"{best} ({scores[best]:.1f} dB)"),
                  ("Edges kept (Lap. var), mean filter", f"{pp.blur_score(images['Mean filter']):.0f}"),
                  ("Edges kept (Lap. var), bilateral", f"{pp.blur_score(images['Bilateral filter']):.0f}")])
    st.markdown("<div class='av-note'>Linear filters (mean, Gaussian) blur edges along with the noise. The median "
                "filter removes salt-and-pepper noise almost completely. The bilateral filter averages only pixels "
                "that are close in both position and intensity, so it smooths noise but keeps face edges, which is "
                "why the pipeline uses it. The noise level is estimated with Immerkær's method and sets the filter "
                "strength.</div>", unsafe_allow_html=True)


def sharpening(frame, full):
    st.latex(r"g = f + a\,(f - G_\sigma * f) \qquad \nabla^2 f = \frac{\partial^2 f}{\partial x^2} + \frac{\partial^2 f}{\partial y^2} \qquad \text{blur score} = \mathrm{Var}(\nabla^2 f)")
    conf, rc = c.cfg()["preprocess"], c.cfg()["recognition"]
    a, b, d = st.columns(3)
    soft = a.slider("Soften the input first (Gaussian σ)", 0.0, 4.0, 1.5, 0.5)
    amount = b.slider("Unsharp amount", 0.0, 2.5, float(conf["unsharp_amount"]), 0.1)
    sigma = d.slider("Unsharp σ", 0.5, 4.0, float(conf["unsharp_sigma"]), 0.5)
    src = cv2.GaussianBlur(frame, (0, 0), soft) if soft > 0 else frame
    images = {"Input": src, f"Unsharp mask (a = {amount:.1f})": pp.unsharp_mask(src, amount, sigma),
              "Laplacian sharpening": pp.laplacian_sharpen(src, 0.6),
              "|Laplacian| response": pp.laplacian_map(src), "Sobel gradient magnitude": pp.sobel_magnitude(src),
              "Unsharp detail layer (f − blur)": cv2.convertScaleAbs(cv2.absdiff(src, cv2.GaussianBlur(src, (0, 0), sigma)), alpha=4)}
    gallery(images)
    score = {n: pp.blur_score(images[n]) for n in list(images)[:3]}
    left, right = st.columns([1.1, 1])
    with left:
        theme.bars(pd.DataFrame({"image": list(score), "v": list(score.values())}), "image", "v",
                   "Variance of the Laplacian (sharpness)", fmt=".0f")
    with right:
        theme.kv([("Blur score of the input", f"{score['Input']:.0f}"),
                  ("Face crop treated as blurry below", f"{rc['blurry_below']:.0f}"),
                  ("Face crop rejected below", f"{rc['reject_below']:.0f}"),
                  ("Match threshold, sharp / blurry", f"{rc['match_threshold']} / {rc['blurry_threshold']}")])
    st.markdown("<div class='av-note'>The Laplacian responds only where intensity changes quickly. A blurred image "
                "has weak, narrow responses, so the variance of the Laplacian falls. The system computes it on every "
                "aligned face crop: very low scores are discarded, low scores are matched with the blurry-face "
                "threshold.</div>", unsafe_allow_html=True)


def frequency(frame, full):
    st.latex(r"F(u,v) = \sum_{x,y} f(x,y)\,e^{-j2\pi(ux/M + vy/N)} \qquad g = \mathcal{F}^{-1}\{H(u,v)\,F(u,v)\}")
    a, b, d = st.columns(3)
    kind = a.selectbox("Filter", ["low-pass", "high-pass"])
    profile = b.selectbox("Transfer function", ["gaussian", "butterworth", "ideal"])
    cutoff = d.slider("Cut-off radius D₀ (pixels in the spectrum)", 5, 150, 40, 5)
    src = fit(frame, 640)
    out, mask = pp.frequency_filter(src, "low" if kind == "low-pass" else "high", cutoff, profile)
    blurred = cv2.GaussianBlur(src, (0, 0), 2.5)
    gallery({"Luminance": pp.luminance(src), "Magnitude spectrum (log)": pp.fft_spectrum(src),
             f"H(u, v): {profile} {kind}": (mask * 255).astype(np.uint8), f"Filtered ({kind})": out,
             "Spectrum of the filtered image": pp.fft_spectrum(out), "Spectrum after a Gaussian blur": pp.fft_spectrum(blurred)})
    rows = pd.DataFrame([{"Image": n, "High-frequency energy %": round(100 * pp.high_frequency_energy(im), 3),
                          "Blur score (Lap. var)": round(pp.blur_score(im), 1)}
                         for n, im in (("Original", src), ("Gaussian blur σ 2.5", blurred),
                                       ("Unsharp mask", pp.unsharp_mask(src, 1.0, 1.5)))])
    st.dataframe(rows, width="stretch", hide_index=True)
    st.markdown("<div class='av-note'>The centre of the spectrum holds the slowly varying content (overall "
                "brightness, large regions); the outer part holds edges and fine detail. Blur removes the outer part, "
                "which is exactly what the Laplacian-variance blur score detects. An ideal filter cuts sharply and "
                "produces ringing; Gaussian and Butterworth transfer functions roll off smoothly.</div>",
                unsafe_allow_html=True)


def colour(frame, full):
    st.latex(r"(B, G, R) \rightarrow (L^*, a^*, b^*) \qquad L^* \text{: lightness} \quad a^*, b^* \text{: colour}")
    l, a_ch, b_ch = pp.lab_channels(frame)
    lab_out = pp.clahe_lab(frame, c.cfg()["preprocess"]["clahe_clip"], c.cfg()["preprocess"]["clahe_grid"])
    rgb_out = pp.equalize_rgb(frame)
    gallery({"Input": frame, "L* (lightness)": l, "a* (green–red)": a_ch, "b* (blue–yellow)": b_ch,
             "Contrast on L* only (pipeline)": lab_out, "Equalising B, G, R separately": rgb_out})

    def shift(img):
        _, a2, b2 = pp.lab_channels(img)
        return float(np.mean(np.hypot(a2.astype(np.float32) - a_ch, b2.astype(np.float32) - b_ch)))
    theme.kv([("Mean colour shift, contrast on L* only", f"{shift(lab_out):.2f}"),
              ("Mean colour shift, per-channel RGB equalisation", f"{shift(rgb_out):.2f}"),
              ("Contrast gain (σ), L* only", f"{contrast(frame):.1f} → {contrast(lab_out):.1f}"),
              ("Contrast gain (σ), per-channel RGB", f"{contrast(frame):.1f} → {contrast(rgb_out):.1f}")])
    st.markdown("<div class='av-note'>Colour shift is the mean distance moved in the a*b* plane. Equalising each RGB "
                "channel on its own changes the balance between channels and tints the image; working on L* changes "
                "only lightness, so skin tones stay as they were.</div>", unsafe_allow_html=True)


def geometric(frame, full):
    st.latex(r"\begin{bmatrix}x'\\y'\end{bmatrix} = s\begin{bmatrix}\cos\theta & -\sin\theta\\ \sin\theta & \cos\theta\end{bmatrix}\begin{bmatrix}x\\y\end{bmatrix} + \begin{bmatrix}t_x\\t_y\end{bmatrix}")
    eng = c.engine()
    faces = sorted(eng.analyze(full, reject_blurry=False), key=lambda f: f.size)
    if not faces:
        st.info("No face was detected in this frame. Choose another frame.", icon=":material/face:")
        return
    a, b = st.columns(2)
    i = a.selectbox("Face", range(len(faces)), index=len(faces) // 2,
                    format_func=lambda k: f"face {k + 1} · {faces[k].size:.0f} px wide")
    factor = b.slider("Upscaling factor", 2, 8, 4)
    f = faces[i]
    x0, y0, x1, y1 = [int(v) for v in f.bbox]
    pad = int(0.25 * (x1 - x0))
    cx0, cy0 = max(0, x0 - pad), max(0, y0 - pad)
    crop = full[cy0:y1 + pad, cx0:x1 + pad]
    marked = pp.resize_with(crop, factor, "lanczos")
    for (px, py), colour_bgr in zip(f.kps, [(80, 220, 80)] * 2 + [(60, 180, 255)] + [(235, 120, 60)] * 2):
        cv2.circle(marked, (int((px - cx0) * factor), int((py - cy0) * factor)), max(2, factor), colour_bgr, -1)
    m = similarity_transform(f.kps, ARCFACE_DST)
    scale, angle = float(np.hypot(m[0, 0], m[1, 0])), float(np.degrees(np.arctan2(m[1, 0], m[0, 0])))
    st.markdown("**Interpolation: the same crop enlarged four ways**")
    gallery({f"{name} · sharpness {pp.blur_score(pp.resize_with(crop, factor, name)):.0f}": pp.resize_with(crop, factor, name)
             for name in pp.INTERPOLATIONS}, per_row=4)
    st.markdown("**Alignment: five landmarks mapped to the recognition template**")
    left, mid, right = st.columns([1, 1, 1.4])
    left.image(rgb(marked), caption="Detected landmarks: eyes, nose, mouth corners", width="stretch")
    mid.image(rgb(cv2.resize(f.crop, (336, 336), interpolation=cv2.INTER_NEAREST)),
              caption="Aligned 112×112 crop given to the recogniser", width="stretch")
    with right:
        theme.kv([("Face width in the frame", f"{f.size:.0f} px"), ("Scale s", f"{scale:.2f}×"),
                  ("Rotation θ", f"{angle:+.1f}°"), ("Translation (tx, ty)", f"{m[0, 2]:.0f}, {m[1, 2]:.0f} px"),
                  ("Blur score of the aligned crop", f"{f.blur:.0f}"),
                  ("Lanczos pre-upscale", "yes" if f.size < c.cfg()["recognition"]["upscale_below_px"] else "not needed")])
    st.markdown("<div class='av-note'>The transform is the least-squares similarity (rotation, uniform scale, shift) "
                "between the five detected landmarks and their positions in the template, found with Umeyama's "
                "method. Nearest-neighbour enlargement is blocky, bilinear is soft, bicubic and Lanczos keep edges; "
                "the benchmark in the report found no measurable recognition gain from Lanczos over direct warping.</div>",
                unsafe_allow_html=True)


def tiling(frame, full):
    eng, det = c.engine(), c.cfg()["detection"]
    h, w = full.shape[:2]
    a, b = st.columns(2)
    cols = a.slider("Tile columns", 1, 5, int(det["tile_grid"][0]))
    rows = b.slider("Tile rows", 1, 4, int(det["tile_grid"][1]))
    plain = eng.analyze(full, tiled=False, reject_blurry=False)
    old = det["tile_grid"]
    det["tile_grid"] = eng.dcfg["tile_grid"] = [cols, rows]
    try:
        tiled = eng.analyze(full, tiled=True, reject_blurry=False)
    finally:
        det["tile_grid"] = eng.dcfg["tile_grid"] = old
    overlay = draw_faces(full, tiled)
    tiles = pp.make_tiles(w, h, (cols, rows), det["tile_overlap"])
    for x0, y0, x1, y1 in tiles:
        cv2.rectangle(overlay, (x0, y0), (x1 - 1, y1 - 1), (0, 190, 255), max(2, w // 700))
    gallery({f"Full frame only: {len(plain)} faces": draw_faces(full, plain),
             f"Full frame + {cols}×{rows} overlapping tiles: {len(tiled)} faces": overlay}, per_row=2)
    tw, th = tiles[0][2] - tiles[0][0], tiles[0][3] - tiles[0][1]
    size = det["input_size"]
    sizes = [f.size for f in tiled]
    theme.kv([("Frame", f"{w}×{h} px"), ("Detector input", f"{size}×{size} px"),
              ("Shrink factor, full-frame pass", f"{max(w, h) / size:.2f}×"),
              ("Shrink factor, one tile", f"{max(tw, th) / size:.2f}×"),
              ("Magnification gained by tiling", f"{max(w, h) / max(tw, th):.2f}×"),
              ("Smallest / median face found", f"{min(sizes):.0f} / {np.median(sizes):.0f} px" if sizes else "–"),
              ("Tile overlap", f"{det['tile_overlap']:.0%}")])
    st.markdown("<div class='av-note'>The detector sees a fixed-size input, so a large frame is shrunk and back-row "
                "faces fall below what it can find. Each tile is shrunk far less. Tiles overlap so that a face cut by "
                "one border is whole in the neighbouring tile; duplicates are removed by non-maximum suppression.</div>",
                unsafe_allow_html=True)


def pipeline(frame, full):
    eng, conf = c.engine(), c.cfg()["preprocess"]
    dark = st.toggle("Simulate a dark, noisy camera frame", value=False)
    src = full
    if dark:
        src = pp.add_noise(pp.gamma_correct(full, 2.4), "gaussian", 6)
    stages, gamma = pp.run_stages(src, conf)
    gallery({name: fit(img, 640) for name, img in stages.items()})
    found = {name: len(eng.detect(img)[0]) for name, img in stages.items()}
    measure(stages, lambda name, img: {"Noise σ": round(pp.estimate_noise(img), 1), "Faces detected": found[name]})
    st.markdown("**Measured on the datasets** (details in the project report)")
    k = st.columns(4)
    k[0].metric("Face found for a labelled person", "90.6%", help="4,603 labelled people in the classroom dataset")
    k[1].metric("Verification accuracy (LFW)", "99.4%", help="500 labelled pairs, no false accepts")
    k[2].metric("Faces found with tiling", "41 vs 23", help="Seminar audience frame: with tiles vs full frame only")
    k[3].metric("Strangers accepted (class of 40)", "0.00%", help="At the 0.35 match threshold")
    st.markdown("<div class='av-note'>Detection runs on the enhanced frame. The recognition embedding is taken from "
                "the original pixels, because the benchmark showed that enhancing the crop lowers recognition "
                "accuracy (99.2% → 96.4% on dark frames).</div>", unsafe_allow_html=True)


RENDER = dict(zip(TOPICS, [intensity, histogram, filtering, sharpening, frequency, colour, geometric, tiling, pipeline]))


def render():
    theme.header("DIP Lab", "The digital image processing fundamentals this system is built on, each demonstrated "
                            "live on a frame of the footage with its formula, parameters and measurements.")
    samples = sorted(c.SAMPLES.rglob("*.mp4"))
    if not samples:
        with theme.card("nosrc"):
            st.info("No sample footage found in data/samples.", icon=":material/movie:")
        return
    left, right = st.columns([1, 3.3])
    with left:
        with theme.card("source"):
            st.subheader("Frame")
            default = next((i for i, p in enumerate(samples) if p.name == "owais-class-2.mp4"), 0)
            path = st.selectbox("Footage", samples, index=default, format_func=lambda p: str(p.relative_to(c.SAMPLES)))
            mtime = path.stat().st_mtime
            n = frame_count(str(path), mtime)
            index = st.slider("Frame number", 0, n - 1, min(10, n - 1)) if n > 1 else 0
            full = read_frame(str(path), index, mtime)
            if full is not None:
                st.image(rgb(fit(full, 480)), width="stretch")
                st.caption(f"{full.shape[1]}×{full.shape[0]} px · mean brightness {pp.mean_brightness(full):.0f}")
        with theme.card("topics"):
            st.subheader("Topic")
            topic = st.radio("Topic", TOPICS, label_visibility="collapsed")
    with right:
        if full is None:
            st.error("That frame could not be read.", icon=":material/error:")
            return
        with theme.card("lab"):
            role, note = ROLE[topic]
            st.markdown(f"<div class='av-eyebrow'>{role}</div>", unsafe_allow_html=True)
            st.subheader(topic)
            st.markdown(f"<div class='av-note' style='margin-bottom:.6rem'>{note}</div>", unsafe_allow_html=True)
            RENDER[topic](fit(full), full)
