"""DIP Lab: the image-processing steps the attendance system applies, shown live on a frame of
the footage in the order the system runs them, with the formula, the system's own parameters
and the measured effect of each step."""
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
import streamlit as st

from src import preprocess as pp
from src.config import ROOT
from src.face_engine import ARCFACE_DST, draw_faces, similarity_transform
from src.heatmap import add_boxes, heat_overlay, new_heat
from ui import common as c
from ui import theme
from ui.lab_helpers import contrast, fit, gallery, hist_frame, measure, note, rgb

# step -> (where it sits in the system, what it does there)
STEPS = {
    "Brightness and gamma correction": ("Pipeline stage 1", "The luminance histogram gives the mean brightness of the frame. A dark frame is brightened with a gamma chosen from that mean."),
    "Noise reduction": ("Pipeline stage 2", "The noise level of the frame is measured, and a bilateral filter (Non-Local Means optional) removes it with a strength that follows the measurement."),
    "Contrast: CLAHE on lightness": ("Pipeline stage 3", "Contrast is raised with CLAHE on the L channel of CIELAB, so that faces stand out and colours are not shifted."),
    "Sharpening and blur score": ("Pipeline stage 4 and face quality check", "Unsharp masking sharpens the frame. The variance of the Laplacian of each face crop is its blur score, which decides whether the face is used."),
    "Face alignment and upscaling": ("Pipeline stage 5", "Each face is warped to a 112×112 template by a similarity transform; small faces are Lanczos-upscaled first."),
    "Multi-scale tiling": ("Pipeline stage 6", "The frame is split into overlapping tiles so that distant faces are magnified for the detector."),
    "Face matching": ("Recognition", "Each face is compared with the enrolled photos of the batch and assigned to a student only if it passes a threshold and a margin."),
    "Seating heat map": ("Session result", "Every face position is accumulated over the session and shown in pseudo-colour on the session result."),
    "Complete pipeline": ("All stages", "The stages in the order the system applies them, with the measured effect of each."),
}
TOPICS = list(STEPS)


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


def chips(*items):
    st.markdown("".join(f"<span class='av-chip'>{i}</span>" for i in items), unsafe_allow_html=True)


# ------------------------------------------------------------------ steps
def brightness(frame, full):
    st.latex(r"s = 255\left(\frac{r}{255}\right)^{\gamma} \qquad \gamma_{auto} = \frac{\ln(\text{target}/255)}{\ln(\text{mean}/255)}")
    conf = c.cfg()["preprocess"]
    a, b = st.columns(2)
    dark = a.toggle("Simulate an under-exposed frame", value=True)
    src = pp.gamma_correct(frame, 2.4) if dark else frame
    out, g_auto = pp.auto_gamma(src, conf["dark_mean"], conf["target_mean"])
    gamma = b.slider("Gamma", 0.2, 1.5, float(round(g_auto, 2)), 0.05, help="Starts at the value the system chooses for this frame.")
    chips(f"mean brightness {pp.mean_brightness(src):.0f}", f"dark below {conf['dark_mean']}", f"target {conf['target_mean']}",
          f"automatic gamma {g_auto:.2f}", "correction applied" if g_auto < 1 else "no correction needed")
    images = {"Input": src, f"Gamma {gamma:.2f}": pp.gamma_correct(src, gamma)}
    gallery(images, per_row=2)
    left, right = st.columns(2)
    with left:
        st.markdown("**Transfer function**")
        r = np.arange(256)
        curves = pd.concat([pd.DataFrame({"input": r, "output": r.astype(float), "curve": "no change"}),
                            pd.DataFrame({"input": r, "output": 255 * (r / 255) ** gamma, "curve": f"gamma {gamma:.2f}"})])
        theme.lines(curves, "input", "output", "curve", "input level r", "output level s")
    with right:
        st.markdown("**Luminance histograms**")
        theme.lines(hist_frame(images), "level", "value", "image", "grey level", "% of pixels", area=True)
    measure(images)
    note("A gamma below 1 lifts the dark grey levels much more than the bright ones, so shadows open up without "
         "burning out the windows. The system applies it only when the histogram mean is below the dark limit, "
         "and never goes below 0.35.")


def noise(frame, full):
    st.latex(r"g(x) = \frac{1}{W}\sum_{y} f(y)\; e^{-\frac{\|x-y\|^2}{2\sigma_s^2}}\; e^{-\frac{(f(x)-f(y))^2}{2\sigma_r^2}} \qquad PSNR = 10\log_{10}\frac{255^2}{MSE}")
    conf = c.cfg()["preprocess"]
    sigma = st.slider("Noise added to the frame (σ)", 0.0, 30.0, 12.0, 1.0)
    clean = fit(frame, 640)
    noisy = pp.add_noise(clean, sigma) if sigma > 0 else clean
    measured = pp.estimate_noise(noisy)
    strength = 1.0 / float(np.clip(conf.get("noise_ok", 4.0) / max(measured, 1e-6), 0.25, 1.0))
    bilateral = pp.denoise(noisy, "bilateral", conf["bilateral_d"] + (4 if strength > 1.5 else 0),
                           min(120, conf["bilateral_sigma_color"] * strength), conf["bilateral_sigma_space"])
    nlm = pp.denoise(noisy, "nlm", nlm_h=min(20, conf["nlm_h"] * strength))
    images = {"Noisy input": noisy, "Bilateral filter (default)": bilateral, "Non-Local Means (optional)": nlm}
    scores = {n: pp.psnr(clean, im) for n, im in images.items()}
    gallery({f"{n} · PSNR {scores[n]:.1f} dB" if sigma > 0 else n: im for n, im in images.items()})
    left, right = st.columns([1.1, 1])
    with left:
        rows = pd.DataFrame({"image": list(images), "noise": [pp.estimate_noise(im) for im in images.values()]})
        theme.bars(rows, "image", "noise", "measured noise σ (lower is better)")
    with right:
        theme.kv([("Measured noise σ of the input", f"{measured:.1f}"), ("Noise level treated as clean", f"{conf.get('noise_ok', 4.0)}"),
                  ("Filter strength used", f"{strength:.1f}×"), ("Filter in use", conf["denoise"]),
                  ("Edges kept (Lap. var), bilateral", f"{pp.blur_score(bilateral):.0f}")])
    note("The bilateral filter averages only pixels that are close in both position and intensity, so it smooths "
         "noise but keeps the edges of a face. The noise level is estimated with Immerkær's method (a kernel that "
         "cancels image structure); a noisier frame gets a stronger filter, and the contrast and sharpening stages "
         "that follow are eased off so they do not bring the noise back.")


def contrast_stage(frame, full):
    st.latex(r"(B, G, R) \rightarrow (L^*, a^*, b^*) \qquad s_k = (L-1)\sum_{j=0}^{k} p_r(r_j)\ \text{per tile, histogram clipped, tiles interpolated}")
    conf = c.cfg()["preprocess"]
    a, b, d = st.columns(3)
    flat = a.toggle("Simulate a low-contrast frame", value=True)
    clip = b.slider("CLAHE clip limit", 1.0, 6.0, float(conf["clahe_clip"]), 0.5)
    grid = d.slider("CLAHE tile grid", 2, 16, int(conf["clahe_grid"]), 2)
    src = (frame.astype(np.float32) * 0.4 + 80).astype(np.uint8) if flat else frame
    out, he = pp.clahe_lab(src, clip, grid), pp.global_hist_eq(src)
    l, a_ch, b_ch = pp.lab_channels(src)
    gallery({"Input": src, f"CLAHE on L* (clip {clip}, {grid}×{grid})": out, "Global equalisation, for comparison": he,
             "L* (lightness): the only channel changed": l, "a* (green–red): untouched": a_ch, "b* (blue–yellow): untouched": b_ch})
    left, right = st.columns([1.3, 1])
    with left:
        st.markdown("**Luminance histograms**")
        theme.lines(hist_frame({"input": src, "CLAHE": out, "global equalisation": he}), "level", "value", "image",
                    "grey level", "% of pixels", area=True)
    with right:
        _, a2, b2 = pp.lab_channels(out)
        shift = float(np.mean(np.hypot(a2.astype(np.float32) - a_ch, b2.astype(np.float32) - b_ch)))
        theme.kv([("Contrast (σ), input", f"{contrast(src):.1f}"), ("Contrast (σ), CLAHE", f"{contrast(out):.1f}"),
                  ("Contrast (σ), global equalisation", f"{contrast(he):.1f}"), ("Mean colour shift after CLAHE", f"{shift:.2f}")])
    note("CLAHE equalises the histogram of small tiles and clips each one, so contrast rises where the detail is. "
         "Equalising the whole frame at once over-brightens walls and windows, which is why the system does not use "
         "it. Working on L* alone changes lightness only, so skin tones stay as they were.")


def sharpening(frame, full):
    st.latex(r"g = f + a\,(f - G_\sigma * f) \qquad \text{blur score} = \mathrm{Var}(\nabla^2 f) \qquad \nabla^2 f = \frac{\partial^2 f}{\partial x^2} + \frac{\partial^2 f}{\partial y^2}")
    conf, rc = c.cfg()["preprocess"], c.cfg()["recognition"]
    a, b, d = st.columns(3)
    soft = a.slider("Soften the input first (Gaussian σ)", 0.0, 4.0, 1.5, 0.5)
    amount = b.slider("Unsharp amount", 0.0, 2.5, float(conf["unsharp_amount"]), 0.1)
    sigma = d.slider("Unsharp σ", 0.5, 4.0, float(conf["unsharp_sigma"]), 0.5)
    src = cv2.GaussianBlur(frame, (0, 0), soft) if soft > 0 else frame
    out = pp.unsharp_mask(src, amount, sigma)
    gallery({"Input": src, f"Unsharp mask (a = {amount:.1f})": out,
             "Detail layer added back (f − blur)": cv2.convertScaleAbs(cv2.absdiff(src, cv2.GaussianBlur(src, (0, 0), sigma)), alpha=4),
             "|Laplacian| of the result": pp.laplacian_map(out)}, per_row=2)
    score = {"Original frame": pp.blur_score(frame), "Input": pp.blur_score(src), "Unsharp mask": pp.blur_score(out)}
    left, right = st.columns([1.1, 1])
    with left:
        theme.bars(pd.DataFrame({"image": list(score), "v": list(score.values())}), "image", "v",
                   "Variance of the Laplacian (sharpness)", fmt=".0f")
    with right:
        theme.kv([("Face crop treated as blurry below", f"{rc['blurry_below']:.0f}"),
                  ("Face crop rejected below", f"{rc['reject_below']:.0f}"),
                  ("Match threshold, sharp face", rc["match_threshold"]), ("Match threshold, blurry face", rc["blurry_threshold"])])
    note("The Laplacian responds only where intensity changes quickly. A blurred image has weak responses, so their "
         "variance falls. The system computes this score on every aligned face crop: very low scores are discarded "
         "and low scores are matched with the blurry-face threshold.")


def alignment(frame, full):
    st.latex(r"\begin{bmatrix}x'\\y'\end{bmatrix} = s\begin{bmatrix}\cos\theta & -\sin\theta\\ \sin\theta & \cos\theta\end{bmatrix}\begin{bmatrix}x\\y\end{bmatrix} + \begin{bmatrix}t_x\\t_y\end{bmatrix}")
    rc = c.cfg()["recognition"]
    faces = sorted(c.engine().analyze(full, reject_blurry=False), key=lambda f: f.size)
    if not faces:
        st.info("No face was detected in this frame. Choose another frame.", icon=":material/face:")
        return
    i = st.selectbox("Face", range(len(faces)), index=len(faces) // 2,
                     format_func=lambda k: f"face {k + 1} · {faces[k].size:.0f} px wide")
    f = faces[i]
    x0, y0, x1, y1 = [int(v) for v in f.bbox]
    pad = int(0.25 * (x1 - x0))
    cx0, cy0 = max(0, x0 - pad), max(0, y0 - pad)
    crop = full[cy0:y1 + pad, cx0:x1 + pad]
    factor = max(2, round(336 / max(crop.shape[1], 1)))
    marked = pp.lanczos_upscale(crop, factor)
    for (px, py), colour_bgr in zip(f.kps, [(80, 220, 80)] * 2 + [(60, 180, 255)] + [(235, 120, 60)] * 2):
        cv2.circle(marked, (int((px - cx0) * factor), int((py - cy0) * factor)), max(2, factor), colour_bgr, -1)
    m = similarity_transform(f.kps, ARCFACE_DST)
    scale, angle = float(np.hypot(m[0, 0], m[1, 0])), float(np.degrees(np.arctan2(m[1, 0], m[0, 0])))
    left, mid, right = st.columns([1, 1, 1.4])
    left.image(rgb(marked), caption=f"Face in the frame (Lanczos ×{factor}) with its five landmarks", width="stretch")
    mid.image(rgb(cv2.resize(f.crop, (336, 336), interpolation=cv2.INTER_NEAREST)),
              caption="Aligned 112×112 crop given to the recogniser", width="stretch")
    with right:
        theme.kv([("Face width in the frame", f"{f.size:.0f} px"), ("Scale s", f"{scale:.2f}×"),
                  ("Rotation θ", f"{angle:+.1f}°"), ("Translation (tx, ty)", f"{m[0, 2]:.0f}, {m[1, 2]:.0f} px"),
                  ("Blur score of the aligned crop", f"{f.blur:.0f}"),
                  ("Lanczos upscale before warping", "yes" if f.size < rc["upscale_below_px"] else "not needed")])
    note("The transform is the least-squares similarity (rotation, uniform scale, shift) between the five detected "
         "landmarks (eyes, nose, mouth corners) and their positions in the template, found with Umeyama's method. "
         f"Faces narrower than {rc['upscale_below_px']} pixels are enlarged with Lanczos interpolation first, which "
         "keeps edges better than bilinear enlargement.")


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
    note("The detector sees a fixed-size input, so a large frame is shrunk and back-row faces fall below what it can "
         "find. Each tile is shrunk far less. Tiles overlap so that a face cut by one border is whole in the "
         "neighbouring tile; duplicates are removed by non-maximum suppression.")


def matching(frame, full):
    st.latex(r"d_j(\mathbf{x}) = \frac{\mathbf{x}\cdot\mathbf{m}_j}{\|\mathbf{x}\|\,\|\mathbf{m}_j\|} \qquad \text{assign } \mathbf{x} \rightarrow \text{student } j \ \text{ if } d_j = \max_k d_k,\; d_j \ge T \ \text{ and } d_j - d_{2nd} \ge m")
    d, conf = c.db(), c.cfg()["recognition"]
    students = d.students()
    batches = sorted(students[students["photos"] > 0]["batch"].unique()) if len(students) else []
    if not batches:
        st.info("No students are enrolled yet, so there is nobody to match against.", icon=":material/group:")
        return
    faces = sorted(c.engine().analyze(full, reject_blurry=False), key=lambda f: -f.size)
    if not faces:
        st.info("No face was detected in this frame. Choose another frame.", icon=":material/face:")
        return
    a, b = st.columns(2)
    default = next((i for i, x in enumerate(batches) if x == "BSCS-A"), 0)
    batch = a.selectbox("Enrolled batch", batches, index=default)
    i = b.selectbox("Face in the frame", range(len(faces)), format_func=lambda k: f"face {k + 1} · {faces[k].size:.0f} px wide")
    f = faces[i]
    rolls, mat = d.gallery(batch)
    uniq = sorted(set(rolls))
    per = np.full(len(uniq), -1.0)
    np.maximum.at(per, [uniq.index(r) for r in rolls], f.embedding @ mat.T)
    order = np.argsort(-per)
    names = students.set_index("roll_no")["name"]
    blurry = f.blur < conf["blurry_below"]
    thr = conf["blurry_threshold"] if blurry else conf["match_threshold"]
    best, second = float(per[order[0]]), float(per[order[1]]) if len(uniq) > 1 else -1.0
    accepted = best >= thr and best - second >= conf["margin"]
    left, right = st.columns([1, 2.2])
    with left:
        st.image(rgb(cv2.resize(f.crop, (224, 224), interpolation=cv2.INTER_CUBIC)), caption="Aligned face (described by 512 numbers)", width="stretch")
        if accepted:
            st.success(f"Matched to {uniq[order[0]]} · {names.get(uniq[order[0]], '')}", icon=":material/check_circle:")
        else:
            st.warning("Not matched: " + ("below the threshold" if best < thr else "too close to the second-best student"), icon=":material/block:")
    with right:
        top = pd.DataFrame({"student": [f"{uniq[j]} · {names.get(uniq[j], '')}" for j in order[:10]], "similarity": [float(per[j]) for j in order[:10]]})
        theme.bars(top, "student", "similarity", "cosine similarity to each student's photos", fmt=".3f")
        theme.kv([("Best student", f"{best:.3f}"), ("Second-best student", f"{second:.3f}"), ("Margin", f"{best - second:.3f} (needs {conf['margin']})"),
                  ("Threshold for this crop", f"{thr} ({'blurry' if blurry else 'sharp'}, blur score {f.blur:.0f})")])
    labels, colours = [""] * len(faces), [(160, 160, 160)] * len(faces)
    labels[i], colours[i] = "x", (80, 220, 80) if accepted else (60, 60, 235)
    st.image(rgb(fit(draw_faces(full, faces, labels, colours))), caption="The selected face in the frame", width="stretch")
    path = ROOT / "docs" / "open_set.csv"
    if path.exists():
        table = pd.read_csv(path)
        sub = table[(table["class_size"] == 40) & (table["margin"] == 0.06) & (table["probe"].isin(["clean, 110 px", "blurred, 40 px"]))]
        curves = pd.concat([sub.assign(value=sub["TAR_%"], curve=sub["probe"] + " · enrolled accepted"),
                            sub.assign(value=sub["FAR_%"] * 100, curve=sub["probe"] + " · strangers accepted ×100")])
        st.markdown("**How the threshold was chosen** (measured, class of 40)")
        theme.lines(curves, "threshold", "value", "curve", "threshold T", "% (strangers ×100)")
    note("Only the students of the batch that the timetable places in the room are compared. The threshold and the "
         "margin let the system say \"nobody\": most faces in a corridor or a shared room belong to no enrolled "
         "student, and a wrong name is worse than an unknown face.")


def heat_map(frame, full):
    st.latex(r"h(x, y) = \sum_{\text{frames}}\ \sum_{\text{faces}} \mathbf{1}\big[(x, y) \in \text{face}\big] \qquad \text{colour} = \mathrm{map}\!\left(\frac{h}{\max h}\right)")
    faces = c.engine().analyze(full, reject_blurry=False)
    heat = add_boxes(new_heat(full.shape), [f.bbox for f in faces])
    gallery({f"Faces found in this frame: {len(faces)}": fit(draw_faces(full, faces)),
             "Their positions in pseudo-colour": fit(heat_overlay(full, cv2.GaussianBlur(heat, (0, 0), 2.0)))}, per_row=2)
    sessions = c.db().sessions()
    saved = [(r, Path(c.paths()["snapshots"]) / f"session_{r['id']}" / "heatmap.jpg") for _, r in sessions.iterrows()] if len(sessions) else []
    saved = [(r, p) for r, p in saved if p.exists()]
    if saved:
        st.markdown("**Accumulated over a whole session**")
        k = st.selectbox("Session", range(len(saved)), format_func=lambda j: c.session_label(saved[j][0]))
        st.image(str(saved[k][1]), width="stretch")
    note("A single frame gives one mark per face. Over a session the marks add up: red shows the seats where a face "
         "was seen in most frames, blue the places seen only a few times. Mapping the count to colour makes small "
         "differences visible that grey levels would hide. The map is meaningful for a fixed camera; if the camera "
         "moved during the recording it is smeared.")


def pipeline(frame, full):
    eng, conf = c.engine(), c.cfg()["preprocess"]
    dark = st.toggle("Simulate a dark, noisy camera frame", value=False)
    src = pp.add_noise(pp.gamma_correct(full, 2.4), 6) if dark else full
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
    note("Detection runs on the enhanced frame. The recognition embedding is taken from the original pixels, because "
         "the benchmark showed that enhancing the crop lowers recognition accuracy (99.2% → 96.4% on dark frames).")


RENDER = dict(zip(TOPICS, [brightness, noise, contrast_stage, sharpening, alignment, tiling, matching, heat_map, pipeline]))


def render():
    theme.header("DIP Lab", "The image-processing steps the system applies to every frame, shown live on the footage "
                            "with the formula, the system's own settings and the measured effect.")
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
            st.subheader("Step")
            topic = st.radio("Step", TOPICS, label_visibility="collapsed")
            st.caption("In the order the system applies them")
    with right:
        if full is None:
            st.error("That frame could not be read.", icon=":material/error:")
            return
        with theme.card("lab"):
            role, text = STEPS[topic]
            st.markdown(f"<div class='av-eyebrow'>{role}</div>", unsafe_allow_html=True)
            st.subheader(topic)
            st.markdown(f"<div class='av-note' style='margin-bottom:.6rem'>{text}</div>", unsafe_allow_html=True)
            RENDER[topic](fit(full), full)
