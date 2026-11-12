"""Measure the syllabus techniques that were added after the first pipeline, on real data.

  python scripts/evaluate_syllabus.py            # everything (about 10 minutes on a GPU)
  python scripts/evaluate_syllabus.py --quick    # fewer pairs / frames

  A  homomorphic filtering   detection against the classroom labels, normal and one-sided lighting
  B  morphological head count  skin segmentation count against the labelled number of students
  C  JPEG compression        recognition accuracy and file size against quality (LFW)
  D  restoration             recognition after blur + noise: none / inverse / Wiener (LFW)
  E  quantization            recognition against bits per channel (LFW)
  F  lossless coding         bits per pixel: Huffman, arithmetic, LZW, PNG

Tables go to outputs/evaluation/syllabus_*.csv and figures to outputs/evaluation/syllabus/.
"""
import argparse
import sys
from pathlib import Path

import cv2
import matplotlib
import numpy as np
import pandas as pd

matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from scripts.evaluate import CANVAS, place
from scripts.evaluate_dataset import DATA, assign
from src import dip_ops as ops
from src import preprocess as pp
from src.config import ROOT, load_config
from src.face_engine import FaceEngine, draw_faces

OUT = ROOT / "outputs" / "evaluation"
FIG = OUT / "syllabus"


def strip(images, path, title, height=3.0):
    """A row of captioned images saved as one figure."""
    n = len(images)
    h, w = next(iter(images.values())).shape[:2]
    fig, axes = plt.subplots(1, n, figsize=(height * w / h * n, height + 0.7), squeeze=False)
    for ax, (name, img) in zip(axes[0], images.items()):
        ax.imshow(cv2.cvtColor(img, cv2.COLOR_GRAY2RGB if img.ndim == 2 else cv2.COLOR_BGR2RGB))
        ax.set_title(name, fontsize=9)
        ax.axis("off")
    fig.suptitle(title, fontsize=11)
    fig.tight_layout()
    fig.savefig(path, dpi=130)
    plt.close(fig)
    print("  saved", path.name, flush=True)


def lfw_pairs(n):
    from sklearn.datasets import fetch_lfw_pairs
    d = fetch_lfw_pairs(subset="test", color=True, resize=1.0, slice_=None, data_home=str(ROOT / "data" / "samples" / "lfw"))
    imgs = d.pairs if d.pairs.max() > 1.5 else d.pairs * 255.0
    idx = np.random.default_rng(0).permutation(len(imgs))[:n]
    pairs = [(cv2.cvtColor(imgs[i, 0].astype(np.uint8), cv2.COLOR_RGB2BGR),
              cv2.cvtColor(imgs[i, 1].astype(np.uint8), cv2.COLOR_RGB2BGR)) for i in idx]
    return pairs, d.target[idx].astype(bool)


def verify(engine, pairs, same, transform, thr):
    """Accuracy at the system threshold when `transform` is applied to both images of each pair."""
    sims = []
    for a, b in pairs:
        embs = []
        for im in (a, b):
            faces = engine.analyze(transform(im), tiled=False, reject_blurry=False)
            if not faces:
                embs.append(None)
                continue
            c = np.array([CANVAS / 2, CANVAS / 2])
            embs.append(min(faces, key=lambda f: np.linalg.norm((f.bbox[:2] + f.bbox[2:]) / 2 - c)).embedding)
        sims.append(-1.0 if embs[0] is None or embs[1] is None else float(embs[0] @ embs[1]))
    sims = np.array(sims)
    pred = sims >= thr
    return {"detected_%": round(100 * float((sims > -1).mean()), 1), "accuracy_%": round(100 * float((pred == same).mean()), 1),
            "false_reject_%": round(100 * float((~pred[same]).mean()), 1), "false_accept_%": round(100 * float(pred[~same].mean()), 1)}


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--quick", action="store_true")
    a = ap.parse_args()
    n_pairs, frame_step = (60, 12) if a.quick else (200, 4)
    FIG.mkdir(parents=True, exist_ok=True)
    cfg = load_config()
    engine = FaceEngine(cfg)
    thr = cfg["recognition"]["match_threshold"]

    # ---------------------------------------------------------------- A + B on the classroom dataset
    images = sorted((DATA / "images").glob("*.jpg"))[::frame_step] if DATA.exists() else []
    rows_a, rows_b = [], []
    ramp = None
    for n, path in enumerate(images, 1):
        img = cv2.imread(str(path))
        h, w = img.shape[:2]
        boxes = []
        for line in (DATA / "labels" / (path.stem + ".txt")).read_text().strip().splitlines():
            _, cx, cy, bw, bh = [float(v) for v in line.split()[:5]]
            boxes.append(((cx - bw / 2) * w, (cy - bh / 2) * h, (cx + bw / 2) * w, (cy + bh / 2) * h))
        if ramp is None or ramp.shape[1] != w:
            ramp = np.linspace(0.3, 1.0, w, dtype=np.float32)[None, :, None]
        shaded = (img * ramp).astype(np.uint8)                       # window light from one side
        for lighting, src in (("as recorded", img), ("one-sided lighting", shaded)):
            for stage, pc in (("pipeline", cfg["preprocess"]), ("pipeline + homomorphic", dict(cfg["preprocess"], homomorphic=True))):
                enhanced, _ = pp.enhance_frame(src, pc)
                hit = assign(boxes, engine.analyze(enhanced, embed_from=src))
                rows_a.append({"lighting": lighting, "stage": stage, "people": len(boxes),
                               "found": sum(x is not None for x in hit), "spread": pp.illumination_spread(enhanced)})
        faces = engine.analyze(pp.enhance_frame(img, cfg["preprocess"])[0], embed_from=img)
        rows_b.append({"labelled": len(boxes), "skin_regions": ops.head_count(img)["count"], "faces_detected": len(faces)})
        if n % 20 == 0:
            print(f"  dataset frames {n}/{len(images)}", flush=True)
    if rows_a:
        da = pd.DataFrame(rows_a).groupby(["lighting", "stage"], sort=False).agg(people=("people", "sum"), found=("found", "sum"),
                                                                                 illumination_spread=("spread", "mean")).reset_index()
        da["detected_%"] = (100 * da["found"] / da["people"]).round(1)
        da["illumination_spread"] = da["illumination_spread"].round(1)
        da.to_csv(OUT / "syllabus_homomorphic.csv", index=False)
        print("\nA  homomorphic filtering\n" + da.to_string(index=False))
        db = pd.DataFrame(rows_b)
        summary = pd.DataFrame([{"method": name, "mean_count": round(db[col].mean(), 2),
                                 "mean_abs_error": round((db[col] - db["labelled"]).abs().mean(), 2),
                                 "correlation": round(float(np.corrcoef(db[col], db["labelled"])[0, 1]), 2)}
                                for name, col in (("Skin segmentation + morphology", "skin_regions"), ("Face detector (SCRFD)", "faces_detected"))])
        summary.insert(1, "labelled_mean", round(db["labelled"].mean(), 2))
        summary.to_csv(OUT / "syllabus_headcount.csv", index=False)
        print("\nB  head count against the labelled number of students\n" + summary.to_string(index=False))

        # figures from one representative frame
        img = cv2.imread(str(next(p for p in images if "owais-class-2" in p.name)))
        shaded = (img * np.linspace(0.3, 1.0, img.shape[1], dtype=np.float32)[None, :, None]).astype(np.uint8)
        small = lambda im: cv2.resize(im, (640, 360), interpolation=cv2.INTER_AREA)
        strip({"one-sided lighting": small(shaded), "homomorphic filter": small(pp.homomorphic(shaded)),
               "global hist. equalisation": small(pp.global_hist_eq(shaded))}, FIG / "homomorphic.png",
              "Homomorphic filtering evens out the illumination")
        hc = ops.head_count(img)
        strip({"skin mask (YCrCb threshold)": small(hc["skin"]), "opening": small(hc["opened"]), "closing": small(hc["closed"]),
               "hole filling": small(hc["filled"]), "connected components": small(ops.colour_labels(hc["labels"])),
               f"regions kept: {hc['count']}": small(ops.draw_regions(img, hc["regions"]))}, FIG / "morphology.png",
              "Morphological head count: segmentation, opening, closing, hole filling, connected components", height=2.2)
        gray = pp.luminance(small(img))
        strip({"Sobel": ops.edges(gray, "sobel"), "Prewitt": ops.edges(gray, "prewitt"), "Roberts": ops.edges(gray, "roberts"),
               "Laplacian of Gaussian": ops.edges(gray, "log"), "Canny": ops.edges(gray, "canny")}, FIG / "edges.png",
              "Edge detection operators", height=2.2)
        lines = ops.hough_lines(ops.edges(gray, "canny"), 80, 90, 8)
        t, otsu = ops.threshold_otsu(gray)
        strip({"Canny edges": ops.edges(gray, "canny"), f"Hough lines ({len(lines)})": ops.draw_lines(small(img), lines),
               f"Otsu threshold T = {t}": otsu, "adaptive threshold": ops.threshold_adaptive(gray)}, FIG / "segmentation.png",
              "Hough transform and thresholding", height=2.4)
        psf = ops.gaussian_psf(2.0)
        blurred = ops.degrade(small(img), psf, 4)
        strip({f"blurred + noise  {pp.psnr(small(img), blurred):.1f} dB": blurred,
               f"inverse filter  {pp.psnr(small(img), ops.inverse_filter(blurred, psf, 0.05)):.1f} dB": ops.inverse_filter(blurred, psf, 0.05),
               f"Wiener filter  {pp.psnr(small(img), ops.wiener_filter(blurred, psf, 0.01)):.1f} dB": ops.wiener_filter(blurred, psf, 0.01)},
              FIG / "restoration.png", "Image restoration: inverse filtering against Wiener filtering")
        strip({f"{b} bits": ops.quantize(small(img), b) for b in (8, 4, 2, 1)} | {"1 pixel in 6": ops.subsample(small(img), 6)},
              FIG / "sampling_quantization.png", "Quantization and sampling", height=2.2)
        strip({f"quality {q}  {ops.jpeg_roundtrip(small(img), q)[1] / 1024:.0f} kB": ops.jpeg_roundtrip(small(img), q)[0] for q in (5, 15, 40, 90)},
              FIG / "jpeg.png", "JPEG compression at four quality settings", height=2.4)
        rows_f, _ = ops.lossless_comparison(cv2.resize(gray, (320, 180)))
        pd.DataFrame(rows_f).to_csv(OUT / "syllabus_lossless.csv", index=False)
        print("\nF  lossless coding of one frame (320 x 180 grey)\n" + pd.DataFrame(rows_f).to_string(index=False))

    # ---------------------------------------------------------------- C, D, E on LFW
    pairs, same = lfw_pairs(n_pairs)
    print(f"\n{len(pairs)} LFW pairs", flush=True)
    rows_c = []
    for q in (95, 70, 40, 20, 10, 5):
        sizes = []

        def jpeg(im, q=q):
            out, size = ops.jpeg_roundtrip(place(im, 40), q)
            sizes.append(size)
            return out
        rows_c.append({"jpeg_quality": q, **verify(engine, pairs, same, jpeg, thr), "mean_kB": round(float(np.mean(sizes)) / 1024, 1)})
        print("  C", rows_c[-1], flush=True)
    pd.DataFrame(rows_c).to_csv(OUT / "syllabus_jpeg.csv", index=False)

    psf = ops.gaussian_psf(1.6)
    blur = lambda im: ops.degrade(place(im, 40), psf, 2.0)
    rows_d = [{"restoration": "none (blurred + noise)", **verify(engine, pairs, same, blur, thr)},
              {"restoration": "inverse filter", **verify(engine, pairs, same, lambda im: ops.inverse_filter(blur(im), psf, 0.1), thr)},
              {"restoration": "Wiener filter, K = 0.01", **verify(engine, pairs, same, lambda im: ops.wiener_filter(blur(im), psf, 0.01), thr)},
              {"restoration": "Wiener filter, K = 0.05", **verify(engine, pairs, same, lambda im: ops.wiener_filter(blur(im), psf, 0.05), thr)}]
    pd.DataFrame(rows_d).to_csv(OUT / "syllabus_restoration.csv", index=False)
    print("\nD  restoration before recognition\n" + pd.DataFrame(rows_d).to_string(index=False), flush=True)

    rows_e = [{"bits_per_channel": b, **verify(engine, pairs, same, lambda im, b=b: ops.quantize(place(im, 40), b), thr)} for b in (8, 4, 3, 2, 1)]
    pd.DataFrame(rows_e).to_csv(OUT / "syllabus_quantization.csv", index=False)
    print("\nE  quantization\n" + pd.DataFrame(rows_e).to_string(index=False))
    print("\nC  JPEG quality\n" + pd.DataFrame(rows_c).to_string(index=False))

    fig, ax = plt.subplots(figsize=(6.5, 3.4))
    dc = pd.DataFrame(rows_c)
    ax.plot(dc["jpeg_quality"], dc["accuracy_%"], "o-", color="#2F5BEA", label="verification accuracy (%)")
    ax.set_xlabel("JPEG quality"); ax.set_ylabel("accuracy (%)"); ax.set_ylim(40, 101); ax.grid(alpha=0.3)
    ax2 = ax.twinx(); ax2.plot(dc["jpeg_quality"], dc["mean_kB"], "s--", color="#F79009", label="file size (kB)"); ax2.set_ylabel("file size (kB)")
    ax.set_title("Recognition accuracy and file size against JPEG quality", fontsize=10)
    fig.legend(loc="lower right", bbox_to_anchor=(0.88, 0.2), frameon=False, fontsize=8)
    fig.tight_layout(); fig.savefig(FIG / "jpeg_accuracy.png", dpi=130); plt.close(fig)
    print(f"\nSaved tables in {OUT} and figures in {FIG}")
