"""Save comparison figures for every DIP stage (for the report).

  python scripts/dip_demo.py --video data/samples/classroom.mp4 --at 5
  python scripts/dip_demo.py --image some_frame.jpg

Written to outputs/dip_demo/:
  1_stages.png          every stage of the pipeline with its histogram
  2_clahe_vs_he.png     global histogram equalisation vs CLAHE
  3_dark_frame.png      a simulated dark, noisy frame recovered by the pipeline
  4_tiled_detection.png full-frame detection vs tiled detection
  5_face_upscale.png    small face crops: nearest / bicubic / Lanczos + blur scores
  metrics.csv           brightness, contrast, entropy, sharpness, faces found per stage
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
from src import preprocess as pp
from src.config import ROOT, load_config
from src.face_engine import FaceEngine, draw_faces


def rgb(img):
    return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)


def entropy(img):
    h = pp.histogram(img)
    h = h[h > 0]
    return float(-(h * np.log2(h)).sum())


def plot_hist(ax, img):
    ax.fill_between(np.arange(256), pp.histogram(img), color="#444", step="mid")
    ax.set_xlim(0, 255)
    ax.set_yticks([])
    ax.set_xlabel(f"mean {pp.mean_brightness(img):.0f}   std {pp.luminance(img).std():.0f}", fontsize=8)


def grid(images, path, title, hist=True):
    """One column per image; second row = luminance histograms."""
    n = len(images)
    h, w = next(iter(images.values())).shape[:2]
    rows = 2 if hist else 1
    fig, axes = plt.subplots(rows, n, figsize=(3.6 * n, 3.6 * h / w * (1.45 if hist else 1) + 0.9),
                             squeeze=False, gridspec_kw={"height_ratios": [3, 1][:rows]})
    for i, (name, img) in enumerate(images.items()):
        axes[0, i].imshow(rgb(img))
        axes[0, i].set_title(name, fontsize=10)
        axes[0, i].axis("off")
        if hist:
            plot_hist(axes[1, i], img)
    fig.suptitle(title, fontsize=12)
    fig.tight_layout()
    fig.savefig(path, dpi=140)
    plt.close(fig)
    print(f"  saved {path.name}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--image")
    ap.add_argument("--video")
    ap.add_argument("--at", type=float, default=5.0, help="second of the video to grab")
    ap.add_argument("--out", default=str(ROOT / "outputs" / "dip_demo"))
    a = ap.parse_args()

    cfg = load_config()
    pc = cfg["preprocess"]
    if a.image:
        frame = cv2.imdecode(np.fromfile(a.image, np.uint8), cv2.IMREAD_COLOR)
    else:
        video = a.video or next(iter(sorted((ROOT / "data" / "samples").rglob("*.mp4"))), None)
        if video is None:
            sys.exit("Give --image or --video (or run scripts/get_sample_video.py first).")
        cap = cv2.VideoCapture(str(video))
        cap.set(cv2.CAP_PROP_POS_MSEC, a.at * 1000)
        ok, frame = cap.read()
        cap.release()
        if not ok:
            sys.exit(f"Could not read a frame at {a.at}s from {video}")
        print(f"Frame at {a.at}s of {video}")
    if frame is None:
        sys.exit("Could not read the image.")
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    engine = FaceEngine(cfg)
    metrics = []

    def measure(group, name, img):
        metrics.append({"figure": group, "stage": name, "mean_brightness": round(pp.mean_brightness(img), 1),
                        "contrast_std": round(float(pp.luminance(img).std()), 1),
                        "entropy_bits": round(entropy(img), 2),
                        "sharpness_lapvar": round(pp.blur_score(img), 1),
                        "faces_detected": len(engine.detect(img)[0])})

    # 1. every stage on the real frame ------------------------------------------------
    stages, gamma = pp.run_stages(frame, pc)
    for name, img in stages.items():
        measure("stages", name, img)
    grid(stages, out / "1_stages.png", "DIP pipeline: output of every stage (luminance histogram below)")

    # 2. global HE vs CLAHE -----------------------------------------------------------
    cmp = {"original": frame, "global hist. equalisation": pp.global_hist_eq(frame),
           "CLAHE (LAB-L)": pp.clahe_lab(frame, pc["clahe_clip"], pc["clahe_grid"])}
    for name, img in cmp.items():
        measure("clahe_vs_he", name, img)
    grid(cmp, out / "2_clahe_vs_he.png", "Global histogram equalisation vs CLAHE")

    # 3. simulated dark + noisy CCTV frame ----------------------------------------------
    rng = np.random.default_rng(0)
    dark = pp.gamma_correct(frame, 2.6).astype(np.float32) * 0.8 + rng.normal(0, 6, frame.shape)
    dark = dark.clip(0, 255).astype(np.uint8)
    dstages, dgamma = pp.run_stages(dark, pc)
    dstages = {("dark input" if k == "original" else k): v for k, v in dstages.items()}
    for name, img in dstages.items():
        measure("dark_frame", name, img)
    grid(dstages, out / "3_dark_frame.png",
         f"Simulated dark, noisy frame: histogram triggers gamma {dgamma:.2f}, then denoise / CLAHE / sharpen")

    # 4. tiled detection --------------------------------------------------------------
    enhanced = next(reversed(stages.values()))
    full = engine.analyze(enhanced, tiled=False, reject_blurry=False)
    tiled = engine.analyze(enhanced, tiled=True, reject_blurry=False)
    tiles_img = draw_faces(enhanced, tiled)
    for x0, y0, x1, y1 in pp.make_tiles(frame.shape[1], frame.shape[0], cfg["detection"]["tile_grid"],
                                        cfg["detection"]["tile_overlap"]):
        cv2.rectangle(tiles_img, (x0, y0), (x1 - 1, y1 - 1), (0, 200, 255), max(1, frame.shape[1] // 900))
    grid({f"full frame only: {len(full)} faces": draw_faces(enhanced, full),
          f"full frame + overlapping tiles: {len(tiled)} faces": tiles_img},
         out / "4_tiled_detection.png", "Tiled detection magnifies distant faces", hist=False)
    metrics.append({"figure": "tiled_detection", "stage": "full frame only", "faces_detected": len(full)})
    metrics.append({"figure": "tiled_detection", "stage": "full frame + tiles", "faces_detected": len(tiled)})

    # 5. small face crops: interpolation comparison --------------------------------------
    small = sorted(tiled, key=lambda f: f.size)[:4]
    if small:
        fig, axes = plt.subplots(len(small), 4, figsize=(10, 2.6 * len(small)), squeeze=False)
        for r, f in enumerate(small):
            x0, y0, x1, y1 = [int(v) for v in f.bbox]
            crop = enhanced[max(0, y0):y1 + 1, max(0, x0):x1 + 1]
            size = (224, round(224 * crop.shape[0] / crop.shape[1]))
            views = {f"crop {crop.shape[1]}x{crop.shape[0]} px (nearest)": cv2.resize(crop, size, interpolation=cv2.INTER_NEAREST),
                     "bicubic": cv2.resize(crop, size, interpolation=cv2.INTER_CUBIC),
                     "Lanczos-4": cv2.resize(crop, size, interpolation=cv2.INTER_LANCZOS4),
                     "aligned 112x112 (Lanczos + warp)": f.crop}
            for c, (name, img) in enumerate(views.items()):
                axes[r, c].imshow(rgb(img))
                axes[r, c].set_title(f"{name}\nLaplacian var {pp.blur_score(img):.0f}", fontsize=8)
                axes[r, c].axis("off")
        fig.suptitle("Smallest faces in the frame: upscaling methods and blur score", fontsize=12)
        fig.tight_layout()
        fig.savefig(out / "5_face_upscale.png", dpi=140)
        plt.close(fig)
        print("  saved 5_face_upscale.png")

    pd.DataFrame(metrics).to_csv(out / "metrics.csv", index=False)
    print("  saved metrics.csv\n")
    print(pd.DataFrame(metrics).to_string(index=False))
