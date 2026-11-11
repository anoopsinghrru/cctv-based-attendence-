"""Measure the face pipeline on a labelled benchmark (LFW pairs: same person / different people).

  python scripts/evaluate.py                # 1000 test pairs, all conditions
  python scripts/evaluate.py --pairs 300    # quicker

Each LFW image is pasted on a 640x640 canvas so the face has a chosen size in pixels, as it
would inside a CCTV frame, and optionally degraded (dark + noise, blur). Reported per condition:
detection rate, verification accuracy at the system threshold, the best achievable accuracy,
and the false-accept / false-reject rates. Written to outputs/evaluation/.
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
from src.face_engine import FaceEngine

CANVAS = 640
NATIVE_FACE = 110.0          # typical face width in a 250x250 LFW image


def place(img, face_px):
    """Shrink the LFW image so its face is ~face_px wide and paste it mid-canvas."""
    scale = face_px / NATIVE_FACE
    if scale < 0.999:
        img = cv2.resize(img, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA)
    canvas = np.full((CANVAS, CANVAS, 3), 115, np.uint8)
    h, w = img.shape[:2]
    y, x = (CANVAS - h) // 2, (CANVAS - w) // 2
    canvas[y:y + h, x:x + w] = img
    return canvas


def darken(img, rng):
    out = pp.gamma_correct(img, 2.6).astype(np.float32) + rng.normal(0, 5, img.shape)
    return out.clip(0, 255).astype(np.uint8)


def embed(engine, cfg, img, mode):
    """Embedding of the most central face (None if nothing is detected).

    mode: 'raw' = no enhancement; 'hybrid' = detect on enhanced, embed from original (the
    system default); 'enhanced' = detect and embed on the enhanced frame.
    """
    if mode == "raw":
        faces = engine.analyze(img, tiled=False, reject_blurry=False)
    else:
        enhanced, _ = pp.enhance_frame(img, cfg["preprocess"])
        faces = engine.analyze(enhanced, tiled=False, reject_blurry=False,
                               embed_from=img if mode == "hybrid" else None)
    if not faces:
        return None
    centre = np.array([CANVAS / 2, CANVAS / 2])
    return min(faces, key=lambda f: np.linalg.norm((f.bbox[:2] + f.bbox[2:]) / 2 - centre)).embedding


def score_pairs(engine, cfg, pairs, face_px, mode, degrade=None):
    rng = np.random.default_rng(0)
    sims, found = [], 0
    for a, b in pairs:
        embs = []
        for im in (a, b):
            im = place(im, face_px)
            if degrade == "dark":
                im = darken(im, rng)
            elif degrade == "blur":
                im = cv2.GaussianBlur(im, (0, 0), 1.6)
            embs.append(embed(engine, cfg, im, mode))
        if embs[0] is None or embs[1] is None:
            sims.append(-1.0)               # an undetected face can never match
        else:
            found += 1
            sims.append(float(embs[0] @ embs[1]))
    return np.array(sims), found / len(pairs)


def metrics(sims, same, thr):
    pred = sims >= thr
    best_t, best_acc = thr, 0.0
    for t in np.arange(0.10, 0.70, 0.01):
        acc = float(((sims >= t) == same).mean())
        if acc > best_acc:
            best_t, best_acc = t, acc
    return {"accuracy_%": round(100 * float((pred == same).mean()), 2),
            "false_accept_%": round(100 * float(pred[~same].mean()), 2),
            "false_reject_%": round(100 * float((~pred[same]).mean()), 2),
            "best_accuracy_%": round(100 * best_acc, 2), "best_threshold": round(float(best_t), 2),
            "mean_sim_same": round(float(sims[same & (sims >= 0)].mean()), 3),
            "mean_sim_diff": round(float(sims[~same & (sims >= 0)].mean()), 3)}


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--pairs", type=int, default=1000, help="number of LFW test pairs (max 1000)")
    a = ap.parse_args()

    from sklearn.datasets import fetch_lfw_pairs
    d = fetch_lfw_pairs(subset="test", color=True, resize=1.0, slice_=None,
                        data_home=str(ROOT / "data" / "samples" / "lfw"))
    imgs = d.pairs if d.pairs.max() > 1.5 else d.pairs * 255.0
    idx = np.random.default_rng(0).permutation(len(imgs))[:a.pairs]
    pairs = [(cv2.cvtColor(imgs[i, 0].astype(np.uint8), cv2.COLOR_RGB2BGR),
              cv2.cvtColor(imgs[i, 1].astype(np.uint8), cv2.COLOR_RGB2BGR)) for i in idx]
    same = d.target[idx].astype(bool)
    print(f"{len(pairs)} pairs ({same.sum()} same person, {(~same).sum()} different)")

    cfg = load_config()
    engine = FaceEngine(cfg)
    thr = cfg["recognition"]["match_threshold"]
    upscale = cfg["recognition"]["upscale_below_px"]
    out = ROOT / "outputs" / "evaluation"
    out.mkdir(parents=True, exist_ok=True)

    # (label, face px, enhancement mode, degradation, Lanczos pre-upscale on?)
    runs = [("clean, face 110 px", 110, "raw", None, True)]
    for px in (64, 40, 28, 20, 14):
        runs.append((f"face {px} px, Lanczos upscale", px, "raw", None, True))
        runs.append((f"face {px} px, plain warp", px, "raw", None, False))
    for mode, name in (("raw", "no enhancement"), ("hybrid", "detect enhanced / embed original"),
                       ("enhanced", "detect + embed enhanced")):
        runs.append((f"dark+noise 40 px, {name}", 40, mode, "dark", True))
    runs.append(("blurred 40 px, no enhancement", 40, "raw", "blur", True))
    runs.append(("blurred 40 px, detect + embed enhanced", 40, "enhanced", "blur", True))

    rows, keep = [], {}
    for label, px, mode, degrade, lanczos in runs:
        engine.rcfg["upscale_below_px"] = upscale if lanczos else 0
        sims, det = score_pairs(engine, cfg, pairs, px, mode, degrade)
        t = cfg["recognition"]["blurry_threshold"] if (degrade or px <= 28) else thr
        row = {"condition": label, "face_px": px, "both_faces_detected_%": round(100 * det, 1),
               "threshold": t, **metrics(sims, same, t)}
        rows.append(row)
        keep[label] = sims
        print(f"  {label:44s} det {row['both_faces_detected_%']:5.1f}%  acc {row['accuracy_%']:6.2f}%  "
              f"FAR {row['false_accept_%']:5.2f}%  FRR {row['false_reject_%']:5.2f}%  "
              f"best {row['best_accuracy_%']:6.2f}% @ {row['best_threshold']}")
    engine.rcfg["upscale_below_px"] = upscale
    df = pd.DataFrame(rows)
    df.to_csv(out / "results.csv", index=False)

    # figure 1: similarity distributions on clean pairs, with the thresholds
    s = keep["clean, face 110 px"]
    fig, ax = plt.subplots(figsize=(7, 3.6))
    bins = np.linspace(-0.2, 1, 61)
    ax.hist(s[~same & (s >= -0.5)], bins, alpha=0.75, label="different people", color="#c44e52")
    ax.hist(s[same & (s >= -0.5)], bins, alpha=0.75, label="same person", color="#4c72b0")
    for t, st in ((thr, "-"), (cfg["recognition"]["blurry_threshold"], "--")):
        ax.axvline(t, color="k", linestyle=st, linewidth=1)
    ax.set_xlabel("cosine similarity"); ax.set_ylabel("pairs"); ax.legend(frameon=False)
    ax.set_title("LFW pairs: similarity distribution (lines: thresholds 0.40 / 0.35)", fontsize=10)
    fig.tight_layout(); fig.savefig(out / "similarity_histogram.png", dpi=140); plt.close(fig)

    # figure 2: accuracy against face size, with and without the Lanczos pre-upscale
    fig, ax = plt.subplots(figsize=(7, 3.6))
    for tag, style in (("Lanczos upscale", "o-"), ("plain warp", "s--")):
        sub = df[df["condition"].str.contains(tag)].sort_values("face_px")
        ax.plot(sub["face_px"], sub["best_accuracy_%"], style, label=tag)
    ax.set_xlabel("face width in the frame (px)"); ax.set_ylabel("best verification accuracy (%)")
    ax.set_title("Accuracy vs face size", fontsize=10); ax.grid(alpha=0.3); ax.legend(frameon=False)
    fig.tight_layout(); fig.savefig(out / "accuracy_vs_face_size.png", dpi=140); plt.close(fig)

    print(f"\nSaved results.csv, similarity_histogram.png, accuracy_vs_face_size.png in {out}")
