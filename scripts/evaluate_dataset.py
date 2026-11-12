"""Score detection and recognition against the annotations of the classroom dataset.

The Kaggle "Classroom Student Engagement Dataset" labels every student in every frame with a
head-and-shoulders box and a behaviour class. That gives ground truth for "a person is here",
so this script reports, per behaviour:

  detected    a face was found inside the labelled box (full pipeline, and full-frame only)
  identified  that face was matched to an enrolled student of the class

Identification is reported separately for the recording a class was enrolled from and for
the other recordings of that class.

  python scripts/evaluate_dataset.py
"""
import re
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
from src.database import Database
from src.face_engine import FaceEngine

DATA = ROOT / "data" / "samples" / "kaggle" / "engagement" / "dataset"
NAMES = ["handrise", "look_forward", "read", "sleep", "stand", "turn_head", "using_device", "write"]
# recording prefix -> (batch, recording used for enrolment)
CLASSES = {"owais-class": ("BSCS-A", "owais-class-1"), "taimoor-class": ("BSCS-B", "taimoor-class-2")}


def assign(boxes, faces):
    """Give each labelled box at most one face whose centre lies inside it (nearest to the box top)."""
    taken, out = set(), [None] * len(boxes)
    for i, (x0, y0, x1, y1) in enumerate(boxes):
        best, dist = None, 1e9
        for j, f in enumerate(faces):
            cx, cy = (f.bbox[0] + f.bbox[2]) / 2, (f.bbox[1] + f.bbox[3]) / 2
            if j not in taken and x0 <= cx <= x1 and y0 <= cy <= y1:
                d = abs(cx - (x0 + x1) / 2) + abs(cy - (y0 + (y1 - y0) * 0.35))
                if d < dist:
                    best, dist = j, d
        if best is not None:
            taken.add(best)
            out[i] = best
    return out


if __name__ == "__main__":
    if not DATA.exists():
        sys.exit(f"Dataset not found at {DATA}")
    cfg = load_config()
    rc = cfg["recognition"]
    db = Database(cfg["paths"]["db"])
    engine = FaceEngine(cfg)
    galleries = {}
    for prefix, (batch, _) in CLASSES.items():
        rolls, mat = db.gallery(batch)
        uniq = sorted(set(rolls))
        galleries[prefix] = (uniq, np.array([uniq.index(r) for r in rolls], dtype=int), mat)

    rows = []
    images = sorted((DATA / "images").glob("*.jpg"))
    for n, path in enumerate(images, 1):
        label = DATA / "labels" / (path.stem + ".txt")
        if not label.exists():
            continue
        img = cv2.imread(str(path))
        h, w = img.shape[:2]
        recording = re.match(r"(.+?)_mp4", path.name).group(1)
        prefix = next((p for p in CLASSES if recording.startswith(p)), None)
        boxes, classes = [], []
        for line in label.read_text().strip().splitlines():
            c, cx, cy, bw, bh = line.split()[:5]
            cx, cy, bw, bh = float(cx) * w, float(cy) * h, float(bw) * w, float(bh) * h
            boxes.append((cx - bw / 2, cy - bh / 2, cx + bw / 2, cy + bh / 2))
            classes.append(NAMES[int(c)])
        enhanced, _ = pp.enhance_frame(img, cfg["preprocess"])
        faces = engine.analyze(enhanced, embed_from=img)
        plain = assign(boxes, engine.analyze(img, tiled=False, reject_blurry=False))
        hit = assign(boxes, faces)
        for i, cls in enumerate(classes):
            f = faces[hit[i]] if hit[i] is not None else None
            identified = None
            if prefix and len(galleries[prefix][2]):
                identified = False
                if f is not None:
                    uniq, idx, mat = galleries[prefix]
                    per = np.full(len(uniq), -1.0)
                    np.maximum.at(per, idx, f.embedding @ mat.T)
                    order = np.argsort(-per)
                    thr = rc["blurry_threshold"] if f.blur < rc["blurry_below"] else rc["match_threshold"]
                    identified = bool(per[order[0]] >= thr and per[order[0]] - per[order[1]] >= rc["margin"])
            rows.append({"recording": recording, "behaviour": cls, "detected": f is not None,
                         "detected_full_frame_only": plain[i] is not None, "identified": identified,
                         "enrolment_recording": bool(prefix and recording == CLASSES[prefix][1]),
                         "face_px": f.size if f is not None else np.nan})
        if n % 60 == 0:
            print(f"  {n}/{len(images)} frames", flush=True)

    df = pd.DataFrame(rows)
    out = ROOT / "outputs" / "evaluation"
    out.mkdir(parents=True, exist_ok=True)
    df.to_csv(out / "dataset_boxes.csv", index=False)

    def table(d):
        other = d[d["identified"].notna() & ~d["enrolment_recording"]]
        enrol = d[d["identified"].notna() & d["enrolment_recording"]]
        return pd.Series({"labelled people": len(d),
                          "detected %": round(100 * d["detected"].mean(), 1),
                          "detected, full frame only %": round(100 * d["detected_full_frame_only"].mean(), 1),
                          "identified, other recordings %": round(100 * other["identified"].astype(bool).mean(), 1) if len(other) else np.nan,
                          "identified, enrolment recording %": round(100 * enrol["identified"].astype(bool).mean(), 1) if len(enrol) else np.nan,
                          "median face px": round(float(d["face_px"].median()), 0)})
    by = df.groupby("behaviour").apply(table, include_groups=False).sort_values("labelled people", ascending=False)
    by.loc["ALL"] = table(df)
    by.to_csv(out / "dataset_by_behaviour.csv")
    rec = df.groupby("recording").apply(table, include_groups=False)
    rec.to_csv(out / "dataset_by_recording.csv")
    pd.set_option("display.width", 220)
    print("\nBy behaviour\n" + by.to_string())
    print("\nBy recording\n" + rec.to_string())

    plot = by.drop(index="ALL")
    fig, ax = plt.subplots(figsize=(8, 3.8))
    x = np.arange(len(plot))
    ax.bar(x - 0.2, plot["detected %"], 0.4, label="face detected", color="#2F5BEA")
    ax.bar(x + 0.2, plot["identified, other recordings %"], 0.4, label="identified (other recordings)", color="#12B76A")
    ax.set_xticks(x, [f"{b}\n(n={int(n)})" for b, n in zip(plot.index, plot["labelled people"])], fontsize=8)
    ax.set_ylim(0, 100); ax.set_ylabel("% of labelled people"); ax.grid(axis="y", alpha=0.3); ax.legend(frameon=False)
    ax.set_title("Classroom dataset: detection and identification by labelled behaviour", fontsize=10)
    fig.tight_layout(); fig.savefig(out / "dataset_by_behaviour.png", dpi=140); plt.close(fig)
    print(f"\nSaved dataset_by_behaviour.csv / .png, dataset_by_recording.csv in {out}")
