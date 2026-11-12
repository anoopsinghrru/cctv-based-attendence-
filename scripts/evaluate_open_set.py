"""Open-set identification on LFW: how low can the match threshold go for a class-sized gallery?

Attendance is not 1:1 verification: every face is compared with a whole batch, and most
faces in a room may belong to people who are not enrolled. This script measures, for a
gallery of N enrolled people:

  TAR    enrolled person, probe accepted as the right student
  misID  enrolled person, probe accepted as a WRONG student
  FAR    person not in the gallery, probe accepted as somebody

for a grid of thresholds and margins and for degraded probes (small, blurred).
Gallery photos are clean (as enrolment photos are); probes are a different photo.

  python scripts/evaluate_open_set.py
"""
import sys
from pathlib import Path

import cv2
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from scripts.evaluate import CANVAS, place
from src.config import ROOT, load_config
from src.face_engine import FaceEngine

LFW = ROOT / "data" / "samples" / "lfw" / "lfw_home" / "lfw_funneled"
CONDITIONS = [("clean, 110 px", 110, 0), ("small, 28 px", 28, 0), ("small, 20 px", 20, 0),
              ("blurred, 40 px", 40, 1.6), ("soft, 30 px", 30, 1.0)]


def embed(engine, path, px, sigma):
    img = place(cv2.imread(str(path)), px)
    if sigma:
        img = cv2.GaussianBlur(img, (0, 0), sigma)
    faces = engine.analyze(img, tiled=False, reject_blurry=False)
    if not faces:
        return np.zeros(512, np.float32)            # undetected: can never be accepted
    c = np.array([CANVAS / 2, CANVAS / 2])
    return min(faces, key=lambda f: np.linalg.norm((f.bbox[:2] + f.bbox[2:]) / 2 - c)).embedding


if __name__ == "__main__":
    if not LFW.exists():
        sys.exit("LFW is not downloaded: run scripts/evaluate.py once first.")
    rng = np.random.default_rng(0)
    people = sorted(p for p in LFW.iterdir() if p.is_dir())
    multi = [p for p in people if len(list(p.glob("*.jpg"))) >= 2]
    single = [p for p in people if len(list(p.glob("*.jpg"))) == 1]
    known = [multi[i] for i in rng.permutation(len(multi))[:200]]
    strangers = [single[i] for i in rng.permutation(len(single))[:300]]
    engine = FaceEngine(load_config())

    gallery = np.stack([embed(engine, sorted(p.glob("*.jpg"))[0], 110, 0) for p in known])
    rows = []
    for label, px, sigma in CONDITIONS:
        gen = np.stack([embed(engine, sorted(p.glob("*.jpg"))[1], px, sigma) for p in known])
        imp = np.stack([embed(engine, next(p.glob("*.jpg")), px, sigma) for p in strangers])
        for n in (15, 40):
            acc = {}
            for _ in range(40):                                  # random classes of n students
                pick = rng.permutation(len(known))[:n]
                g = gallery[pick]
                for kind, probes, truth in (("gen", gen[pick], np.arange(n)), ("imp", imp, None)):
                    s = probes @ g.T
                    order = np.argsort(-s, axis=1)
                    best = s[np.arange(len(s)), order[:, 0]]
                    gap = best - s[np.arange(len(s)), order[:, 1]]
                    for t in (0.25, 0.28, 0.30, 0.32, 0.35, 0.40):
                        for m in (0.06, 0.10):
                            ok = (best >= t) & (gap >= m)
                            a = acc.setdefault((t, m), {"tar": [], "mis": [], "far": []})
                            if kind == "gen":
                                a["tar"].append((ok & (order[:, 0] == truth)).mean())
                                a["mis"].append((ok & (order[:, 0] != truth)).mean())
                            else:
                                a["far"].append(ok.mean())
            for (t, m), a in acc.items():
                rows.append({"probe": label, "class_size": n, "threshold": t, "margin": m,
                             "TAR_%": round(100 * np.mean(a["tar"]), 2), "misID_%": round(100 * np.mean(a["mis"]), 3),
                             "FAR_%": round(100 * np.mean(a["far"]), 3)})
        print(f"done: {label}", flush=True)
    df = pd.DataFrame(rows)
    out = ROOT / "outputs" / "evaluation"
    out.mkdir(parents=True, exist_ok=True)
    df.to_csv(out / "open_set.csv", index=False)
    pd.set_option("display.width", 220)
    for n in (15, 40):
        for m in (0.06, 0.10):
            sub = df[(df.class_size == n) & (df.margin == m)]
            print(f"\nclass of {n}, margin {m}:  TAR % / misID % / FAR %")
            t = sub.assign(cell=sub["TAR_%"].map("{:.1f}".format) + " / " + sub["misID_%"].map("{:.2f}".format) +
                           " / " + sub["FAR_%"].map("{:.2f}".format)).pivot(index="probe", columns="threshold", values="cell")
            print(t.reindex([c[0] for c in CONDITIONS]).to_string())
    print(f"\nSaved {out / 'open_set.csv'}")
