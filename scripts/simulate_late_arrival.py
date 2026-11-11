"""Make a demo clip in which one enrolled student "arrives late".

The student's face is located in the video and that region is blurred out until
--until seconds, so the system first sees them only after that moment. This is a
simulation for demonstrating the Late status on a short clip - say so when presenting it.

  python scripts/simulate_late_arrival.py --video data/samples/seminar_class.mp4 --roll S003 --until 75
"""
import argparse
import sys
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from src.config import load_config
from src.database import Database
from src.face_engine import FaceEngine

if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--video", required=True)
    ap.add_argument("--roll", required=True, help="enrolled student to hide")
    ap.add_argument("--until", type=float, default=75, help="hidden until this second")
    ap.add_argument("--out", help="output file (default: <video>_late.mp4)")
    a = ap.parse_args()

    cfg = load_config()
    db = Database(cfg["paths"]["db"])
    row = db.df("SELECT batch FROM students WHERE roll_no=?", (a.roll,))
    if not len(row):
        sys.exit(f"{a.roll} is not a student.")
    rolls, gallery = db.gallery(row.iloc[0]["batch"])
    mine = gallery[[r == a.roll for r in rolls]]
    if not len(mine):
        sys.exit(f"{a.roll} has no enrolled photos.")
    engine = FaceEngine(cfg)

    cap = cv2.VideoCapture(a.video)
    fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    w, h = int(cap.get(3)), int(cap.get(4))
    hide_frames = int(a.until * fps)

    # 1. where is this student while hidden? sample a frame every 2 s and collect their boxes
    boxes = []
    for idx in range(0, hide_frames, max(1, int(fps * 2))):
        cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
        ok, frame = cap.read()
        if not ok:
            break
        for f in engine.analyze(frame):
            if float((mine @ f.embedding).max()) >= 0.45:
                boxes.append(f.bbox)
    if not boxes:
        sys.exit(f"{a.roll} was not found in the first {a.until:.0f} s of the video.")
    boxes = np.array(boxes)
    x0, y0, x1, y1 = boxes[:, 0].min(), boxes[:, 1].min(), boxes[:, 2].max(), boxes[:, 3].max()
    px, py = 0.5 * (x1 - x0), 0.5 * (y1 - y0)       # padding covers small head movements
    x0, y0 = int(max(0, x0 - px)), int(max(0, y0 - py))
    x1, y1 = int(min(w, x1 + px)), int(min(h, y1 + py))
    print(f"{a.roll} found in {len(boxes)} sampled frames; hiding region x {x0}-{x1}, y {y0}-{y1} "
          f"for the first {a.until:.0f} s")

    # 2. rewrite the video with that region blurred until --until
    out = Path(a.out) if a.out else Path(a.video).with_name(Path(a.video).stem + "_late.mp4")
    writer = cv2.VideoWriter(str(out), cv2.VideoWriter_fourcc(*"mp4v"), fps, (w, h))
    cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
    k = max(31, ((x1 - x0) // 2) | 1)
    idx = 0
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        if idx < hide_frames:
            frame[y0:y1, x0:x1] = cv2.GaussianBlur(frame[y0:y1, x0:x1], (k, k), 0)
        writer.write(frame)
        idx += 1
    writer.release()
    cap.release()
    print(f"Wrote {out} ({idx} frames)")
