"""Build a demo class out of any video: cluster the faces in it into S001, S002, ...

  python scripts/auto_enroll_from_video.py --video data/samples/classroom.mp4 --batch CSE-A

Faces are sampled from the video, embedded, and grouped with DBSCAN (cosine distance).
Each cluster becomes one "student": its best crops are saved to data/faces/<roll_no>/,
a row is added to students.csv and the embeddings go straight into the database.
This is only for demos on internet footage - real students are enrolled from photos.
"""
import argparse
import shutil
import sys
from pathlib import Path

import cv2
import numpy as np
from sklearn.cluster import DBSCAN

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from src import preprocess as pp
from src.attendance import face_snapshot
from src.config import ROOT, load_config
from src.database import Database
from src.face_engine import FaceEngine


def pick_diverse(embs, quality, k):
    """Best-quality crop first, then the ones least similar to those already chosen."""
    chosen = [int(np.argmax(quality))]
    pool = list(np.argsort(-quality)[:max(k * 4, 12)])
    while len(chosen) < min(k, len(pool)):
        rest = [i for i in pool if i not in chosen]
        sims = embs[rest] @ embs[chosen].T
        chosen.append(rest[int(np.argmin(sims.max(axis=1)))])
    return chosen


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--video", required=True)
    ap.add_argument("--batch", default="CSE-A")
    ap.add_argument("--prefix", default="S", help="roll number prefix")
    ap.add_argument("--every", type=float, default=1.0, help="seconds between sampled frames")
    ap.add_argument("--max-frames", type=int, default=400)
    ap.add_argument("--min-face", type=int, default=28, help="ignore faces smaller than this (px)")
    ap.add_argument("--min-frontal", type=float, default=0.25,
                    help="reject side / back views: minimum eye distance / face width (0 = keep all)")
    ap.add_argument("--eps", type=float, default=0.45, help="DBSCAN radius in cosine distance")
    ap.add_argument("--min-samples", type=int, default=4, help="sightings needed to form a student")
    ap.add_argument("--photos", type=int, default=5, help="crops kept per student")
    ap.add_argument("--keep", action="store_true", help="keep students already in this batch")
    a = ap.parse_args()

    cfg = load_config()
    db = Database(cfg["paths"]["db"])
    db.sync_students_csv(cfg["paths"]["students"])
    engine = FaceEngine(cfg)
    rc = cfg["recognition"]

    cap = cv2.VideoCapture(a.video)
    if not cap.isOpened():
        sys.exit(f"Cannot open {a.video}")
    fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    step = max(1, round(fps * a.every))
    embs, snaps, aligned, quality = [], [], [], []
    n = 0
    while n < a.max_frames:
        ok, frame = cap.read()
        if not ok:
            break
        enhanced, _ = pp.enhance_frame(frame, cfg["preprocess"])
        for f in engine.analyze(enhanced, embed_from=frame):
            if f.score < 0.65 or f.size < a.min_face or f.blur < rc["blurry_below"] / 2:
                continue
            # eye distance relative to face width: ~0.45 frontal, near 0 for a side / back view
            eyes = np.linalg.norm(f.kps[1] - f.kps[0]) / max(f.bbox[2] - f.bbox[0], 1)
            if eyes < a.min_frontal:
                continue
            embs.append(f.embedding)
            snaps.append(face_snapshot(frame, f.bbox))
            aligned.append(f.crop)
            quality.append(f.score * min(f.size, 112) * min(f.blur, 300) ** 0.5)
        n += 1
        print(f"\r  sampled {n} frames, {len(embs)} usable faces", end="", flush=True)
        for _ in range(step - 1):
            if not cap.grab():
                break
    cap.release()
    print()
    if len(embs) < a.min_samples:
        sys.exit("Not enough faces found - try a different video or a smaller --min-face.")

    embs, quality = np.stack(embs), np.array(quality)
    labels = DBSCAN(eps=a.eps, min_samples=a.min_samples, metric="cosine").fit(embs).labels_
    clusters = sorted((c for c in set(labels) if c >= 0), key=lambda c: -(labels == c).sum())
    print(f"  {len(clusters)} people found ({(labels < 0).sum()} faces left unclustered)")

    if not a.keep:                                  # replace the previous demo class of this batch
        for roll in db.students(a.batch)["roll_no"]:
            db.delete_student(roll)
            shutil.rmtree(Path(cfg["paths"]["faces"]) / roll, ignore_errors=True)

    start = len(db.students(a.batch)) + 1
    sheet = []
    for i, c in enumerate(clusters, start):
        idx = np.where(labels == c)[0]
        roll = f"{a.prefix}{i:03d}"
        folder = Path(cfg["paths"]["faces"]) / roll
        shutil.rmtree(folder, ignore_errors=True)
        folder.mkdir(parents=True)
        db.upsert_student(roll, f"Student {i:03d}", a.batch)
        db.clear_embeddings(roll)
        picks = idx[pick_diverse(embs[idx], quality[idx], a.photos)]
        for j, p in enumerate(picks, 1):
            cv2.imwrite(str(folder / f"{j:02d}.jpg"), snaps[p])
            db.add_embedding(roll, embs[p], f"auto:{j:02d}.jpg")
        row = [aligned[p] for p in picks] + [np.zeros((112, 112, 3), np.uint8)] * (a.photos - len(picks))
        row = np.hstack(row)
        cv2.putText(row, f"{roll} ({len(idx)})", (4, 14), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 255, 255), 1)
        sheet.append(row)
        print(f"  {roll}: {len(idx)} sightings, {len(picks)} photos saved")
    db.export_students_csv(cfg["paths"]["students"])

    out = ROOT / "outputs" / "auto_enroll_clusters.jpg"
    out.parent.mkdir(exist_ok=True)
    cols = [np.vstack(sheet[k:k + 12]) for k in range(0, len(sheet), 12)]
    tallest = max(c.shape[0] for c in cols)
    cols = [cv2.copyMakeBorder(c, 0, tallest - c.shape[0], 0, 8, cv2.BORDER_CONSTANT) for c in cols]
    cv2.imwrite(str(out), np.hstack(cols))
    print(f"\nEnrolled {len(clusters)} students into batch {a.batch}. Review sheet: {out}")
