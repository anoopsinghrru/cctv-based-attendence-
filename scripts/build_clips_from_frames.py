"""Turn a folder of extracted video frames back into one clip per source recording.

Frame datasets (e.g. Roboflow / Kaggle exports) name their images
<recording>_mp4-<frame number>_jpg.rf.<hash>.jpg. Images are grouped by <recording>,
ordered by frame number and written as a 1 frame-per-second video, so run_session.py
can treat each recording as a class (use --every 1).

  python scripts/build_clips_from_frames.py --frames data/samples/kaggle/engagement/dataset/images --out data/samples/kaggle_clips
"""
import argparse
import re
import sys
from collections import defaultdict
from pathlib import Path

import cv2
import numpy as np

if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--frames", required=True, help="folder with the frame images")
    ap.add_argument("--out", required=True, help="folder for the clips")
    ap.add_argument("--min-frames", type=int, default=15, help="skip recordings with fewer frames")
    a = ap.parse_args()

    groups = defaultdict(list)
    for p in Path(a.frames).iterdir():
        m = re.match(r"(.+?)(?:_mp4)?[-_](\d+)_(?:jpg|png)", p.name)
        if m and p.suffix.lower() in (".jpg", ".jpeg", ".png"):
            groups[m.group(1)].append((int(m.group(2)), p))
    if not groups:
        sys.exit("No frames named <recording>-<number>_jpg... found.")
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    for name, items in sorted(groups.items(), key=lambda kv: -len(kv[1])):
        if len(items) < a.min_frames:
            print(f"  skip {name}: only {len(items)} frames")
            continue
        items.sort()
        writer, size = None, None
        for _, p in items:
            img = cv2.imdecode(np.fromfile(str(p), np.uint8), cv2.IMREAD_COLOR)
            if img is None:
                continue
            if writer is None:
                size = (img.shape[1], img.shape[0])
                writer = cv2.VideoWriter(str(out / f"{name}.mp4"), cv2.VideoWriter_fourcc(*"mp4v"), 1, size)
            writer.write(img if (img.shape[1], img.shape[0]) == size else cv2.resize(img, size))
        writer.release()
        print(f"  {name}.mp4: {len(items)} frames, {size[0]}x{size[1]}")
