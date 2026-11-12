"""Process one class session (video file or live stream) into Present / Late / Absent."""
import io
import time
from datetime import datetime, timedelta
from pathlib import Path

import cv2
import numpy as np
import pandas as pd

from . import preprocess as pp
from .heatmap import add_boxes, heat_overlay, new_heat
from .face_engine import draw_faces
from .timetable import slot_bounds

GREEN, RED = (80, 220, 80), (60, 60, 235)
HEAT_CELL = 8       # frame pixels per heat-map cell


def enroll_photo(engine, db, roll_no, img, source=""):
    """Embed the largest face of an enrollment photo and store it. Returns the Face or None."""
    face = engine.largest_face(img)
    if face is not None:
        db.add_embedding(roll_no, face.embedding, source)
    return face


def enroll_folder(engine, db, faces_dir, roll_no):
    """(Re)build one student's embeddings from data/faces/<roll_no>/*. Returns (ok, total)."""
    folder = Path(faces_dir) / roll_no
    files = sorted(p for p in folder.glob("*") if p.suffix.lower() in (".jpg", ".jpeg", ".png", ".bmp"))
    db.clear_embeddings(roll_no)
    ok = 0
    for p in files:
        img = cv2.imdecode(np.fromfile(str(p), np.uint8), cv2.IMREAD_COLOR)
        if img is not None and enroll_photo(engine, db, roll_no, img, p.name) is not None:
            ok += 1
    return ok, len(files)


def face_snapshot(frame, bbox, pad=0.35, min_side=160):
    h, w = frame.shape[:2]
    bw, bh = bbox[2] - bbox[0], bbox[3] - bbox[1]
    x0, y0 = int(max(0, bbox[0] - pad * bw)), int(max(0, bbox[1] - pad * bh))
    x1, y1 = int(min(w, bbox[2] + pad * bw)), int(min(h, bbox[3] + pad * bh))
    crop = frame[y0:y1, x0:x1]
    if crop.size and min(crop.shape[:2]) < min_side:
        crop = pp.lanczos_upscale(crop, min_side / min(crop.shape[:2]))
    return crop


class SessionProcessor:
    """Feed it frames with timestamps; call finalize() to get the attendance table."""

    def __init__(self, engine, db, cfg, slot, when, source=""):
        self.engine, self.db, self.cfg, self.slot = engine, db, cfg, slot
        self.acfg, self.rcfg = cfg["attendance"], cfg["recognition"]
        self.slot_start, self.slot_end = slot_bounds(slot, when)
        self.students = db.students(slot["batch"])
        self.names = dict(zip(self.students["roll_no"], self.students["name"]))
        # only this batch's students are loaded: smaller gallery, fewer wrong matches
        rolls, self.gallery = db.gallery(slot["batch"])
        self.rolls = sorted(set(rolls))
        self.roll_idx = np.array([self.rolls.index(r) for r in rolls], dtype=int)
        self.session_id = db.create_session(slot, self.slot_start.strftime("%Y-%m-%d"), source)
        self.snap_dir = Path(cfg["paths"]["snapshots"]) / f"session_{self.session_id}"
        (self.snap_dir / "unknown").mkdir(parents=True, exist_ok=True)
        self.seen = {}        # roll -> {"times": [...], "best": float, "snapshot": str}
        self.unknown = []     # dicts: emb, seen_at, sightings, nearest, score, snapshot, size
        self.frames = 0
        self.first_t = self.last_t = None
        self.finalized = False
        self.heat = None      # where faces were seen, accumulated on a coarse grid (pseudo-colour map)
        self.backdrop = None  # a small copy of the latest frame to draw the heat map on

    # -------------------------------------------------------------- matching
    def match(self, faces):
        """Per face: (roll_no or None, best similarity, nearest roll_no).

        A face is accepted when its best student beats the threshold (lower for blurry
        crops) AND beats the second-best student by a margin. Each student is given to
        at most one face per frame - the one with the highest similarity.
        """
        if not faces:
            return []
        if not len(self.gallery):
            return [(None, 0.0, "")] * len(faces)
        sims = np.stack([f.embedding for f in faces]) @ self.gallery.T        # faces x photos
        per_student = np.full((len(faces), len(self.rolls)), -1.0, np.float32)
        np.maximum.at(per_student.T, self.roll_idx, sims.T)                   # best photo per student
        order = np.argsort(-per_student, axis=1)
        result, owner = [], {}
        for i, f in enumerate(faces):
            best = float(per_student[i, order[i, 0]])
            second = float(per_student[i, order[i, 1]]) if len(self.rolls) > 1 else -1.0
            roll = self.rolls[order[i, 0]]
            thr = self.rcfg["blurry_threshold"] if f.blur < self.rcfg["blurry_below"] \
                else self.rcfg["match_threshold"]
            ok = best >= thr and best - second >= self.rcfg["margin"]
            result.append([roll if ok else None, best, roll])
            if ok:
                if roll in owner and result[owner[roll]][1] >= best:
                    result[i][0] = None                 # another face owns this student
                else:
                    if roll in owner:
                        result[owner[roll]][0] = None
                    owner[roll] = i
        return [tuple(r) for r in result]

    # -------------------------------------------------------------- per frame
    def process_frame(self, frame, t):
        """Enhance -> detect -> recognise one frame taken at datetime `t`. Returns annotated frame."""
        enhanced, info = pp.enhance_frame(frame, self.cfg["preprocess"])
        # detect on the enhanced frame, embed from the original pixels (see FaceEngine.analyze)
        faces = self.engine.analyze(enhanced, embed_from=frame)
        matches = self.match(faces)
        self.frames += 1
        self.first_t = self.first_t or t
        self.last_t = t
        h, w = frame.shape[:2]
        if self.heat is None or self.heat.shape != new_heat(frame.shape, HEAT_CELL).shape:
            self.heat = new_heat(frame.shape, HEAT_CELL)
        add_boxes(self.heat, [f.bbox for f in faces], HEAT_CELL)
        self.backdrop = cv2.resize(frame, (min(w, 960), round(h * min(w, 960) / w)), interpolation=cv2.INTER_AREA)
        labels, colors = [], []
        for f, (roll, score, nearest) in zip(faces, matches):
            if roll:
                rec = self.seen.setdefault(roll, {"times": [], "best": -1.0, "snapshot": ""})
                rec["times"].append(t)
                if score > rec["best"]:
                    rec["best"] = score
                    path = self.snap_dir / f"{roll}.jpg"
                    cv2.imwrite(str(path), face_snapshot(frame, f.bbox))
                    rec["snapshot"] = str(path)
                labels.append(f"{roll} {score:.2f}")
                colors.append(GREEN)
            else:
                self._add_unknown(frame, f, t, nearest, score)
                labels.append("?")
                colors.append(RED)
        out = draw_faces(frame, faces, labels, colors)
        text = (f"{t:%Y-%m-%d %H:%M:%S}  {self.slot['room']}  {self.slot['batch']}  "
                f"{self.slot['subject']}  faces {len(faces)}  recognised {len(self.seen)}"
                f"/{len(self.students)}  gamma {info['gamma']:.2f}")
        s = max(out.shape[1] / 1400, 0.45)
        cv2.rectangle(out, (0, 0), (out.shape[1], int(34 * s)), (0, 0, 0), -1)
        cv2.putText(out, text, (8, int(24 * s)), cv2.FONT_HERSHEY_SIMPLEX, 0.6 * s,
                    (255, 255, 255), max(1, round(1.5 * s)))
        return out

    def _add_unknown(self, frame, face, t, nearest, score):
        if face.score < 0.6:
            return
        sims = [float(u["emb"] @ face.embedding) for u in self.unknown]
        if sims and max(sims) > 0.40:               # same stranger seen again?
            u = self.unknown[int(np.argmax(sims))]
            # keep a running mean embedding: one noisy first sighting no longer defines the person
            mean = u["emb"] * u["sightings"] + face.embedding
            u["emb"] = mean / max(np.linalg.norm(mean), 1e-9)
            u["sightings"] += 1
            if face.size > u["size"]:
                u["size"] = face.size
                cv2.imwrite(u["snapshot"], face_snapshot(frame, face.bbox))
            return
        if len(self.unknown) >= 300:
            return
        path = str(self.snap_dir / "unknown" / f"unknown_{len(self.unknown) + 1:03d}.jpg")
        cv2.imwrite(path, face_snapshot(frame, face.bbox))
        self.unknown.append({"emb": face.embedding, "seen_at": t.strftime("%H:%M:%S"), "sightings": 1,
                             "nearest": nearest, "score": round(score, 3), "snapshot": path,
                             "size": face.size})

    def _merge_unknowns(self):
        """Merge unknown entries that turned out to be the same person, and drop one-off
        detections, so the unknown list approximates a head count instead of a sighting log."""
        import os
        import shutil

        def discard(path):
            try:
                os.remove(path)
            except OSError:             # locked by a viewer or a virus scanner: leave the stray file
                pass

        def move(src, dst):
            """Replace dst with src. Windows can briefly lock a file that was just written."""
            for attempt in range(4):
                try:
                    os.replace(src, dst)
                    return
                except OSError:
                    time.sleep(0.15 * (attempt + 1))
            try:
                shutil.copyfile(src, dst)
            except OSError:
                return
            discard(src)

        merged = []
        for u in sorted(self.unknown, key=lambda x: -x["sightings"]):
            home = next((m for m in merged if float(m["emb"] @ u["emb"]) > 0.40), None)
            if home is None:
                merged.append(u)
                continue
            mean = home["emb"] * home["sightings"] + u["emb"] * u["sightings"]
            home["emb"] = mean / max(np.linalg.norm(mean), 1e-9)
            home["sightings"] += u["sightings"]
            home["seen_at"] = min(home["seen_at"], u["seen_at"])
            if u["size"] > home["size"]:            # keep the larger, clearer snapshot
                move(u["snapshot"], home["snapshot"])
                home["size"] = u["size"]
            else:
                discard(u["snapshot"])
        keep = []
        for u in merged:
            if u["sightings"] >= 2 or self.frames < 5:
                keep.append(u)
            else:
                discard(u["snapshot"])
        self.unknown = keep

    def save_heatmap(self):
        """Pseudo-colour map of where faces were seen during the session (hot = seen most often)."""
        if self.heat is None or self.backdrop is None or not self.heat.any():
            return None
        smooth = cv2.GaussianBlur(self.heat, (0, 0), 2.0)
        path = self.snap_dir / "heatmap.jpg"
        cv2.imwrite(str(path), heat_overlay(self.backdrop, smooth))
        return str(path)

    # -------------------------------------------------------------- decision
    def finalize(self):
        """Apply the Present / Late / Absent rules and store the session."""
        a = self.acfg
        self._merge_unknowns()
        start = max(self.slot_start, self.first_t) if self.first_t else self.slot_start
        end = min(self.slot_end, self.last_t) if self.last_t else self.slot_end
        span = max((end - start).total_seconds(), 1.0)
        n = int(a["time_buckets"])
        rows = []
        for roll in self.students["roll_no"]:
            rec = self.seen.get(roll)
            row = {"roll_no": roll, "name": self.names.get(roll, roll), "status": "Absent",
                   "first_seen": "", "last_seen": "", "sightings": 0, "buckets": 0,
                   "best_score": None, "snapshot": ""}
            if rec:
                times = rec["times"]
                buckets = {min(n - 1, max(0, int((t - start).total_seconds() / span * n))) for t in times}
                row.update(first_seen=times[0].strftime("%H:%M:%S"), last_seen=times[-1].strftime("%H:%M:%S"),
                           sightings=len(times), buckets=len(buckets),
                           best_score=round(rec["best"], 3), snapshot=rec["snapshot"])
                if len(times) >= a["min_sightings"] and len(buckets) / n >= a["min_bucket_fraction"]:
                    late = times[0] > start + timedelta(minutes=a["late_after_min"])
                    row["status"] = "Late" if late else "Present"
            rows.append(row)
        self.db.save_results(self.session_id, self.frames, rows, self.unknown)
        self.save_heatmap()
        self.finalized = True
        return pd.DataFrame(rows)


# ------------------------------------------------------------------ runners
def run_video(proc, path, start_dt, every=None, on_frame=None, should_stop=None, frame_seconds=None):
    """Recorded-video mode: frame k is stamped start_dt + k / fps (a virtual clock).

    `frame_seconds`: for time-lapse recordings (NVR exports, frame datasets) - the real time
    between two consecutive frames of the file. Overrides the file's own frame rate.
    """
    cap = cv2.VideoCapture(str(path))
    if not cap.isOpened():
        raise IOError(f"Cannot open video: {path}")
    fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    if not 0.01 <= fps <= 240:
        fps = 25.0
    sparse = fps <= 2 and not frame_seconds         # frame dataset / time-lapse export
    if frame_seconds:
        fps = 1.0 / frame_seconds
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    step = max(1, round(fps * (every or proc.acfg["sample_every_sec"])))
    if sparse and every is None:
        step = 1                                    # frames are already sparse samples: use them all
    idx = 0
    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            t = start_dt + timedelta(seconds=idx / fps)
            if t >= proc.slot_end or (should_stop and should_stop()):
                break
            annotated = proc.process_frame(frame, t)
            if on_frame:
                on_frame(annotated, min(1.0, idx / total) if total else 0.0, t)
            for _ in range(step - 1):               # skip to the next sample
                if not cap.grab():
                    break
            idx += step
    finally:
        cap.release()
    return proc.finalize()


def video_fps(path):
    """Frame rate of a video file (0 if it cannot be read)."""
    cap = cv2.VideoCapture(str(path))
    fps = cap.get(cv2.CAP_PROP_FPS) if cap.isOpened() else 0
    cap.release()
    return float(fps or 0)


def open_capture(source):
    """Open a webcam index or stream url. Some Windows webcams only open through DirectShow,
    so a webcam index is retried with that backend when the default one fails."""
    if isinstance(source, str) and source.lower().startswith("rtsp"):
        # RTSP over TCP: no smeared frames from lost UDP packets (can be overridden in the environment)
        import os
        os.environ.setdefault("OPENCV_FFMPEG_CAPTURE_OPTIONS", "rtsp_transport;tcp")
    cap = cv2.VideoCapture(source)
    if not cap.isOpened() and isinstance(source, int):
        cap.release()
        cap = cv2.VideoCapture(source, cv2.CAP_DSHOW)
    return cap


def run_live(proc, source, every=None, on_frame=None, should_stop=None):
    """Live mode (webcam index or RTSP url): real clock, runs until the slot ends."""
    cap = open_capture(source)
    if not cap.isOpened():
        hint = (" Check that the webcam is plugged in, enabled in Device Manager and not in use "
                "by another app.") if isinstance(source, int) else ""
        raise IOError(f"Cannot open camera / stream: {source}.{hint}")
    every = every or proc.acfg["sample_every_sec"]
    total = max((proc.slot_end - datetime.now()).total_seconds(), 1.0)
    began = time.time()
    misses = 0
    try:
        while datetime.now() < proc.slot_end and not (should_stop and should_stop()):
            ok, frame = cap.read()
            if not ok:
                misses += 1
                time.sleep(0.5)
                if misses % 10 == 0:                # stream dropped: reconnect
                    cap.release()
                    cap = open_capture(source)
                continue
            misses = 0
            tick = time.time()
            annotated = proc.process_frame(frame, datetime.now())
            if on_frame:
                on_frame(annotated, min(1.0, (time.time() - began) / total), datetime.now())
            while time.time() - tick < every:       # drain the buffer so the next frame is fresh
                if should_stop and should_stop():
                    break
                if not cap.grab():
                    time.sleep(0.05)
    finally:
        cap.release()
    return proc.finalize()


# ------------------------------------------------------------------ export
def session_info(db, session_id):
    s = db.df("SELECT * FROM sessions WHERE id=?", (session_id,))
    return s.iloc[0].to_dict() if len(s) else {}


def excel_bytes(sheets):
    """{sheet name: DataFrame} -> xlsx file content."""
    buf = io.BytesIO()
    with pd.ExcelWriter(buf, engine="openpyxl") as xl:
        for name, df in sheets.items():
            df.to_excel(xl, sheet_name=name[:31], index=False)
            ws = xl.sheets[name[:31]]
            for i, col in enumerate(df.columns, 1):
                width = max([len(str(col))] + [len(str(v)) for v in df[col].head(200)]) + 2
                ws.column_dimensions[ws.cell(1, i).column_letter].width = min(width, 50)
    return buf.getvalue()


def export_session(db, cfg, session_id):
    """Write outputs/reports/session_<id>_... .xlsx and .csv. Returns both paths."""
    info = session_info(db, session_id)
    att = db.session_attendance(session_id).drop(columns=["snapshot"])
    counts = att["status"].value_counts()
    summary = pd.DataFrame({
        "field": ["session", "date", "time", "room", "batch", "subject", "faculty", "source",
                  "frames processed", "Present", "Late", "Absent", "attendance %"],
        "value": [session_id, info["date"], f"{info['start']}-{info['end']}", info["room"], info["batch"],
                  info["subject"], info["faculty"], info["source"], info["frames"],
                  int(counts.get("Present", 0)), int(counts.get("Late", 0)), int(counts.get("Absent", 0)),
                  round(100 * (att["status"] != "Absent").mean(), 1) if len(att) else 0]})
    unknown = db.unknowns(session_id).drop(columns=["id", "session_id"], errors="ignore")
    stem = f"session_{session_id}_{info['batch']}_{info['subject']}_{info['date']}".replace(" ", "_")
    stem = "".join(ch for ch in stem if ch.isalnum() or ch in "_-")
    base = Path(cfg["paths"]["reports"]) / stem
    xlsx, csv = base.with_suffix(".xlsx"), base.with_suffix(".csv")
    xlsx.write_bytes(excel_bytes({"Attendance": att, "Summary": summary, "Unknown faces": unknown}))
    att.to_csv(csv, index=False)
    return str(xlsx), str(csv)
