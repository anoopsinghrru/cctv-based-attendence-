"""Run attendance for one class session.

Recorded video (virtual clock - the video is treated as starting at --start):
  python scripts/run_session.py --room R101 --start "2026-11-12 10:00" --video data/samples/classroom.mp4

Live (webcam / RTSP from data/cameras.csv, real clock, runs until the slot ends):
  python scripts/run_session.py --room R101 --live
  python scripts/run_session.py --room R101 --live --source rtsp://user:pass@ip:554/stream1
"""
import argparse
import sys
from datetime import datetime, timedelta
from pathlib import Path

import cv2

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from src import timetable as tt
from src.attendance import SessionProcessor, export_session, run_live, run_video
from src.config import load_config
from src.database import Database
from src.face_engine import FaceEngine

if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--room", required=True)
    ap.add_argument("--start", help='"YYYY-MM-DD HH:MM" - when the recording started (default: now)')
    ap.add_argument("--video", help="recorded video file")
    ap.add_argument("--live", action="store_true", help="use the room's camera from cameras.csv")
    ap.add_argument("--source", help="override camera: webcam index or RTSP url")
    ap.add_argument("--every", type=float, help="seconds between sampled frames")
    ap.add_argument("--timelapse", type=float,
                    help="time-lapse recording: real seconds between consecutive frames of the file")
    ap.add_argument("--late-after", type=float,
                    help="minutes after the start that count as Late (default: config, 10)")
    ap.add_argument("--duration", type=float, help="live mode: stop after this many seconds (for testing)")
    ap.add_argument("--show", action="store_true", help="show the annotated preview window")
    ap.add_argument("--save-video", help="write the annotated frames to this .mp4")
    ap.add_argument("--no-enhance", action="store_true", help="skip the DIP enhancement stage")
    ap.add_argument("--batch", help="no timetable slot? run ad hoc for this batch")
    ap.add_argument("--subject", default="Ad hoc session")
    a = ap.parse_args()
    if not a.video and not a.live:
        ap.error("give --video FILE or --live")

    cfg = load_config()
    if a.no_enhance:
        cfg["preprocess"]["enabled"] = False
    if a.late_after is not None:
        cfg["attendance"]["late_after_min"] = a.late_after
    when = datetime.strptime(a.start, "%Y-%m-%d %H:%M") if a.start else datetime.now().replace(microsecond=0)
    table = tt.load_timetable(cfg["paths"]["timetable"])
    slot = tt.find_slot(table, a.room, when)
    if slot is None and a.batch:
        slot = {"day": tt.DAYS[when.weekday()], "start": when.strftime("%H:%M"),
                "end": (when + timedelta(hours=1)).strftime("%H:%M"), "room": a.room,
                "batch": a.batch, "subject": a.subject, "faculty": ""}
    if slot is None:
        today = tt.slots_for_day(table, when)
        today = today[today["room"].str.lower() == a.room.lower()]
        print(f"No class is scheduled in {a.room} at {when:%a %Y-%m-%d %H:%M}.")
        print(today.to_string(index=False) if len(today) else "(no slots in this room on that day)")
        sys.exit(1)
    print(f"Slot : {slot['day']} {slot['start']}-{slot['end']}  {slot['room']}  "
          f"batch {slot['batch']}  {slot['subject']}  ({slot['faculty']})")

    db = Database(cfg["paths"]["db"])
    db.sync_students_csv(cfg["paths"]["students"])
    engine = FaceEngine(cfg)
    source = a.video or (a.source if a.source else tt.load_cameras(cfg["paths"]["cameras"]).get(a.room, 0))
    if isinstance(source, str) and source.isdigit():
        source = int(source)
    proc = SessionProcessor(engine, db, cfg, slot, when, source=str(source))
    enrolled = int((proc.students["photos"] > 0).sum())
    print(f"Batch: {len(proc.students)} students, {enrolled} enrolled, models on {engine.device}")
    if not enrolled:
        print("Nobody in this batch is enrolled - run scripts/enroll.py first.")

    writer = None
    state = {"stop": False}

    def on_frame(img, progress, t):
        global writer
        print(f"\r  {t:%H:%M:%S}  frame {proc.frames:4d}  recognised {len(proc.seen)}/{len(proc.students)}"
              f"  unknown {len(proc.unknown)}  {progress * 100:5.1f}%", end="", flush=True)
        if a.save_video:
            if writer is None:
                Path(a.save_video).parent.mkdir(parents=True, exist_ok=True)
                writer = cv2.VideoWriter(a.save_video, cv2.VideoWriter_fourcc(*"mp4v"), 2,
                                         (img.shape[1], img.shape[0]))
            writer.write(img)
        if a.show:
            try:
                scale = min(1.0, 1280 / img.shape[1])
                cv2.imshow("Attendance (q = stop)", cv2.resize(img, None, fx=scale, fy=scale))
                if cv2.waitKey(1) & 0xFF == ord("q"):
                    state["stop"] = True
            except cv2.error:
                a.show = False
                print("\n  (this OpenCV build has no GUI - use --save-video or the Streamlit app for a preview)")

    try:
        if a.video:
            result = run_video(proc, a.video, when, a.every, on_frame, lambda: state["stop"],
                               frame_seconds=a.timelapse)
        else:
            import time
            deadline = time.time() + a.duration if a.duration else float("inf")
            result = run_live(proc, source, a.every, on_frame,
                              lambda: state["stop"] or time.time() >= deadline)
    except KeyboardInterrupt:
        result = proc.finalize()
    except IOError as e:                # source could not be opened: leave no empty session behind
        db.delete_session(proc.session_id)
        sys.exit(str(e))
    finally:
        if writer is not None:
            writer.release()
        if a.show:
            cv2.destroyAllWindows()

    print(f"\n\nSession {proc.session_id}: {proc.frames} frames processed\n")
    print(result.drop(columns=["snapshot"]).to_string(index=False))
    counts = result["status"].value_counts()
    print(f"\nPresent {counts.get('Present', 0)}  Late {counts.get('Late', 0)}  "
          f"Absent {counts.get('Absent', 0)}  |  unknown faces saved: {len(proc.unknown)}")
    xlsx, csv = export_session(db, cfg, proc.session_id)
    print(f"Report   : {xlsx}\n           {csv}\nSnapshots: {proc.snap_dir}")
