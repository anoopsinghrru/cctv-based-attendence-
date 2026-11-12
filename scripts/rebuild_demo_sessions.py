"""Delete every recorded session and re-run the standard demo sessions.

  python scripts/rebuild_demo_sessions.py          # asks before deleting
  python scripts/rebuild_demo_sessions.py --yes

Students and enrolments are kept; only sessions, their reports and snapshots are rebuilt.
Sessions whose video is missing are skipped.
"""
import shutil
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from src.config import load_config
from src.database import Database

S = "data/samples/"
K = S + "kaggle_clips/"
# room, recording start, video, extra arguments
DEMO = [
    # 1 frame-per-second clips rebuilt from the frame dataset: every frame is analysed automatically
    ("R201", "2026-11-12 09:00", K + "owais-class-2.mp4", []),
    ("R201", "2026-11-13 11:00", K + "owais-class-3.mp4", []),
    ("R202", "2026-11-12 09:00", K + "taimoor-class-3.mp4", []),
    ("R202", "2026-11-13 11:00", K + "taimoor-class-4.mp4", []),
    ("R202", "2026-11-16 10:00", K + "taimoor-class-5.mp4", []),
    # three recordings joined into one 40-minute time-lapse class; one student's face is hidden for
    # the first 13 minutes (simulate_late_arrival.py) to show Late at the real 10-minute threshold
    ("R202", "2026-11-20 11:00", K + "composite_long_class_late.mp4", ["--timelapse", "40"]),
    ("R103", "2026-11-12 12:00", S + "cctv_exam_class.mp4", ["--every", "1"]),
    ("R101", "2026-11-12 10:00", S + "seminar_class_late.mp4", ["--every", "3", "--late-after", "1"]),
    ("R102", "2026-11-12 11:00", S + "test_set/hamilton_clip.mp4", ["--every", "1"]),
]

if __name__ == "__main__":
    cfg = load_config()
    db = Database(cfg["paths"]["db"])
    n = len(db.sessions())
    if "--yes" not in sys.argv and n:
        if input(f"Delete {n} recorded sessions and rebuild the demo? [y/N] ").strip().lower() != "y":
            sys.exit("Nothing changed.")
    for sid in db.sessions()["id"]:
        db.delete_session(int(sid))
    db.conn.execute("DELETE FROM sqlite_sequence WHERE name = 'sessions'")     # numbering restarts at 1
    db.conn.commit()
    for folder in (cfg["paths"]["snapshots"], cfg["paths"]["reports"]):
        shutil.rmtree(folder, ignore_errors=True)
        Path(folder).mkdir(parents=True, exist_ok=True)
    db.conn.close()

    for room, start, video, extra in DEMO:
        if not (ROOT / video).exists():
            print(f"skip  {video} (file missing)")
            continue
        out = ROOT / "outputs" / "annotated" / (Path(video).stem + ".mp4")
        cmd = [sys.executable, str(ROOT / "scripts" / "run_session.py"), "--room", room, "--start", start,
               "--video", str(ROOT / video), "--save-video", str(out), *extra]
        r = subprocess.run(cmd, capture_output=True, text=True, encoding="utf-8", errors="replace")
        line = next((ln for ln in r.stdout.splitlines() if ln.startswith("Present ")), None)
        print(f"{'ok  ' if r.returncode == 0 and line else 'FAIL'}  {room}  {start}  {Path(video).name:28s} "
              f"{line or r.stderr.strip().splitlines()[-1] if (line or r.stderr.strip()) else ''}")
