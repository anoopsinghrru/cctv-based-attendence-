"""Enroll students: students.csv -> database, data/faces/<roll_no>/*.jpg -> embeddings.

  python scripts/enroll.py              # everyone
  python scripts/enroll.py --roll S001  # one student
Use 3-5 clear photos per student (front + slight left/right turns work best).
"""
import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from src.attendance import enroll_folder
from src.config import load_config
from src.database import Database
from src.face_engine import FaceEngine

if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--roll", help="enroll only this roll number")
    a = ap.parse_args()

    cfg = load_config()
    db = Database(cfg["paths"]["db"])
    db.sync_students_csv(cfg["paths"]["students"])
    engine = FaceEngine(cfg)
    print(f"Models loaded on {engine.device}")

    students = db.students()
    known = set(students["roll_no"])
    for folder in sorted(Path(cfg["paths"]["faces"]).iterdir()):
        if folder.is_dir() and folder.name not in known:
            print(f"  ! {folder.name}: photo folder has no row in students.csv - skipped")
    for r in students.itertuples():
        if a.roll and r.roll_no != a.roll:
            continue
        if not (Path(cfg["paths"]["faces"]) / r.roll_no).is_dir():
            print(f"  - {r.roll_no} {r.name}: no photo folder (kept {r.photos} existing embeddings)")
            continue
        ok, total = enroll_folder(engine, db, cfg["paths"]["faces"], r.roll_no)
        flag = " " if ok else "!"
        print(f"  {flag} {r.roll_no} {r.name} [{r.batch}]: {ok}/{total} photos enrolled")
    print("\n" + db.students().to_string(index=False))
