"""Apply names from a CSV (roll_no,name) to the students in the database and students.csv.

  python scripts/apply_names.py                       # data/demo_names.csv
  python scripts/apply_names.py my_class_list.csv

The demo classes are built from public footage of people whose identities are unknown, so
data/demo_names.csv holds invented names (initial + surname) for presentation only.
Run this again after auto_enroll_from_video.py, which creates students as "Student 001", ...
"""
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from src.config import ROOT, load_config
from src.database import Database

if __name__ == "__main__":
    source = Path(sys.argv[1]) if len(sys.argv) > 1 else ROOT / "data" / "demo_names.csv"
    names = pd.read_csv(source, dtype=str).dropna()
    cfg = load_config()
    db = Database(cfg["paths"]["db"])
    db.sync_students_csv(cfg["paths"]["students"])
    known = db.students().set_index("roll_no")
    changed = 0
    for r in names.itertuples():
        if r.roll_no in known.index and known.loc[r.roll_no, "name"] != r.name.strip():
            db.upsert_student(r.roll_no, r.name.strip(), known.loc[r.roll_no, "batch"])
            changed += 1
    db.export_students_csv(cfg["paths"]["students"])
    missing = sorted(set(names["roll_no"]) - set(known.index))
    print(f"{changed} names updated from {source.name}" + (f"; not enrolled: {', '.join(missing)}" if missing else ""))
