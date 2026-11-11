from datetime import datetime

import numpy as np
import pandas as pd

from src import timetable as tt
from tests.conftest import SLOT, unit

CSV = """day,start,end,room,batch,subject,faculty
Thursday,9:00,10:00,R101,CSE-A,Networks,Dr. M
thu,10:00,11:00,R101,CSE-A,DIP,Dr. R
Thu,10:00,11:00,R102,CSE-B,OS,Dr. I
Fri,23:00,01:00,R101,CSE-A,Night Lab,Dr. N
"""
THU = datetime(2026, 11, 12)      # a Thursday


def table(tmp_path):
    p = tmp_path / "tt.csv"
    p.write_text(CSV)
    return tt.load_timetable(p)


def at(h, m=0, day=THU):
    return day.replace(hour=h, minute=m)


def test_load_normalises_days_and_times(tmp_path):
    df = table(tmp_path)
    assert list(df["day"]) == ["Thu", "Thu", "Thu", "Fri"]
    assert df.iloc[0]["start"] == "09:00"


def test_find_slot_by_room_and_time(tmp_path):
    df = table(tmp_path)
    assert tt.find_slot(df, "R101", at(10, 30))["subject"] == "DIP"
    assert tt.find_slot(df, "R102", at(10, 30))["batch"] == "CSE-B"     # same time, other room
    assert tt.find_slot(df, "r101", at(9, 15))["subject"] == "Networks"  # room is case-insensitive


def test_slot_boundaries_start_inclusive_end_exclusive(tmp_path):
    df = table(tmp_path)
    assert tt.find_slot(df, "R101", at(10, 0))["subject"] == "DIP"
    assert tt.find_slot(df, "R101", at(9, 59))["subject"] == "Networks"
    assert tt.find_slot(df, "R101", at(11, 0)) is None


def test_no_slot_for_wrong_day_room_or_time(tmp_path):
    df = table(tmp_path)
    assert tt.find_slot(df, "R101", at(10, 30, datetime(2026, 11, 11))) is None    # Wednesday
    assert tt.find_slot(df, "R999", at(10, 30)) is None
    assert tt.find_slot(df, "R101", at(15)) is None


def test_slot_bounds_and_midnight_wrap(tmp_path):
    df = table(tmp_path)
    s, e = tt.slot_bounds(tt.find_slot(df, "R101", at(10, 30)), at(10, 30))
    assert (s, e) == (at(10), at(11))
    fri = datetime(2026, 11, 13, 23, 30)
    s, e = tt.slot_bounds(tt.find_slot(df, "R101", fri), fri)
    assert e.day == 14 and (e - s).total_seconds() == 7200
    sat = datetime(2026, 11, 14, 0, 30)                  # after midnight, still Friday's slot
    slot = tt.find_slot(df, "R101", sat)
    assert slot["subject"] == "Night Lab" and tt.slot_bounds(slot, sat) == (s, e)
    assert tt.find_slot(df, "R101", datetime(2026, 11, 14, 1, 0)) is None


def test_slots_for_day_sorted(tmp_path):
    day = tt.slots_for_day(table(tmp_path), THU)
    assert list(day["start"]) == ["09:00", "10:00", "10:00"]


def test_save_roundtrip_drops_blank_rows(tmp_path):
    df = table(tmp_path)
    df.loc[len(df)] = ["", "", "", "", "", "", ""]
    out = tmp_path / "saved.csv"
    tt.save_timetable(df, out)
    again = tt.load_timetable(out)
    assert len(again) == 4 and list(again.columns) == tt.COLUMNS


def test_missing_files_give_empty_results(tmp_path):
    assert len(tt.load_timetable(tmp_path / "nope.csv")) == 0
    assert tt.load_cameras(tmp_path / "nope.csv") == {}


def test_cameras_numeric_source_becomes_webcam_index(tmp_path):
    p = tmp_path / "cams.csv"
    p.write_text("room,source,description\nR101,0,webcam\nR102,rtsp://cam/1,ip camera\n")
    assert tt.load_cameras(p) == {"R101": 0, "R102": "rtsp://cam/1"}


# ------------------------------------------------------------------ database
def test_students_upsert_and_photo_count(db):
    db.upsert_student("S1", "Asha", "B1")
    db.upsert_student("S1", "Asha K", "B1")             # update, not duplicate
    db.add_embedding("S1", unit(1), "a.jpg")
    db.add_embedding("S1", unit(2), "b.jpg")
    s = db.students()
    assert len(s) == 1 and s.iloc[0]["name"] == "Asha K" and s.iloc[0]["photos"] == 2


def test_gallery_only_contains_the_requested_batch(db):
    db.upsert_student("S1", "A", "B1")
    db.upsert_student("S2", "B", "B2")
    db.add_embedding("S1", unit(1))
    db.add_embedding("S2", unit(2))
    rolls, mat = db.gallery("B1")
    assert rolls == ["S1"] and mat.shape == (1, 512)
    assert np.allclose(mat[0], unit(1), atol=1e-6)       # float32 blob round-trip
    assert db.gallery("nobody")[1].shape == (0, 512)


def test_delete_student_cascades_to_embeddings(db):
    db.upsert_student("S1", "A", "B1")
    db.add_embedding("S1", unit(1))
    db.delete_student("S1")
    assert db.gallery("B1")[0] == []


def test_csv_sync_and_export(db, tmp_path):
    src = tmp_path / "in.csv"
    src.write_text("roll_no,name,batch\nS1,Asha,B1\nS2,,B1\n")
    db.sync_students_csv(src)
    s = db.students("B1")
    assert list(s["roll_no"]) == ["S1", "S2"] and s.iloc[1]["name"] == "S2"   # blank name -> roll
    out = tmp_path / "out.csv"
    db.export_students_csv(out)
    assert list(pd.read_csv(out)["roll_no"]) == ["S1", "S2"]
    assert db.sync_students_csv(tmp_path / "missing.csv") == 0


def test_session_results_and_cascade_delete(db):
    db.upsert_student("S1", "A", "B1")
    sid = db.create_session(SLOT, "2026-11-12", "video.mp4")
    row = {"roll_no": "S1", "status": "Present", "first_seen": "10:00:00", "last_seen": "10:30:00",
           "sightings": 9, "buckets": 6, "best_score": 0.71234, "snapshot": ""}
    unk = {"seen_at": "10:01:00", "sightings": 3, "nearest": "S1", "score": 0.2, "snapshot": "u.jpg"}
    db.save_results(sid, 40, [row], [unk])
    db.save_results(sid, 40, [row], [unk])               # saving twice must not duplicate
    att = db.session_attendance(sid)
    assert len(att) == 1 and att.iloc[0]["name"] == "A" and att.iloc[0]["best_score"] == 0.712
    assert len(db.unknowns(sid)) == 1
    s = db.sessions().iloc[0]
    assert (s["present"], s["late"], s["absent"], s["frames"]) == (1, 0, 0, 40)
    assert len(db.attendance_long()) == 1
    db.delete_session(sid)
    assert len(db.attendance_long()) == 0 and len(db.unknowns(sid)) == 0
