"""Drive the Streamlit dashboard headlessly: every page and every action button."""
import shutil
from datetime import datetime
from pathlib import Path

import cv2
import pandas as pd
import pytest
import streamlit as st
import yaml
from streamlit.testing.v1 import AppTest

from src.attendance import enroll_photo
from src.config import load_config
from src.database import Database
from tests.conftest import ROOT, TEST_SET, needs_models, needs_samples

pytestmark = [needs_models, needs_samples]
DAYS = ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"]
PAGES = ["overview", "attendance", "students", "timetable", "reports", "dip", "system"]


@pytest.fixture
def app_cfg(tmp_path, monkeypatch):
    """A private config + data folder; the app is pointed at it through DIP_CONFIG."""
    raw = yaml.safe_load((ROOT / "config.yaml").read_text(encoding="utf-8"))
    raw["paths"] = {k: str(tmp_path / v) for k, v in {
        "db": "t.db", "faces": "faces", "timetable": "timetable.csv", "students": "students.csv",
        "cameras": "cameras.csv", "snapshots": "snapshots", "reports": "reports"}.items()}
    raw["models"]["dir"] = str(ROOT / "models")
    cfg_file = tmp_path / "config.yaml"
    cfg_file.write_text(yaml.safe_dump(raw), encoding="utf-8")
    rows = "".join(f"{d},00:00,23:59,R102,DEMO,Face Lab,Dr. S\n" for d in DAYS)
    (tmp_path / "timetable.csv").write_text("day,start,end,room,batch,subject,faculty\n" + rows)
    (tmp_path / "cameras.csv").write_text("room,source,description\nR102,0,webcam\n")
    (tmp_path / "students.csv").write_text("roll_no,name,batch\nT3,Lin-Manuel Miranda,DEMO\nT2,Joe Biden,DEMO\n")
    monkeypatch.setenv("DIP_CONFIG", str(cfg_file))
    st.cache_resource.clear()
    st.cache_data.clear()
    yield load_config(cfg_file)
    st.cache_resource.clear()


def get_engine():
    from src.face_engine import FaceEngine
    return FaceEngine(load_config(ROOT / "config.yaml"))


def go(at, page):
    at.session_state["_goto"] = page
    at.run()
    assert not at.exception, f"{page}: {at.exception}"
    return at


def launch(page="overview"):
    at = AppTest.from_file(str(ROOT / "app.py"), default_timeout=300).run()
    assert not at.exception, at.exception
    return go(at, page) if page != "overview" else at


def by_label(widgets, label):
    return next(w for w in widgets if w.label.startswith(label))


def texts(at):
    return " | ".join(str(e.value) for kind in ("success", "error", "warning", "info", "markdown", "title", "subheader")
                      for e in getattr(at, kind))


def metrics(at):
    return {m.label: m.value for m in at.metric}


def test_every_page_loads_on_an_empty_installation(app_cfg):
    at = launch()
    assert metrics(at)["Students"] == "2" and metrics(at)["Sessions recorded"] == "0"
    assert "Getting started" in texts(at) and "Face Lab" in str([d.value for d in at.dataframe])
    titles = {}
    for page in PAGES:
        go(at, page)
        titles[page] = at.title[0].value
    assert titles == {"overview": "Overview", "attendance": "Take attendance", "students": "Students",
                      "timetable": "Timetable", "reports": "Reports", "dip": "DIP Lab", "system": "System"}
    go(at, "reports")
    assert "No sessions yet" in texts(at)
    go(at, "system")
    assert metrics(at)["Face models"] == "Ready"


def test_dip_lab_renders_every_topic(app_cfg):
    from ui.dip_lab import TOPICS, UNITS
    assert len(UNITS) == 6 and len(TOPICS) == 20
    db = Database(app_cfg["paths"]["db"])              # a gallery, so the recognition topic has classes
    db.sync_students_csv(app_cfg["paths"]["students"])
    enroll_photo(get_engine(), db, "T3", cv2.imread(str(TEST_SET / "lin-manuel-miranda.png")), "p.png")
    at = launch("dip")
    assert at.title[0].value == "DIP Lab"
    for unit, topics in UNITS.items():
        by_label(at.selectbox, "Unit").set_value(unit).run()
        for topic in topics:
            by_label(at.radio, "Topic").set_value(topic).run()
            assert not at.exception, f"{topic}: {at.exception}"
            assert topic in [s.value for s in at.subheader]
    # controls change the result
    by_label(at.selectbox, "Unit").set_value("Unit 3 · Restoration and colour").run()
    by_label(at.radio, "Topic").set_value("Noise and spatial filtering").run()
    by_label(at.slider, "Noise level").set_value(30.0).run()
    assert not at.exception
    by_label(at.selectbox, "Unit").set_value("Unit 4 · Compression and morphology").run()
    by_label(at.radio, "Topic").set_value("JPEG compression").run()
    by_label(at.slider, "JPEG quality").set_value(5).run()
    assert not at.exception


def test_overview_button_opens_take_attendance(app_cfg):
    at = launch()
    by_label(at.button, "Take attendance").click().run()
    assert not at.exception and at.title[0].value == "Take attendance"
    assert "Face Lab" in texts(at)                       # the timetable slot is resolved for the room


def test_add_student_and_validation(app_cfg):
    at = launch("students")
    by_label(at.text_input, "Roll number").set_value("N 1/x")
    by_label(at.text_input, "Batch").set_value("DEMO")
    at.run()
    by_label(at.button, "Save student").click().run()
    assert not at.exception and "Roll number may only contain" in texts(at)

    by_label(at.text_input, "Roll number").set_value("N100")
    by_label(at.text_input, "Name").set_value("New Student")
    at.run()
    by_label(at.button, "Save student").click().run()
    assert not at.exception and "Saved N100" in texts(at)
    db = Database(app_cfg["paths"]["db"])
    assert "N100" in set(db.students("DEMO")["roll_no"])
    assert "N100" in set(pd.read_csv(app_cfg["paths"]["students"])["roll_no"])   # students.csv kept in sync


def test_roster_search_filters_the_table(app_cfg):
    at = launch("students")
    by_label(at.text_input, "Search").set_value("biden")
    at.run()
    assert not at.exception and "1 of 2 students shown" in " ".join(c.value for c in at.caption)


def test_reenroll_and_delete_student(app_cfg, engine):
    for roll, photo in (("T3", "lin-manuel-miranda.png"), ("T2", "biden.jpg")):
        folder = Path(app_cfg["paths"]["faces"]) / roll
        folder.mkdir(parents=True)
        shutil.copy(TEST_SET / photo, folder / photo)
    at = launch("students")
    by_label(at.button, "Re-enroll everyone").click().run()
    assert not at.exception
    db = Database(app_cfg["paths"]["db"])
    assert dict(zip(db.students()["roll_no"], db.students()["photos"])) == {"T2": 1, "T3": 1}

    by_label(at.selectbox, "Photos of").set_value("T2")
    at.run()
    by_label(at.button, "Delete T2").click().run()
    assert not at.exception
    assert list(db.students()["roll_no"]) == ["T3"]
    assert list(pd.read_csv(app_cfg["paths"]["students"])["roll_no"]) == ["T3"]


def test_save_timetable_and_cameras(app_cfg):
    at = launch("timetable")
    assert metrics(at)["Weekly slots"] == "7"
    assert "Face Lab · DEMO" in str(at.dataframe[0].value)             # week view grid
    by_label(at.segmented_control, "Timetable view").set_value("Edit timetable").run()
    by_label(at.button, "Save timetable").click().run()
    assert not at.exception and "Timetable saved" in texts(at)
    assert len(pd.read_csv(app_cfg["paths"]["timetable"])) == 7
    by_label(at.segmented_control, "Timetable view").set_value("Cameras").run()
    by_label(at.button, "Save cameras").click().run()
    assert not at.exception and "Cameras saved" in texts(at)
    assert list(pd.read_csv(app_cfg["paths"]["cameras"], dtype=str)["room"]) == ["R102"]


def test_run_session_then_reports_then_delete(app_cfg, engine):
    """The whole dashboard journey: run a recorded class, read the reports, delete the session."""
    db = Database(app_cfg["paths"]["db"])
    db.sync_students_csv(app_cfg["paths"]["students"])
    enroll_photo(engine, db, "T3", cv2.imread(str(TEST_SET / "lin-manuel-miranda.png")), "p.png")
    enroll_photo(engine, db, "T2", cv2.imread(str(TEST_SET / "biden.jpg")), "b.jpg")

    at = launch("attendance")
    by_label(at.radio, "Source").set_value("Sample video")
    at.run()
    by_label(at.selectbox, "File").set_value(next(
        p for p in sorted((ROOT / "data" / "samples").rglob("*.mp4")) if p.name == "short_hamilton_clip.mp4"))
    by_label(at.slider, "Sample one frame every").set_value(0.5)
    at.run()
    assert "2 of 2 students enrolled" in texts(at)
    by_label(at.button, "Start session").click().run()
    assert not at.exception, at.exception

    sessions = db.sessions()
    assert len(sessions) == 1 and sessions.iloc[0]["batch"] == "DEMO" and sessions.iloc[0]["frames"] >= 6
    att = db.session_attendance(int(sessions.iloc[0]["id"])).set_index("roll_no")
    assert att.loc["T3", "status"] == "Present" and att.loc["T2", "status"] == "Absent"
    assert sessions.iloc[0]["date"] == datetime.now().strftime("%Y-%m-%d")
    assert any(Path(app_cfg["paths"]["reports"]).glob("session_*.xlsx"))

    # the result panel on the same page
    shown = metrics(at)
    assert shown["Present"] == "1" and shown["Absent"] == "1" and shown["Attendance"] == "50%"
    assert "Download Excel" in [b.label for b in at.get("download_button")]

    go(at, "overview")
    assert "Attendance taken" in str([d.value for d in at.dataframe])      # today's slot is marked
    assert metrics(at)["Sessions recorded"] == "1" and metrics(at)["Average attendance"] == "50%"

    go(at, "students")
    roster = at.dataframe[0].value.set_index("roll_no")["attendance"]      # roster shows attendance %
    assert roster["T2"] == 0.0 and roster["T3"] == 100.0

    go(at, "reports")
    assert "Download full report (Excel)" in [b.label for b in at.get("download_button")]
    assert metrics(at)["Sessions"] == "1" and metrics(at)["Below 75%"] == "1"
    by_label(at.selectbox, "Batch ").set_value("DEMO")
    at.run()
    assert not at.exception
    for view in ("By subject", "Session archive"):
        by_label(at.segmented_control, "Report view").set_value(view).run()
        assert not at.exception, f"{view}: {at.exception}"
    assert metrics(at)["Present"] == "1"                               # the session panel is shown
    by_label(at.button, "Delete session").click().run()
    assert not at.exception and len(db.sessions()) == 0
    assert "No sessions yet" in texts(at)


def test_take_attendance_warns_when_no_class_is_scheduled(app_cfg):
    Path(app_cfg["paths"]["timetable"]).write_text("day,start,end,room,batch,subject,faculty\n")
    at = launch("attendance")
    assert by_label(at.button, "Start session").disabled
    assert "No class is scheduled" in texts(at)
