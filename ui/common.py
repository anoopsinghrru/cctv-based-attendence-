"""Shared state and helpers for the dashboard pages."""
import base64
import copy
from datetime import datetime
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
import streamlit as st

from src import timetable as tt
from src.attendance import export_session, session_info
from src.config import ROOT, load_config
from src.database import Database
from src.face_engine import FaceEngine
from ui import theme


@st.cache_resource
def cfg():
    return load_config()


@st.cache_resource(show_spinner="Loading face models ...")
def engine():
    return FaceEngine(copy.deepcopy(cfg()))


@st.cache_resource
def db():
    d = Database(cfg()["paths"]["db"])
    d.sync_students_csv(cfg()["paths"]["students"])
    return d


def paths():
    return cfg()["paths"]


def timetable():
    return tt.load_timetable(paths()["timetable"])


def goto(page):
    """Jump to another page (handled in app.py)."""
    st.session_state["_goto"] = page
    st.rerun()


def to_rgb(img, max_w=1280):
    if img.shape[1] > max_w:
        img = cv2.resize(img, None, fx=max_w / img.shape[1], fy=max_w / img.shape[1])
    return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)


@st.cache_data(show_spinner=False)
def _thumb(path, mtime, size=72):
    img = cv2.imdecode(np.fromfile(path, np.uint8), cv2.IMREAD_COLOR)
    if img is None:
        return None
    h, w = img.shape[:2]
    s = min(h, w)
    img = cv2.resize(img[(h - s) // 2:(h + s) // 2, (w - s) // 2:(w + s) // 2], (size, size), interpolation=cv2.INTER_AREA)
    return "data:image/jpeg;base64," + base64.b64encode(cv2.imencode(".jpg", img)[1].tobytes()).decode()


def photos_of(roll):
    folder = Path(paths()["faces"]) / str(roll)
    return sorted(p for p in folder.glob("*.*") if p.suffix.lower() in (".jpg", ".jpeg", ".png", ".bmp")) \
        if folder.is_dir() else []


def thumbnail(roll):
    """Small square preview of a student's first photo, as a data URI (None if no photo)."""
    files = photos_of(roll)
    return _thumb(str(files[0]), files[0].stat().st_mtime) if files else None


def summarise(long, by):
    """Classes / Present / Late / Absent / attendance % grouped by the given columns."""
    cols = list(by) + ["classes", "Present", "Late", "Absent", "attendance"]
    if not len(long):
        return pd.DataFrame(columns=cols)
    g = long.groupby(list(by))["status"].value_counts().unstack(fill_value=0)
    for s in ("Present", "Late", "Absent"):
        if s not in g:
            g[s] = 0
    g["classes"] = g[["Present", "Late", "Absent"]].sum(axis=1)
    g["attendance"] = (100 * (g["Present"] + g["Late"]) / g["classes"]).round(1)
    return g.reset_index()[cols]


def slot_states(day_slots, sessions, now):
    """Today's slots with a status: Attendance taken / Running now / Upcoming / Not recorded."""
    done = set(zip(sessions["date"], sessions["room"], sessions["start"])) if len(sessions) else set()
    hhmm, today = now.strftime("%H:%M"), now.strftime("%Y-%m-%d")

    def state(r):
        if (today, r["room"], r["start"]) in done:
            return "Attendance taken"
        if r["start"] <= hhmm < r["end"]:
            return "Running now"
        return "Upcoming" if hhmm < r["start"] else "Not recorded"
    return day_slots.assign(status=day_slots.apply(state, axis=1)) if len(day_slots) else day_slots.assign(status=[])


def session_label(row):
    return f"#{row['id']} · {row['date']} {row['start']} · {row['room']} · {row['batch']} · {row['subject']}"


def show_session(session_id, key=""):
    """The result panel of one session: summary, split, register, downloads, snapshots."""
    d = db()
    info = session_info(d, session_id)
    if not info:
        st.info("This session no longer exists.")
        return
    att = d.session_attendance(session_id)
    counts = att["status"].value_counts().to_dict()
    rate = 100 * (att["status"] != "Absent").mean() if len(att) else 0

    st.markdown(f"<div class='av-eyebrow'>Session #{session_id}</div>", unsafe_allow_html=True)
    st.subheader(f"{info['subject']} · {info['batch']}")
    st.markdown(
        f"<span class='av-chip'>{info['date']}</span><span class='av-chip'>{info['start']}–{info['end']}</span>"
        f"<span class='av-chip'>Room {info['room']}</span>"
        + (f"<span class='av-chip'>{info['faculty']}</span>" if info["faculty"] else "")
        + f"<span class='av-chip'>{info['frames']} frames analysed</span>", unsafe_allow_html=True)

    m = st.columns(4)
    m[0].metric("Present", int(counts.get("Present", 0)))
    m[1].metric("Late", int(counts.get("Late", 0)))
    m[2].metric("Absent", int(counts.get("Absent", 0)))
    m[3].metric("Attendance", f"{rate:.0f}%" if len(att) else "–")

    left, right = st.columns([1, 2.4])
    with left:
        if len(att):
            theme.donut(counts)
        xlsx, csv = export_session(d, cfg(), session_id)
        st.download_button("Download Excel", Path(xlsx).read_bytes(), Path(xlsx).name, icon=":material/download:",
                           key=f"x{key}{session_id}", width="stretch")
        st.download_button("Download CSV", Path(csv).read_bytes(), Path(csv).name, icon=":material/download:",
                           key=f"c{key}{session_id}", width="stretch")
    with right:
        table = att.drop(columns=["snapshot"]).rename(columns={
            "roll_no": "Roll no", "name": "Name", "status": "Status", "first_seen": "First seen",
            "last_seen": "Last seen", "sightings": "Sightings", "buckets": "Time buckets", "best_score": "Best match"})
        st.dataframe(theme.style_status(table, "Status"), width="stretch", hide_index=True,
                     height=min(460, 38 * (len(table) + 1) + 3),
                     column_config={"Best match": st.column_config.NumberColumn(format="%.2f")})

    snaps = [(r.snapshot, f"{r.roll_no} · {r.status}") for r in att.itertuples()
             if r.snapshot and Path(r.snapshot).exists()]
    unk = d.unknowns(session_id)
    unk = unk[unk["snapshot"].map(lambda p: Path(p).exists())] if len(unk) else unk
    if snaps or len(unk):
        tabs = st.tabs([f"Recognised students ({len(snaps)})", f"Unknown faces ({len(unk)})"])
        with tabs[0]:
            if snaps:
                cols = st.columns(8)
                for i, (path, cap) in enumerate(snaps):
                    cols[i % 8].image(path, caption=cap, width="stretch")
            else:
                st.caption("No enrolled student was recognised in this session.")
        with tabs[1]:
            if len(unk):
                st.caption("People seen in the room who are not enrolled in this batch. "
                           "The caption shows how often each was seen.")
                cols = st.columns(10)
                for i, r in enumerate(unk.sort_values("sightings", ascending=False).head(60).itertuples()):
                    cols[i % 10].image(r.snapshot, caption=f"seen ×{r.sightings}", width="stretch")
            else:
                st.caption("No unknown faces were recorded.")


def now():
    return datetime.now()


SAMPLES = ROOT / "data" / "samples"
