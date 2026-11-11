"""Streamlit dashboard:  streamlit run app.py"""
import copy
from datetime import datetime, timedelta
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
import streamlit as st

from src import timetable as tt
from src.attendance import (SessionProcessor, enroll_folder, enroll_photo, excel_bytes,
                            export_session, run_live, run_video, session_info)
from src.config import ROOT, load_config
from src.database import Database
from src.face_engine import FaceEngine

st.set_page_config(page_title="CCTV Attendance", page_icon="🎓", layout="wide")
CFG = load_config()
P = CFG["paths"]


@st.cache_resource(show_spinner="Loading face models ...")
def get_engine():
    return FaceEngine(copy.deepcopy(CFG))


@st.cache_resource
def get_db():
    db = Database(P["db"])
    db.sync_students_csv(P["students"])
    return db


db = get_db()
table = tt.load_timetable(P["timetable"])


def to_rgb(img, max_w=1280):
    if img.shape[1] > max_w:
        img = cv2.resize(img, None, fx=max_w / img.shape[1], fy=max_w / img.shape[1])
    return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)


def color_status(v):
    return {"Present": "background-color:#d6f5d6;color:#0b3d0b", "Late": "background-color:#fff0c2;color:#4d3800",
            "Absent": "background-color:#f8d7da;color:#58151c"}.get(v, "")


def finish(proc):
    proc.finalize()
    export_session(db, proc.cfg, proc.session_id)
    st.session_state["last_session"] = proc.session_id
    st.session_state.pop("proc", None)


# A session interrupted by the Stop button (which reruns the script) is closed here.
if "proc" in st.session_state:
    _p = st.session_state["proc"]
    if not _p.finalized:
        if _p.frames:
            finish(_p)
        else:
            db.delete_session(_p.session_id)
            st.session_state.pop("proc", None)


def show_session(session_id, key=""):
    info = session_info(db, session_id)
    if not info:
        st.info("Session not found.")
        return
    att = db.session_attendance(session_id)
    st.markdown(f"**Session {session_id}** · {info['date']} {info['start']}–{info['end']} · room "
                f"{info['room']} · batch {info['batch']} · {info['subject']} · {info['frames']} frames")
    counts = att["status"].value_counts()
    c = st.columns(4)
    c[0].metric("Present", int(counts.get("Present", 0)))
    c[1].metric("Late", int(counts.get("Late", 0)))
    c[2].metric("Absent", int(counts.get("Absent", 0)))
    c[3].metric("Attendance", f"{100 * (att['status'] != 'Absent').mean():.0f}%" if len(att) else "–")
    st.dataframe(att.drop(columns=["snapshot"]).style.map(color_status, subset=["status"]),
                 width="stretch", hide_index=True)
    xlsx, csv = export_session(db, CFG, session_id)
    d = st.columns([1, 1, 4])
    d[0].download_button("Download Excel", Path(xlsx).read_bytes(), Path(xlsx).name, key=f"x{key}{session_id}")
    d[1].download_button("Download CSV", Path(csv).read_bytes(), Path(csv).name, key=f"c{key}{session_id}")
    snaps = [(r.snapshot, f"{r.roll_no} · {r.status}") for r in att.itertuples()
             if r.snapshot and Path(r.snapshot).exists()]
    if snaps:
        with st.expander(f"Best snapshot per student ({len(snaps)})", expanded=True):
            cols = st.columns(8)
            for i, (path, cap) in enumerate(snaps):
                cols[i % 8].image(path, caption=cap, width="stretch")
    unk = db.unknowns(session_id)
    unk = unk[unk["snapshot"].map(lambda p: Path(p).exists())] if len(unk) else unk
    if len(unk):
        with st.expander(f"Unknown faces ({len(unk)})"):
            cols = st.columns(10)
            for i, r in enumerate(unk.sort_values("sightings", ascending=False).head(60).itertuples()):
                cols[i % 10].image(r.snapshot, caption=f"×{r.sightings} ~{r.nearest} {r.score:.2f}",
                                   width="stretch")


st.title("CCTV Face-Recognition Attendance")
tab_dash, tab_run, tab_students, tab_time, tab_reports = st.tabs(
    ["Dashboard", "Run Session", "Students", "Timetable", "Reports"])

# =============================================================== Dashboard
with tab_dash:
    now = datetime.now()
    students = db.students()
    sessions = db.sessions()
    long = db.attendance_long()
    c = st.columns(4)
    c[0].metric("Students", len(students))
    c[1].metric("Enrolled (have photos)", int((students["photos"] > 0).sum()) if len(students) else 0)
    c[2].metric("Sessions recorded", len(sessions))
    c[3].metric("Overall attendance", f"{100 * (long['status'] != 'Absent').mean():.0f}%" if len(long) else "–")

    st.subheader(f"Today's timetable · {now:%A %d %b %Y}")
    today = tt.slots_for_day(table, now)
    if len(today):
        done = set(zip(sessions["date"], sessions["room"], sessions["start"])) if len(sessions) else set()
        hhmm = now.strftime("%H:%M")

        def state(r):
            if (now.strftime("%Y-%m-%d"), r["room"], r["start"]) in done:
                return "Attendance taken"
            return "Running now" if r["start"] <= hhmm < r["end"] else ("Upcoming" if hhmm < r["start"] else "Missed")
        today = today.assign(status=today.apply(state, axis=1)).drop(columns=["day"])
        st.dataframe(today, width="stretch", hide_index=True)
    else:
        st.info("No classes scheduled today.")

    if len(long):
        left, right = st.columns(2)
        pct = long.assign(attended=(long["status"] != "Absent") * 100.0)
        left.subheader("Attendance % by subject")
        left.bar_chart(pct.groupby("subject")["attended"].mean().round(1), y_label="%")
        right.subheader("Attendance % by batch")
        right.bar_chart(pct.groupby("batch")["attended"].mean().round(1), y_label="%")
        st.subheader("Recent sessions")
        st.dataframe(sessions.head(10).drop(columns=["source"]), width="stretch", hide_index=True)

# =============================================================== Run Session
with tab_run:
    rooms = sorted(table["room"].unique()) or ["R101"]
    left, right = st.columns([1, 2])
    with left:
        room = st.selectbox("Room", rooms)
        mode = st.radio("Source", ["Upload video", "Sample video", "Webcam", "RTSP stream"], horizontal=True)
        live = mode in ("Webcam", "RTSP stream")
        video_path, source = None, None
        if mode == "Upload video":
            up = st.file_uploader("Class recording", type=["mp4", "avi", "mov", "mkv"])
            if up:
                updir = ROOT / "outputs" / "uploads"
                updir.mkdir(parents=True, exist_ok=True)
                video_path = updir / Path(up.name).name
                if not video_path.exists() or video_path.stat().st_size != up.size:
                    video_path.write_bytes(up.getbuffer())
        elif mode == "Sample video":
            samples = sorted((ROOT / "data" / "samples").rglob("*.mp4"))
            if samples:
                video_path = st.selectbox("File", samples, format_func=lambda p: str(p.relative_to(ROOT)))
            else:
                st.info("No samples yet - run scripts/get_sample_video.py")
        elif mode == "Webcam":
            source = int(st.number_input("Webcam index", 0, 10, 0))
        else:
            default = tt.load_cameras(P["cameras"]).get(room, "")
            source = st.text_input("RTSP url", value=default if isinstance(default, str) else "")
        if live:
            when = datetime.now().replace(microsecond=0)
            st.caption(f"Live mode uses the real clock ({when:%a %H:%M}) and runs until the slot ends.")
        else:
            d = st.date_input("Recording date", value=datetime.now().date())
            day = tt.slots_for_day(table, datetime.combine(d, datetime.min.time()))
            day = day[day["room"] == room].reset_index(drop=True)
            when = datetime.combine(d, datetime.min.time())
            if len(day):
                i = st.selectbox("Class", range(len(day)), format_func=lambda k: "{start}–{end} · {batch} · {subject}"
                                 .format(**day.iloc[k].to_dict()))
                offset = st.number_input("Recording began this many minutes after class start", 0, 600, 0)
                begin = datetime.strptime(day.iloc[i]["start"], "%H:%M").time()
                when = datetime.combine(d, begin) + timedelta(minutes=int(offset))
        every = st.slider("Sample one frame every (s)", 0.5, 10.0, float(CFG["attendance"]["sample_every_sec"]), 0.5)
        timelapse = 0.0
        if not live:
            timelapse = st.number_input("Time-lapse: real seconds per video frame (0 = normal video)",
                                        0.0, 600.0, 0.0, 1.0)
        late_after = st.number_input("Late if first seen after (min)", 0.5, 120.0,
                                     float(CFG["attendance"]["late_after_min"]), 0.5)
        enhance = st.checkbox("DIP enhancement (gamma / denoise / CLAHE / sharpen)", value=True)
        tiled = st.checkbox("Tiled detection (small back-row faces)", value=CFG["detection"]["tiled"])
        slot = tt.find_slot(table, room, when)
        if slot:
            n = db.students(slot["batch"])
            st.success(f"**{slot['subject']}** · batch **{slot['batch']}** · {slot['start']}–{slot['end']} · "
                       f"{slot['faculty']}\n\n{int((n['photos'] > 0).sum())} of {len(n)} students enrolled")
        else:
            st.warning(f"No class is scheduled in {room} on {when:%A}" + (f" at {when:%H:%M}" if live else "") +
                       ". Pick another date or room, or add the slot in the Timetable tab.")
        ready = slot is not None and (video_path is not None or (live and source not in (None, "")))
        start = st.button("Start session", type="primary", disabled=not ready)
    with right:
        if start:
            engine = get_engine()
            engine.cfg["preprocess"]["enabled"] = enhance
            engine.dcfg["tiled"] = tiled
            engine.cfg["attendance"]["late_after_min"] = late_after
            proc = SessionProcessor(engine, db, engine.cfg, slot, when, source=str(video_path or source))
            st.session_state["proc"] = proc
            st.button("Stop and save", key="stop")      # clicking reruns the script -> session is finalised
            status, bar, view = st.empty(), st.progress(0.0), st.empty()

            def on_frame(img, progress, ts):
                view.image(to_rgb(img), width="stretch")
                bar.progress(progress)
                status.markdown(f"`{ts:%H:%M:%S}` · frame {proc.frames} · recognised "
                                f"**{len(proc.seen)}/{len(proc.students)}** · unknown {len(proc.unknown)} · "
                                f"{engine.device}")
            try:
                if live:
                    run_live(proc, source, every, on_frame)
                else:
                    run_video(proc, video_path, when, every, on_frame, frame_seconds=timelapse or None)
                export_session(db, engine.cfg, proc.session_id)
                st.session_state["last_session"] = proc.session_id
                st.session_state.pop("proc", None)
                bar.progress(1.0)
                st.rerun()              # refresh the Dashboard and Reports tabs with the new session
            except IOError as e:
                db.delete_session(proc.session_id)
                st.session_state.pop("proc", None)
                st.error(str(e))
        if "last_session" in st.session_state:
            st.divider()
            show_session(st.session_state["last_session"], key="run")
        elif not start:
            st.info("Choose a room, a source and a time, then press **Start session**. The annotated "
                    "preview appears here.")

# =============================================================== Students
with tab_students:
    students = db.students()
    left, right = st.columns([3, 2])
    with left:
        batches = ["All"] + sorted(students["batch"].unique())
        pick = st.selectbox("Batch", batches)
        view = students if pick == "All" else students[students["batch"] == pick]
        st.dataframe(view, width="stretch", hide_index=True, height=360)
        if st.button("Re-enroll everyone from data/faces"):
            engine = get_engine()
            with st.spinner("Embedding photos ..."):
                done = [enroll_folder(engine, db, P["faces"], r)[0] for r in students["roll_no"]
                        if (Path(P["faces"]) / r).is_dir()]
            st.success(f"Re-enrolled {len(done)} students ({sum(done)} photos).")
            st.rerun()
        if len(view):
            sel = st.selectbox("Photos of", view["roll_no"],
                               format_func=lambda r: f"{r} · {students.set_index('roll_no').loc[r, 'name']}")
            photos = sorted((Path(P["faces"]) / sel).glob("*.*")) if sel else []
            if photos:
                cols = st.columns(6)
                for i, p in enumerate(photos[:12]):
                    cols[i % 6].image(str(p), width="stretch")
            if st.button(f"Delete {sel}"):
                db.delete_student(sel)
                db.export_students_csv(P["students"])
                st.rerun()
    with right:
        st.subheader("Add / enroll a student")
        roll = st.text_input("Roll number").strip()
        name = st.text_input("Name").strip()
        batch = st.text_input("Batch", value=pick if pick != "All" else "").strip()
        ups = st.file_uploader("Photos (3-5 clear face photos)", type=["jpg", "jpeg", "png"],
                               accept_multiple_files=True)
        shot = st.camera_input("Or take a photo") if st.toggle("Use camera") else None
        if st.button("Save student", type="primary", disabled=not (roll and batch)):
            safe = "".join(ch for ch in roll if ch.isalnum() or ch in "-_")
            if safe != roll:
                st.error("Roll number may only contain letters, digits, - and _.")
            else:
                engine = get_engine()
                db.upsert_student(roll, name or roll, batch)
                folder = Path(P["faces"]) / roll
                folder.mkdir(parents=True, exist_ok=True)
                files = [(f.name, f.getvalue()) for f in ups or []]
                if shot:
                    files.append((f"camera_{datetime.now():%Y%m%d_%H%M%S}.jpg", shot.getvalue()))
                good = 0
                for fname, data in files:
                    img = cv2.imdecode(np.frombuffer(data, np.uint8), cv2.IMREAD_COLOR)
                    if img is None or enroll_photo(engine, db, roll, img, fname) is None:
                        st.warning(f"No face found in {fname} - skipped.")
                        continue
                    (folder / Path(fname).name).write_bytes(data)
                    good += 1
                db.export_students_csv(P["students"])
                st.success(f"Saved {roll}. {good} new photo(s) enrolled.")

# =============================================================== Timetable
with tab_time:
    st.caption("day = Mon..Sun, times as HH:MM (24 h). A session is matched to the slot whose room and time "
               "contain it; that slot's batch is the only gallery searched.")
    edited = st.data_editor(table, num_rows="dynamic", width="stretch", hide_index=True, key="tt_edit",
                            column_config={"day": st.column_config.SelectboxColumn(options=tt.DAYS)})
    if st.button("Save timetable", type="primary"):
        try:
            tt.save_timetable(edited, P["timetable"])
            st.success("Timetable saved.")
            st.rerun()
        except (ValueError, IndexError):
            st.error("Could not save: check that every row has a day and HH:MM start / end times.")
    st.subheader("Cameras")
    cam_path = Path(P["cameras"])
    cams = pd.read_csv(cam_path, dtype=str).fillna("") if cam_path.exists() else \
        pd.DataFrame(columns=["room", "source", "description"])
    cams_edit = st.data_editor(cams, num_rows="dynamic", width="stretch", hide_index=True, key="cam_edit")
    if st.button("Save cameras"):
        cams_edit.fillna("").to_csv(cam_path, index=False)
        st.success("Cameras saved.")

# =============================================================== Reports
with tab_reports:
    long = db.attendance_long()
    if not len(long):
        st.info("No sessions yet. Run one from the Run Session tab.")
    else:
        f = st.columns(2)
        fb = f[0].selectbox("Batch ", ["All"] + sorted(long["batch"].unique()))
        fs = f[1].selectbox("Subject", ["All"] + sorted(long["subject"].unique()))
        data = long[((long["batch"] == fb) | (fb == "All")) & ((long["subject"] == fs) | (fs == "All"))]

        def summarise(by):
            g = data.groupby(by)["status"].value_counts().unstack(fill_value=0)
            for s in ("Present", "Late", "Absent"):
                if s not in g:
                    g[s] = 0
            g["classes"] = g[["Present", "Late", "Absent"]].sum(axis=1)
            g["attendance %"] = (100 * (g["Present"] + g["Late"]) / g["classes"]).round(1)
            return g[["classes", "Present", "Late", "Absent", "attendance %"]].reset_index()
        per_student = summarise(["roll_no", "name", "batch"])
        per_subject = summarise(["subject", "batch"])
        per_subject.insert(2, "sessions", per_subject.apply(
            lambda r: data[(data["subject"] == r["subject"]) & (data["batch"] == r["batch"])]["session_id"].nunique(),
            axis=1).values)
        per_subject = per_subject.rename(columns={"classes": "student-classes"})
        a, b = st.columns(2)
        a.subheader("Per student")
        a.dataframe(per_student, width="stretch", hide_index=True,
                    column_config={"attendance %": st.column_config.ProgressColumn(min_value=0, max_value=100,
                                                                                  format="%.1f%%")})
        b.subheader("Per subject")
        b.dataframe(per_subject, width="stretch", hide_index=True)
        sessions = db.sessions()
        st.download_button(
            "Download full report (Excel)",
            excel_bytes({"Per student": per_student, "Per subject": per_subject,
                         "Sessions": sessions, "All records": data}),
            f"attendance_report_{datetime.now():%Y%m%d}.xlsx", type="primary")
        st.subheader("Sessions")
        st.dataframe(sessions, width="stretch", hide_index=True)
        sid = st.selectbox("Open session", sessions["id"],
                           format_func=lambda i: "{id} · {date} {start} · {room} · {batch} · {subject}".format(
                               **sessions.set_index("id", drop=False).loc[i].to_dict()))
        show_session(int(sid), key="rep")
        if st.button(f"Delete session {sid}"):
            db.delete_session(int(sid))
            if st.session_state.get("last_session") == int(sid):
                st.session_state.pop("last_session")
            st.rerun()
