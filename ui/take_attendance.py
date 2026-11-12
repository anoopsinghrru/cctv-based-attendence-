"""Take attendance: pick the class and the video source, watch it run, read the result."""
from datetime import datetime, timedelta
from pathlib import Path

import streamlit as st

from src import timetable as tt
from src.attendance import SessionProcessor, export_session, run_live, run_video, video_fps
from src.config import ROOT
from ui import common as c
from ui import theme

SOURCES = ["Upload video", "Sample video", "Webcam", "RTSP stream"]


def render():
    d, conf, P = c.db(), c.cfg(), c.paths()
    table = c.timetable()
    theme.header("Take attendance", "Choose the class and the camera or recording. The timetable decides which "
                                    "batch is expected, and only those students are searched.")

    left, right = st.columns([1, 1.9])
    with left:
        with theme.card("setup"):
            st.subheader("Session setup")
            rooms = sorted(table["room"].unique()) or ["R101"]
            room = st.selectbox("Room", rooms)
            mode = st.radio("Source", SOURCES, horizontal=True)
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
                samples = sorted(c.SAMPLES.rglob("*.mp4"))
                if samples:
                    video_path = st.selectbox("File", samples, format_func=lambda p: str(p.relative_to(c.SAMPLES)))
                else:
                    st.info("No sample videos found in data/samples.")
            elif mode == "Webcam":
                source = int(st.number_input("Webcam index", 0, 10, 0))
            else:
                default = tt.load_cameras(P["cameras"]).get(room, "")
                source = st.text_input("RTSP url", value=default if isinstance(default, str) else "",
                                       placeholder="rtsp://user:password@camera-ip:554/stream1")

            if live:
                when = datetime.now().replace(microsecond=0)
                st.caption(f"Live mode uses the current time ({when:%a %H:%M}) and runs until the class ends.")
            else:
                day_date = st.date_input("Recording date", value=datetime.now().date())
                when = datetime.combine(day_date, datetime.min.time())
                day = tt.slots_for_day(table, when)
                day = day[day["room"] == room].reset_index(drop=True)
                if len(day):
                    i = st.selectbox("Class", range(len(day)),
                                     format_func=lambda k: "{start}–{end} · {batch} · {subject}".format(**day.iloc[k].to_dict()))
                    offset = st.number_input("Recording began (minutes after class start)", 0, 600, 0)
                    begin = datetime.strptime(day.iloc[i]["start"], "%H:%M").time()
                    when = datetime.combine(day_date, begin) + timedelta(minutes=int(offset))

            sparse = bool(video_path) and 0 < video_fps(video_path) <= 2
            if sparse:
                st.caption("This file has one frame per second or less (a frame dataset or time-lapse), "
                           "so every frame is analysed.")
            with st.expander("Advanced settings"):
                every = st.slider("Sample one frame every (s)", 0.5, 10.0, float(conf["attendance"]["sample_every_sec"]), 0.5)
                timelapse = 0.0
                if not live:
                    timelapse = st.number_input("Time-lapse: real seconds per video frame (0 = normal video)",
                                                0.0, 600.0, 0.0, 1.0)
                late_after = st.number_input("Late if first seen after (min)", 0.5, 120.0,
                                             float(conf["attendance"]["late_after_min"]), 0.5)
                enhance = st.checkbox("Image enhancement (gamma, denoise, CLAHE, sharpen)", value=True)
                tiled = st.checkbox("Tiled detection for small back-row faces", value=conf["detection"]["tiled"])

        slot = tt.find_slot(table, room, when)
        with theme.card("slot"):
            if slot:
                roster = d.students(slot["batch"])
                enrolled = int((roster["photos"] > 0).sum())
                st.markdown("<div class='av-eyebrow'>Expected class</div>", unsafe_allow_html=True)
                st.subheader(slot["subject"])
                theme.kv([("Batch", slot["batch"]), ("Time", f"{slot['start']}–{slot['end']}"),
                          ("Faculty", slot["faculty"] or "–"),
                          ("Roster", f"{enrolled} of {len(roster)} students enrolled")])
                if not enrolled:
                    st.warning("Nobody in this batch has photos yet, so no one can be recognised.",
                               icon=":material/warning:")
            else:
                st.warning(f"No class is scheduled in {room} on {when:%A}" + (f" at {when:%H:%M}" if live else "") +
                           ". Pick another date or room, or add the slot on the Timetable page.",
                           icon=":material/event_busy:")
            ready = slot is not None and (video_path is not None or (live and source not in (None, "")))
            start = st.button("Start session", type="primary", disabled=not ready, width="stretch",
                              icon=":material/play_arrow:")

    with right:
        if start:
            with theme.card("monitor"):
                eng = c.engine()
                eng.cfg["preprocess"]["enabled"] = enhance
                eng.dcfg["tiled"] = tiled
                eng.cfg["attendance"]["late_after_min"] = late_after
                proc = SessionProcessor(eng, d, eng.cfg, slot, when, source=str(video_path or source))
                st.session_state["proc"] = proc
                top = st.columns([3, 1])
                top[0].subheader(f"Live monitor · {slot['subject']}")
                top[1].button("Stop and save", key="stop", icon=":material/stop_circle:", width="stretch")
                status, bar, view = st.empty(), st.progress(0.0), st.empty()

                def on_frame(img, progress, ts):
                    view.image(c.to_rgb(img), width="stretch")
                    bar.progress(progress)
                    status.markdown(
                        f"<span class='av-chip'>{ts:%H:%M:%S}</span><span class='av-chip'>frame {proc.frames}</span>"
                        f"<span class='av-chip'>recognised {len(proc.seen)} / {len(proc.students)}</span>"
                        f"<span class='av-chip'>unknown {len(proc.unknown)}</span>"
                        f"<span class='av-chip'>{eng.device}</span>", unsafe_allow_html=True)
                try:
                    if live:
                        run_live(proc, source, every, on_frame)
                    else:
                        run_video(proc, video_path, when, None if sparse and not timelapse else every, on_frame,
                                  frame_seconds=timelapse or None)
                    export_session(d, eng.cfg, proc.session_id)
                    st.session_state["last_session"] = proc.session_id
                    st.session_state.pop("proc", None)
                    st.rerun()              # refresh every page with the new session
                except IOError as e:
                    d.delete_session(proc.session_id)
                    st.session_state.pop("proc", None)
                    st.error(str(e), icon=":material/error:")
        elif "last_session" in st.session_state:
            with theme.card("result"):
                c.show_session(st.session_state["last_session"], key="run")
        else:
            with theme.card("idle"):
                st.subheader("Live monitor")
                st.markdown("<div class='av-note'>The annotated camera view appears here once a session starts. "
                            "Recognised students are boxed in green with their roll number and match score; "
                            "unrecognised faces are boxed in red.</div>", unsafe_allow_html=True)
                st.markdown("&nbsp;")
                steps = st.columns(3)
                steps[0].markdown("**1 · Choose the class**  \nRoom and date select the timetable slot.")
                steps[1].markdown("**2 · Choose the source**  \nA recording, the webcam or an RTSP camera.")
                steps[2].markdown("**3 · Start**  \nAttendance is saved when the video or class ends.")
