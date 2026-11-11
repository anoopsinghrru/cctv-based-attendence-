"""Timetable: the weekly schedule that decides which batch is expected, and the room cameras."""
from pathlib import Path

import pandas as pd
import streamlit as st

from src import timetable as tt
from ui import common as c
from ui import theme


def render():
    P = c.paths()
    table = c.timetable()
    theme.header("Timetable", "A session is matched to the slot whose room and time contain it. "
                              "That slot's batch is the only group of students searched.")

    k = st.columns(4)
    k[0].metric("Weekly slots", len(table))
    k[1].metric("Rooms", table["room"].nunique() if len(table) else 0)
    k[2].metric("Batches", table["batch"].nunique() if len(table) else 0)
    k[3].metric("Subjects", table["subject"].nunique() if len(table) else 0)

    view = theme.switch("Timetable view", ["Week view", "Edit timetable", "Cameras"], "timetable_view")

    if view == "Week view":
        if len(table):
            rooms = ["All rooms"] + sorted(table["room"].unique())
            room = st.selectbox("Room", rooms, key="week_room")
            view = table if room == "All rooms" else table[table["room"] == room]
            cell = view.assign(slot=view["start"] + "–" + view["end"],
                               text=view["subject"] + " · " + view["batch"] +
                               ("" if room != "All rooms" else " (" + view["room"] + ")"))
            grid = cell.pivot_table(index="slot", columns="day", values="text",
                                    aggfunc=lambda s: "  |  ".join(s)).reindex(columns=tt.DAYS).dropna(axis=1, how="all")
            st.dataframe(grid.fillna(""), width="stretch", height=min(520, 38 * (len(grid) + 1) + 3),
                         column_config={"slot": st.column_config.TextColumn("Time", width="small")})
        else:
            st.info("The timetable is empty. Add slots on the Edit timetable tab.", icon=":material/event_busy:")

    elif view == "Edit timetable":
        st.markdown("<div class='av-note'>Days are Mon to Sun; times are 24-hour HH:MM. "
                    "Add a row at the bottom, or select rows to delete them.</div>", unsafe_allow_html=True)
        edited = st.data_editor(table, num_rows="dynamic", width="stretch", hide_index=True, key="tt_edit",
                                column_config={"day": st.column_config.SelectboxColumn("Day", options=tt.DAYS),
                                               "start": "Start", "end": "End", "room": "Room", "batch": "Batch",
                                               "subject": "Subject", "faculty": "Faculty"})
        if st.button("Save timetable", type="primary", icon=":material/save:"):
            try:
                tt.save_timetable(edited, P["timetable"])
                st.success("Timetable saved.", icon=":material/check_circle:")
            except (ValueError, IndexError):
                st.error("Could not save: every row needs a day and HH:MM start and end times.",
                         icon=":material/error:")

    else:
        st.markdown("<div class='av-note'>The source used by live mode for each room: a webcam index (0, 1, ...) "
                    "or an RTSP address such as rtsp://user:password@camera-ip:554/stream1.</div>",
                    unsafe_allow_html=True)
        cam_path = Path(P["cameras"])
        current = pd.read_csv(cam_path, dtype=str).fillna("") if cam_path.exists() else \
            pd.DataFrame(columns=["room", "source", "description"])
        cams_edit = st.data_editor(current, num_rows="dynamic", width="stretch", hide_index=True, key="cam_edit",
                                   column_config={"room": "Room", "source": "Source", "description": "Description"})
        if st.button("Save cameras", icon=":material/save:"):
            cams_edit.fillna("").to_csv(cam_path, index=False)
            st.success("Cameras saved.", icon=":material/check_circle:")
