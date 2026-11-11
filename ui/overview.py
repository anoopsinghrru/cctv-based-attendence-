"""Overview: today's classes and the state of attendance at a glance."""
import streamlit as st

from src import timetable as tt
from ui import common as c
from ui import theme


def render():
    d, now = c.db(), c.now()
    students, sessions, long = d.students(), d.sessions(), d.attendance_long()
    table = c.timetable()
    today = c.slot_states(tt.slots_for_day(table, now), sessions, now)

    theme.header("Overview", f"{now:%A, %d %B %Y} · {len(today)} class{'es' if len(today) != 1 else ''} scheduled today")

    enrolled = int((students["photos"] > 0).sum()) if len(students) else 0
    k = st.columns(4)
    k[0].metric("Students", len(students), help="Students registered across all batches")
    k[1].metric("Enrolled with photos", f"{enrolled} / {len(students)}",
                help="Students who have at least one face photo and can be recognised")
    k[2].metric("Sessions recorded", len(sessions))
    k[3].metric("Average attendance", f"{100 * (long['status'] != 'Absent').mean():.0f}%" if len(long) else "–",
                help="Present and Late count as attended")

    left, right = st.columns([1.35, 1])
    with left, theme.card("today"):
        head = st.columns([3, 1.2])
        head[0].subheader("Today's schedule")
        if head[1].button("Take attendance", type="primary", icon=":material/videocam:", width="stretch"):
            c.goto("attendance")
        if len(today):
            view = today.drop(columns=["day"]).rename(columns={
                "start": "Start", "end": "End", "room": "Room", "batch": "Batch", "subject": "Subject",
                "faculty": "Faculty", "status": "Status"})
            st.dataframe(theme.style_status(view, "Status"), width="stretch", hide_index=True)
        else:
            st.info("No classes are scheduled today. Add slots on the Timetable page.", icon=":material/event_busy:")
    with right, theme.card("subject"):
        st.subheader("Attendance by subject")
        if len(long):
            theme.bar_pct(c.summarise(long, ["subject"]), "subject")
        else:
            st.caption("Appears after the first session is recorded.")

    if not len(long):
        with theme.card("empty"):
            st.subheader("Getting started")
            st.markdown("1. Add your classes on the **Timetable** page.\n"
                        "2. Register students and their photos on the **Students** page.\n"
                        "3. Record a class on the **Take attendance** page.")
        return

    left, right = st.columns([1.35, 1])
    with left, theme.card("trend"):
        st.subheader("Attendance trend")
        by_day = long.assign(att=(long["status"] != "Absent") * 100.0).groupby("date").agg(
            attendance=("att", "mean"), sessions=("session_id", "nunique")).round(1).reset_index()
        theme.trend(by_day)
    with right, theme.card("risk"):
        st.subheader("Students below 75%")
        per = c.summarise(long, ["roll_no", "name", "batch"])
        risk = per[per["attendance"] < 75].sort_values("attendance")
        if len(risk):
            st.dataframe(risk[["roll_no", "name", "batch", "classes", "attendance"]].rename(columns={
                "roll_no": "Roll no", "name": "Name", "batch": "Batch", "classes": "Classes"}),
                width="stretch", hide_index=True, height=230, column_config={"attendance": theme.pct_column()})
        else:
            st.success("Every student is at or above 75%.", icon=":material/verified:")

    with theme.card("recent"):
        st.subheader("Recent sessions")
        recent = sessions.head(8).copy()
        total = (recent["present"] + recent["late"] + recent["absent"]).replace(0, 1)
        recent["attendance"] = (100 * (recent["present"] + recent["late"]) / total).round(0)
        st.dataframe(recent[["id", "date", "start", "room", "batch", "subject", "present", "late", "absent", "attendance"]]
                     .rename(columns={"id": "#", "date": "Date", "start": "Start", "room": "Room", "batch": "Batch",
                                      "subject": "Subject", "present": "Present", "late": "Late", "absent": "Absent"}),
                     width="stretch", hide_index=True, column_config={"attendance": theme.pct_column()})
