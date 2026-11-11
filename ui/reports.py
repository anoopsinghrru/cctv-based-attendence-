"""Reports: attendance per student and per subject, the session archive, and exports."""
from datetime import datetime

import streamlit as st

from src.attendance import excel_bytes
from ui import common as c
from ui import theme


def render():
    d = c.db()
    long = d.attendance_long()
    theme.header("Reports", "Attendance per student and per subject, with every recorded session. "
                            "Present and Late both count as attended.")
    if not len(long):
        with theme.card("none"):
            st.info("No sessions yet. Record a class on the Take attendance page.", icon=":material/insights:")
        return

    with theme.card("filters"):
        f = st.columns([1, 1, 1.2])
        fb = f[0].selectbox("Batch ", ["All batches"] + sorted(long["batch"].unique()))
        scope = long if fb == "All batches" else long[long["batch"] == fb]
        fs = f[1].selectbox("Subject", ["All subjects"] + sorted(scope["subject"].unique()))
        data = scope if fs == "All subjects" else scope[scope["subject"] == fs]
        per_student = c.summarise(data, ["roll_no", "name", "batch"])
        per_subject = c.summarise(data, ["subject", "batch"])
        per_subject.insert(2, "sessions", [
            data[(data["subject"] == r.subject) & (data["batch"] == r.batch)]["session_id"].nunique()
            for r in per_subject.itertuples()])
        sessions = d.sessions()
        shown = sessions[sessions["id"].isin(data["session_id"].unique())]
        f[2].markdown("<div style='height:1.75rem'></div>", unsafe_allow_html=True)
        f[2].download_button(
            "Download full report (Excel)",
            excel_bytes({"Per student": per_student, "Per subject": per_subject,
                         "Sessions": shown, "All records": data}),
            f"attendance_report_{datetime.now():%Y%m%d}.xlsx", type="primary", icon=":material/download:",
            width="stretch")

    k = st.columns(4)
    k[0].metric("Sessions", data["session_id"].nunique())
    k[1].metric("Students", data["roll_no"].nunique())
    k[2].metric("Average attendance", f"{100 * (data['status'] != 'Absent').mean():.0f}%")
    k[3].metric("Below 75%", int((per_student["attendance"] < 75).sum()))

    view = theme.switch("Report view", ["By student", "By subject", "Session archive"], "report_view")

    if view == "By student":
        st.dataframe(per_student.sort_values(["attendance", "roll_no"]).rename(columns={
            "roll_no": "Roll no", "name": "Name", "batch": "Batch", "classes": "Classes"}),
            width="stretch", hide_index=True, height=min(520, 38 * (len(per_student) + 1) + 3),
            column_config={"attendance": theme.pct_column()})

    elif view == "By subject":
        a, b = st.columns([1, 1.2])
        with a:
            theme.bar_pct(per_subject.assign(label=per_subject["subject"] + " · " + per_subject["batch"]), "label")
        with b:
            st.dataframe(per_subject.rename(columns={
                "subject": "Subject", "batch": "Batch", "sessions": "Sessions", "classes": "Student-classes"}),
                width="stretch", hide_index=True, column_config={"attendance": theme.pct_column()})

    else:
        if not len(shown):
            st.caption("No sessions match the filters.")
            return
        listing = shown.copy()
        total = (listing["present"] + listing["late"] + listing["absent"]).replace(0, 1)
        listing["attendance"] = (100 * (listing["present"] + listing["late"]) / total).round(0)
        st.dataframe(listing[["id", "date", "start", "end", "room", "batch", "subject", "faculty", "frames",
                              "present", "late", "absent", "attendance"]].rename(columns={
                                  "id": "#", "date": "Date", "start": "Start", "end": "End", "room": "Room",
                                  "batch": "Batch", "subject": "Subject", "faculty": "Faculty", "frames": "Frames",
                                  "present": "Present", "late": "Late", "absent": "Absent"}),
                     width="stretch", hide_index=True, height=min(330, 38 * (len(listing) + 1) + 3),
                     column_config={"attendance": theme.pct_column()})
        rows = listing.set_index("id", drop=False)
        pick = st.columns([3, 1])
        sid = pick[0].selectbox("Open session", listing["id"], format_func=lambda i: c.session_label(rows.loc[i]))
        pick[1].markdown("<div style='height:1.75rem'></div>", unsafe_allow_html=True)
        if pick[1].button("Delete session", icon=":material/delete:", width="stretch"):
            d.delete_session(int(sid))
            if st.session_state.get("last_session") == int(sid):
                st.session_state.pop("last_session")
            st.rerun()
        with theme.card("session"):
            c.show_session(int(sid), key="rep")
