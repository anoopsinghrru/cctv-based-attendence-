"""Students: the roster, photo enrolment, and each student's record."""
from datetime import datetime
from pathlib import Path

import cv2
import numpy as np
import streamlit as st

from src.attendance import enroll_folder, enroll_photo
from ui import common as c
from ui import theme


def render():
    d, P = c.db(), c.paths()
    students, long = d.students(), d.attendance_long()
    theme.header("Students", "The roster of every batch. A student needs at least one face photo to be recognised; "
                             "three to five photos give the most reliable results.")

    enrolled = int((students["photos"] > 0).sum()) if len(students) else 0
    k = st.columns(4)
    k[0].metric("Students", len(students))
    k[1].metric("Enrolled with photos", enrolled)
    k[2].metric("Awaiting photos", len(students) - enrolled)
    k[3].metric("Batches", students["batch"].nunique() if len(students) else 0)

    left, right = st.columns([1.7, 1])
    with left:
        with theme.card("roster"):
            st.subheader("Roster")
            f = st.columns([1, 1.6])
            pick = f[0].selectbox("Batch filter", ["All batches"] + sorted(students["batch"].unique()))
            query = f[1].text_input("Search", placeholder="Roll number or name").strip().lower()
            view = students if pick == "All batches" else students[students["batch"] == pick]
            if query:
                view = view[view["roll_no"].str.lower().str.contains(query, regex=False) |
                            view["name"].str.lower().str.contains(query, regex=False)]
            per = c.summarise(long, ["roll_no"]).set_index("roll_no")["attendance"] if len(long) else {}
            show = view.assign(photo=[c.thumbnail(r) for r in view["roll_no"]],
                               attendance=[per.get(r) if len(long) else None for r in view["roll_no"]],
                               state=["Enrolled" if p else "No photos" for p in view["photos"]])
            st.dataframe(
                show[["photo", "roll_no", "name", "batch", "photos", "state", "attendance"]],
                width="stretch", hide_index=True, height=430, row_height=44,
                column_config={"photo": st.column_config.ImageColumn("", width="small"),
                               "roll_no": "Roll no", "name": "Name", "batch": "Batch", "photos": "Photos",
                               "state": "Enrolment", "attendance": theme.pct_column()})
            st.caption(f"{len(view)} of {len(students)} students shown")

        if len(view):
            with theme.card("detail"):
                st.subheader("Student record")
                names = students.set_index("roll_no")["name"]
                sel = st.selectbox("Photos of", view["roll_no"], format_func=lambda r: f"{r} · {names[r]}")
                row = students.set_index("roll_no").loc[sel]
                mine = long[long["roll_no"] == sel] if len(long) else long
                a, b = st.columns([1, 1.3])
                with a:
                    theme.kv([("Roll number", sel), ("Name", row["name"]), ("Batch", row["batch"]),
                              ("Enrolled photos", int(row["photos"])),
                              ("Classes recorded", len(mine)),
                              ("Attendance", f"{100 * (mine['status'] != 'Absent').mean():.0f}%" if len(mine) else "–")])
                with b:
                    photos = c.photos_of(sel)
                    if photos:
                        cols = st.columns(5)
                        for i, p in enumerate(photos[:10]):
                            cols[i % 5].image(str(p), width="stretch")
                    else:
                        st.caption("No photos on file for this student.")
                if len(mine):
                    hist = mine.sort_values(["date", "start"], ascending=False)[["date", "start", "subject", "room", "status"]]
                    st.dataframe(theme.style_status(hist.rename(columns={
                        "date": "Date", "start": "Start", "subject": "Subject", "room": "Room", "status": "Status"}),
                        "Status"), width="stretch", hide_index=True, height=min(250, 38 * (len(hist) + 1) + 3))
                if st.button(f"Delete {sel}", icon=":material/delete:"):
                    d.delete_student(sel)
                    d.export_students_csv(P["students"])
                    st.rerun()

    with right:
        with theme.card("add"):
            st.subheader("Add or update a student")
            st.markdown("<div class='av-note'>Use an existing roll number to add more photos to that student.</div>",
                        unsafe_allow_html=True)
            roll = st.text_input("Roll number").strip()
            name = st.text_input("Name").strip()
            batch = st.text_input("Batch", value=pick if pick != "All batches" else "").strip()
            ups = st.file_uploader("Photos", type=["jpg", "jpeg", "png"], accept_multiple_files=True,
                                   help="Clear, front-facing photos. One face per photo.")
            shot = st.camera_input("Take a photo") if st.toggle("Use the camera instead") else None
            if st.button("Save student", type="primary", disabled=not (roll and batch), width="stretch",
                         icon=":material/person_add:"):
                safe = "".join(ch for ch in roll if ch.isalnum() or ch in "-_")
                if safe != roll:
                    st.error("Roll number may only contain letters, digits, - and _.", icon=":material/error:")
                else:
                    eng = c.engine()
                    existing = students.set_index("roll_no")["name"].get(roll)
                    d.upsert_student(roll, name or existing or roll, batch)
                    folder = Path(P["faces"]) / roll
                    folder.mkdir(parents=True, exist_ok=True)
                    files = [(f.name, f.getvalue()) for f in ups or []]
                    if shot:
                        files.append((f"camera_{datetime.now():%Y%m%d_%H%M%S}.jpg", shot.getvalue()))
                    good = 0
                    for fname, data in files:
                        img = cv2.imdecode(np.frombuffer(data, np.uint8), cv2.IMREAD_COLOR)
                        if img is None or enroll_photo(eng, d, roll, img, fname) is None:
                            st.warning(f"No face found in {fname}; it was skipped.", icon=":material/warning:")
                            continue
                        (folder / Path(fname).name).write_bytes(data)
                        good += 1
                    d.export_students_csv(P["students"])
                    st.success(f"Saved {roll}. {good} new photo(s) enrolled.", icon=":material/check_circle:")

        with theme.card("maint"):
            st.subheader("Maintenance")
            st.markdown("<div class='av-note'>Rebuild every student's face data from the photo folders. "
                        "Use this after copying photos in by hand.</div>", unsafe_allow_html=True)
            if st.button("Re-enroll everyone from the photo folders", icon=":material/refresh:", width="stretch"):
                eng = c.engine()
                with st.spinner("Reading photos ..."):
                    done = [enroll_folder(eng, d, P["faces"], r)[0] for r in students["roll_no"]
                            if (Path(P["faces"]) / r).is_dir()]
                st.success(f"Re-enrolled {len(done)} students ({sum(done)} photos).", icon=":material/check_circle:")
