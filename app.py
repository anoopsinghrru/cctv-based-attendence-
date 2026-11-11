"""AttendVision dashboard:  python -m streamlit run app.py"""
import onnxruntime as ort
import streamlit as st

from src.attendance import export_session
from src.config import ROOT
from ui import common as c
from ui import overview, reports, students, system, take_attendance, theme, timetable_page

st.set_page_config(page_title="AttendVision", page_icon=str(ROOT / "assets" / "icon.svg"), layout="wide",
                   initial_sidebar_state="expanded")
theme.apply()

# A session interrupted by "Stop and save" (which reruns the script) is closed here.
if "proc" in st.session_state:
    _p = st.session_state["proc"]
    if not _p.finalized:
        if _p.frames:
            _p.finalize()
            export_session(c.db(), _p.cfg, _p.session_id)
            st.session_state["last_session"] = _p.session_id
        else:
            c.db().delete_session(_p.session_id)
    st.session_state.pop("proc", None)

PAGES = {
    "overview": st.Page(overview.render, title="Overview", icon=":material/space_dashboard:",
                        url_path="overview", default=True),
    "attendance": st.Page(take_attendance.render, title="Take attendance", icon=":material/videocam:",
                          url_path="attendance"),
    "students": st.Page(students.render, title="Students", icon=":material/group:", url_path="students"),
    "timetable": st.Page(timetable_page.render, title="Timetable", icon=":material/calendar_month:",
                         url_path="timetable"),
    "reports": st.Page(reports.render, title="Reports", icon=":material/insights:", url_path="reports"),
    "system": st.Page(system.render, title="System", icon=":material/settings:", url_path="system"),
}
nav = st.navigation({"": [PAGES["overview"]],
                     "Operations": [PAGES["attendance"], PAGES["students"], PAGES["timetable"]],
                     "Insights": [PAGES["reports"]],
                     "Administration": [PAGES["system"]]})
st.logo(str(ROOT / "assets" / "logo.svg"), size="large", icon_image=str(ROOT / "assets" / "icon.svg"))

if "_goto" in st.session_state:                 # set by ui.common.goto()
    st.switch_page(PAGES[st.session_state.pop("_goto")])

with st.sidebar:
    gpu = "CUDAExecutionProvider" in ort.get_available_providers() and c.cfg()["models"].get("use_gpu")
    st.markdown(
        "<div class='av-side' style='margin-top:1.2rem'>"
        f"<span class='av-dot' style='background:#12B76A'></span><b>System ready</b><br>"
        f"Inference on {'GPU (CUDA)' if gpu else 'CPU'}<br>{c.now():%a %d %b %Y}</div>",
        unsafe_allow_html=True)

nav.run()
