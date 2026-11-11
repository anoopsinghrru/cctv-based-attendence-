"""Look and feel: colours, CSS, and the small building blocks every page uses."""
import altair as alt
import pandas as pd
import streamlit as st

PRIMARY = "#2F5BEA"
INK = "#1B2333"
MUTED = "#667085"
LINE = "#E4E7EF"
STATUS = {"Present": "#12B76A", "Late": "#F79009", "Absent": "#F04438"}
STATUS_BG = {"Present": "#E7F8F0", "Late": "#FEF3E2", "Absent": "#FEECEB"}
STATUS_FG = {"Present": "#05603A", "Late": "#93370D", "Absent": "#912018"}
SLOT_STYLE = {"Attendance taken": ("#E7F8F0", "#05603A"), "Running now": ("#E8EEFF", "#1E3FAE"),
              "Upcoming": ("#F2F4F7", "#475467"), "Not recorded": ("#FEF3E2", "#93370D")}

CSS = f"""
<style>
.block-container {{ padding-top: 2.2rem; padding-bottom: 3rem; max-width: 1380px; }}
h1 {{ font-size: 1.65rem !important; font-weight: 700 !important; letter-spacing: -0.015em; padding-bottom: 0 !important; }}
h3 {{ font-size: 1.02rem !important; font-weight: 650 !important; letter-spacing: -0.005em; padding-top: 0 !important; }}
.av-sub {{ color: {MUTED}; font-size: 0.95rem; margin: -0.35rem 0 1.1rem 0; }}
.av-eyebrow {{ color: {MUTED}; font-size: 0.72rem; font-weight: 650; letter-spacing: 0.08em; text-transform: uppercase; }}
.av-note {{ color: {MUTED}; font-size: 0.86rem; }}

/* cards */
[class*="st-key-card"] {{ background: #FFFFFF; border: 1px solid {LINE}; border-radius: 14px;
    padding: 1.05rem 1.2rem 1.15rem 1.2rem; box-shadow: 0 1px 2px rgba(16, 24, 40, 0.04); }}
[data-testid="stMetric"] {{ background: #FFFFFF; border: 1px solid {LINE}; border-radius: 14px;
    padding: 0.9rem 1.1rem; box-shadow: 0 1px 2px rgba(16, 24, 40, 0.04); }}
[data-testid="stMetricLabel"] p {{ color: {MUTED}; font-size: 0.8rem; font-weight: 600; }}
[data-testid="stMetricValue"] {{ font-size: 1.75rem; font-weight: 700; color: {INK}; }}

/* pills */
.av-pill {{ display: inline-block; padding: 2px 10px; border-radius: 999px; font-size: 0.76rem; font-weight: 650;
    line-height: 1.5; white-space: nowrap; }}
.av-chip {{ display: inline-block; padding: 3px 10px; border-radius: 8px; font-size: 0.8rem; font-weight: 600;
    background: #F2F4F7; color: #344054; margin: 0 6px 6px 0; }}
.av-kv {{ display: flex; justify-content: space-between; padding: 7px 0; border-bottom: 1px solid #F0F2F6; font-size: 0.9rem; }}
.av-kv span:first-child {{ color: {MUTED}; }}
.av-kv span:last-child {{ font-weight: 600; color: {INK}; text-align: right; }}

/* controls */
.stButton > button, .stDownloadButton > button {{ border-radius: 9px; font-weight: 600; }}
[data-testid="stSidebar"] [data-testid="stSidebarNavLink"] {{ border-radius: 9px; }}
[data-testid="stSidebar"] .av-side {{ color: #9AA6C0; font-size: 0.78rem; line-height: 1.55; }}
[data-testid="stSidebar"] .av-side b {{ color: #E6EAF2; font-weight: 600; }}
.av-dot {{ display: inline-block; width: 8px; height: 8px; border-radius: 50%; margin-right: 6px; }}
[data-testid="stDataFrame"] {{ border-radius: 10px; }}
[data-testid="stExpander"] details {{ border-radius: 12px; border-color: {LINE}; background: #FFFFFF; }}
</style>
"""


def apply():
    st.markdown(CSS, unsafe_allow_html=True)


def header(title, subtitle=None):
    st.title(title)
    if subtitle:
        st.markdown(f"<div class='av-sub'>{subtitle}</div>", unsafe_allow_html=True)


def card(key):
    """A white panel. `key` must be unique on the page."""
    return st.container(key=f"card_{key}")


def switch(label, options, key):
    """A segmented view switch. Only the chosen view is rendered, which keeps tables correctly
    sized (tables inside hidden st.tabs are laid out with zero width)."""
    choice = st.segmented_control(label, options, default=options[0], key=key, label_visibility="collapsed")
    return choice or options[0]


def pill(text, bg="#F2F4F7", fg="#344054"):
    return f"<span class='av-pill' style='background:{bg};color:{fg}'>{text}</span>"


def status_pill(status):
    return pill(status, STATUS_BG.get(status, "#F2F4F7"), STATUS_FG.get(status, "#344054"))


def kv(rows):
    """Key / value list (dict or list of pairs) rendered as a compact definition table."""
    items = rows.items() if isinstance(rows, dict) else rows
    st.markdown("".join(f"<div class='av-kv'><span>{k}</span><span>{v}</span></div>" for k, v in items),
                unsafe_allow_html=True)


def style_status(df, column="status"):
    """Colour a status column as soft badges."""
    def css(v):
        if v in STATUS_BG:
            return f"background-color:{STATUS_BG[v]};color:{STATUS_FG[v]};font-weight:600"
        if v in SLOT_STYLE:
            return f"background-color:{SLOT_STYLE[v][0]};color:{SLOT_STYLE[v][1]};font-weight:600"
        return ""
    return df.style.map(css, subset=[column]) if column in df.columns and len(df) else df


def pct_column(label="Attendance"):
    return st.column_config.ProgressColumn(label, min_value=0, max_value=100, format="%.0f%%")


# ------------------------------------------------------------------ charts
def _base(chart, height):
    return (chart.properties(height=height, background="#FFFFFF")
            .configure_view(strokeWidth=0)
            .configure_axis(labelColor="#475467", titleColor=MUTED, gridColor="#EEF0F5", domainColor=LINE,
                            tickColor=LINE, labelFontSize=12, titleFontSize=12, labelLimit=220)
            .configure_legend(labelColor="#475467", titleColor=MUTED, labelFontSize=12))


def bar_pct(df, category, value="attendance", height=None):
    """Horizontal bars of an attendance percentage per category."""
    chart = alt.Chart(df).mark_bar(cornerRadiusEnd=5, color=PRIMARY, size=18).encode(
        x=alt.X(f"{value}:Q", scale=alt.Scale(domain=[0, 100]), title="Attendance %"),
        y=alt.Y(f"{category}:N", sort="-x", title=None),
        tooltip=[alt.Tooltip(f"{category}:N"), alt.Tooltip(f"{value}:Q", format=".1f", title="Attendance %")])
    st.altair_chart(_base(chart, height or max(110, 38 * len(df) + 40)), width="stretch")


def trend(df, x="date", value="attendance"):
    """Attendance % per day."""
    base = alt.Chart(df).encode(
        x=alt.X(f"{x}:O", title=None, axis=alt.Axis(labelAngle=0)),
        y=alt.Y(f"{value}:Q", scale=alt.Scale(domain=[0, 100]), title="Attendance %"),
        tooltip=[alt.Tooltip(f"{x}:O", title="Date"), alt.Tooltip(f"{value}:Q", format=".1f", title="Attendance %"),
                 alt.Tooltip("sessions:Q", title="Sessions")])
    chart = base.mark_line(color=PRIMARY, strokeWidth=2.5) + base.mark_point(color=PRIMARY, size=70, filled=True)
    st.altair_chart(_base(chart, 230), width="stretch")


def donut(counts, height=210):
    """Present / Late / Absent split. `counts` is a dict."""
    df = pd.DataFrame({"status": list(STATUS), "count": [int(counts.get(s, 0)) for s in STATUS]})
    chart = alt.Chart(df[df["count"] > 0]).mark_arc(innerRadius=48, outerRadius=74, cornerRadius=3, padAngle=0.02).encode(
        theta=alt.Theta("count:Q", stack=True),
        color=alt.Color("status:N", scale=alt.Scale(domain=list(STATUS), range=list(STATUS.values())),
                        legend=alt.Legend(title=None, orient="bottom")),
        tooltip=["status:N", "count:Q"])
    st.altair_chart(_base(chart, height), width="stretch")
