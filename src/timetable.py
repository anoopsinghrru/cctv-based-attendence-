"""Timetable lookup: (room, datetime) -> the slot (batch / subject) scheduled there."""
from datetime import datetime, timedelta
from pathlib import Path

import pandas as pd

COLUMNS = ["day", "start", "end", "room", "batch", "subject", "faculty"]
DAYS = ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"]


def _norm_day(d):
    return str(d).strip()[:3].title()


def _norm_time(t):
    """'9:00', '09:00:00', '9.00' -> '09:00'"""
    t = str(t).strip().replace(".", ":")
    parts = t.split(":")
    return f"{int(parts[0]):02d}:{int(parts[1]) if len(parts) > 1 else 0:02d}"


def load_timetable(path):
    if not Path(path).exists():
        return pd.DataFrame(columns=COLUMNS)
    df = pd.read_csv(path, dtype=str).fillna("")
    df = df[[c for c in COLUMNS if c in df.columns]]
    df = df[df["day"].str.strip() != ""].copy()
    df["day"] = df["day"].map(_norm_day)
    df["start"] = df["start"].map(_norm_time)
    df["end"] = df["end"].map(_norm_time)
    for c in ("room", "batch", "subject", "faculty"):
        df[c] = df[c].str.strip()
    return df.reset_index(drop=True)


def save_timetable(df, path):
    df = df[COLUMNS].fillna("").astype(str)
    df = df[df["day"].str.strip().str.len() >= 3].copy()
    df["day"] = df["day"].map(_norm_day)
    df["start"] = df["start"].map(_norm_time)
    df["end"] = df["end"].map(_norm_time)
    df.to_csv(path, index=False)


def slots_for_day(df, when):
    day = DAYS[when.weekday()]
    return df[df["day"] == day].sort_values(["start", "room"]).reset_index(drop=True)


def find_slot(df, room, when):
    """Slot running in `room` at `when` (a slot covers start <= t < end). None if free."""
    hhmm = when.strftime("%H:%M")
    in_room = df["room"].str.lower() == str(room).lower()
    today = df["day"] == DAYS[when.weekday()]
    yesterday = df["day"] == DAYS[(when.weekday() - 1) % 7]
    wraps = df["end"] <= df["start"]                    # slot runs past midnight
    hit = df[in_room & ((today & ~wraps & (df["start"] <= hhmm) & (hhmm < df["end"])) |
                        (today & wraps & (df["start"] <= hhmm)) |
                        (yesterday & wraps & (hhmm < df["end"])))]
    return hit.iloc[0].to_dict() if len(hit) else None


def slot_bounds(slot, when):
    """Start / end datetimes of the slot occurrence that contains `when`."""
    day = when.date()
    start_t = datetime.strptime(slot["start"], "%H:%M").time()
    end_t = datetime.strptime(slot["end"], "%H:%M").time()
    if end_t <= start_t and when.time() < end_t:        # after midnight: the slot began yesterday
        day -= timedelta(days=1)
    start = datetime.combine(day, start_t)
    end = datetime.combine(day, end_t)
    if end <= start:
        end += timedelta(days=1)
    return start, end


def load_cameras(path):
    """cameras.csv: room,source,description. A numeric source is a webcam index."""
    if not Path(path).exists():
        return {}
    df = pd.read_csv(path, dtype=str).fillna("")
    return {r.room: (int(r.source) if r.source.isdigit() else r.source) for r in df.itertuples()}
