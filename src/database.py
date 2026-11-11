"""SQLite storage: students, face embeddings, sessions, attendance."""
import sqlite3
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd

SCHEMA = """
CREATE TABLE IF NOT EXISTS students (
    roll_no TEXT PRIMARY KEY,
    name    TEXT NOT NULL,
    batch   TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS embeddings (
    id        INTEGER PRIMARY KEY AUTOINCREMENT,
    roll_no   TEXT NOT NULL REFERENCES students(roll_no) ON DELETE CASCADE,
    embedding BLOB NOT NULL,
    source    TEXT,
    created   TEXT
);
CREATE TABLE IF NOT EXISTS sessions (
    id       INTEGER PRIMARY KEY AUTOINCREMENT,
    room     TEXT, batch TEXT, subject TEXT, faculty TEXT,
    date     TEXT, start TEXT, end TEXT,
    source   TEXT,
    frames   INTEGER DEFAULT 0,
    created  TEXT
);
CREATE TABLE IF NOT EXISTS attendance (
    session_id INTEGER NOT NULL REFERENCES sessions(id) ON DELETE CASCADE,
    roll_no    TEXT NOT NULL,
    status     TEXT NOT NULL,
    first_seen TEXT, last_seen TEXT,
    sightings  INTEGER, buckets INTEGER,
    best_score REAL,
    snapshot   TEXT,
    PRIMARY KEY (session_id, roll_no)
);
CREATE TABLE IF NOT EXISTS unknown_faces (
    id         INTEGER PRIMARY KEY AUTOINCREMENT,
    session_id INTEGER NOT NULL REFERENCES sessions(id) ON DELETE CASCADE,
    seen_at    TEXT, sightings INTEGER,
    nearest    TEXT, score REAL,
    snapshot   TEXT
);
"""


class Database:
    def __init__(self, path):
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        self.conn = sqlite3.connect(str(path), check_same_thread=False)
        self.conn.execute("PRAGMA foreign_keys = ON")
        self.conn.executescript(SCHEMA)
        self.conn.commit()

    def df(self, sql, params=()):
        return pd.read_sql_query(sql, self.conn, params=params)

    # ---------------------------------------------------------- students
    def upsert_student(self, roll_no, name, batch):
        self.conn.execute(
            "INSERT INTO students(roll_no, name, batch) VALUES(?,?,?) "
            "ON CONFLICT(roll_no) DO UPDATE SET name=excluded.name, batch=excluded.batch",
            (str(roll_no).strip(), str(name).strip(), str(batch).strip()))
        self.conn.commit()

    def delete_student(self, roll_no):
        self.conn.execute("DELETE FROM students WHERE roll_no=?", (roll_no,))
        self.conn.commit()

    def students(self, batch=None):
        sql = ("SELECT s.roll_no, s.name, s.batch, COUNT(e.id) AS photos FROM students s "
               "LEFT JOIN embeddings e ON e.roll_no = s.roll_no ")
        if batch:
            return self.df(sql + "WHERE s.batch=? GROUP BY s.roll_no ORDER BY s.roll_no", (batch,))
        return self.df(sql + "GROUP BY s.roll_no ORDER BY s.batch, s.roll_no")

    def sync_students_csv(self, csv_path):
        """Import students.csv (roll_no,name,batch) into the database."""
        if not Path(csv_path).exists():
            return 0
        df = pd.read_csv(csv_path, dtype=str).fillna("")
        for r in df.itertuples():
            if r.roll_no:
                self.upsert_student(r.roll_no, r.name or r.roll_no, r.batch)
        return len(df)

    def export_students_csv(self, csv_path):
        self.students()[["roll_no", "name", "batch"]].to_csv(csv_path, index=False)

    # ---------------------------------------------------------- embeddings
    def add_embedding(self, roll_no, emb, source=""):
        self.conn.execute(
            "INSERT INTO embeddings(roll_no, embedding, source, created) VALUES(?,?,?,?)",
            (roll_no, np.asarray(emb, np.float32).tobytes(), source,
             datetime.now().isoformat(timespec="seconds")))
        self.conn.commit()

    def clear_embeddings(self, roll_no):
        self.conn.execute("DELETE FROM embeddings WHERE roll_no=?", (roll_no,))
        self.conn.commit()

    def gallery(self, batch):
        """All embeddings of one batch -> (rolls list, matrix (N, 512))."""
        rows = self.conn.execute(
            "SELECT e.roll_no, e.embedding FROM embeddings e JOIN students s "
            "ON s.roll_no = e.roll_no WHERE s.batch=? ORDER BY e.roll_no", (batch,)).fetchall()
        rolls = [r[0] for r in rows]
        mat = (np.stack([np.frombuffer(r[1], np.float32) for r in rows])
               if rows else np.zeros((0, 512), np.float32))
        return rolls, mat

    # ---------------------------------------------------------- sessions
    def create_session(self, slot, date, source):
        cur = self.conn.execute(
            "INSERT INTO sessions(room, batch, subject, faculty, date, start, end, source, created) "
            "VALUES(?,?,?,?,?,?,?,?,?)",
            (slot["room"], slot["batch"], slot["subject"], slot.get("faculty", ""), date,
             slot["start"], slot["end"], source, datetime.now().isoformat(timespec="seconds")))
        self.conn.commit()
        return cur.lastrowid

    def delete_session(self, session_id):
        self.conn.execute("DELETE FROM sessions WHERE id=?", (session_id,))
        self.conn.commit()

    def save_results(self, session_id, frames, rows, unknowns):
        self.conn.execute("UPDATE sessions SET frames=? WHERE id=?", (frames, session_id))
        self.conn.execute("DELETE FROM attendance WHERE session_id=?", (session_id,))
        self.conn.execute("DELETE FROM unknown_faces WHERE session_id=?", (session_id,))
        self.conn.executemany(
            "INSERT INTO attendance VALUES(?,?,?,?,?,?,?,?,?)",
            [(session_id, r["roll_no"], r["status"], r["first_seen"], r["last_seen"],
              r["sightings"], r["buckets"], r["best_score"], r["snapshot"]) for r in rows])
        self.conn.executemany(
            "INSERT INTO unknown_faces(session_id, seen_at, sightings, nearest, score, snapshot) "
            "VALUES(?,?,?,?,?,?)",
            [(session_id, u["seen_at"], u["sightings"], u["nearest"], u["score"], u["snapshot"])
             for u in unknowns])
        self.conn.commit()

    def sessions(self):
        return self.df(
            "SELECT s.id, s.date, s.start, s.end, s.room, s.batch, s.subject, s.faculty, s.source, "
            "s.frames, SUM(a.status='Present') AS present, SUM(a.status='Late') AS late, "
            "SUM(a.status='Absent') AS absent FROM sessions s LEFT JOIN attendance a "
            "ON a.session_id = s.id GROUP BY s.id ORDER BY s.date DESC, s.start DESC, s.id DESC")

    def session_attendance(self, session_id):
        return self.df(
            "SELECT a.roll_no, COALESCE(st.name, a.roll_no) AS name, a.status, a.first_seen, "
            "a.last_seen, a.sightings, a.buckets, ROUND(a.best_score, 3) AS best_score, a.snapshot "
            "FROM attendance a LEFT JOIN students st ON st.roll_no = a.roll_no "
            "WHERE a.session_id=? ORDER BY a.roll_no", (session_id,))

    def unknowns(self, session_id):
        return self.df("SELECT * FROM unknown_faces WHERE session_id=?", (session_id,))

    # ---------------------------------------------------------- reports
    def attendance_long(self):
        """One row per (session, student) - the base table for every report."""
        return self.df(
            "SELECT s.id AS session_id, s.date, s.start, s.room, s.batch, s.subject, a.roll_no, "
            "COALESCE(st.name, a.roll_no) AS name, a.status FROM attendance a "
            "JOIN sessions s ON s.id = a.session_id LEFT JOIN students st ON st.roll_no = a.roll_no")
