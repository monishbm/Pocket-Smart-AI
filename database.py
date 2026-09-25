from __future__ import annotations

import json
import os
import sqlite3
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterator

BASE_DIR = Path(__file__).resolve().parent
DATABASE_PATH = Path(os.getenv("DATABASE_PATH", "data/pocketsmart.db"))
if not DATABASE_PATH.is_absolute():
    DATABASE_PATH = BASE_DIR / DATABASE_PATH

def utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()

@contextmanager
def get_connection() -> Iterator[sqlite3.Connection]:
    DATABASE_PATH.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(DATABASE_PATH)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA foreign_keys = ON")
    try:
        yield conn
        conn.commit()
    finally:
        conn.close()

def init_db() -> None:
    with get_connection() as conn:
        conn.executescript("""
        CREATE TABLE IF NOT EXISTS users (
            username TEXT PRIMARY KEY,
            email TEXT UNIQUE NOT NULL,
            full_name TEXT NOT NULL,
            hashed_password TEXT NOT NULL,
            created_at TEXT NOT NULL
        );
        CREATE TABLE IF NOT EXISTS sessions (
            token_id TEXT PRIMARY KEY,
            username TEXT NOT NULL,
            login_time TEXT NOT NULL,
            last_activity TEXT NOT NULL,
            expires_at TEXT NOT NULL,
            user_data TEXT NOT NULL DEFAULT '{}',
            revoked INTEGER NOT NULL DEFAULT 0,
            FOREIGN KEY(username) REFERENCES users(username) ON DELETE CASCADE
        );
        CREATE TABLE IF NOT EXISTS recommendations (
            id TEXT PRIMARY KEY,
            username TEXT NOT NULL,
            timestamp TEXT NOT NULL,
            recommendation_type TEXT NOT NULL,
            input_summary TEXT NOT NULL,
            result_summary TEXT NOT NULL,
            full_result TEXT NOT NULL,
            FOREIGN KEY(username) REFERENCES users(username) ON DELETE CASCADE
        );
        CREATE INDEX IF NOT EXISTS idx_recommendations_user_time
        ON recommendations(username, timestamp DESC);
        """)

def create_user(username: str, email: str, full_name: str, hashed_password: str) -> None:
    with get_connection() as conn:
        conn.execute(
            "INSERT INTO users(username,email,full_name,hashed_password,created_at) VALUES(?,?,?,?,?)",
            (username, email, full_name, hashed_password, utc_now_iso()),
        )

def get_user(username: str) -> dict[str, Any] | None:
    with get_connection() as conn:
        row = conn.execute("SELECT * FROM users WHERE username=?", (username,)).fetchone()
        return dict(row) if row else None

def get_user_by_email(email: str) -> dict[str, Any] | None:
    with get_connection() as conn:
        row = conn.execute("SELECT * FROM users WHERE email=?", (email,)).fetchone()
        return dict(row) if row else None

def create_session(token_id: str, username: str, login_time: str, last_activity: str, expires_at: str) -> None:
    with get_connection() as conn:
        conn.execute(
            "INSERT INTO sessions(token_id,username,login_time,last_activity,expires_at,user_data,revoked) VALUES(?,?,?,?,?,'{}',0)",
            (token_id, username, login_time, last_activity, expires_at),
        )

def get_session(token_id: str) -> dict[str, Any] | None:
    with get_connection() as conn:
        row = conn.execute("SELECT * FROM sessions WHERE token_id=?", (token_id,)).fetchone()
    if not row:
        return None
    data = dict(row)
    data["user_data"] = json.loads(data.get("user_data") or "{}")
    return data

def touch_session(token_id: str) -> None:
    with get_connection() as conn:
        conn.execute("UPDATE sessions SET last_activity=? WHERE token_id=? AND revoked=0", (utc_now_iso(), token_id))

def update_session_data(token_id: str, user_data: dict[str, Any]) -> None:
    with get_connection() as conn:
        conn.execute("UPDATE sessions SET user_data=? WHERE token_id=? AND revoked=0",
                     (json.dumps(user_data, ensure_ascii=False), token_id))

def revoke_session(token_id: str) -> None:
    with get_connection() as conn:
        conn.execute("UPDATE sessions SET revoked=1 WHERE token_id=?", (token_id,))

def cleanup_sessions(cutoff_iso: str) -> int:
    with get_connection() as conn:
        cur = conn.execute("DELETE FROM sessions WHERE revoked=1 OR expires_at<? OR last_activity<?",
                           (utc_now_iso(), cutoff_iso))
        return cur.rowcount

def save_recommendation(rec_id: str, username: str, recommendation_type: str,
                        input_summary: dict[str, Any], result_summary: dict[str, Any],
                        full_result: dict[str, Any]) -> None:
    with get_connection() as conn:
        conn.execute("""
            INSERT INTO recommendations(id,username,timestamp,recommendation_type,input_summary,result_summary,full_result)
            VALUES(?,?,?,?,?,?,?)
        """, (
            rec_id, username, utc_now_iso(), recommendation_type,
            json.dumps(input_summary, ensure_ascii=False),
            json.dumps(result_summary, ensure_ascii=False),
            json.dumps(full_result, ensure_ascii=False),
        ))

def list_recommendations(username: str) -> list[dict[str, Any]]:
    with get_connection() as conn:
        rows = conn.execute("""
            SELECT id,timestamp,recommendation_type,input_summary,result_summary
            FROM recommendations WHERE username=? ORDER BY timestamp DESC
        """, (username,)).fetchall()
    out = []
    for row in rows:
        item = dict(row)
        item["input_summary"] = json.loads(item["input_summary"])
        item["result_summary"] = json.loads(item["result_summary"])
        out.append(item)
    return out

def get_recommendation(rec_id: str, username: str) -> dict[str, Any] | None:
    with get_connection() as conn:
        row = conn.execute("SELECT * FROM recommendations WHERE id=? AND username=?",
                           (rec_id, username)).fetchone()
    if not row:
        return None
    item = dict(row)
    item["input_summary"] = json.loads(item["input_summary"])
    item["result_summary"] = json.loads(item["result_summary"])
    item["full_result"] = json.loads(item["full_result"])
    return item
