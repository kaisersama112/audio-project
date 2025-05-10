"""
@Project ：audio-split-src 
@File    ：database.py.py
@IDE     ：PyCharm 
@Author  ：panshangguo
@Date    ：10/5/2025 下午4:27 
"""
# database.py
import sqlite3
from datetime import datetime
from contextlib import contextmanager

DATABASE_NAME = "audio_tasks.db"


@contextmanager
def get_db_connection():
    conn = sqlite3.connect(DATABASE_NAME)
    try:
        yield conn
    finally:
        conn.close()


def init_db():
    with get_db_connection() as conn:
        conn.execute("""
        CREATE TABLE IF NOT EXISTS tasks (
            task_id TEXT PRIMARY KEY,
            status TEXT NOT NULL,
            message TEXT,
            progress INTEGER DEFAULT 0,
            original_path TEXT NOT NULL,
            segments_path TEXT,
            start_time TEXT,
            complete_time TEXT,
            duration REAL,
            error TEXT,
            created_at TEXT NOT NULL
        )
        """)


def create_task(conn, task_data):
    conn.execute("""
    INSERT INTO tasks 
    (task_id, status, message, progress, original_path, segments_path, 
     start_time, complete_time, duration, error, created_at)
    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
    """, (
        task_data["task_id"],
        task_data["status"],
        task_data["message"],
        task_data["progress"],
        task_data["original_path"],
        task_data.get("segments_path"),
        task_data.get("start_time"),
        task_data.get("complete_time"),
        task_data.get("duration"),
        task_data.get("error"),
        task_data["created_at"]
    ))


def update_task(conn, task_data):
    conn.execute("""
    UPDATE tasks SET
        status = ?,
        message = ?,
        progress = ?,
        segments_path = ?,
        start_time = ?,
        complete_time = ?,
        duration = ?,
        error = ?
    WHERE task_id = ?
    """, (
        task_data["status"],
        task_data["message"],
        task_data["progress"],
        task_data.get("segments_path"),
        task_data.get("start_time"),
        task_data.get("complete_time"),
        task_data.get("duration"),
        task_data.get("error"),
        task_data["task_id"]
    ))


def get_task(conn, task_id):
    cursor = conn.execute("SELECT * FROM tasks WHERE task_id = ?", (task_id,))
    return cursor.fetchone()
