"""
@Project ：audio-split-src 
@File    ：mysql_db.py
@IDE     ：PyCharm 
@Author  ：panshangguo
@Date    ：14/5/2025 下午4:24 
"""
import asyncio
import os
import shutil

import pymysql
from pymysql.cursors import DictCursor
from contextlib import contextmanager
from datetime import datetime

from services.audio_service import audio_service
# 测试站
"""
DB_CONFIG = {
    "host": "123.57.150.136",
    "user": "broadcast_ai",
    "password": "eMRtryH6LcpidGRR",
    "database": "broadcast_ai",
    "port": 3306,
    "charset": "utf8mb4",
    "cursorclass": DictCursor
}
"""
# 正式站
DB_CONFIG = {
    "host": "1.14.127.39",
    "user": "broadcast_ai",
    "password": "2Afsp2cGCdk7dRf8",
    "database": "broadcast_ai",
    "port": 3388,
    "charset": "utf8mb4",
    "cursorclass": DictCursor
}


@contextmanager
def get_db_connection():
    connection = None
    try:
        connection = pymysql.connect(**DB_CONFIG)
        yield connection
    finally:
        if connection:
            connection.close()


def init_db():
    with get_db_connection() as conn:
        with conn.cursor() as cursor:
            # 创建任务表
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS ai_tasks (
                    task_id VARCHAR(36) PRIMARY KEY,
                    status VARCHAR(20) NOT NULL DEFAULT 'pending',
                    message TEXT,
                    progress INT NOT NULL DEFAULT 0,
                    original_path TEXT,
                    segments_path TEXT,
                    start_time DATETIME,
                    complete_time DATETIME,
                    duration FLOAT,
                    error TEXT,
                    created_at DATETIME NOT NULL DEFAULT CURRENT_TIMESTAMP
                ) CHARACTER SET utf8mb4 COLLATE utf8mb4_unicode_ci;
            ''')

            # 创建结果表
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS ai_task_results (
                              id INT AUTO_INCREMENT PRIMARY KEY,
                              task_id VARCHAR(36) NOT NULL,
                              `index` INT NOT NULL,
                              start FLOAT NOT NULL,
                              `end` FLOAT NOT NULL,
                              text TEXT NOT NULL,
                              speaker VARCHAR(100),
                              `url` VARCHAR(1000),
                              FOREIGN KEY (task_id) REFERENCES ai_tasks(task_id)
                          ) CHARACTER SET utf8mb4 COLLATE utf8mb4_unicode_ci;
                      ''')
        conn.commit()


def create_task(conn, task_data):
    with conn.cursor() as cursor:
        cursor.execute('''
            INSERT INTO ai_tasks (task_id, status, message, progress, original_path)
            VALUES (%s, %s, %s, %s, %s)
        ''', (
            task_data["task_id"],
            task_data["status"],
            task_data["message"],
            task_data["progress"],
            task_data["original_path"]
        ))
    conn.commit()


def update_task_status(task_data: dict):
    """更新任务状态（数据库集成版）"""
    with get_db_connection() as conn:
        with conn.cursor() as cursor:
            # 构造完整任务数据
            full_data = {
                "status": task_data.get("status", "pending"),
                "message": task_data.get("message", ""),
                "progress": task_data.get("progress", 0),
                "segments_path": task_data.get("segments_path"),
                "start_time": task_data.get("start_time"),
                "complete_time": task_data.get("complete_time"),
                "duration": task_data.get("duration"),
                "error": task_data.get("error"),
                "task_id": task_data["task_id"]
            }

            # 处理时间计算
            if full_data["status"] in ["completed", "failed"]:
                full_data["complete_time"] = datetime.now().isoformat()
                if full_data.get("start_time"):
                    start = datetime.fromisoformat(full_data["start_time"])
                    end = datetime.fromisoformat(full_data["complete_time"])
                    full_data["duration"] = round((end - start).total_seconds(), 2)

            # 更新数据库
            cursor.execute('''
                UPDATE ai_tasks SET
                    status = %s,
                    message = %s,
                    progress = %s,
                    segments_path = %s,
                    start_time = %s,
                    complete_time = %s,
                    duration = %s,
                    error = %s
                WHERE task_id = %s
            ''', (
                full_data["status"],
                full_data["message"],
                full_data["progress"],
                full_data["segments_path"],
                full_data["start_time"],
                full_data["complete_time"],
                full_data["duration"],
                full_data["error"],
                full_data["task_id"]
            ))
        conn.commit()


def get_task(conn, task_id):
    with conn.cursor() as cursor:
        cursor.execute(
            "SELECT * FROM ai_tasks WHERE task_id = %s",
            (task_id,)
        )
        result = cursor.fetchone()
    return result


def get_task_results(conn, task_id, keyword=None, speaker=None, page=1, per_page=10):
    with conn.cursor() as cursor:
        query = '''
            SELECT * FROM ai_task_results 
            WHERE task_id = %s
        '''
        params = [task_id]

        if keyword:
            query += " AND (text LIKE %s OR speaker LIKE %s)"
            params.extend([f"%{keyword}%", f"%{keyword}%"])

        if speaker:
            query += " AND speaker LIKE %s"
            params.append(f"%{speaker}%")

        query += " ORDER BY `index`"

        cursor.execute(query, params)
        results = cursor.fetchall()

        # 分页处理
        total = len(results)
        total_pages = (total + per_page - 1) // per_page if per_page > 0 else 1
        start = (page - 1) * per_page
        end = start + per_page

        paginated_results = results[start:end]

        return {
            "items": paginated_results,
            "total": total,
            "page": page,
            "per_page": per_page,
            "total_pages": total_pages
        }


def get_all_task_results(conn, task_id): # 获取所有任务结果
    with conn.cursor() as cursor:
        cursor.execute(
            "SELECT * FROM ai_task_results WHERE task_id = %s",
            (task_id,)
        )
        results = cursor.fetchall()
    return results