"""
@Project ：audio-split-src 
@File    ：file_utils.py
@IDE     ：PyCharm 
@Author  ：panshangguo
@Date    ：10/5/2025 下午3:16 
"""
import asyncio
import json
import os
import uuid
import shutil
import time

from utils.database import get_db_connection, get_task

TEMP_DIR = "temp_audio_files"


async def cleanup_task(task_id: str):
    await asyncio.sleep(3600 * 24 * 30)  # 30天后清理

    with get_db_connection() as conn:
        task = get_task(conn, task_id)
        if not task:
            return

        # 删除文件
        task_dir = os.path.dirname(task[4])
        shutil.rmtree(task_dir, ignore_errors=True)

        # 标记任务为已清理
        conn.execute("UPDATE tasks SET status = 'cleaned' WHERE task_id = ?", (task_id,))
        conn.commit()
