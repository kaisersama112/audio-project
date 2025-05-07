"""
@Project ：src 
@File    ：file_utils.py
@IDE     ：PyCharm 
@Author  ：panshangguo
@Date    ：29/4/2025 下午4:54 
"""
import json
import os
import uuid
import shutil
import time

TEMP_DIR = "temp_audio_files"


def cleanup_task(task_id: str):
    """清理临时文件"""
    time.sleep(3600)
    task_dir = os.path.join(TEMP_DIR, task_id)
    if os.path.exists(task_dir):
        shutil.rmtree(task_dir)
