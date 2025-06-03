"""
@Project ：audio-split-src 
@File    ：transcribe_curl.py
@IDE     ：PyCharm 
@Author  ：panshangguo
@Date    ：30/5/2025 上午10:38 
"""
import re
from datetime import datetime
from fastapi import HTTPException
from sqlalchemy import any_
from sqlalchemy.orm import Session
from models.schemas import Segment
from utils.mysql_db import AiTask, AiTaskResult


def create_task(db: Session, task_data):
    # 检查任务 ID 是否已经存在
    existing_task = db.query(AiTask).filter(AiTask.task_id == task_data["task_id"]).first()
    if existing_task:
        raise HTTPException(400, detail="任务 ID 已经存在")

    db_task = AiTask(
        task_id=task_data["task_id"],
        status=task_data["status"],
        message=task_data["message"],
        progress=task_data["progress"],
        original_path=task_data["original_path"],
        created_at=task_data.get("created_at", datetime.now())
    )
    db.add(db_task)
    db.commit()
    db.refresh(db_task)
    return db_task


def update_task_status(db: Session, task_data: dict):
    db_task = db.query(AiTask).filter(AiTask.task_id == task_data["task_id"]).first()
    if not db_task:
        return None

    db_task.status = task_data.get("status", db_task.status)
    db_task.message = task_data.get("message", db_task.message)
    db_task.progress = task_data.get("progress", db_task.progress)
    db_task.segments_path = task_data.get("segments_path", db_task.segments_path)
    db_task.start_time = task_data.get("start_time", db_task.start_time)
    db_task.complete_time = task_data.get("complete_time", db_task.complete_time)
    db_task.duration = task_data.get("duration", db_task.duration)
    db_task.error = task_data.get("error", db_task.error)

    if db_task.status in ["completed", "failed"]:
        db_task.complete_time = datetime.now()
        if db_task.start_time:
            start = db_task.start_time
            end = db_task.complete_time
            db_task.duration = round((end - start).total_seconds(), 2)

    db.commit()
    db.refresh(db_task)
    return db_task


def get_task(db: Session, task_id: str):
    return db.query(AiTask).filter(AiTask.task_id == task_id).first()


def get_task_results(db: Session, task_id: str, keyword: str = None, speaker: str = None, page: int = 1,
                     per_page: int = 10):
    query = db.query(AiTaskResult).filter(AiTaskResult.task_id == task_id, AiTaskResult.is_deleted == False)

    if keyword:
        keywords = re.split(r'[;,；，]', keyword)
        keywords = [k.strip() for k in keywords if k.strip()]
        conditions = []
        for keyword in keywords:
            conditions.append(AiTaskResult.text.like(f"%{keyword}%"))
            conditions.append(AiTaskResult.speaker.like(f"%{keyword}%"))
        query = query.filter(any_(conditions))

    if speaker:
        query = query.filter(AiTaskResult.speaker.like(f"%{speaker}%"))

    query = query.order_by(AiTaskResult.index)

    total = query.count()
    total_pages = (total + per_page - 1) // per_page if per_page > 0 else 1
    paginated_results = query.offset((page - 1) * per_page).limit(per_page).all()

    return {
        "items": paginated_results,
        "total": total,
        "page": page,
        "per_page": per_page,
        "total_pages": total_pages
    }


def delete_task_by_id(db: Session, task_id: str):
    db_task = db.query(AiTask).filter(AiTask.task_id == task_id).first()
    if not db_task:
        return None

    db.query(AiTaskResult).filter(AiTaskResult.task_id == task_id).update({AiTaskResult.is_deleted: True})
    db_task.status = "completed"

    db.commit()
    db.refresh(db_task)
    return db_task


def load_segments_if_completed(db: Session, task_id: str):
    """
    检查任务是否已完成，如果完成则加载结果
    """
    try:
        # 查询任务结果
        results = db.query(AiTaskResult).filter(
            AiTaskResult.task_id == task_id,
            AiTaskResult.is_deleted == False
        ).order_by(
            AiTaskResult.index
        ).all()

        # 将结果转换为 Segment 对象列表
        return [Segment(**val.__dict__) for val in results]
    except Exception as e:
        # 捕获异常并返回错误信息
        return f"Error loading segments: {str(e)}"


def delete_segments_by_keywords(db: Session, task_id: str, keyword: str):
    """
    删除指定任务中包含任意关键词的分段数据
    """
    # 检查任务是否存在及状态
    task = get_task(db, task_id)
    if not task:
        raise HTTPException(status_code=404, detail="任务不存在")
    if task.status != "completed":
        raise HTTPException(status_code=425, detail="任务尚未完成")

    # 解析关键词
    keywords = re.split(r'[;,；，]', keyword)
    keywords = [f"%{k.strip()}%" for k in keywords if k.strip()]
    if not keywords:
        raise HTTPException(status_code=400, detail="关键词不能为空")

    # 构建查询条件
    conditions = []
    params = []
    for k in keywords:
        conditions.append((AiTaskResult.text.like(k) | AiTaskResult.speaker.like(k)))
        params.extend([k, k])

    # 删除操作
    query = db.query(AiTaskResult).filter(
        AiTaskResult.task_id == task_id,
        *conditions
    )
    results = query.all()
    if not results:
        raise HTTPException(status_code=404, detail="未找到匹配的分段")

    query.delete(synchronize_session=False)
    db.commit()

    return len(results)


def get_speakers_list(db: Session, task_id: str):
    """
    获取指定任务的所有发音人列表
    """
    # 检查任务是否存在
    task = get_task(db, task_id)
    if not task:
        raise HTTPException(status_code=404, detail="任务不存在")

    # 查询发音人列表
    speakers = db.query(AiTaskResult.speaker).filter(
        AiTaskResult.task_id == task_id
    ).distinct().all()

    return [speaker[0] for speaker in speakers]


def delete_segments_by_indices(db: Session, task_id: str, segment_indices: list):
    """
    删除指定任务的多个分段数据
    """
    task = get_task(db, task_id)
    if not task:
        raise HTTPException(status_code=404, detail="任务不存在")
    if task.status != "completed":
        raise HTTPException(status_code=425, detail="任务尚未完成")
    # 删除操作
    query = db.query(AiTaskResult).filter(
        AiTaskResult.task_id == task_id,
        AiTaskResult.index.in_(segment_indices)
    )
    results = query.all()
    if not results:
        raise HTTPException(status_code=404, detail=f"任务 {task_id} 下指定的分段不存在")

    query.delete(synchronize_session=False)
    db.commit()

    return len(results)


def get_all_task_results(db: Session, task_id: str):
    """
    获取指定任务的所有结果片段
    """
    # 查询任务结果
    results = db.query(AiTaskResult).filter(
        AiTaskResult.task_id == task_id,
        AiTaskResult.is_deleted == False
    ).all()

    return results


def get_task_results_by_indices(db: Session, task_id: str, indices: list):
    """
    根据任务ID和索引列表获取任务结果片段
    """
    return db.query(AiTaskResult).filter(
        AiTaskResult.task_id == task_id,
        AiTaskResult.index.in_(indices),
        AiTaskResult.is_deleted == False
    ).all()


def get_task_result_by_index(db: Session, task_id: str, index: int):
    """
    根据任务ID和索引获取单个任务结果片段
    """
    return db.query(AiTaskResult).filter(
        AiTaskResult.task_id == task_id,
        AiTaskResult.index == index,
        AiTaskResult.is_deleted == False
    ).first()


def get_task_result(db: Session, task_id: str, segment_index: int):
    """
    根据任务ID和片段索引获取任务结果片段
    """
    return db.query(AiTaskResult).filter(
        AiTaskResult.task_id == task_id,
        AiTaskResult.index == segment_index,
        AiTaskResult.is_deleted == False
    ).first()


def get_task_all_results(db: Session, task_id: str):
    """
    获取指定任务的所有分段结果
    """
    return db.query(AiTaskResult).filter(
        AiTaskResult.task_id == task_id,
        AiTaskResult.is_deleted == False
    ).all()


def insert_task_results(db: Session, task_id: str, segments_paths: list, base_url: str):
    """
    批量插入任务结果片段
    """
    for idx, segment_path, merged_seg in segments_paths:
        url = segment_path.replace("\\", "/").replace("/root/autodl-fs", "")
        db.add(AiTaskResult(
            task_id=task_id,
            index=idx,
            start=merged_seg.get("start"),
            end=merged_seg.get("end"),
            text=merged_seg.get("text"),
            speaker=str(merged_seg.get("spk")),
            url=base_url + url
        ))
    db.commit()
