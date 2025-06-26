from contextlib import asynccontextmanager, contextmanager
from datetime import datetime

from curd import models
from curd.models import AITask, AITaskResult

from sqlalchemy import or_
from sqlalchemy.orm import Session


# 获取数据库会话
""""""
@contextmanager
def get_db():
    SessionLocal = models.init_db()
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()



# 创建任务
def create_task(db: Session, task_data: dict):
    db_task = AITask(
        task_id=task_data["task_id"],
        status=task_data["status"],
        message=task_data["message"],
        progress=task_data["progress"],
        original_path=task_data["original_path"],
        created_at=datetime.now()
    )
    db.add(db_task)
    db.commit()
    db.refresh(db_task)
    return db_task


# 更新任务状态
def update_task_status(db: Session, task_data: dict):
    db_task = db.query(AITask).filter(AITask.task_id == task_data["task_id"]).first()
    if not db_task:
        return None

    # 更新任务字段
    db_task.status = task_data.get("status", db_task.status)
    db_task.message = task_data.get("message", db_task.message)
    db_task.progress = task_data.get("progress", db_task.progress)
    db_task.segments_path = task_data.get("segments_path", db_task.segments_path)
    db_task.start_time = task_data.get("start_time", db_task.start_time)
    db_task.complete_time = task_data.get("complete_time", db_task.complete_time)
    db_task.duration = task_data.get("duration", db_task.duration)
    db_task.error = task_data.get("error", db_task.error)

    # 如果任务状态为 completed 或 failed，更新完成时间
    if db_task.status in ["completed", "failed"] and not db_task.complete_time:
        db_task.complete_time = datetime.now()
        if db_task.start_time:
            duration = (db_task.complete_time - db_task.start_time).total_seconds()
            db_task.duration = round(duration, 2)

    db.commit()
    db.refresh(db_task)
    return db_task


# 获取任务
def get_task(db: Session, task_id: str):
    return db.query(AITask).filter(AITask.task_id == task_id).first()


def get_task_results(db: Session, task_id: str, keyword: str = None, speaker: str = None, page: int = 1,
                     per_page: int = 10):
    """

    """
    query = db.query(AITaskResult).filter(AITaskResult.task_id == task_id)

    # 筛选关键词
    if keyword:
        keywords = keyword.split(',')  # 假设关键词用逗号分隔
        conditions = []
        for kw in keywords:
            kw = kw.strip()
            if kw:
                conditions.append(
                    (AITaskResult.text.like(f"%{kw}%")) | (AITaskResult.speaker.like(f"%{kw}%"))
                )
        if conditions:
            # 使用 OR 组合多个关键词条件
            query = query.filter(or_(*conditions))

    # 筛选说话人
    if speaker:
        query = query.filter(AITaskResult.speaker.like(f"%{speaker}%"))

    # 排序和分页
    query = query.order_by(AITaskResult.index)
    total = query.count()
    results = query.offset((page - 1) * per_page).limit(per_page).all()

    return {
        "items": results,
        "total": total,
        "page": page,
        "per_page": per_page,
        "total_pages": (total + per_page - 1) // per_page if per_page > 0 else 1
    }


# 获取所有任务结果
def get_all_task_results(db: Session, task_id: str):
    return db.query(AITaskResult).filter(AITaskResult.task_id == task_id).all()


# 删除任务
def delete_task(db: Session, task_id: str):
    db.query(AITaskResult).filter(AITaskResult.task_id == task_id).delete()
    db.query(AITask).filter(AITask.task_id == task_id).delete()
    db.commit()


def get_segments_by_indices(db: Session, task_id: str, indices: list[int]):
    """
    根据原始索引列表获取对应的结果片段。
    用于 /download/bulk 接口判断是否真的有数据可以下载。
    """
    return (
        db.query(AITaskResult)
        .filter(
            AITaskResult.task_id == task_id,
            AITaskResult.index.in_(indices)
        )
        .all()
    )


def get_all_segments(db: Session, task_id: str):
    """
    获取某个任务下的所有结果片段。
    用于 /download/all 接口判断是否有数据可下载。
    """
    return (
        db.query(AITaskResult)
        .filter(AITaskResult.task_id == task_id)
        .all()
    )


def delete_segments_by_keywords(db: Session, task_id: str, keywords: list):
    """
    根据任务ID和关键词列表删除包含任意关键词的分段数据
    :param db: 数据库会话
    :param task_id: 任务ID
    :param keywords: 关键词列表
    :return: 被删除的分段索引列表
    """
    # 构建查询条件
    query = db.query(AITaskResult).filter(AITaskResult.task_id == task_id)

    # 使用 or_ 将多个关键词条件组合起来
    conditions = []
    for k in keywords:
        conditions.append((AITaskResult.text.like(k)) | (AITaskResult.speaker.like(k)))

    if conditions:
        query = query.filter(or_(*conditions))
    else:
        return []  # 没有关键词条件，直接返回空列表

    # 查询匹配的记录
    results = query.all()
    if not results:
        return []  # 没有匹配的记录

    # 记录被删除的分段索引
    deleted_indices = [seg.index for seg in results]

    # 删除数据库记录
    query.delete(synchronize_session=False)
    db.commit()

    return deleted_indices


def delete_segments_by_indices(db: Session, task_id: str, indices: list):
    """
    根据任务ID和分段索引列表删除分段数据
    :param db: 数据库会话
    :param task_id: 任务ID
    :param indices: 分段索引列表
    :return: 被删除的分段索引列表
    """
    # 查询要删除的分段
    results = db.query(AITaskResult).filter(
        AITaskResult.task_id == task_id,
        AITaskResult.index.in_(indices)
    ).all()

    if not results:
        return []

    # 记录被删除的分段索引
    deleted_indices = [seg.index for seg in results]

    # 删除数据库记录
    db.query(AITaskResult).filter(
        AITaskResult.task_id == task_id,
        AITaskResult.index.in_(indices)
    ).delete(synchronize_session=False)
    db.commit()

    return deleted_indices
